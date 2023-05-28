import apex.normalization
import flash_attn.flash_attn_interface
import flash_attn.ops.fused_dense
import lib.utils
import mup
import numpy as np
import lib.rotary
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from einops import rearrange
from torch import nn, optim

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]

def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, causal, residual_scale):
        super().__init__()

        self.causal = causal
        self.dim = dim
        self.n_heads = n_heads
        self.residual_scale = residual_scale

        self.rmsnorm1 = apex.normalization.FusedRMSNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.rmsnorm2 = apex.normalization.FusedRMSNorm(dim)
        self.mlp = flash_attn.ops.fused_dense.FusedMLP(
            dim, 4*dim, bias1=False, bias2=False, checkpoint_lvl=1)

    def forward(self, x, rotary_cos_sin, cu_seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Self-attention block
        x_skip = x
        x = self.rmsnorm1(x)
        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3, h=self.n_heads
        )
        half_dtype = qkv.dtype
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = lib.rotary.apply_rotary_pos_emb(
                qkv, cos.to(half_dtype), sin.to(half_dtype)
            )
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        x = flash_attn.flash_attn_interface.flash_attn_unpadded_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=self.causal)
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        x = residual_linear(
            x, self.attn_out.weight, x_skip, self.residual_scale
        )

        # Feedforward block
        x_skip = x
        x = self.rmsnorm2(x)
        x = self.mlp(x)
        x = torch.add(x_skip, x, alpha=self.residual_scale)

        return x

class EmbeddingMatrix(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.matrix.data /= self.matrix.data.norm(p=2, dim=1, keepdim=True)
    def forward(self):
        norm = torch.linalg.norm(self.matrix, dim=1, keepdim=True)
        return (self.matrix / (norm + 1e-8))

class NoiseSchedule(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(1024, 1))
        self.b1 = nn.Parameter(torch.randn(1024))
        self.W2 = nn.Parameter(torch.randn(1, 1024))
    def forward(self, t):
        """t.shape: [n]"""
        W1 = F.softplus(self.W1.double())
        W2 = 0.01 * F.softplus(self.W2.double())
        def gamma_tilde(t):
            h = t[:,None] - 0.5
            h = (h @ W1.T) + self.b1[None,:].double()
            h = torch.tanh(h)
            h = (h @ W2.T)[:,0]
            return h
        gamma_tilde_0 = gamma_tilde(torch.tensor([0.], device='cuda'))
        gamma_tilde_1 = gamma_tilde(torch.tensor([1.], device='cuda'))
        gamma_tilde_t = gamma_tilde(t)
        return (
            (gamma_tilde_t - gamma_tilde_0) /
            (gamma_tilde_1 - gamma_tilde_0)
        )

class GammaBounds(nn.Module):
    def __init__(self, gamma_0, gamma_1):
        super().__init__()
        self.gamma_0 = nn.Parameter(torch.tensor(float(gamma_0)))
        self.gamma_1 = nn.Parameter(torch.tensor(float(gamma_1)))
    def forward(self):
        return self.gamma_0.clone().double(), self.gamma_1.clone().double()

class DiffusionModel(nn.Module):
    def __init__(self, dim, embed_dim, n_blocks, n_heads, vocab_size):
        super().__init__()

        self.input_linear = nn.Linear(embed_dim, dim, bias=False)
        self.selfcond_linear = nn.Linear(embed_dim, dim, bias=False)
        self.selfcond_linear.weight.data.zero_()
        self.gamma_linear = nn.Linear(64, dim, bias=False)
        self.gamma_linear.weight.data.zero_()

        self.rotary_emb = lib.rotary.Rotary(dim // n_heads)

        residual_scale = float(1./np.sqrt(n_blocks))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, False, residual_scale)
            for i in range(n_blocks)
        ])

        self.output_norm = lib.models.LayerNorm(dim)
        self.output_linear = mup.MuReadout(dim, vocab_size)
        self.output_linear.weight.data.zero_()
        self.output_linear.bias.data.zero_()

        self.dim = dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def forward(self, z, gamma, embedding_matrix, bias_scale, x_selfcond,
        selfcond_mask=None, cu_seqlens=None):

        if selfcond_mask is None:
            selfcond_mask = torch.ones(z.shape[0], device='cuda')

        alpha_squared = torch.sigmoid(-gamma)[:,None,None]
        sigma_squared = torch.sigmoid(gamma)[:,None,None]
        alpha = alpha_squared.sqrt()

        # Rescale input to stdev 1
        z_variance = (alpha_squared / self.embed_dim) + sigma_squared
        x = z / z_variance.sqrt().float()

        x = self.input_linear(x)

        x = x + self.selfcond_linear(
            x_selfcond * float(np.sqrt(self.embed_dim))
        )

        gamma_embed = torch.linspace(-5., 5., 64 // 2, device='cuda')
        gamma_embed = gamma_embed.exp()[None,:] * gamma[:,None]
        gamma_embed = torch.cat([gamma_embed.sin(), gamma_embed.cos()], dim=1)
        gamma_embed = self.gamma_linear(gamma_embed.float())[:,None,:]
        x = x + gamma_embed

        rotary_cos_sin = self.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, cu_seqlens=cu_seqlens)

        x = self.output_norm(x.float())

        x *= self.output_linear.output_mult/self.output_linear.width_mult()

        W = torch.cat([
            self.output_linear.weight.T,
            embedding_matrix.T,
            embedding_matrix.T.detach()
        ], dim=0)
        z_scaled_for_bias = bias_scale * (alpha/sigma_squared).float() * z
        x = torch.cat([
            x,
            z_scaled_for_bias * (1 - selfcond_mask.float()[:,None,None]),
            z_scaled_for_bias * selfcond_mask.float()[:,None,None]
        ], dim=2)
        logits = torch.addmm(
            self.output_linear.bias.view(1, self.vocab_size),
            x.view(-1, self.dim + 2*self.embed_dim),
            W.view(self.dim + 2*self.embed_dim, self.vocab_size)
        ).view(x.shape[0], x.shape[1], self.vocab_size)

        # Comment for 'no categorical reparameterization' ablation
        x_reconst = F.softmax(logits, dim=2)
        x_reconst = x_reconst @ torch.cat([
            embedding_matrix, embedding_matrix.detach()], dim=1)
        x_reconst = torch.lerp(
            x_reconst[:,:,:self.embed_dim],
            x_reconst[:,:,self.embed_dim:],
            selfcond_mask.float()[:,None,None]
        )

        return logits, x_reconst

class AutoregressiveModel(nn.Module):
    def __init__(self, dim, n_blocks, n_heads, vocab_size, tie_embeddings):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        if not tie_embeddings:
            self.input_embedding = nn.Embedding(vocab_size, dim)
        self.rotary_emb = lib.rotary.Rotary(dim // n_heads)

        residual_scale = float(1./np.sqrt(n_blocks))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, True, residual_scale)
            for i in range(n_blocks)
        ])
        self.output_norm = apex.normalization.FusedRMSNorm(dim)
        self.output_linear = mup.MuReadout(dim, vocab_size)
        self.first_token_logits = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        if self.tie_embeddings:
            x = F.embedding(x, self.output_linear.weight) * float(np.sqrt(3*256))
        else:
            x = self.input_embedding(x)
        rotary_cos_sin = self.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin)
        x = x.float()
        x = self.output_norm(x)
        logits = self.output_linear(x)
        logits = torch.cat([
            self.first_token_logits[None,None,:].expand(x.shape[0],-1,-1),
            logits[:,:-1,:]
        ], dim=1)
        return logits