import contextlib
import fire
import flash_attn.flash_attn_interface
import flash_attn.ops.fused_dense
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mup
import numpy as np
import lib.ddp
import lib.datasets
import lib.models
import lib.ops
import lib.utils
import os
import time
import torch
import torch.distributed.optim
import torch.nn.functional as F
import tqdm
import wandb
from einops import rearrange
from torch import nn, optim, autograd
from torch.nn.parallel import DistributedDataParallel as DDP

def main(**args):
    args = lib.utils.AttributeDict(args)
    args.setdefault('batch_size', 256)
    args.setdefault('dataset', 'openwebtext2')
    args.setdefault('grad_accum_steps', 1)
    # Only run eval once, at the end of training
    args.setdefault('hook_freq', 85714)
    args.setdefault('lr', 1.4e-3)
    args.setdefault('lr_warmup_steps', 5000)
    args.setdefault('lr_decay', True)
    args.setdefault('print_freq', 1000)
    args.setdefault('save_weights', False)
    # Compensates for extra compute used by self-conditioning
    args.setdefault('steps', 85714)
    args.setdefault('weights_path', None)
    args.setdefault('reconst_weight', 1.0)
    args.setdefault('log_to_wandb', True)
    args.setdefault('dim', 384)
    args.setdefault('n_blocks', 16)
    args.setdefault('n_heads', 6)
    args.setdefault('t_min', 2.)
    args.setdefault('t_max', 300.)
    args.setdefault('embed_dim', 16)
    args.setdefault('seq_len', 256)
    args.setdefault('val_steps', 10000)
    args.setdefault('val_batch_size', 64)
    args.setdefault('weight_decay', 4e-5)
    args.setdefault('selfcond', True)
    args.setdefault('embed_init_std', 0.25)
    args.setdefault('clip_quantile', 0.95)

    lib.utils.print_args(args)

    if args.log_to_wandb and (lib.ddp.rank() == 0):
        wandb.init(
            project="simplex-diffusion",
            name=os.getcwd().split('/')[-1],
            config=args,
            settings=wandb.Settings(start_method="fork")            
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_device('cuda')

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp32/bf16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    class EmbeddingMatrix(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.matrix = nn.Parameter(torch.randn(vocab_size, embed_dim))
            self.matrix.data *= args.embed_init_std
        def forward(self):
            norm = torch.linalg.norm(self.matrix, dim=1, keepdim=True)
            return self.matrix / norm

    class LossCDF(nn.Module):
        def __init__(self, n_bins):
            super().__init__()
            self.l_t = nn.Parameter(torch.zeros([n_bins]) - float(np.log(n_bins)))
            self.l_u = nn.Parameter(torch.zeros([n_bins]) - float(np.log(n_bins)))
        def forward(self, t=None, u=None, normalized=True):
            """t.shape: [n]"""
            w_t = F.softmax(self.l_t, dim=0)
            w_t = w_t + 1e-3
            w_t = w_t / w_t.sum()
            w_u = self.l_u.exp()
            w_u = w_u + 1e-3
            if normalized:
                w_u = w_u / w_u.sum()
            e_t = torch.cat([torch.zeros([1]).cuda(), w_t.cumsum(dim=0)])
            e_u = torch.cat([torch.zeros([1]).cuda(), w_u.cumsum(dim=0)])
            if t is not None:
                t_prime = (t - args.t_min) / (args.t_max - args.t_min)
                t_idx = (e_t[None,:] <= t_prime[:,None]).long().sum(dim=1) - 1
                t_idx = t_idx.clamp(min=0, max=w_t.shape[0]-1)
                u = e_u[t_idx] + (e_u[t_idx+1] - e_u[t_idx])*((t_prime - e_t[t_idx])/(e_t[t_idx+1] - e_t[t_idx]))
                return u
            elif u is not None:
                u_idx = (e_u[None,:] <= u[:,None]).long().sum(dim=1) - 1
                u_idx = u_idx.clamp(min=0, max=w_u.shape[0]-1)
                t_prime = e_t[u_idx] + (e_t[u_idx+1] - e_t[u_idx])*((u - e_u[u_idx])/(e_u[u_idx+1] - e_u[u_idx]))
                t = t_prime * (args.t_max - args.t_min) + args.t_min
                return t

    class CondLayerNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.cond_linear = nn.Linear(128, dim)
            self.cond_linear.weight.data.zero_()
            self.cond_linear.bias.data.zero_()
            self.weight = nn.Parameter(torch.ones([dim]))
            self.dim = dim
        def forward(self, x, cond):
            with torch.cuda.amp.autocast(enabled=False):
                x = F.layer_norm(x.float(), [self.dim])
                bias = self.cond_linear(cond)
                x = (x * self.weight) + bias
            return x

    class CDCDTransformerBlock(nn.Module):
        def __init__(self, dim, n_heads, residual_scale):
            super().__init__()

            self.dim = dim
            self.n_heads = n_heads
            self.residual_scale = residual_scale

            self.norm1 = CondLayerNorm(dim)
            self.attn_qkv = nn.Linear(dim, 3*dim, bias=False)
            self.attn_out = nn.Linear(dim, dim, bias=False)

            self.norm2 = CondLayerNorm(dim)
            self.mlp = flash_attn.ops.fused_dense.FusedMLP(
                dim, 4*dim, bias1=False, bias2=False, checkpoint_lvl=1)

        def forward(self, x, t_embed, rotary_cos_sin, cu_seqlens=None):
            batch_size, seq_len = x.shape[0], x.shape[1]

            # Self-attention block
            x_skip = x
            x = self.norm1(x, t_embed)
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
                qkv, cu_seqlens, seq_len, 0., causal=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
            x = lib.models.residual_linear(
                x, self.attn_out.weight, x_skip, self.residual_scale
            )

            # Feedforward block
            x_skip = x
            x = self.norm2(x, t_embed)
            x = self.mlp(x)
            x = torch.add(x_skip, x, alpha=self.residual_scale)

            return x


    class CDCDModel(nn.Module):
        def __init__(self, dim, embed_dim, n_blocks, n_heads, vocab_size):
            super().__init__()

            self.input_linear = nn.Linear(embed_dim, dim, bias=False)
            self.selfcond_linear = nn.Linear(embed_dim, dim, bias=False)

            self.timestep_embed = nn.Sequential(
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU()
            )

            self.rotary_emb = lib.rotary.Rotary(dim // n_heads)

            residual_scale = float(1./np.sqrt(n_blocks))
            self.blocks = nn.ModuleList([
                CDCDTransformerBlock(dim, n_heads, residual_scale)
                for i in range(n_blocks)
            ])

            self.output_norm = nn.LayerNorm(dim)
            self.output_linear = mup.MuReadout(dim, vocab_size)
            self.output_linear.bias.data.zero_()

            self.dim = dim
            self.embed_dim = embed_dim
            self.vocab_size = vocab_size

        def forward(self, z, gamma, embedding_matrix, x_reconst):
            alpha_squared = torch.sigmoid(-gamma)[:,None,None]
            sigma_squared = torch.sigmoid(gamma)[:,None,None]
            alpha = alpha_squared.sqrt()

            t = (256. / (-gamma).exp()).sqrt()
            t_embed = torch.sin(
                torch.linspace(-6., 6., 128, device='cuda').exp()[None,:]
                * t[:,None]
            ).float()
            t_embed = self.timestep_embed(t_embed)[:,None,:]

            # Rescale input to stdev 1
            z_variance = (alpha_squared / self.embed_dim) + sigma_squared
            x = z / z_variance.sqrt().float()

            x = self.input_linear(x)
            x = x + self.selfcond_linear(x_reconst) * float(np.sqrt(self.embed_dim))

            rotary_cos_sin = self.rotary_emb(x)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, cache_enabled=False):
                for i in range(len(self.blocks)):
                    x = self.blocks[i](x, t_embed, rotary_cos_sin)

            x = self.output_norm(x.float())
            logits = self.output_linear(x)
            x_reconst = F.softmax(logits, dim=2) @ embedding_matrix

            return logits, x_reconst


    dataset = lib.datasets.REGISTRY[args.dataset](
        args.batch_size, args.val_batch_size, args.seq_len
    )
    (train_iterator,val_iterator,test_iterator), (word2idx, idx2word) = dataset

    vocab_size = len(word2idx)
    print(f'vocab_size: {vocab_size}')

    @torch.jit.script
    def gaussian_kl(mu_p, sigma_p, mu_q, sigma_q):
        """KL(p||q)"""
        return (
            sigma_q.log() - sigma_p.log()
            + (sigma_p**2 + (mu_p - mu_q)**2)/(2*sigma_q**2)
            - 0.5
        )

    def create_modules(dim, n_heads):
        return {
            'loss_cdf': LossCDF(100).float(),
            'embedding_matrix': EmbeddingMatrix(vocab_size, args.embed_dim).float(),
            'model': CDCDModel(dim, args.embed_dim, args.n_blocks, n_heads, vocab_size).float()
        }
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.cuda()
        print(key+':')
        lib.utils.print_model(main)

    if args.weights_path is not None:
        print('Loading weights...')
        for name, module in modules.items():
            module.load_state_dict(torch.load(
                os.path.join(args.weights_path, f'{name}.pt'),
                map_location=torch.device('cuda')
            ))

    ddp_modules = {
        name: DDP(module)
        for name, module in modules.items()
    }

    def forward(x=None, selfcond_iters=1):
        if x is None:
            x = next(train_iterator)

        embedding_matrix = ddp_modules['embedding_matrix']()

        reconst_bs = 1
        u = torch.empty([x.shape[0]], device='cuda')
        # First entries of t are used for reconst_loss below
        u[:reconst_bs] = 0
        # Low-discrepancy sampler for the remaining entries of t
        u[reconst_bs:] = torch.arange(
            x.shape[0] - reconst_bs, device='cuda')
        u[reconst_bs:] += torch.rand(1, device='cuda')
        u[reconst_bs:] /= float(x.shape[0] - reconst_bs)
        u.requires_grad = True
        with torch.enable_grad():
            t = modules['loss_cdf'](u=u, normalized=True)
            gamma = -torch.log(256. / t**2)
            gamma_prime = autograd.grad(gamma.sum(), [u], create_graph=True)[0]
            t, gamma, gamma_prime = t.detach(), gamma.detach(), gamma_prime.detach()

        # Quantities derived from gamma and gamma_prime:
        alpha_squared = torch.sigmoid(-gamma)
        sigma_squared = torch.sigmoid(gamma)
        alpha = alpha_squared.sqrt()
        sigma = sigma_squared.sqrt()
        snr_prime = -(-gamma).exp() * gamma_prime # SNR = exp(-gamma)
        gamma_1 = -(torch.tensor(256. / args.t_max**2).cuda().log())
        alpha_1 = torch.sigmoid(-gamma_1).sqrt()
        sigma_1 = torch.sigmoid(gamma_1).sqrt()

        # Construct z (with reparam. trick gradients)
        x_embed = embedding_matrix[x]
        z = torch.randn(
            [x.shape[0], x.shape[1], args.embed_dim],
            dtype=torch.float32, device='cuda'
        )
        z.mul_(sigma[:,None,None])
        z.add_(alpha[:,None,None] * x_embed)

        # Model forward pass
        x_reconst = torch.zeros_like(z)
        for i in range(selfcond_iters):
            with torch.set_grad_enabled(i == selfcond_iters - 1):
                logits, x_reconst = ddp_modules['model'](
                    z, gamma, embedding_matrix, x_reconst
                )

        xent = lib.ops.cross_entropy(logits, x).mean(dim=1).double()
        reconst_loss = xent[:reconst_bs]

        prior_loss = gaussian_kl(
            (alpha_1 * x_embed),
            sigma_1,
            torch.tensor(0., device='cuda'),
            torch.tensor(1., device='cuda')
        ).sum(dim=2).mean()

        diffusion_loss = xent

        nll_diffusion_loss = (x_embed - x_reconst).pow(2)
        nll_diffusion_loss = nll_diffusion_loss.mean(dim=1).double().sum(dim=1)
        nll_diffusion_loss = -0.5*(snr_prime * nll_diffusion_loss)

        grad_hook_loss = diffusion_loss # Used above (weird variable scope)
        nll = reconst_loss.mean() + prior_loss + nll_diffusion_loss[reconst_bs:].mean()
        loss = diffusion_loss[reconst_bs:].mean()

        # CDF loss
        t2 = t.clone()
        t2.requires_grad = True
        with torch.enable_grad():
            xent_pred = modules['loss_cdf'](t=t2, normalized=False)
            imp_weights = 1. / autograd.grad(xent_pred.sum(), [t2])[0]
        imp_weights = imp_weights.detach() * 1e-5
        cdf_loss = (imp_weights * (modules['loss_cdf'](t=t, normalized=False) - xent.detach()).pow(2)).mean()
        loss = loss + cdf_loss

        return (
            loss,
            nll,
            reconst_loss.mean(),
            prior_loss,
            cdf_loss,
            torch.tensor(reconst_bs).cuda(),
        )

    def train_forward(*_):
        if args.selfcond:
            x = next(train_iterator)
            results_1 = forward(x[::2], selfcond_iters=1)
            results_2 = forward(x[1::2], selfcond_iters=2)
            return (
                0.5 * (results_1[0] + results_2[0]), # loss
                *results_2[1:] # NLL and all other results
            )
        else:
            return forward()

    learning_rates = {
        'model': args.lr,
        'loss_cdf': 1e-2,
        'embedding_matrix': 1e-2,
    }

    weight_decays = {
        'model': args.weight_decay,
        'loss_cdf': 0,
        'embedding_matrix': 0.,
    }

    def optimizer_impl(param_groups, **kwargs):
        assert('weight_decay' not in kwargs)
        modules_seen = set()
        for i, param_group in enumerate(param_groups):
            weight_decay_set = False
            for name in modules:
                group_params = param_group['params']
                module_params = list(modules[name].parameters())
                if all([any([p is p2 for p2 in module_params]) for p in group_params]):
                    assert(not weight_decay_set)
                    assert(param_group['weight_decay'] == 0.)
                    param_group['weight_decay'] = (
                        weight_decays[name] / (param_group['lr']+1e-16)
                    )
                    weight_decay_set = True
                    modules_seen.add(name)
            assert(weight_decay_set)
        assert(all([name in modules_seen for name in modules]))

        return torch.distributed.optim.ZeroRedundancyOptimizer(param_groups,
            optimizer_class=optim.AdamW, **kwargs)

    param_groups = [
        {'params': modules[name].parameters(), 'lr': learning_rates[name]}
        for name in modules
    ]
    opt = mup.MuAdam(param_groups, impl=optimizer_impl, betas=(0.9, 0.99))

    def compute_nll(data_iterator, steps, selfcond_iters=1):
        with torch.no_grad():
            total_nll = 0.
            total_tokens = 0
            for i, X in enumerate(data_iterator):
                X = X.cuda()
                nll = forward(x=X, selfcond_iters=selfcond_iters)[1]
                total_nll += (nll.item() * args.seq_len * X.shape[0])
                total_tokens += args.seq_len * X.shape[0]
                if i == steps:
                    break

        return total_nll / total_tokens

    def hook(step):
        if step % args.hook_freq == (args.hook_freq - 1):
            for selfcond_iters in [1,2,4,8]:
                val_nll = compute_nll(val_iterator, args.val_steps, selfcond_iters)
                print(f'NLL (val, selfcond_iters={selfcond_iters}): {val_nll}')
                if not args.selfcond:
                    break

            if lib.ddp.rank() == 0:
                # Save weights
                if args.save_weights:
                    for name in modules:
                        torch.save(modules[name].state_dict(), f'{name}.pt')

                # Save time warping plots
                with torch.no_grad():
                    t = torch.linspace(args.t_min, args.t_max, 1024).cuda()
                    u = modules['loss_cdf'](t=t, normalized=False)
                    plt.clf()
                    plt.plot(t.detach().cpu().numpy(), u.detach().cpu().numpy())
                    plt.savefig(f'loss_cdf_{step}.jpg')

                    u = torch.linspace(0., 1., 1024).cuda()
                    t = modules['loss_cdf'](u=u, normalized=True)
                    gamma = -torch.log(256. / t**2)
                    plt.clf()
                    plt.plot(u.detach().cpu().numpy(), gamma.detach().cpu().numpy())
                    plt.savefig(f'gamma_{step}.jpg')


    print('Starting train loop...')
    lib.utils.train_loop(
        train_forward,
        opt,
        args.steps,
        names=['nll','reconst','prior','cdf_loss','reconst_bs'],
        hook=hook,
        print_freq=args.print_freq,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        amp_grad_scaler=False,
        grad_accum_steps=args.grad_accum_steps,
        ddp_models=ddp_modules.values(),
        clip_params=[
            param
            for module in modules.values()
            for param in module.parameters()
        ],
        log_to_wandb=args.log_to_wandb,
        clip_quantile=args.clip_quantile,
    )

    if args.log_to_wandb and (lib.ddp.rank() == 0):
        wandb.finish(quiet=True)

if __name__ == '__main__':
    fire.Fire(lib.ddp.wrap_main(main))