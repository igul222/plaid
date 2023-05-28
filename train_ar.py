import contextlib
import fire
import functools
import mup
import numpy as np
import lib.ddp
import lib.ema
import lib.datasets
import lib.models
import lib.ops
import lib.utils
import os
import time
import torch
import torch.nn.functional as F
import tqdm
import wandb
from torch import nn, optim, autograd
from torch.nn.parallel import DistributedDataParallel as DDP

def main(**args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = lib.utils.AttributeDict(args)
    args.setdefault('batch_size', 64)
    args.setdefault('dataset', 'openwebtext2')
    args.setdefault('grad_accum_steps', 1)
    args.setdefault('hook_freq', 10000)
    args.setdefault('lr', 8e-3)
    args.setdefault('lr_warmup_steps', 1000)
    args.setdefault('lr_decay', True)
    args.setdefault('print_freq', 1000)
    args.setdefault('save_weights', False)
    args.setdefault('steps', 104000)
    args.setdefault('weights_path', None)
    args.setdefault('log_to_wandb', True)
    args.setdefault('dim', 768)
    args.setdefault('n_blocks', 12)
    args.setdefault('n_heads', 12)
    args.setdefault('seq_len', 256)
    args.setdefault('val_steps', 1000)
    args.setdefault('val_batch_size', 64)
    args.setdefault('weight_decay', 4e-5)
    args.setdefault('ema', 0.)
    args.setdefault('tie_embeddings', False)

    lib.utils.print_args(args)

    if args.log_to_wandb and (lib.ddp.rank() == 0):
        wandb.init(
            project="simplex-diffusion",
            name=os.getcwd().split('/')[-1],
            config=args
        )

    dataset = lib.datasets.REGISTRY[args.dataset](
        args.batch_size, args.val_batch_size, args.seq_len
    )
    (train_iterator,val_iterator,test_iterator), (word2idx, idx2word) = dataset

    seq_len = args.seq_len
    vocab_size = len(word2idx)
    print(f'seq_len: {seq_len}, vocab_size: {vocab_size}')

    model       = lib.models.AutoregressiveModel(args.dim, args.n_blocks, args.n_heads, vocab_size, args.tie_embeddings)
    base_model  = lib.models.AutoregressiveModel(256,      args.n_blocks, 4,            vocab_size, args.tie_embeddings)
    delta_model = lib.models.AutoregressiveModel(128,      args.n_blocks, 2,            vocab_size, args.tie_embeddings)
    mup.set_base_shapes(model, base_model, delta=delta_model)
    model = model.cuda()

    lib.utils.print_model(model)

    if args.weights_path is not None:
        model.load_state_dict(torch.load(
            os.path.join(args.weights_path, 'model.pt')
        ))

    ddp_model = DDP(model)

    ema = lib.ema.EMA(model, args.ema)

    def forward(*_):
        X = next(train_iterator).cuda().long()
        logits = ddp_model(X)
        loss = lib.ops.cross_entropy(logits, X).mean()
        return loss

    def compute_nll(data_iterator, steps, eval_seq_len=seq_len):
        with torch.no_grad():
            with ema.enabled():
                total_nll = 0.
                total_tokens = 0
                for i, X in enumerate(data_iterator):
                    X = X.cuda()[:,:eval_seq_len]
                    logits = ddp_model(X)
                    loss = lib.ops.cross_entropy(logits, X).mean()
                    total_nll += loss.item() * X.numel()
                    total_tokens += X.numel()
                    if i == steps:
                        break
        return total_nll / total_tokens

    all_val_nlls = []
    def hook(step):
        ema.step()
        if step % args.hook_freq == args.hook_freq - 1:
            for eval_seq_len in [256, 1024]:
                val_nll = compute_nll(val_iterator, args.val_steps, eval_seq_len)
                print(f'NLL (val, seq len {eval_seq_len}): {val_nll}')
                if eval_seq_len == seq_len:
                    all_val_nlls.append(val_nll)

            if (lib.ddp.rank() == 0) and args.save_weights:
                torch.save(model.state_dict(), 'model.pt')

    def impl(param_groups, **kwargs):
        assert('weight_decay' not in kwargs)
        for param_group in param_groups:
            param_group['weight_decay'] = (
                args.weight_decay / (param_group['lr'] + 1e-16)
            )
        return optim.AdamW(param_groups, **kwargs)
    opt = mup.MuAdam(
        model.parameters(),
        impl=impl,
        lr=args.lr,
        betas=(0.9, 0.99)
    )

    lib.utils.train_loop(
        forward,
        opt,
        args.steps,
        hook=hook,
        print_freq=args.print_freq,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        amp_grad_scaler=False,
        grad_accum_steps=args.grad_accum_steps,
        ddp_models=[ddp_model],
        clip_params=[
            param for param in model.parameters()
        ],
        log_to_wandb=args.log_to_wandb
    )

    final_val_nll = compute_nll(val_iterator, 3000)
    print('Final val NLL:', final_val_nll)

    return all_val_nlls, final_val_nll

if __name__ == '__main__':
    fire.Fire(lib.ddp.wrap_main(main))