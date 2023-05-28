# Plaid: Likelihood-Based Diffusion Language Models

This repository contains code for training and evaluating the models in the paper *Likelihood-Based Diffusion Language Models*.

## Generating samples from Plaid 1B

First download the weights from here and extract them:

```
cat plaid1b_weights.tar.gz.* | tar xvzf -
```

Then run the sampling code:

```
python sample.py --weights_path=/path/to/plaid1b_weights --dim=2048 --n_blocks=24 --n_heads=32 --seq_len=1024
```

## Computing zero-shot likelihoods

This repository supports computing zero-shot likelihoods on six datasets: Penn TreeBank, enwik8, text8, WikiText2, WikiText103, and the 1 Billion Word corpus.
To compute likelihood for one of these datasets, specify the dataset path in the corresponding constant at the top of `lib/datasets.py`. Then run this command (e.g. for WikiText103):

```
python train.py --weights_path=/path/to/plaid1b_weights --dim=2048 --n_blocks=24 --n_heads=32 --seq_len=1024 --dataset=wikitext103
```

## Training Plaid models

First, download OpenWebText2 from [here](https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar) and update the `OPENWEBTEXT2_DATA_DIR` constant in `lib/datasets.py` with the path to the extracted files.

Then run the OpenWebText2 preprocessing script:

```
python -m misc.owt2_preprocess --data_dir=/path/to/openwebtext2
```

Finally, run the training script:

```
python train.py
```