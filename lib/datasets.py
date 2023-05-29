import collections
import orjson
import functools
import lib.ddp
import numpy as np
import os
import random
import itertools
import re
import socket
import subprocess
import torch
import torch.nn.utils.rnn
from tokenizers import Tokenizer

# Generate using misc/owt2_preprocess.py
OPENWEBTEXT2_DATA_DIR = '/REPLACE_ME'
# https://mattmahoney.net/dc/enwik8.zip
ENWIK8_PATH           = '/REPLACE_ME/data/enwik8'
# https://mattmahoney.net/dc/text8.zip
TEXT8_PATH            = '/REPLACE_ME/data/text8'
# https://github.com/wojzaremba/lstm/blob/master/data/ptb.test.txt
PTB_PATH              = '/REPLACE_ME/data/ptb/ptb.test.txt'
# https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
WIKITEXT2_PATH        = '/REPLACE_ME/data/wikitext2.test.tokens'
WIKITEXT103_PATH      = '/REPLACE_ME/data/wikitext103.test.tokens'
# https://www.statmt.org/lm-benchmark/
BILLIONWORD_PATH      = '/REPLACE_ME/billionword_test_all.txt'

def _openwebtext2_shard_iterator(shard_name):
    data_dir = OPENWEBTEXT2_DATA_DIR
    path = os.path.join(data_dir, f'en_shuffled_{shard_name}.jsonl')

    with open(path, 'r') as f:
        num_lines = sum(1 for _ in f)
        start_line = random.randint(0, num_lines - 1)
        f.seek(0)

        for i, line in enumerate(itertools.islice(f, start_line, None)):
            yield orjson.loads(line[:-1])

        # Continue reading from the beginning of the file, up to the initial start_line
        f.seek(0)
        for i, line in enumerate(itertools.islice(f, start_line)):
            yield orjson.loads(line[:-1])


def openwebtext2_train_iterator(infinite=True, rank=0, world_size=1):
    n_shards = 8
    if (world_size % 8) == 0:
        rank = rank % 8
        world_size = 8
    assert(n_shards % world_size == 0)
    offset = rank * (n_shards // world_size)
    shards = [f'train_{(i + offset) % n_shards}' for i in range(n_shards)]
    while True:
        for shard_name in shards:
            for x in _openwebtext2_shard_iterator(shard_name):
                yield x
        if not infinite:
            break

def _openwebtext2_val_iterator():
    while True:
        for x in _openwebtext2_shard_iterator('val'):
            yield x

def _rolling_shuffle(iterator, buffer_size):
    # set seed if you want deterministic shuffling
    rng = np.random.RandomState()
    buffer = []
    for x1 in iterator:
        if len(buffer) < buffer_size:
            buffer.append(x1)
        else:
            idx = rng.randint(0, buffer_size - 1)
            x2, buffer[idx] = buffer[idx], x1
            yield x2
    rng.shuffle(buffer)
    for x in buffer:
        yield x

def _tokenize(iterator, tokenizer):
    batch_size = 64
    batch = [None] * batch_size
    for i, x in enumerate(iterator):
        i = i % batch_size
        batch[i] = x
        if i == (batch_size - 1):
            tokenized = tokenizer.encode_batch(batch)
            for x in tokenized:
                yield torch.tensor(x.ids)


def _to_chunks(iterator, chunk_size):
    buffer = torch.tensor([], dtype=torch.int64)
    eot_token = torch.tensor([0], dtype=torch.int64)
    for x in iterator:
        buffer = torch.cat([buffer, x, eot_token], dim=0)
        while buffer.shape[0] >= chunk_size:
            yield buffer[:chunk_size].clone()
            buffer = buffer[chunk_size:]

def _batch(iterator, batch_size):
    batch = [None] * batch_size
    for i, x in enumerate(iterator):
        i = i % batch_size
        batch[i] = x
        if i == (batch_size - 1):
            yield torch.stack(batch)

class _OWT2IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, train, batch_size, seq_len, tokenizer):
        self.args = (train, batch_size, seq_len, tokenizer)
        self.rank = lib.ddp.rank()
        self.world_size = lib.ddp.world_size()
    def __iter__(self):
        train, batch_size, seq_len, tokenizer = self.args
        if train:
            iterator = openwebtext2_train_iterator(
                rank=self.rank, world_size=self.world_size
            )
        else:
            iterator = _openwebtext2_val_iterator()
        iterator = _rolling_shuffle(iterator, 10)
        iterator = _tokenize(iterator, tokenizer)
        iterator = _to_chunks(iterator, seq_len)
        if 'OWT2_DEBUG_MODE' in os.environ:
            iterator = _rolling_shuffle(iterator, 2_560_000 // seq_len)
        else:
            iterator = _rolling_shuffle(iterator, 256_000_000 // seq_len)
        iterator = _batch(iterator, batch_size)
        return iterator

def openwebtext2_tokenizer():
    data_dir = OPENWEBTEXT2_DATA_DIR
    tokenizer_path = os.path.join('misc/owt2_tokenizer.json')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def openwebtext2(batch_size, val_batch_size, seq_len):
    if seq_len is None:
        seq_len = 1024

    tokenizer = openwebtext2_tokenizer()
    word2idx = {k.encode('utf-8'):v for k,v in tokenizer.get_vocab().items()}
    idx2word = {v:k for k,v in word2idx.items()}

    train_iterator = iter(torch.utils.data.DataLoader(
        _OWT2IterableDataset(True, batch_size, seq_len, tokenizer),
        batch_size=None, num_workers=1, prefetch_factor=1024//batch_size))

    val_iterator = iter(torch.utils.data.DataLoader(
        _OWT2IterableDataset(False, val_batch_size, seq_len, tokenizer),
        batch_size=None, num_workers=1, prefetch_factor=1024//batch_size))

    test_iterator = iter(torch.utils.data.DataLoader(
        _OWT2IterableDataset(False, val_batch_size, seq_len, tokenizer),
        batch_size=None, num_workers=1, prefetch_factor=1024//batch_size))

    return (train_iterator, val_iterator, test_iterator), (word2idx, idx2word)

def ptb_untokenized():
    with open(PTB_PATH, 'r') as f:
        dataset = f.read()[:-1] # drop trailing newline
    total_words = len(dataset.split(' '))
    # Verified invertible
    def detokenize(x):
        x = x.replace(" 's", "'s")
        x = x.replace("s ' ", "s' ")
        x = x.replace(" n't", "n't")
        x = x.replace(" \n ", "\n")
        x = x.replace("\\/", "/")
        for _ in range(10):
            x = x.replace(" N ", " 1 ")
        x = x.replace("$ 1", "$1")
        x = x.replace("# 1", "#1")
        x = x.replace("<unk>", "?")
        return x
    dataset = detokenize(dataset)
    return dataset, total_words

def _wikitext_untokenized(wikitext_path):
    with open(wikitext_path, 'r') as f:
        dataset = f.read()[1:] # [1:] to drop an initial space
    total_words = len(dataset.split(' '))
    # From https://github.com/EleutherAI/megatron-3d
    def detokenizer(string):
        # contractions
        string = string.replace("s '", "s'")
        string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
        # number separators
        string = string.replace(" @-@ ", "-")
        string = string.replace(" @,@ ", ",")
        string = string.replace(" @.@ ", ".")
        # punctuation
        string = string.replace(" : ", ": ")
        string = string.replace(" ; ", "; ")
        string = string.replace(" . ", ". ")
        string = string.replace(" ! ", "! ")
        string = string.replace(" ? ", "? ")
        string = string.replace(" , ", ", ")
        # double brackets
        string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
        string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
        string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
        string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
        string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
        # miscellaneous
        string = string.replace("= = = =", "====")
        string = string.replace("= = =", "===")
        string = string.replace("= =", "==")
        string = string.replace(" " + chr(176) + " ", chr(176))
        string = string.replace(" \n", "\n")
        string = string.replace("\n ", "\n")
        string = string.replace(" N ", " 1 ")
        string = string.replace(" 's", "'s")
        return string
    dataset = detokenizer(dataset)
    return dataset, total_words

def wikitext2_untokenized():
    return _wikitext_untokenized(WIKITEXT2_PATH)

def wikitext103_untokenized():
    return _wikitext_untokenized(WIKITEXT103_PATH)

def enwik8_untokenized():
    with open(ENWIK8_PATH, 'r') as f:
        dataset = f.read()[-5_000_000:]
    return dataset, len(dataset)

def text8_untokenized():
    with open(TEXT8_PATH, 'r') as f:
        dataset = f.read()[-5_000_000:]
    return dataset, len(dataset)

def billionword_untokenized():
    with open(BILLIONWORD_PATH, 'r') as f:
        dataset = f.read()[:-1]
    total_words = len(dataset.split())

    def retokenize(x):
        x, _ = subprocess.Popen(
            'misc/bw_tokenizer/normalize-punctuation.perl -l en | misc/bw_tokenizer/tokenizer.perl -l en',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        ).communicate(x)
        return x[:-1] # tokenizer.py emits a trailing newline; drop it.

    def detokenize(orig):
        # Apply detokenizer heuristics to each line
        orig_lines = orig.split("\n")
        detok_lines = []
        for x in orig_lines:
            x = x.replace('http : / / ', 'http://')
            x = x.replace('https : / / ', 'https://')
            x = re.sub(r' \'(\w+)', r"'\1", x)
            x = re.sub(r' (\w+) \. ', r' \1. ', x)
            x = re.sub(r' (\w+) \.$', r' \1.', x)
            x = x.replace(' ? ', '? ')
            x = re.sub(r' \?$', '?', x)
            x = x.replace(' ! ', '! ')
            x = re.sub(r' \!$', '!', x)
            x = x.replace(' , ', ', ')
            x = x.replace(' : ', ': ')
            x = x.replace(' ; ', '; ')
            x = x.replace(' / ', '/')
            x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
            x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
            x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
            x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
            x = x.replace('$ ', '$')
            x = x.replace('£ ', '£')
            detok_lines.append(x)
        # Guarantee invertibility by verbatim-copying lines which aren't retokenized
        # correctly, with an escape sequence prepended
        retok_lines = retokenize("\n".join(detok_lines)).split("\n")
        assert(len(orig_lines) == len(detok_lines) == len(retok_lines))
        detok_lines = [
            detok if (retok == orig) else ("$$$ "+orig)
            for orig, detok, retok in zip(orig_lines, detok_lines, retok_lines)
        ]
        print('billionword detokenized:',
            np.mean([(retok == orig) for orig, retok in zip(orig_lines, retok_lines)])
        )
        return "\n".join(detok_lines)

    dataset = detokenize(dataset)
    return dataset, total_words

def _eval_dataset(
    dataset_name,
    batch_size,
    val_batch_size,
    seq_len
    ):
    assert(val_batch_size == 1)

    untokenized_dataset, total_words = UNTOKENIZED_REGISTRY[dataset_name]()

    tokenizer = openwebtext2_tokenizer()
    word2idx = {k.encode('utf-8'):v for k,v in tokenizer.get_vocab().items()}
    idx2word = {v:k for k,v in word2idx.items()}

    dataset = torch.tensor(
        tokenizer.encode(untokenized_dataset).ids,
        dtype=torch.int64
    )
    total_tokens = len(dataset)

    print(f'Words per token ({dataset_name}):', total_words / total_tokens)

    seqs = [dataset[None,i:i+seq_len] for i in range(0, len(dataset), seq_len)]
    def test_iterator():
        while True:
            np.random.shuffle(seqs)
            for seq in seqs:
                yield seq

    return (None, test_iterator(), None), (word2idx, idx2word)

ptb = functools.partial(_eval_dataset, 'ptb')
wikitext2 = functools.partial(_eval_dataset, 'wikitext2')
wikitext103 = functools.partial(_eval_dataset, 'wikitext103')
enwik8 = functools.partial(_eval_dataset, 'enwik8')
text8 = functools.partial(_eval_dataset, 'text8')
billionword = functools.partial(_eval_dataset, 'billionword')

UNTOKENIZED_REGISTRY = {
    'ptb': ptb_untokenized,
    'wikitext2': wikitext2_untokenized,
    'wikitext103': wikitext103_untokenized,
    'enwik8': enwik8_untokenized,
    'text8': text8_untokenized,
    'billionword': billionword_untokenized
}

REGISTRY = {
    'openwebtext2': openwebtext2,
    'ptb': ptb,
    'wikitext2': wikitext2,
    'wikitext103': wikitext103,
    'enwik8': enwik8,
    'text8': text8,
    'billionword': billionword
}