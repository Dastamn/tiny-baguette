import argparse
import os
import re
from typing import List, Tuple, Union

import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchtext.data.metrics import bleu_score as bleu
from torchtext.legacy.data import Field, TabularDataset
from tqdm import tqdm

import config
from model import Seq2Seq


def tokenize(text: Union[str, List[str]], tokenizer: spacy.Language) -> List[str]:
    return [t.text.lower() for t in tokenizer.tokenizer(text)] if isinstance(text, str) \
        else [t.lower() for t in text]


def translate_one(sentence: str, model: Seq2Seq, src_tokenizer: spacy.Language, src_lan: Field, tar_lan: Field, as_str: bool = False, max_len: int = 80) -> Union[str, List[str]]:
    tokens = tokenize(sentence, src_tokenizer)
    tokens = [src_lan.init_token, *tokens, tar_lan.eos_token]
    idx = [src_lan.vocab.stoi[token] for token in tokens]
    x = torch.LongTensor(idx).unsqueeze(1).to(config.DEVICE)
    with torch.no_grad():
        hidden, cell = model.encoder(x)
    # prediction
    preds = [tar_lan.vocab.stoi['<sos>']]
    for _ in range(1, max(len(idx)-1, max_len+1)):
        pred = torch.LongTensor([preds[-1]]).to(config.DEVICE)
        with torch.no_grad():
            output, (hidden, cell) = model.decoder(pred, hidden, cell)
            out = output.argmax(1).item()
        preds.append(out)
        # end of sentence
        if out == tar_lan.vocab.stoi['<eos>']:
            break
    translated = [tar_lan.vocab.itos[idx]
                  for idx in preds][1:]  # remove start token
    if translated[-1] == '<eos>':
        translated = translated[:-1]  # remove end token
    if as_str:
        translated = prettify(translated)
    return translated


def translate_many(data: TabularDataset, model: Seq2Seq, src_tokenizer: spacy.Language, src_lan: Field, tar_lan: Field, max_len: int = 80, bleu_score: bool = False) -> Union[List[Tuple[str, str, str]],  Tuple[List[Tuple[str, str, str]], float]]:
    translations = [(datum.src, translate_one(datum.src, model, src_tokenizer, src_lan,
                                              tar_lan, max_len=max_len), [datum.tar]) for datum in tqdm(data)]
    as_sentences = [(prettify(src), prettify(translated), prettify(tar[0]))
                    for src, translated, tar in translations]
    if not bleu_score:
        return as_sentences
    return as_sentences, bleu(*list(zip(*translations))[1:])


def prettify(text: Union[str, List[str]]) -> str:
    if isinstance(text, list):
        text = ' '.join(text)
    out = re.sub(r' *([?!.,;:]) *', r'\1 ', text)
    out = re.sub(r' *([\'-]) *', r'\1', out)
    return out.rstrip()


def bleu_score(data: TabularDataset, model: Seq2Seq, src_tokenizer: spacy.Language, src_lan: Field, tar_lan: Field, max_len: int = 80) -> float:
    translations = [(translate_one(datum.src, model, src_tokenizer, src_lan,
                     tar_lan, max_len=max_len), [datum.tar]) for datum in tqdm(data)]
    return bleu(*list(zip(*translations)))


def to_df(src: Tuple[str, List], tar: Tuple[str, List]) -> pd.DataFrame:
    data = {src[0]: [line for line in src[1]],
            tar[0]: [line for line in tar[1]]}
    return pd.DataFrame(data, columns=[src[0], tar[0]])


def load_checkpoint(model: nn.Module = None, optimizer: Optimizer = None, lr: float = None, filename: str = 'model.pt', dir: str = 'checkpoint') -> int:
    if dir:
        filename = os.path.join(dir, filename)
    print(f"=> Loading checkpoint from '{filename}'...")
    try:
        checkpoint = torch.load(filename, map_location=config.DEVICE)
    except:
        print('No checkpoint found.')
        return None
    if model:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    print('Done.')
    return checkpoint


def save_checkpoint(model: nn.Module, optimizer: Optimizer, epoch: int, filename: str = 'model.pt', dir: str = 'checkpoint', **kwargs):
    if dir:
        check_dir(dir)
        filename = os.path.join(dir, filename)
    print(f"=> Saving checkpoint to '{filename}'...")
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        **kwargs
    }
    torch.save(checkpoint, filename)
    print('Done.')


def check_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def check_csv(fn: str) -> str:
    _, ext = os.path.splitext(fn)
    if ext.lower() != '.csv':
        raise argparse.ArgumentTypeError('File must be .csv')
    return fn
