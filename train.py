import argparse

import spacy
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import BucketIterator, Field, TabularDataset
from tqdm import tqdm

import config
from model import Decoder, Encoder, Seq2Seq
from utils import (bleu_score, check_csv, load_checkpoint, save_checkpoint,
                   tokenize, translate)


def train(args: argparse.Namespace):
    en_tokenizer = spacy.load('en_core_web_sm')
    fr_tokenizer = spacy.load('fr_core_news_sm')

    # preprocessing
    en = Field(init_token='<sos>', eos_token='<eos>',
               tokenize=lambda t: tokenize(t, en_tokenizer), lower=True)
    fr = Field(init_token='<sos>', eos_token='<eos>',
               tokenize=lambda t: tokenize(t, fr_tokenizer), lower=True)

    # data
    fields = [('src', en), ('tar', fr)]
    train_data, val_data = TabularDataset.splits(
        path=args.datadir, train='train.csv', validation='val.csv', format='csv', fields=fields)

    # vocab
    print('Building vocabulary...')
    en.build_vocab(train_data)
    fr.build_vocab(train_data)
    print('Done.')

    # iterator
    train_iter = BucketIterator(
        train_data,
        batch_size=config.BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),  # minimize padding
        device=config.DEVICE)

    # models, loss and optim
    encoder = Encoder(len(en.vocab), args.embedsize, args.hiddensize,
                      args.encoder_layers, config.DROPOUT).to(config.DEVICE)
    decoder = Decoder(len(fr.vocab), args.embedsize, args.hiddensize,
                      args.decoder_layers, config.DROPOUT).to(config.DEVICE)
    model = Seq2Seq(encoder, decoder, len(fr.vocab)).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pad_idx = en.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    last_epoch = load_checkpoint(
        model, optimizer, args.lr) if args.resume else 0

    assert last_epoch < args.epochs
    for epoch in range(last_epoch+1, args.epochs+1):
        print(f'Epoch {epoch}')
        model.train()
        for batch in tqdm(train_iter):
            x = batch.src
            y = batch.tar
            output = model(x, y)
            output = output[1:]  # skip start token
            # (sen_size, batch_size, vocab_size) => (sen_size*batch_size, vocab_size)
            output = output.reshape(-1, output.shape[2])
            y = y[1:].reshape(-1)
            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        # eval
        model.eval()
        translated = translate(model, args.evalsen,
                               en_tokenizer, en, fr, as_str=True)
        print(f'Sentence: {args.evalsen}\nTranslation: {translated}')
        if epoch % 5 == 0:
            print('Computing bleu score...')
            bleu = bleu_score(val_data[1:], model, en_tokenizer, en, fr)
            print(f'Bleu score: {bleu}')
        if args.save:
            save_checkpoint(model, optimizer, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='English to French Translator')
    parser.add_argument('--datadir', type=str, default='data/',
                        help='training data directory')
    parser.add_argument('--traindata', type=check_csv, default='train.csv')
    parser.add_argument('--valdata', type=check_csv, default='val.csv')
    parser.add_argument('--lr', type=float, default=config.LR)
    parser.add_argument('--encoder_layers', type=float,
                        default=config.NUM_LAYERS)
    parser.add_argument('--decoder_layers', type=float,
                        default=config.NUM_LAYERS)
    parser.add_argument('--embedsize', type=int, default=config.EMBEDDING_SIZE)
    parser.add_argument('--hiddensize', type=int, default=config.HIDDEN_SIZE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batchsize', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--save', action='store_true',
                        help='save model at each epoch')
    parser.add_argument('--model', type=str, default='model.pt',
                        help='model name')
    parser.add_argument('--evalsen', type=str,
                        default='A man in a blue shirt is standing on a ladder cleaning a window.')
    args = parser.parse_args()
    print('------------ Arguments -------------')
    for k, v in sorted(vars(args).items()):
        print(f'{k}: {v}')
    print('------------------------------------')
    train(args)
