import argparse
from typing import Tuple

import pandas as pd
import spacy
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import BucketIterator, Field, TabularDataset
from tqdm import tqdm

import config
from model import Decoder, Encoder, Seq2Seq
from utils import (bleu_score, check_csv, load_checkpoint, save_checkpoint,
                   tokenize, translate_many, translate_one)


def get_text_processors(src_lan_model: str, tar_lan_model: str) -> Tuple[Tuple[spacy.Language, spacy.Language], Tuple[Field, Field]]:
    src_tokenizer = spacy.load(src_lan_model)
    tar_tokenizer = spacy.load(tar_lan_model)
    src_lan = Field(init_token='<sos>', eos_token='<eos>',
                    tokenize=lambda t: tokenize(t, src_tokenizer), lower=True)
    tar_lan = Field(init_token='<sos>', eos_token='<eos>',
                    tokenize=lambda t: tokenize(t, tar_tokenizer), lower=True)
    return (src_tokenizer, tar_tokenizer), (src_lan, tar_lan)


def build_model(src_vocab_len: int, tar_vocab_len: int, embed_size: int, hidden_size, encoder_layers: int, decoder_layers: int, dropout: float) -> Seq2Seq:
    encoder = Encoder(src_vocab_len, embed_size, hidden_size,
                      encoder_layers, dropout).to(config.DEVICE)
    decoder = Decoder(tar_vocab_len, embed_size, hidden_size,
                      decoder_layers, dropout).to(config.DEVICE)
    model = Seq2Seq(encoder, decoder, tar_vocab_len).to(config.DEVICE)
    return model


def load_data(args: argparse.Namespace, src_lan: Field, tar_lan: Field):
    fields = [('src', src_lan), ('tar', tar_lan)]
    return TabularDataset.splits(
        path=args.datadir, train=args.traindata, validation=args.valdata, test=args.testdata, format='csv', fields=fields)


def train(args: argparse.Namespace):
    (src_tokenizer, _), (src_lan, tar_lan) = get_text_processors(
        'en_core_web_sm', 'fr_core_news_sm')

    train_data, val_data, test_data = load_data(args, src_lan, tar_lan)

    src_lan.build_vocab(train_data, val_data, test_data)
    tar_lan.build_vocab(train_data, val_data, test_data)

    train_iter = BucketIterator(
        train_data,
        batch_size=config.BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),  # minimize padding
        device=config.DEVICE)

    model = build_model(len(src_lan.vocab), len(tar_lan.vocab), args.embedsize,
                        args.hiddensize, args.encoder_layers, args.decoder_layers, args.dropout)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pad_idx = src_lan.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    last_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, args.lr)
        if checkpoint:
            last_epoch = checkpoint['epoch']

    assert last_epoch < args.epochs
    for epoch in range(last_epoch+1, args.epochs+1):
        print(f'Epoch {epoch}/{args.epochs}')
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
        translated = translate_one(args.evalsen, model,
                                   src_tokenizer, src_lan, tar_lan, as_str=True)
        print(f'Sentence: {args.evalsen}\nTranslation: {translated}')
        if epoch % 5 == 0:
            print('Computing bleu score...')
            bleu = bleu_score(val_data[1:], model,
                              src_tokenizer, src_lan, tar_lan)
            print(f'Bleu score: {bleu}')
        if args.save:
            save_checkpoint(model, optimizer, epoch, encoder_layers=args.encoder_layers,
                            decoder_layers=args.decoder_layers, embed_size=args.embedsize, hidden_size=args.hiddensize, dropout=args.dropout)


def test(args: argparse.Namespace):
    # data
    (src_tokenizer, _), (src_lan, tar_lan) = get_text_processors(
        'en_core_web_sm', 'fr_core_news_sm')
    train_data, val_data, test_data = load_data(args, src_lan, tar_lan)
    # keep the same vocab size as the training => same model input size
    src_lan.build_vocab(train_data, val_data, test_data)
    tar_lan.build_vocab(train_data, val_data, test_data)
    # model
    checkpoint = load_checkpoint()
    model = build_model(len(src_lan.vocab), len(tar_lan.vocab), checkpoint['embed_size'],
                        checkpoint['hidden_size'], checkpoint['encoder_layers'], checkpoint['decoder_layers'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # translation
    translations, bleu = translate_many(
        test_data[1:], model, src_tokenizer, src_lan, tar_lan, bleu_score=True)
    df = pd.DataFrame(translations, columns=[
                      'source', 'translation', 'target'])
    df.to_csv('translations.csv', index=False)
    print(f'Bleu score: {bleu}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='English to French Translator')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')
    # training
    train_parser = subparsers.add_parser('train', help='train the model')
    train_parser.add_argument('--datadir', type=str, default='data/',
                              help='training data directory')
    train_parser.add_argument(
        '--traindata', type=check_csv, default='train.csv')
    train_parser.add_argument('--valdata', type=check_csv, default='val.csv')
    train_parser.add_argument('--testdata', type=check_csv, default='test.csv')
    train_parser.add_argument('--lr', type=float, default=config.LR)
    train_parser.add_argument('--encoder_layers', type=float,
                              default=config.NUM_LAYERS)
    train_parser.add_argument('--decoder_layers', type=float,
                              default=config.NUM_LAYERS)
    train_parser.add_argument('--embedsize', type=int,
                              default=config.EMBEDDING_SIZE)
    train_parser.add_argument('--hiddensize', type=int,
                              default=config.HIDDEN_SIZE)
    train_parser.add_argument('--dropout', type=float,
                              default=config.DROPOUT)
    train_parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    train_parser.add_argument('--batchsize', type=int,
                              default=config.BATCH_SIZE)
    train_parser.add_argument('--resume', action='store_true',
                              help='resume from checkpoint')
    train_parser.add_argument('--save', action='store_true',
                              help='save model at each epoch')
    train_parser.add_argument('--evalsen', type=str,
                              default='A man in a blue shirt is standing on a ladder cleaning a window.')
    # testing
    test_parser = subparsers.add_parser('test', help='test the model')
    test_parser.add_argument('--datadir', type=str, default='data/',
                             help='training data directory')
    test_parser.add_argument(
        '--traindata', type=check_csv, default='train.csv')
    test_parser.add_argument('--valdata', type=check_csv, default='val.csv')
    test_parser.add_argument('--testdata', type=check_csv, default='test.csv')

    args = parser.parse_args()
    print('------------ Arguments -------------')
    for k, v in sorted(vars(args).items()):
        print(f'{k}: {v}')
    print('------------------------------------')
    if (args.subcommand == 'train'):
        train(args)
    elif (args.subcommand == 'test'):
        test(args)
