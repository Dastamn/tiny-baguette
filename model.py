import random

import torch
import torch.nn as nn

import config


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_p):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.nn = nn.LSTM(embedding_size, hidden_size,
                          num_layers, dropout=dropout_p)

    def forward(self, x):  # x: (d, batch)
        embedding = self.embedding(x)  # (d, batch, embedding_size)
        embedding = self.dropout(embedding)
        _, (hidden, cell) = self.nn(embedding)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_p):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.nn = nn.LSTM(embedding_size, hidden_size,
                          num_layers, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden, cell):
        expanded = x.unsqueeze(0)  # (1, batch)
        embedding = self.embedding(expanded)  # (1, batch, embedding_size)
        embedding = self.dropout(embedding)
        output, (hidden, cell) = self.nn(embedding, (hidden, cell))
        pred = self.fc(output)  # (1, batch, vocab_size)
        return pred.squeeze(0), (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_len, teacher_forcing=.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing = teacher_forcing
        self.vocab_len = vocab_len

    def forward(self, x, y):
        batch_size = x.shape[1]
        y_len = y.shape[0]
        hidden, cell = self.encoder(x)
        # prediction
        token = y[0]  # start token
        preds = torch.zeros(
            (y_len, batch_size, self.vocab_len)).to(config.DEVICE)
        for i in range(1, y_len):
            output, (hidden, cell) = self.decoder(token, hidden, cell)
            preds[i] = output
            pred = torch.argmax(output, dim=1)
            token = y[i] if random.random() < self.teacher_forcing else pred
        return preds
