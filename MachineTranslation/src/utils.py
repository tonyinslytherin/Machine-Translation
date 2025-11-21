import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD_IDX = 1  # same constant used in main

# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        emb = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=True)
        _, (hidden, cell) = self.rnn(packed)
        return hidden, cell


# ---------- Decoder ----------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        emb = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(emb, (hidden, cell))
        return self.fc_out(output.squeeze(1)), hidden, cell


# ---------- Seq2Seq ----------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
