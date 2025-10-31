import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3


class Encoder(nn.Module):
    """
    Mô hình Encoder (LSTM 2 tầng với Packed Sequences).
    Đầu ra là Vector Ngữ cảnh Cố định (Fixed Context Vector) (hidden/cell cuối cùng).
    """
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        
        embedded = self.dropout(self.embedding(src))
        
        packed_embedded = pack_padded_sequence(
            embedded, 
            src_len.cpu(), 
            batch_first=True, 
            enforce_sorted=True
        )
        
        _, (hidden, cell) = self.rnn(packed_embedded)
        
        return hidden, cell


class Decoder(nn.Module):
    """
    Mô hình Decoder (LSTM 2 tầng).
    Đầu vào là token trước đó và trạng thái ẩn/tế bào từ bước t-1.
    """
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        input = input.unsqueeze(1) 
        
        embedded = self.dropout(self.embedding(input)) 
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        output_1d = output.squeeze(1) 
        
        prediction = self.fc_out(output_1d)
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Mô hình Encoder-Decoder hoàn chỉnh.
    Thực hiện vòng lặp Decoder từng bước với Teacher Forcing.
    """
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Number of layers must be equal!"

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src, src_len)
        
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            outputs[:, t] = output
            
            top1 = output.argmax(1) 
            
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1
            
        return outputs


def init_weights(m):
    """Khởi tạo trọng số Normal(0, 0.01) cho LSTM và 0 cho Bias."""
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)