import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 
import nltk
from nltk.translate.bleu_score import corpus_bleu
import os

# Cài đặt thư viện (Chỉ chạy khi cần thiết)
# print("Đang cài đặt các thư viện cần thiết...")
# !pip install torchtext==0.16.0 torch==2.1.0 spacy numpy nltk tqdm matplotlib

# Tải mô hình spacy (Chỉ chạy một lần ban đầu)
# print("Đang tải mô hình Spacy...")
# !python -m spacy download en_core_web_sm
# !python -m spacy download fr_core_news_sm

# Thiết lập Seed và Thiết bị
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")


UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
MAX_VOCAB_SIZE = 10000
BATCH_SIZE = 32
MAX_SENTENCE_LEN = 50 

# Tokenizers
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

DATA_DIR = 'data/' 

def read_pairs(file_en, file_fr):
    """Đọc dữ liệu từ tệp Multi30K và lọc theo độ dài câu."""
    
    path_en = os.path.join(DATA_DIR, file_en)
    path_fr = os.path.join(DATA_DIR, file_fr)
    
    data = []

    try:
        with open(path_en, 'r', encoding='utf-8') as f_en, open(path_fr, 'r', encoding='utf-8') as f_fr:
            for en, fr in zip(f_en, f_fr):
                en_clean = en.strip()
                fr_clean = fr.strip()
                
                if 0 < len(en_clean.split()) <= MAX_SENTENCE_LEN and 0 < len(fr_clean.split()) <= MAX_SENTENCE_LEN:
                    data.append((en_clean, fr_clean))

        print(f"Đã tải {len(data)} cặp câu từ {file_en} & {file_fr}.")
        return data
            
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp. Kiểm tra đường dẫn: {path_en} và {path_fr}")
        return []

train_data = read_pairs('train.en', 'train.fr')
val_data = read_pairs('val.en', 'val.fr')
test_data = read_pairs('test.en', 'test.fr')

if not train_data:
    raise RuntimeError("Không thể tải dữ liệu đào tạo. Vui lòng kiểm tra thư mục 'data/' và các tệp Multi30K.")


def yield_tokens(data_iter, tokenizer, src_idx=0):
    for pair in data_iter:
        yield tokenizer(pair[src_idx])

en_vocab = build_vocab_from_iterator(
    yield_tokens(train_data, en_tokenizer, 0),
    min_freq=1, specials=special_tokens, max_tokens=MAX_VOCAB_SIZE
)
en_vocab.set_default_index(UNK_IDX)

fr_vocab = build_vocab_from_iterator(
    yield_tokens(train_data, fr_tokenizer, 1),
    min_freq=1, specials=special_tokens, max_tokens=MAX_VOCAB_SIZE
)
fr_vocab.set_default_index(UNK_IDX)

print(f"Kích thước từ vựng EN: {len(en_vocab)}, FR: {len(fr_vocab)}")

def data_process(data, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer):
    data_index = []
    for en_sentence, fr_sentence in data:
        en_tensor = torch.tensor(en_vocab(en_tokenizer(en_sentence)), dtype=torch.long)
        fr_tensor = torch.tensor([SOS_IDX] + fr_vocab(fr_tokenizer(fr_sentence)) + [EOS_IDX], dtype=torch.long)
        data_index.append((en_tensor, fr_tensor))
    return data_index

train_data_indices = data_process(train_data, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer)
val_data_indices = data_process(val_data, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer)
test_data_indices = data_process(test_data, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer)

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    en_batch, fr_batch = zip(*batch)
    
    en_lengths = torch.tensor([len(seq) for seq in en_batch], dtype=torch.long)
    
    en_padded = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=True)
    fr_padded = pad_sequence(fr_batch, padding_value=PAD_IDX, batch_first=True)
    
    return en_padded.to(device), en_lengths.to(device), fr_padded.to(device)

train_iterator = DataLoader(TranslationDataset(train_data_indices), batch_size=BATCH_SIZE, collate_fn=collate_fn)
valid_iterator = DataLoader(TranslationDataset(val_data_indices), batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_iterator = DataLoader(TranslationDataset(test_data_indices), batch_size=1, collate_fn=collate_fn)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        
        packed_embedded = pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=True)
        
        _, (hidden, cell) = self.rnn(packed_embedded)
        
        return hidden, cell

class Decoder(nn.Module):
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
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim
        assert encoder.n_layers == decoder.n_layers

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
            
            # Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1
            
        return outputs


INPUT_DIM = len(en_vocab)
OUTPUT_DIM = len(fr_vocab)
EMB_DIM = 256 
HID_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5
TRG_PAD_IDX = PAD_IDX

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
CLIP = 1.0 # Gradient Clipping

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, src_len, trg) in enumerate(tqdm(iterator, desc="Training")):
        optimizer.zero_grad()
        output = model(src, src_len, trg, teacher_forcing_ratio=0.5)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, src_len, trg) in enumerate(tqdm(iterator, desc="Evaluating")):
            output = model(src, src_len, trg, 0) 
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

N_EPOCHS = 20
best_valid_loss = float('inf')
train_losses = []
valid_losses = []
patience = 0 

print("\n--- Bắt đầu Huấn luyện ---")
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience = 0
        print(f'*** Epoch {epoch+1:02}: Lưu mô hình tốt nhất! ***')
    else:
        patience += 1
    
    print(f'Epoch: {epoch+1:02} | Patience: {patience}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):7.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {np.exp(valid_loss):7.3f}')

    if patience >= 3:
        print(f"\n--- Dừng sớm tại Epoch {epoch+1} ---")
        break

model.load_state_dict(torch.load('best_model.pth'))
print("\nĐã tải mô hình tốt nhất ('best_model.pth').")


def translate(sentence, en_vocab, fr_vocab, en_tokenizer, model, device, max_len=50):
    model.eval()
    
    tokens = en_tokenizer(sentence)
    src_indexes = [en_vocab[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device) 
    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)

    input = torch.LongTensor([SOS_IDX]).to(device) 
    trg_indexes = []

    for t in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input, hidden, cell)
        
        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == EOS_IDX:
            break
            
        input = torch.LongTensor([pred_token]).to(device)
        
    trg_tokens = [fr_vocab.lookup_token(i) for i in trg_indexes]
    
    translation = ' '.join(trg_tokens[1:-1])
    return translation

def calculate_bleu_score(test_data_indices, en_vocab, fr_vocab, model, device):
    trgs = [] 
    prds = [] 
    
    for src_tensor, trg_tensor in tqdm(test_data_indices, desc="Tính BLEU"):
        
        src_sentence = [en_vocab.lookup_token(i.item()) for i in src_tensor]
        sentence_str = ' '.join([t for t in src_sentence if t not in special_tokens])
        predicted_str = translate(sentence_str, en_vocab, fr_vocab, en_tokenizer, model, device)
        predicted_tokens = predicted_str.split()
        
        target_tokens = [fr_vocab.lookup_token(i.item()) for i in trg_tensor]
        clean_target = [t for t in target_tokens if t not in special_tokens]
        
        trgs.append([clean_target]) 
        prds.append(predicted_tokens)

    bleu_score = corpus_bleu(trgs, prds)
    return bleu_score

print("\n--- 6.5. Đánh giá ---")
bleu_score = calculate_bleu_score(test_data_indices, en_vocab, fr_vocab, model, device)
print(f"BLEU Score trên tập Test: {bleu_score*100:.2f}")

print("\n--- Ví dụ Dịch ---")
example_sentence = "A woman is selling various kinds of vegetables."
translation_output = translate(example_sentence, en_vocab, fr_vocab, en_tokenizer, model, device)
print(f"Câu nguồn (EN): {example_sentence}")
print(f"Dịch (FR): {translation_output}")


if train_losses and valid_losses:
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Train vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
