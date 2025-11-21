import torch, gzip, os, random, numpy as np
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from nltk.translate.bleu_score import corpus_bleu
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from utils import Encoder, Decoder, Seq2Seq  # import from utils.py

# ===============================
# ⚙️ Setup
# ===============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
MAX_SENTENCE_LEN = 50

# ===============================
# 🧠 Tokenizers
# ===============================
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# ===============================
# 📂 Load Multi30K (Local path)
# ===============================
def read_pairs(file_en, file_fr, data_dir='../data/data', max_len=50):
    path_en = os.path.join(data_dir, file_en)
    path_fr = os.path.join(data_dir, file_fr)
    pairs = []
    with open(path_en, 'r', encoding='utf-8') as f_en, open(path_fr, 'r', encoding='utf-8') as f_fr:
        for en, fr in zip(f_en, f_fr):
            en, fr = en.strip(), fr.strip()
            if 0 < len(en.split()) <= max_len and 0 < len(fr.split()) <= max_len:
                pairs.append((en, fr))
    print(f"✅ Loaded {len(pairs)} pairs from {file_en} & {file_fr}")
    return pairs

train_data = read_pairs('train.en', 'train.fr')
val_data   = read_pairs('val.en', 'val.fr')
test_data  = read_pairs('test.en', 'test.fr')

# ===============================
# 🧱 Build Vocabulary
# ===============================
def yield_tokens(data_iter, tokenizer, idx):
    for pair in data_iter:
        yield tokenizer(pair[idx])

en_vocab = build_vocab_from_iterator(yield_tokens(train_data, en_tokenizer, 0), specials=special_tokens, min_freq=1)
fr_vocab = build_vocab_from_iterator(yield_tokens(train_data, fr_tokenizer, 1), specials=special_tokens, min_freq=1)
en_vocab.set_default_index(UNK_IDX)
fr_vocab.set_default_index(UNK_IDX)
print(f"✅ Vocab sizes: EN={len(en_vocab)} | FR={len(fr_vocab)}")

# ===============================
# 🔢 Convert to Tensors
# ===============================
def data_process(data):
    out = []
    for en, fr in data:
        en_t = torch.tensor(en_vocab(en_tokenizer(en)), dtype=torch.long)
        fr_t = torch.tensor([SOS_IDX] + fr_vocab(fr_tokenizer(fr)) + [EOS_IDX], dtype=torch.long)
        out.append((en_t, fr_t))
    return out

test_data_indices = data_process(test_data)

# ===============================
# 🧩 Load Model
# ===============================
INPUT_DIM = len(en_vocab)
OUTPUT_DIM = len(fr_vocab)
EMB_DIM, HID_DIM, N_LAYERS, DROPOUT = 256, 512, 2, 0.5

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

checkpoint = torch.load("../checkpoints/best_model.pth", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()
print("✅ Loaded model from checkpoints/best_model.pth")

# ===============================
# 🌍 Translation
# ===============================
def translate(sentence, max_len=50):
    model.eval()
    tokens = en_tokenizer(sentence)
    src_idx = [en_vocab[t] for t in tokens]
    src_tensor = torch.LongTensor(src_idx).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(src_idx)]).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)

    input = torch.LongTensor([SOS_IDX]).to(device)
    out_idx = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input, hidden, cell)
        pred = output.argmax(1).item()
        if pred == EOS_IDX:
            break
        out_idx.append(pred)
        input = torch.LongTensor([pred]).to(device)

    return ' '.join(fr_vocab.lookup_token(i) for i in out_idx)

# ===============================
# 🧪 BLEU Evaluation
# ===============================
def calculate_bleu_score():
    trgs, prds = [], []
    for en_t, fr_t in tqdm(test_data_indices, desc="Testing"):
        src = [en_vocab.lookup_token(i.item()) for i in en_t if i.item() > 3]
        pred_tokens = translate(' '.join(src)).split()
        tgt = [fr_vocab.lookup_token(i.item()) for i in fr_t if i.item() > 3]
        trgs.append([tgt]); prds.append(pred_tokens)
    return corpus_bleu(trgs, prds)

bleu = calculate_bleu_score()
print(f"✅ BLEU Score: {bleu * 100:.2f}")

# ===============================
# 💬 Example
# ===============================
example = "A woman is selling various kinds of vegetables."
print(f"\nEN: {example}\nFR: {translate(example)}")
