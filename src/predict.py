from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score
from data_utils import load_official_split, load_test_data

MODEL_NAME = "FacebookAI/roberta-large"
CKPT_PATH = Path("BestModel/best_model.pt")

BATCH_SIZE = 32
MAX_LENGTH = 256
BEST_THR = 0.2 # Get from training process

DEV_OUT = Path("dev.txt")
TEST_OUT = Path("test.txt")

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0), # (1, seq_len) -> (seq_len, )
            "attention_mask": enc["attention_mask"].squeeze(0), # (seq_len, )
        }

        return item
    
@torch.no_grad()
def predict_binary(model, loader, thr, device):
    model.eval()
    preds = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits #(batch, 2)
        p1 = torch.softmax(logits.float(), dim=-1)[:, 1] # Get p of y = 1 (batch, 1)
        pred = (p1 > thr).long()

        preds.extend(pred.cpu().tolist())

    return np.array(preds, dtype=np.int64)

def write_txt(preds: np.array, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for p in preds.tolist():
            f.write(f"{int(p)}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    # Load data
    train_df, dev_df = load_official_split()
    dev_texts = dev_df["text"].astype(str).tolist()
    test_df = load_test_data()
    test_texts = test_df["text"].astype(str).tolist()

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    print("Loaded:", CKPT_PATH, "thr=", BEST_THR)

    # DataLoader
    dev_loader = DataLoader(
        TextDataset(dev_texts, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        TextDataset(test_texts, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Predict and Write
    dev_pred = predict_binary(model, dev_loader, BEST_THR, device)

    dev_labels = dev_df["binary_label"].astype(int).to_numpy()

    dev_f1 = f1_score(dev_labels, dev_pred, average="binary", pos_label=1)
    dev_acc = accuracy_score(dev_labels, dev_pred)

    print(f"Official DEV F1 (pos=1): {dev_f1:.4f}")
    print(f"Official DEV Accuracy: {dev_acc:.4f}")

    write_txt(dev_pred, DEV_OUT)
    print(f"Write {DEV_OUT}, lines = {len(dev_texts)}")
    test_pred = predict_binary(model, test_loader, BEST_THR, device)
    write_txt(test_pred, TEST_OUT)
    print(f"Write {TEST_OUT}, lines = {len(test_texts)}")

if __name__ == "__main__":
    main()





