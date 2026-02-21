import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_utils import load_official_split, PCLDataset
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
from torch.optim import AdamW
from sklearn.metrics import f1_score
import torch.nn.functional as F

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_df, dev_df = load_official_split()
train_texts = train_df["text"].tolist()
train_labels = train_df["binary_label"].tolist()
dev_texts = dev_df["text"].tolist()
dev_labels = dev_df["binary_label"].tolist()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# DataLoader
train_dataset = PCLDataset(train_texts,tokenizer, train_labels, MAX_LENGTH)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
    )

dev_dataset = PCLDataset(dev_texts, tokenizer, dev_labels, MAX_LENGTH)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=2*BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))

@torch.no_grad()
def eval_and_collect_preds(model, loader):
    model.eval()
    all_labels = []
    all_preds = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = torch.argmax(logits, dim=-1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1_pos = f1_score(all_labels, all_preds, pos_label=1)
    return f1_pos, all_preds

# Start training
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader)
    dev_f1, _ = eval_and_collect_preds(model, dev_loader)
    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | dev_pos_f1={dev_f1:.4f}")

dev_f1, dev_preds = eval_and_collect_preds(model, dev_loader)
print("Final baseline dev pos F1:", dev_f1)

with open("error_analysis/baseline_dev.txt", "w", encoding="utf-8") as f:
    for p in dev_preds.tolist():
        f.write(f"{int(p)}\n")
print("Saved baseline_dev.txt")



