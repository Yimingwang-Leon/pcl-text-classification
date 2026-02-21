import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
import numpy as np
import math
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.data_utils import load_official_split, PCLDataset
import pandas as pd

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--ablation", required=True,
                    choices=["no_freeze", "no_threshold"],
                    help="Which component to ablate")
args = parser.parse_args()
ABLATION = args.ablation
print(f"\n{'='*60}\nAblation: {ABLATION}\n{'='*60}\n")

# Seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Config
MODEL_NAME     = "FacebookAI/roberta-large"
BATCH_SIZE     = 16
EPOCHS         = 20
MAX_EPOCHS     = 20
PATIENCE       = 5
MIN_DELTA      = 1e-4
max_lr         = 1e-5
min_lr         = 2e-6
FREEZE_LAYERS  = 12   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_df, val_df = load_official_split()
train_texts_full  = train_df["text"].tolist()
train_labels_full = train_df["binary_label"].tolist()

train_texts, internal_texts, train_labels, internal_labels = train_test_split(
    train_texts_full, train_labels_full,
    test_size=0.1, random_state=42, stratify=train_labels_full
)

aug_df = pd.read_csv("data/aug/pcl_aug.csv")
train_texts  = train_texts  + aug_df["text"].tolist()
train_labels = train_labels + aug_df["binary_label"].tolist()
print(f"Aug data loaded: {len(aug_df)} examples → train size now {len(train_texts)}")

official_dev_texts  = val_df["text"].tolist()
official_dev_labels = val_df["binary_label"].tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset        = PCLDataset(train_texts, tokenizer, train_labels)
internal_dataset     = PCLDataset(internal_texts, tokenizer, internal_labels)
official_dev_dataset = PCLDataset(official_dev_texts, tokenizer, official_dev_labels)

train_loader        = DataLoader(train_dataset,        batch_size=BATCH_SIZE,   shuffle=True,  num_workers=0, pin_memory=True)
internal_loader     = DataLoader(internal_dataset,     batch_size=2*BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
official_dev_loader = DataLoader(official_dev_dataset, batch_size=2*BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

# Ablation: no_freeze: skip freezing; otherwise freeze embeddings + first 12 layers
if ABLATION == "no_freeze":
    print("Ablation: ALL layers trainable (no freezing)")
else:
    base = getattr(model, model.base_model_prefix)
    for param in base.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(base.encoder.layer):
        if i < FREEZE_LAYERS:
            for param in layer.parameters():
                param.requires_grad = False
    print(f"Froze embeddings + first {FREEZE_LAYERS} layers")

# Optimizer
param_dict     = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params   = [p for p in param_dict.values() if p.dim() >= 2]
nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay_params,   "weight_decay": 0.01},
     {"params": nodecay_params, "weight_decay": 0.0}],
    lr=max_lr, betas=(0.9, 0.95), eps=1e-8, fused=False
)

steps_per_epoch = len(train_loader)
max_steps       = EPOCHS * steps_per_epoch
warmup_steps    = int(0.2 * max_steps)

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Loss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def train_one_epoch(model, loader, optimizer, criterion, global_step):
    model.train()
    total_loss = 0.0
    for batch in loader:
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(global_step)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(input_ids=input_ids, attention_mask=attention_mask).logits.float(), labels)
        loss.backward()
        optimizer.step()
        total_loss  += loss.item()
        global_step += 1
    return total_loss / max(len(loader), 1), global_step

@torch.no_grad()
def collect_probs(model, loader):
    model.eval()
    probs, golds = [], []
    for batch in loader:
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        golds.extend(batch["label"].tolist())
    return np.array(probs, dtype=np.float32), np.array(golds, dtype=np.int64)

def best_threshold_by_f1(probs, golds):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(golds, (probs >= thr).astype(int), average="binary", pos_label=1)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1

# Training starts
best_dev_f1 = -1.0
best_thr     = 0.5
bad_epochs   = 0
global_step  = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, global_step = train_one_epoch(model, train_loader, optimizer, criterion, global_step)

    internal_probs, internal_golds = collect_probs(model, internal_loader)

    # Ablation: no_threshold → fixed thr=0.5; otherwise search on internal val
    if ABLATION == "no_threshold":
        thr = 0.5
        internal_f1 = f1_score(internal_golds, (internal_probs >= thr).astype(int),
                                average="binary", pos_label=1)
    else:
        thr, internal_f1 = best_threshold_by_f1(internal_probs, internal_golds)

    dev_probs, dev_golds = collect_probs(model, official_dev_loader)
    dev_f1 = f1_score(dev_golds, (dev_probs >= thr).astype(int), average="binary", pos_label=1)

    if dev_f1 > best_dev_f1 + MIN_DELTA:
        best_dev_f1 = dev_f1
        best_thr    = thr
        bad_epochs  = 0
        pass  
    else:
        bad_epochs += 1

    print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | loss={train_loss:.4f} | thr={thr:.2f} | "
          f"internal_f1={internal_f1:.4f} | dev_f1={dev_f1:.4f} | bad={bad_epochs}/{PATIENCE}")

    if bad_epochs >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

print(f"\n[{ABLATION}] Best dev_f1={best_dev_f1:.4f}, thr={best_thr:.2f}")

results_path = os.path.join(os.path.dirname(__file__), "results.txt")
with open(results_path, "a", encoding="utf-8") as f:
    f.write(f"{ABLATION:<20} best_dev_f1={best_dev_f1:.4f}  thr={best_thr:.2f}\n")
print(f"Result appended to {results_path}")
