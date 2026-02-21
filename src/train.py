import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from data_utils import load_official_split, PCLDataset
from sklearn.model_selection import train_test_split
import inspect
import math
from pathlib import Path
import random
import pandas as pd

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEBUG_SMOKE = False
SMOKE_TRAIN_N = 256
SMOKE_VAL_N = 128
BATCH_SIZE = 8 if DEBUG_SMOKE else 16
EPOCHS = 1 if DEBUG_SMOKE else 20
MAX_EPOCHS = 20 
PATIENCE = 5
MIN_DELTA = 1e-4        


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "FacebookAI/roberta-large"

# Load data
train_df, val_df = load_official_split()

# Convert data into List type
train_texts_full = train_df["text"].to_list()
train_labels_full = train_df["binary_label"].to_list()

# Split a internal val data for hyper tunning
train_texts, internal_texts, train_labels, internal_labels = train_test_split(
    train_texts_full,
    train_labels_full,
    test_size=0.1,
    random_state=42,
    stratify=train_labels_full
)

# Augmented data
aug_df = pd.read_csv("data/aug/pcl_aug.csv")
train_texts  = train_texts  + aug_df["text"].tolist()
train_labels = train_labels + aug_df["binary_label"].tolist()
print(f"Aug data loaded: {len(aug_df)} examples")

official_dev_texts = val_df["text"].to_list()
official_dev_labels = val_df["binary_label"].to_list()

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if DEBUG_SMOKE:
    train_texts = train_texts[:SMOKE_TRAIN_N]
    train_labels = train_labels[:SMOKE_TRAIN_N]

    internal_texts = internal_texts[:SMOKE_VAL_N]
    internal_labels = internal_labels[:SMOKE_VAL_N]

    official_dev_texts = official_dev_texts[:SMOKE_VAL_N]
    official_dev_labels = official_dev_labels[:SMOKE_VAL_N]

# Build dataset
train_dataset = PCLDataset(train_texts, tokenizer, train_labels)
internal_dataset = PCLDataset(internal_texts, tokenizer, internal_labels)
official_dev_dataset = PCLDataset(official_dev_texts, tokenizer, official_dev_labels)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
internal_loader = DataLoader(
    internal_dataset, 
    batch_size=2*BATCH_SIZE, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=True
)
official_dev_loader = DataLoader(
    official_dev_dataset,
    batch_size=2*BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)

decay, no_decay = [], []
def configure_optimizers(model, weight_decay, learning_rate, device):
    # Collect parameters that require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Create parameter groups: apply weight decay only to 2D (and higher) tensors
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} params")
    print(f"num nodecayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} params")

    # Use fused AdamW if available and on CUDA
    # fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    # use_fused = fused_available and (device.type == "cuda")
    # print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=False
        )
    return optimizer

max_lr = 1e-5
min_lr = 2e-6
steps_per_epoch = len(train_loader)  
max_steps = EPOCHS * steps_per_epoch  
warmup_steps = int(0.2 * max_steps)  

def get_lr(it):
    # 1) Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # 2) After training steps, keep min lr
    if it > max_steps:
        return min_lr

    # 3) Cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

base_model_prefix = model.base_model_prefix
base_model = getattr(model, base_model_prefix)
for param in base_model.embeddings.parameters():
    param.requires_grad = False
freeze_until_layer = 12
for i, layer in enumerate(base_model.encoder.layer):
    if i < freeze_until_layer:
        for param in layer.parameters():
            param.requires_grad = False

print(f"Successfully froze {base_model_prefix} embeddings and first {freeze_until_layer} layers.")

optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=2e-5, device=device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
print('Loss: CrossEntropyLoss(label_smoothing=0.1)')

def train_one_epoch(model, loader, optimizer, criterion, global_step):
    model.train()
    total_loss = 0.0

    for batch in loader:
        lr = get_lr(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits.float(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, global_step

@torch.no_grad()
def evaluate_pos_f1(model, loader):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = torch.argmax(logits, dim=-1)

        preds.extend(pred.cpu().tolist())
        golds.extend(labels.cpu().tolist())

    return f1_score(golds, preds, pos_label=1, average="binary")

from sklearn.metrics import precision_recall_fscore_support

@torch.no_grad()
def evaluate_pos_metrics(model, loader):
    model.eval()
    preds, golds = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

        pred = torch.argmax(logits, dim=-1)

        preds.extend(pred.cpu().tolist())
        golds.extend(labels.cpu().tolist())

    p, r, f1, _ = precision_recall_fscore_support(
        golds,
        preds,
        average="binary",
        pos_label=1
    )

    return p, r, f1

@torch.no_grad()
def collect_pos_probs_and_labels(model, loader, device):
    """Collect p(y=1) and gold labels from a dataloader."""
    model.eval()
    probs, golds = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        p1 = torch.softmax(logits.float(), dim=-1)[:, 1]   

        probs.extend(p1.cpu().tolist())
        golds.extend(labels.cpu().tolist())

    return np.array(probs, dtype=np.float32), np.array(golds, dtype=np.int64)


def best_threshold_by_f1(probs, golds, thresholds=None):
    """Find threshold that maximizes positive-class F1 on given probs/golds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)  

    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        f1 = f1_score(golds, preds, average="binary", pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, float(best_f1)

# Start training
import time
Path("BestModel").mkdir(parents=True, exist_ok=True)
run_tag = time.strftime("%m%d_%H%M")

best_dev_f1 = -1.0
best_thr = 0.5
bad_epochs = 0
global_step = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, global_step = train_one_epoch(
        model, train_loader, optimizer, criterion, global_step
    )

    # Threshold search on INTERNAL
    internal_probs, internal_golds = collect_pos_probs_and_labels(model, internal_loader, device)
    thr, internal_f1_thr = best_threshold_by_f1(internal_probs, internal_golds)

    internal_preds = (internal_probs >= thr).astype(int)
    tp = int(((internal_preds == 1) & (internal_golds == 1)).sum())
    fp = int(((internal_preds == 1) & (internal_golds == 0)).sum())
    fn = int(((internal_preds == 0) & (internal_golds == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)

    # DEV evaluation using SAME threshold
    dev_probs, dev_golds = collect_pos_probs_and_labels(model, official_dev_loader, device)
    dev_preds = (dev_probs >= thr).astype(int)
    official_dev_f1_thr = f1_score(dev_golds, dev_preds, average="binary", pos_label=1)

    if official_dev_f1_thr > best_dev_f1 + MIN_DELTA:
        best_dev_f1 = official_dev_f1_thr
        best_thr = thr
        bad_epochs = 0
        if epoch > 3:
            ckpt_path = f"BestModel/model_{run_tag}_ep{epoch:02d}_dev{official_dev_f1_thr:.4f}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  >> Saved: {ckpt_path}")
    else:
        bad_epochs += 1

    print(
        f"Epoch {epoch}/{MAX_EPOCHS} | "
        f"loss={train_loss:.4f} | "
        f"thr={thr:.2f} | pos_p={p:.4f} | pos_r={r:.4f} | "
        f"internal_pos_f1={internal_f1_thr:.4f} | "
        f"official_dev_pos_f1={official_dev_f1_thr:.4f} | "
        f"bad_epochs={bad_epochs}/{PATIENCE}"
    )

    if bad_epochs >= PATIENCE:
        print(
            f"Early stopping triggered at epoch {epoch}. "
            f"Best dev_f1={best_dev_f1:.4f}, best_thr={best_thr:.2f}"
        )
        break

print(f"Best dev_f1={best_dev_f1:.4f}, best_thr={best_thr:.2f}")