from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import re

def load_raw_data(path: str = "data/raw/dontpatronizeme_pcl.tsv") -> pd.DataFrame:
    data = pd.read_csv(path, skiprows=4, sep="\t", header=None, engine="python")
    data.columns = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    data = data.dropna(subset=["text"])
    return data


def binarize(data: pd.DataFrame) -> pd.DataFrame:
    data = data.set_index("par_id")
    data["binary_label"] = data["label"].apply(lambda x: 1 if x >= 2 else 0)
    return data[["text", "binary_label"]]


def split_data(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, dev_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["binary_label"],
    )
    return train_df, dev_df


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)  
    text = re.sub(r"\s+", " ", text)       
    return text.strip()

def save_splits(
    train_df: pd.DataFrame, dev_df: pd.DataFrame, out_dir: str = "data/clean"
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_path / "train.csv")
    dev_df.to_csv(out_path / "dev.csv")


def load_official_split(
    raw_path: str = "data/raw/dontpatronizeme_pcl.tsv",
    train_ids_path: str = "train_semeval_parids-labels.csv",
    dev_ids_path: str = "dev_semeval_parids-labels.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data using the official SemEval train/dev split."""
    data = load_raw_data(raw_path)
    data = binarize(data)
    data["text"] = data["text"].apply(clean_text)

    train_ids = pd.read_csv(train_ids_path)["par_id"].values
    dev_ids = pd.read_csv(dev_ids_path)["par_id"].values

    train_df = data.loc[data.index.isin(train_ids)]
    dev_df = data.loc[data.index.isin(dev_ids)]
    return train_df, dev_df


def load_test_data(
    test_path: str = "dontpatronizeme/semeval-2022/TEST/task4_test.tsv",
) -> pd.DataFrame:
    """Load the official test set (no labels)."""
    data = pd.read_csv(test_path, sep="\t", header=None, engine="python")
    data.columns = ["test_id", "art_id", "keyword", "country_code", "text"]
    data["text"] = data["text"].apply(clean_text)
    return data

class PCLDataset(Dataset):
    def __init__(self, texts, tokenizer, labels, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length = self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0), # (seq_len, )
            "attention_mask": enc["attention_mask"].squeeze(0), # (seq_len, )
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        return item
if __name__ == "__main__":
    train_df, dev_df = load_official_split()
    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}")
    print(train_df["binary_label"].value_counts(normalize=True))
    print(dev_df["binary_label"].value_counts(normalize=True))
    save_splits(train_df, dev_df)
