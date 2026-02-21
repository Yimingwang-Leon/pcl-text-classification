import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_official_split, load_raw_data

DATA_RAW = PROJECT_ROOT / "data" / "raw"


def load_analysis_data():
    """Return (dev_df, true_labels, main_preds, base_preds)."""
    _, dev_df = load_official_split(
        raw_path=str(DATA_RAW / "dontpatronizeme_pcl.tsv"),
        train_ids_path=str(DATA_RAW / "train_semeval_parids-labels.csv"),
        dev_ids_path=str(DATA_RAW / "dev_semeval_parids-labels.csv"),
    )
    dev_df = dev_df.reset_index()

    raw = load_raw_data(str(DATA_RAW / "dontpatronizeme_pcl.tsv"))
    dev_df = dev_df.merge(raw[["par_id", "keyword"]], on="par_id", how="left")

    true_labels = dev_df["binary_label"].to_numpy()
    main_preds = np.array([int(l.strip()) for l in open(PROJECT_ROOT / "dev.txt")])
    base_preds = np.array([int(l.strip()) for l in open(PROJECT_ROOT / "error_analysis" / "baseline_dev.txt")])

    assert len(main_preds) == len(base_preds) == len(true_labels), \
        f"Length mismatch: main={len(main_preds)}, base={len(base_preds)}, true={len(true_labels)}"

    return dev_df, true_labels, main_preds, base_preds
