# Patronising and Condescending Language (PCL) Detection
**Task:** Binary classification — detect PCL toward vulnerable social groups (SemEval 2022 Task 4)
**Leaderboard name:** `[TidySleet]`
**Repository:** https://github.com/Yimingwang-Leon/pcl-text-classification.git

---

## Results

| Split | F1 (positive class) |
|-------|-------------------|
| Official dev set | [0.6185] |
| Official test set | [0.6140] |

Prediction files:
- **[`dev.txt`](dev.txt)** — 2,093 predictions on the official dev set
- **[`test.txt`](test.txt)** — 3,832 predictions on the official test set

---

## Repository Structure

```
.
├── dev.txt                     # dev set predictions (0/1 per line)
├── test.txt                    # test set predictions (0/1 per line)
│
├── src/
│   ├── data_utils.py           # Data loading, binarisation, cleaning, Dataset class
│   ├── train.py                # BestModel training script (RoBERTa-large)
│   ├── augment.py              # Data augmentation via OpenAI API (GPT-4o-mini)
│   └── predict.py              # Load checkpoint → generate dev.txt / test.txt
│
├── EDA/
│   └── eda.ipynb               # Exploratory Data Analysis
│
├── error_analysis/
│   ├── error_analysis.ipynb    # Error analysis notebook
│   ├── baseline_model.py       # Baseline (RoBERTa-base, 2 epochs, no augmentation)
│   ├── baseline_dev.txt        # Baseline predictions on dev set
│   ├── ea_utils.py             # Shared utilities for error analysis
│   └── error_examples.txt      # Categorised error examples (FP/FN/both wrong/both right)
│
├── ablation/
│   ├── ablation_train.py       # Ablation study script
│   └── results.txt             # Ablation results summary
│
└── img/                        # Figures used in the report
```

> **BestModel code:** [`src/train.py`](src/train.py) — see *Training* section below.
> The model checkpoint (`BestModel/best_model.pt`, ~1.4 GB) is excluded from version control due to file size.

---

## Approach (BestModel)

**Model:** `FacebookAI/roberta-large` fine-tuned for binary sequence classification

**Key components:**

1. **Partial layer freezing** — embeddings and first 12 encoder layers frozen; only the top 12 layers and classification head are trained. Reduces overfitting on the small PCL-positive class.

2. **Data augmentation** — GPT-4o-mini generates synthetic PCL examples via two strategies:
   - *Strategy A:* paraphrase label-3/4 examples across 5 PCL rhetorical patterns (savior, charity, pity, praise, othering)
   - *Strategy B:* generate fresh examples for underrepresented keywords (poor-families, hopeless, women, homeless, immigrant, migrant)

3. **Threshold optimisation** — after each epoch, the classification threshold is searched over [0.05, 0.95] on an internal validation split (10% of train) to maximise positive-class F1. The best threshold is then applied to the official dev set.

4. **Training details:**
   - Loss: CrossEntropyLoss with label smoothing 0.1
   - Optimiser: AdamW (weight decay 0.01 on 2D+ params only)
   - LR schedule: linear warmup (20% of steps) → cosine decay, peak 1e-5
   - Early stopping: patience 5 epochs, min delta 1e-4
   - Seed: 42

---

## Setup

```bash
pip install torch transformers scikit-learn pandas numpy openai
```

Data files expected under `data/raw/` (download from the [SemEval-2022 Task 4 official repository](https://github.com/Perez-AlmendrosC/dontpatronizeme)):
- `dontpatronizeme_pcl.tsv`
- `train_semeval_parids-labels.csv`
- `dev_semeval_parids-labels.csv`
- `task4_test.tsv`

---

## Training

```bash
cd src
python train.py
```

This will:
1. Load the official SemEval train/dev split
2. Load augmented data from `data/aug/pcl_aug.csv`
3. Train RoBERTa-large with the settings above
4. Save the best checkpoint to `BestModel/model_MMDD_HHMM_epXX_devF1.pt`

**To regenerate augmented data** (requires OpenAI API key in environment):
```bash
python src/augment.py
```

---

## Generating Predictions

Update `CKPT_PATH` and `BEST_THR` in [`src/predict.py`](src/predict.py), then:

```bash
cd src
python predict.py
```

Outputs `dev.txt` and `test.txt` in the project root.

---

## Ablation Study

Two ablation conditions (see [`ablation/results.txt`](ablation/results.txt)):

| Condition | Dev F1 | Notes |
|-----------|--------|-------|
| Full model (w/ augmentation) | [YOUR DEV F1] | BestModel |
| no_aug | 0.6009 | training data without augmentation |
| no_freeze | 0.6125 | all layers trainable |
| no_threshold | 0.5926 | fixed threshold 0.5 |

To re-run experiments (seed is fixed at 42, but results may vary slightly due to GPU non-determinism):
```bash
python ablation/ablation_train.py --ablation no_freeze
python ablation/ablation_train.py --ablation no_threshold
```

The `no_aug` condition was run using `src/train.py` with the augmentation data loading commented out.

---

## Error Analysis

See [`error_analysis/error_analysis.ipynb`](error_analysis/error_analysis.ipynb) for a comparison of BestModel vs. RoBERTa-base baseline across four error categories:

| Category | Count |
|----------|-------|
| Both correct | 1,873 |
| Both wrong | 97 |
| Main correct, Baseline wrong | 67 |
| Main wrong, Baseline correct | 56 |
