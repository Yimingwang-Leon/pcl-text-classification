"""
Generate synthetic PCL examples via OpenAI API.
  Strategy A: pattern-diverse paraphrase of label-3/4 examples 
  Strategy B: fresh generation for hard/underrepresented keywords 
Output: data/aug/pcl_aug.csv  (columns: text, binary_label)
Usage:  python src/augment.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import argparse
import pandas as pd
from pathlib import Path
from openai import OpenAI
from src.data_utils import load_raw_data

parser = argparse.ArgumentParser()
parser.add_argument("--model",         default="gpt-4o-mini")
parser.add_argument("--n_per_example", type=int, default=1,
                    help="Paraphrases per source example (Strategy A)")
parser.add_argument("--n_fresh",       type=int, default=5,
                    help="Fresh examples per (keyword, pattern) in Strategy B")
parser.add_argument("--out",           default="data/aug/pcl_aug.csv")
parser.add_argument("--sleep",         type=float, default=0.3)
args = parser.parse_args()

client = OpenAI()

# Keyword descriptions
KEYWORD_DESC = {
    "homeless":      "homeless people",
    "in-need":       "people in need / living in poverty",
    "hopeless":      "people in desperate or hopeless situations",
    "poor-families": "poor families",
    "refugee":       "refugees",
    "vulnerable":    "vulnerable groups",
    "disabled":      "people with disabilities",
    "women":         "women (in disadvantaged contexts)",
    "migrant":       "migrants",
    "immigrant":     "immigrants",
}

# Five PCL patterns
PCL_PATTERNS = {
    "savior": (
        "Savior / unequal-power framing: the author presents themselves or "
        "society as a powerful benefactor swooping in to rescue the helpless group "
        "(e.g., \"we must save these poor people\", \"lucky to have us helping them\")."
    ),
    "charity": (
        "Charity / self-congratulatory framing: describes charitable or "
        "volunteer work toward the group in a way that centres the helper's "
        "goodness rather than the group's agency "
        "(e.g., \"dedicating her life to these unfortunate families\")."
    ),
    "pity": (
        "Pity / victimhood emphasis: portrays the group as passive, suffering "
        "objects of pity with no agency, often with excessive emotional language "
        "(e.g., \"these poor souls have no choice but to...\")."
    ),
    "praise": (
        "Condescending praise: praises the group in a way that implies "
        "low prior expectations "
        "(e.g., \"remarkable what disabled people can achieve\", "
        "\"you would never guess a refugee wrote this\")."
    ),
    "othering": (
        "Othering / distancing language: creates an us-vs-them divide, "
        "treats the group as exotic or fundamentally different "
        "(e.g., \"these people's culture\", \"they just don't understand how things work here\")."
    ),
}

# Keywords with most false negatives + underrepresented in label-3/4
HARD_KEYWORDS = ["poor-families", "hopeless", "women", "homeless", "immigrant", "migrant"]

# Load source data
train_ids = pd.read_csv("data/raw/train_semeval_parids-labels.csv")["par_id"].values
raw       = load_raw_data()
train_raw = raw[raw["par_id"].isin(train_ids)].copy()
train_raw["text"] = train_raw["text"].str.strip()

src_A     = train_raw[train_raw["label"].isin([3, 4])].reset_index(drop=True)
src_B_ref = train_raw[train_raw["label"] == 2].reset_index(drop=True)

print(f"Strategy A source (label 3+4): {len(src_A)}")
print(f"Strategy B reference (label 2): {len(src_B_ref)}")

OUT_A     = Path("data/aug/pcl_aug_A.csv")
OUT_B     = Path("data/aug/pcl_aug_B.csv")
out_final = Path(args.out)
OUT_A.parent.mkdir(parents=True, exist_ok=True)

# Strategy A: pattern-diverse paraphrase of label-3/4
SYS_A = (
    "You are a data augmentation assistant for NLP research on patronizing and "
    "condescending language (PCL) toward vulnerable groups.\n\n"
    "PCL is language that, while often appearing well-intentioned or charitable, "
    "subtly talks DOWN to vulnerable people rather than treating them as equals.\n\n"
    "Your task: given an original PCL sentence and a target PCL pattern, produce ONE "
    "rewritten version that:\n"
    "  1. Preserves the core patronizing meaning\n"
    "  2. Uses the specified PCL pattern as the dominant rhetorical device\n"
    "  3. Changes the wording, structure, and/or perspective from the original\n"
    "  4. Sounds realistic — newspaper, social media, or charity organisation style\n"
    "  5. Is subtle rather than overt (no explicit slurs or hate speech)\n"
    "  6. Keeps roughly the same length as the original\n\n"
    "Return ONLY the rewritten sentence. No explanation, no quotes, no prefix."
)

def paraphrase_A(text: str, keyword: str, pattern_name: str) -> str:
    group = KEYWORD_DESC.get(keyword, keyword)
    user_msg = (
        f"Target group: {group}\n"
        f"PCL pattern to use: {pattern_name} — {PCL_PATTERNS[pattern_name]}\n"
        f"Original: {text}\n"
        "Rewrite:"
    )
    resp = client.chat.completions.create(
        model=args.model, max_tokens=512, temperature=0.9,
        messages=[{"role": "system", "content": SYS_A},
                  {"role": "user",   "content": user_msg}],
    )
    return resp.choices[0].message.content.strip()


if OUT_A.exists():
    results_A = pd.read_csv(OUT_A).to_dict("records")
    done = len(results_A)
    print(f"\nResuming Strategy A from {done} already done")
else:
    results_A, done = [], 0

pattern_names = list(PCL_PATTERNS.keys())
total_A  = len(src_A) * args.n_per_example
errors_A = 0

print(f"\n── Strategy A: {total_A} calls ──")
for i, row in src_A.iterrows():
    for j in range(args.n_per_example):
        idx = i * args.n_per_example + j
        if idx < done:
            continue
        pattern = pattern_names[idx % len(pattern_names)]
        try:
            aug_text = paraphrase_A(row["text"], row["keyword"], pattern)
            results_A.append({
                "text": aug_text, "binary_label": 1,
                "keyword": row["keyword"], "src_label": int(row["label"]),
                "strategy": "A_paraphrase", "pcl_pattern": pattern,
            })
            time.sleep(args.sleep)
            if (idx + 1) % 50 == 0 or (idx + 1) == total_A:
                pd.DataFrame(results_A).to_csv(OUT_A, index=False)
                print(f"  A [{idx+1:>4}/{total_A}] saved  (errors: {errors_A})")
        except Exception as e:
            errors_A += 1
            print(f"  A [{idx+1:>4}/{total_A}] ERROR: {e}")

pd.DataFrame(results_A).to_csv(OUT_A, index=False)
print(f"Strategy A done. {len(results_A)} examples (errors: {errors_A})")

# Strategy B: fresh generation for hard/underrepresented keywords
SYS_B = (
    "You are writing synthetic training data for an NLP classifier that detects "
    "patronizing and condescending language (PCL) toward vulnerable social groups.\n\n"
    "PCL is language that sounds compassionate or neutral on the surface but actually "
    "talks DOWN to vulnerable people — treating them as objects of pity, charity cases, "
    "or inferiors rather than as autonomous individuals.\n\n"
    "IMPORTANT characteristics of good PCL examples for training:\n"
    "  - Subtle, not extreme: avoid hate speech, slurs, or obvious insults\n"
    "  - Sound like real text: news articles, NGO reports, social media posts\n"
    "  - The patronizing tone comes from framing and word choice, not explicit statements\n"
    "  - 1-3 sentences, newspaper/charity register\n\n"
    "Return ONLY the sentences, one per line, no numbering, no explanation."
)

def generate_fresh(keyword: str, pattern_name: str, anchors: list) -> list:
    group = KEYWORD_DESC.get(keyword, keyword)
    anchor_str = ""
    if anchors:
        anchor_str = (
            "Real examples of subtle PCL about this group (for style reference):\n"
            + "\n".join(f"  - {ex}" for ex in anchors[:3])
            + "\n\n"
        )
    user_msg = (
        f"Target group: {group}\n"
        f"PCL pattern: {pattern_name} — {PCL_PATTERNS[pattern_name]}\n"
        f"{anchor_str}"
        f"Write {args.n_fresh} distinct sentences about {group} that exhibit this PCL pattern."
    )
    resp = client.chat.completions.create(
        model=args.model, max_tokens=1024, temperature=1.0,
        messages=[{"role": "system", "content": SYS_B},
                  {"role": "user",   "content": user_msg}],
    )
    lines = [l.strip("- •\t ") for l in resp.choices[0].message.content.strip().splitlines() if l.strip()]
    return lines[:args.n_fresh]


if OUT_B.exists():
    results_B = pd.read_csv(OUT_B).to_dict("records")
    done_B = {(r["keyword"], r["pcl_pattern"]) for r in results_B}
    print(f"\nResuming Strategy B: {len(done_B)} (keyword, pattern) pairs already done")
else:
    results_B, done_B = [], set()

anchors     = {kw: src_B_ref[src_B_ref["keyword"] == kw]["text"].tolist() for kw in HARD_KEYWORDS}
total_B     = len(HARD_KEYWORDS) * len(PCL_PATTERNS)
done_count  = 0
errors_B    = 0

print(f"\n── Strategy B: {total_B} calls ──")
for kw in HARD_KEYWORDS:
    for pattern in PCL_PATTERNS:
        if (kw, pattern) in done_B:
            done_count += 1
            continue
        try:
            fresh = generate_fresh(kw, pattern, anchors.get(kw, []))
            for sent in fresh:
                if len(sent) > 20:
                    results_B.append({
                        "text": sent, "binary_label": 1,
                        "keyword": kw, "src_label": -1,
                        "strategy": "B_fresh", "pcl_pattern": pattern,
                    })
            done_count += 1
            print(f"  B [{done_count}/{total_B}] {kw} × {pattern}: {len(fresh)} sentences")
            time.sleep(args.sleep)
        except Exception as e:
            errors_B += 1
            print(f"  B [{done_count}/{total_B}] ERROR {kw}×{pattern}: {e}")

pd.DataFrame(results_B).to_csv(OUT_B, index=False)
print(f"Strategy B done. {len(results_B)} examples (errors: {errors_B})")

# Merge & deduplicate
aug    = pd.concat([pd.read_csv(OUT_A), pd.read_csv(OUT_B)], ignore_index=True)
before = len(aug)
aug    = aug.drop_duplicates(subset="text").reset_index(drop=True)
aug[["text", "binary_label"]].to_csv(out_final, index=False)

print(f"\n{'='*50}")
print(f"  Total (before dedup): {before}")
print(f"  Total (after dedup) : {len(aug)}")
print(f"  Strategy A          : {(aug['strategy']=='A_paraphrase').sum()}")
print(f"  Strategy B          : {(aug['strategy']=='B_fresh').sum()}")
print(f"  Saved to            : {out_final}")
print(f"{'='*50}")
