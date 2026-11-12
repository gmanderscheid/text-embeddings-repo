"""hotel_reviews.csv - the "Text to features" and "Padding & pooling" pages.

1,000 TripAdvisor hotel reviews of 50 to 100 words, balanced over 3 classes, from the Hugging Face
dataset jniimi/tripadvisor-review-rating (201,295 reviews, Apache-2.0). Nothing to download by
hand: its parquet file (~220 MB) is fetched once into the Hugging Face cache.

    label  sentiment  stars      reviews
    0      neutral    3          334
    1      positive   4 or 5     333
    2      negative   1 or 2     333

Columns: text, label, sentiment, stars. The text is kept raw (case, punctuation, digits): cleaning it
is the point of the lab.

Usage:  python make_hotel_reviews.py [--seed 42] [--out ../assets]
"""
import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET = "jniimi/tripadvisor-review-rating"
PARQUET = "default/train/0000.parquet"          # the parquet conversion of the dataset (branch below)
TARGET = {0: 334, 1: 333, 2: 333}
NAMES = {0: "neutral", 1: "positive", 2: "negative"}
MIN_WORDS, MAX_WORDS = 50, 100


def label_of(stars):
    return 0 if stars == 3 else 1 if stars >= 4 else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="../assets")
    args = parser.parse_args()

    path = hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision="refs/convert/parquet")
    raw = pd.read_parquet(path, columns=["text", "overall", "lang"])
    raw = raw[raw["lang"] == "__label__en"].copy()
    raw["text"] = raw["text"].astype(str).str.split().str.join(" ")          # one line per review
    raw = raw.drop_duplicates("text")
    words = raw["text"].str.split().str.len()
    raw = raw[(words >= MIN_WORDS) & (words <= MAX_WORDS)]
    raw["label"] = raw["overall"].map(label_of)
    print(f"{len(raw):,} English reviews of {MIN_WORDS}-{MAX_WORDS} words:",
          raw["label"].map(NAMES).value_counts().to_dict())

    parts = [raw[raw["label"] == k].sample(n, random_state=args.seed) for k, n in TARGET.items()]
    df = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df = pd.DataFrame({"text": df["text"], "label": df["label"], "sentiment": df["label"].map(NAMES),
                       "stars": df["overall"].astype(int)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hotel_reviews.csv", index=False)
    words = df["text"].str.split().str.len()
    print(f"\n{out / 'hotel_reviews.csv'}: {len(df)} reviews, {words.min()}-{words.max()} words "
          f"(median {words.median():.0f})")
    print(df["sentiment"].value_counts().to_string())
