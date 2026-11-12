"""hotel_glove50.npz - the "Padding & pooling" and "Pretrained embeddings" pages.

The GloVe vectors of lecture 02 (glove-wiki-gigaword-50: 400,000 words, 50 numbers each, trained on
Wikipedia + Gigaword, public domain), cut down to what the demo needs so that the app has no gensim
dependency and no 66 MB download:

    - every word of the 1,000 hotel reviews (assets/hotel_reviews.csv) that GloVe knows,
    - plus the 50,000 most frequent GloVe words, so that sentences typed in the app are covered too.

Stored as float16 (~5 MB). Words are tokenized like GloVe: lower-case, "didn't" -> "did" + "n't".

Usage:  python make_hotel_glove.py [--reviews ../assets/hotel_reviews.csv]
                                   [--out ../assets] [--top 50000]
"""
import argparse
import gzip
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
TOKEN_RE = re.compile(r"n't|[a-z]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower().replace("’", "'").replace("n't", " n't"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", default="../assets/hotel_reviews.csv")
    parser.add_argument("--out", default="../assets")
    parser.add_argument("--top", type=int, default=50_000)
    parser.add_argument("--raw", default="raw/glove-wiki-gigaword-50.gz", help="downloaded here if missing")
    args = parser.parse_args()

    raw = Path(args.raw)
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL} (66 MB)...")
        urllib.request.urlretrieve(URL, raw)

    corpus = {w for text in pd.read_csv(args.reviews)["text"] for w in tokenize(text)}
    words, vectors = [], []
    with gzip.open(raw, "rt", encoding="utf-8") as f:
        first = f.readline().split()
        if len(first) != 2:                     # no "count dim" header line: the first line is a vector
            f.seek(0)
        for rank, line in enumerate(f):
            word, *values = line.rstrip().split(" ")
            if rank < args.top or word in corpus:
                words.append(word)
                vectors.append(np.asarray(values, dtype=np.float32))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "hotel_glove50.npz", words=np.array(words), vectors=np.stack(vectors).astype(np.float16))
    covered = len(corpus & set(words))
    print(f"{out / 'hotel_glove50.npz'}: {len(words):,} words × {len(vectors[0])} "
          f"({covered:,} of the {len(corpus):,} corpus words have a vector, {covered / len(corpus):.1%})")
