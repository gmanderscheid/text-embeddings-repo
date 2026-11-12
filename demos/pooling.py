"""Padding and pooling: from token vectors to one vector per review, then a classic model (no Streamlit here).

Embeddings: GloVe-50 (lecture 02), cut down by data_prep/make_hotel_glove.py to assets/hotel_glove50.npz.
Corpus: the same 1,000 hotel reviews and the same 80/20 split as the "Text to features" page, so the
results of the two pages can be compared.
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CLASSES = ["neutral", "positive", "negative"]
TOKEN_RE = re.compile(r"n't|[a-z]+")
SEED = 42


def tokenize(text):
    "GloVe's tokenization: lower-case words, and “didn't” -> “did” + “n't” (so the negation keeps a vector)."
    return TOKEN_RE.findall(text.lower().replace("’", "'").replace("n't", " n't"))


class Glove:
    def __init__(self):
        z = np.load(ASSETS / "hotel_glove50.npz")
        self.words = z["words"]
        self.vectors = z["vectors"].astype(np.float32)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.dim = self.vectors.shape[1]

    def embed(self, tokens):
        "(length, dim) matrix and the mask of tokens that have a vector (unknown words get zeros, mask 0)."
        known = np.array([t in self.index for t in tokens], dtype=bool)
        M = np.zeros((len(tokens), self.dim), dtype=np.float32)
        for i, t in enumerate(tokens):
            if known[i]:
                M[i] = self.vectors[self.index[t]]
        return M, known


def pad_batch(matrices, masks, max_len=None):
    "Stack variable-length (length, dim) matrices into (batch, L, dim) with zeros, and the (batch, L) mask."
    L = max_len or max(len(m) for m in matrices)
    dim = matrices[0].shape[1]
    X = np.zeros((len(matrices), L, dim), dtype=np.float32)
    mask = np.zeros((len(matrices), L), dtype=bool)
    for i, (m, k) in enumerate(zip(matrices, masks)):
        n = min(L, len(m))
        X[i, :n], mask[i, :n] = m[:n], k[:n]
    return X, mask


def pool(X, mask, mode):
    "(batch, L, dim) -> (batch, dim). mean/max respect the mask; the *_nomask variants show what goes wrong."
    m = mask[..., None]
    if mode == "mean":
        return (X * m).sum(1) / np.maximum(m.sum(1), 1)
    if mode == "mean_nomask":                         # pads (zeros) are averaged in
        return X.mean(1)
    if mode == "max":
        return np.where(m, X, -np.inf).max(1).clip(min=-1e9)   # pads can never win
    if mode == "max_zeropad":                         # pads at 0 win wherever every word is negative
        return X.max(1)
    if mode == "mean+max":
        return np.concatenate([pool(X, mask, "mean"), pool(X, mask, "max")], axis=1)
    raise ValueError(mode)


# ------------------------------------------------------------------ the corpus
def load_reviews():
    return pd.read_csv(ASSETS / "hotel_reviews.csv")


def split(labels):
    "The 80/20 stratified split of the Text to features page (same seed, same reviews)."
    return train_test_split(np.arange(len(labels)), test_size=.2, stratify=labels, random_state=SEED)


def corpus_tensor(glove, texts, max_len=None):
    embedded = [glove.embed(tokenize(t)) for t in texts]
    return pad_batch([e[0] for e in embedded], [e[1] for e in embedded], max_len)


def xgboost():
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.05, subsample=.8, colsample_bytree=.8,
                         random_state=0, n_jobs=-1, verbosity=0)


def logreg():
    return LogisticRegression(solver="newton-cg", max_iter=500)


def _fit(name, family, X, y, train, test, model):
    t0 = time.perf_counter()
    model.fit(X[train], y[train])
    return {"representation": name, "family": family, "model": type(model).__name__.replace("Classifier", ""),
            "features": X.shape[1], "train": model.score(X[train], y[train]), "test": model.score(X[test], y[test]),
            "seconds": time.perf_counter() - t0}


def compare(glove, df, max_len=None, n_train=None):
    """Every representation x model on the same split. n_train < 800 uses a random subset of the train reviews."""
    y = df["label"].to_numpy()
    train, test = split(y)
    if n_train and n_train < len(train):
        train = np.random.default_rng(0).choice(train, n_train, replace=False)
    texts = df["text"].tolist()
    rows = []
    for vec, name in [(CountVectorizer(token_pattern=r"[a-z']+"), "bag of words (counts)"),
                      (TfidfVectorizer(token_pattern=r"[a-z']+"), "bag of words (TF-IDF)")]:
        vec.fit([texts[i] for i in train])
        B = vec.transform(texts)
        rows.append(_fit(name, "one axis per word", B, y, train, test, logreg()))
        if name.endswith("(counts)"):
            rows.append(_fit(name, "one axis per word", B, y, train, test, xgboost()))
    X, mask = corpus_tensor(glove, texts, max_len)
    for mode, name in [("mean", "GloVe, mean pooling"), ("max", "GloVe, max pooling"), ("mean+max", "GloVe, mean + max")]:
        P = pool(X, mask, mode)
        rows.append(_fit(name, "pooled embeddings", P, y, train, test, xgboost()))
        if mode == "mean":
            rows.append(_fit(name, "pooled embeddings", P, y, train, test, logreg()))
    return rows


def learning_curve(glove, df, sizes=(50, 100, 200, 400, 800), repeats=3):
    "Test accuracy vs number of train reviews: BoW + logistic regression vs mean-pooled GloVe + XGBoost / logreg."
    y = df["label"].to_numpy()
    train, test = split(y)
    texts = df["text"].tolist()
    X, mask = corpus_tensor(glove, texts)
    G = pool(X, mask, "mean")
    rng = np.random.default_rng(0)
    out = []
    for n in sizes:
        for r in range(repeats if n < len(train) else 1):
            sub = rng.choice(train, n, replace=False) if n < len(train) else train
            vec = CountVectorizer(token_pattern=r"[a-z']+").fit([texts[i] for i in sub])
            B = vec.transform(texts)
            out.append({"n": n, "model": "bag of words + logistic regression",
                        "test": logreg().fit(B[sub], y[sub]).score(B[test], y[test])})
            out.append({"n": n, "model": "GloVe mean + XGBoost", "test": xgboost().fit(G[sub], y[sub]).score(G[test], y[test])})
            out.append({"n": n, "model": "GloVe mean + logistic regression",
                        "test": logreg().fit(G[sub], y[sub]).score(G[test], y[test])})
    return pd.DataFrame(out).groupby(["n", "model"], as_index=False)["test"].mean()


# ------------------------------------------------------------------ the models used on typed sentences
def fit_sentence_models(glove, df):
    "Models trained on the 800 train reviews, to predict sentences typed in the app."
    y = df["label"].to_numpy()
    train, _ = split(y)
    texts = df["text"].tolist()
    X, mask = corpus_tensor(glove, texts)
    models = {mode: xgboost().fit(pool(X, mask, mode)[train], y[train]) for mode in ("mean", "max")}
    vec = CountVectorizer(token_pattern=r"[a-z']+").fit([texts[i] for i in train])
    models["bow"] = (vec, logreg().fit(vec.transform([texts[i] for i in train]), y[train]))
    return models


def predict_sentences(glove, models, sentences):
    tokens = [tokenize(s) for s in sentences]
    X, mask = pad_batch(*zip(*[glove.embed(t) if t else (np.zeros((1, glove.dim), np.float32), np.zeros(1, bool))
                              for t in tokens]))
    vec, bow = models["bow"]
    seen = set(vec.vocabulary_)
    return {
        "mean": models["mean"].predict_proba(pool(X, mask, "mean")),
        "max": models["max"].predict_proba(pool(X, mask, "max")),
        "bow": bow.predict_proba(vec.transform(sentences)),
        "unseen": [[t for t in re.findall(r"[a-z']+", s.lower()) if t not in seen] for s in sentences],
    }


# ------------------------------------------------------------------ what the animation needs
def animation_data(glove, models, sentences, max_tokens=10, shown_dims=8):
    tokens = [tokenize(s)[:max_tokens] for s in sentences]
    tokens = [t for t in tokens if t] or [["empty"]]
    embedded = [glove.embed(t) for t in tokens]
    X, mask = pad_batch([e[0] for e in embedded], [e[1] for e in embedded])

    # the sentence the pooling close-up focuses on: the first one with a dim where every word is negative
    # (if any), to show why a max pooling must not let a 0 pad win; that dim is among the dims shown
    focus, negative = 0, []
    for i in range(len(tokens)):
        real = X[i, mask[i]]
        neg = [d for d in range(glove.dim) if len(real) and (real[:, d] < 0).all()]
        if neg:
            focus, negative = i, neg
            break
    chosen = set(negative[:1])
    for d in range(glove.dim):
        if len(chosen) == shown_dims:
            break
        chosen.add(d)
    dims = sorted(chosen)
    neg_dim = dims.index(negative[0]) if negative else -1

    preds = predict_sentences(glove, models, [" ".join(t) for t in tokens])
    r2 = lambda a: np.round(a[:, dims].astype(np.float64), 2).tolist()
    return {
        "sents": [{"tokens": t, "known": e[1].tolist(), "vecs": np.round(e[0][:, dims].astype(np.float64), 2).tolist()}
                  for t, e in zip(tokens, embedded)],
        "L": X.shape[1], "dims": dims, "D": glove.dim, "negDim": neg_dim, "focus": focus,
        "pooled": {m: r2(pool(X, mask, m)) for m in ("mean", "mean_nomask", "max", "max_zeropad")},
        "preds": {m: np.round(preds[m].astype(np.float64), 3).tolist() for m in ("mean", "max")},
        "classes": CLASSES,
    }
