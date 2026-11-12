"""Bag of words on 1,000 hotel reviews: one-hot, counts, TF-IDF and their limits (no Streamlit here).

The corpus is assets/hotel_reviews.csv (built by data_prep/make_hotel_reviews.py): 1,000 TripAdvisor
reviews of 50-100 words, label 0 = neutral (3 stars), 1 = positive (4-5), 2 = negative (1-2).

The cleaning is done here, step by step, and not inside scikit-learn, so that the page can show
what every step removes. The vectorizers then only split on spaces.
"""
import re
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "assets" / "hotel_reviews.csv"
CLASSES = ["neutral", "positive", "negative"]                     # label 0, 1, 2
NEGATIONS = frozenset({"not", "no", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere",
                       "cannot", "without"})
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
APOSTROPHE = re.compile(r"[’']")
PUNCT = re.compile(r"[^\w\s]|_")
DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class Cleaning:
    lowercase: bool = True
    punctuation: bool = True
    digits: bool = False
    stop_words: bool = False
    keep_negations: bool = True
    lemmatize: bool = False          # spaCy, applied first, on the raw text: rooms -> room, was -> be
    def stop_list(self):
        if not self.stop_words:
            return frozenset()
        return STOP_WORDS - NEGATIONS if self.keep_negations else STOP_WORDS


@dataclass(frozen=True)
class Vectorizing:
    kind: str = "count"              # "count" or "tfidf"
    ngram_max: int = 1
    min_df: int = 1                  # keep a word only if it is in at least min_df reviews
    max_df: float = 1.0              # ... and in at most max_df of them
    max_features: int | None = None  # ... and among the max_features most frequent


def load_reviews():
    return pd.read_csv(DATA)


# ------------------------------------------------------------------ lemmatization (spaCy)
@lru_cache(maxsize=1)
def nlp():
    "spaCy's small English model, without the parser and the named entities (only the lemmas are needed)."
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


@lru_cache(maxsize=4)
def lemmatize_all(texts):
    "Every text replaced by the lemmas of its tokens (`texts` is a tuple, so the result is cached)."
    return tuple(" ".join(t.lemma_ for t in doc) for doc in nlp().pipe(texts, batch_size=64))


# ------------------------------------------------------------------ cleaning
def normalise(text, c: Cleaning, lemmatized=False):
    "The text after lemmas, case, punctuation and digits, before stop words (still a string)."
    if c.lemmatize and not lemmatized:
        text = lemmatize_all((text,))[0]
    if c.lowercase:
        text = text.lower()
    if c.punctuation:
        text = PUNCT.sub(" ", APOSTROPHE.sub("", text))     # "didn't" -> "didnt", "room." -> "room"
    if c.digits:
        text = DIGITS.sub(" ", text)
    return text


def clean_tokens(text, c: Cleaning, lemmatized=False):
    stop = c.stop_list()
    return [t for t in normalise(text, c, lemmatized).split() if t not in stop]


def clean_corpus(texts, c: Cleaning):
    "One string per review: its tokens after cleaning, joined by spaces (what the vectorizers read)."
    if c.lemmatize:
        return [" ".join(clean_tokens(t, c, lemmatized=True)) for t in lemmatize_all(tuple(texts))]
    return [" ".join(clean_tokens(t, c)) for t in texts]


def explain(text, c: Cleaning, vocabulary):
    """Every raw word of `text` and what happened to it, for the colour-coded view.

    Returns [(raw word, cleaned token or None, status)], status in
    kept / stop word / out of vocabulary / removed by cleaning. With lemmatization, the raw words are spaCy's
    tokens (it splits "didn't" into "did" + "n't") and each one is shown with its lemma.
    """
    stop, out = c.stop_list(), []
    words = [(t.text, t.lemma_) for t in nlp()(text)] if c.lemmatize else [(w, w) for w in text.split()]
    for raw, base in words:
        pieces = normalise(base, c, lemmatized=True).split()
        if not pieces:
            out.append((raw, None, "removed by cleaning"))
        for piece in pieces:
            status = ("stop word" if piece in stop
                      else "kept" if vocabulary is None or piece in vocabulary else "out of vocabulary")
            out.append((raw, piece, status))
    return out


# ------------------------------------------------------------------ vectorizing
def make_vectorizer(v: Vectorizing):
    cls = TfidfVectorizer if v.kind == "tfidf" else CountVectorizer
    return cls(lowercase=False, token_pattern=r"\S+", ngram_range=(1, v.ngram_max),
               min_df=v.min_df, max_df=v.max_df, max_features=v.max_features)


def fit_matrix(docs, v: Vectorizing):
    """The document-term matrix of all reviews (sparse) and its vocabulary, or (None, []) when the
    filters leave no word at all."""
    vec = make_vectorizer(v)
    try:
        X = vec.fit_transform(docs)
    except ValueError:                                   # "empty vocabulary" / "max_df < min_df"
        return None, np.array([])
    return X, vec.get_feature_names_out()


def word_counts(docs):
    "token -> number of occurrences, and token -> number of reviews it appears in."
    tf, df = Counter(), Counter()
    for doc in docs:
        tokens = doc.split()
        tf.update(tokens)
        df.update(set(tokens))
    return tf, df


# ------------------------------------------------------------------ classification
def classify(docs, labels, v: Vectorizing, seed=42):
    """Logistic regression on an 80/20 stratified split. The vocabulary is learned on the train part only."""
    d_train, d_test, y_train, y_test = train_test_split(docs, labels, test_size=.2, stratify=labels,
                                                        random_state=seed)
    vec = make_vectorizer(v)
    try:
        X_train = vec.fit_transform(d_train)
    except ValueError:
        return None
    X_test = vec.transform(d_test)
    # newton-cg: same solution as the default lbfgs on these sparse counts, ~100x faster (< 0.1 s)
    model = LogisticRegression(solver="newton-cg", max_iter=300).fit(X_train, y_train)
    vocab = set(vec.get_feature_names_out())

    # words of the test reviews the model has never seen: they are silently dropped
    test_tokens = [t for d in d_test for t in d.split()]
    unseen = [t for t in test_tokens if t not in vocab]
    return {
        "train_acc": model.score(X_train, y_train), "test_acc": model.score(X_test, y_test),
        "n_features": X_train.shape[1], "n_train": X_train.shape[0],
        "confusion": confusion_matrix(y_test, model.predict(X_test), labels=[0, 1, 2]),
        "coef": model.coef_, "features": vec.get_feature_names_out(),
        "empty_test": int((X_test.sum(axis=1) == 0).sum()),
        "unseen_share": len(unseen) / max(1, len(test_tokens)), "unseen_examples": Counter(unseen).most_common(12),
        "vectorizer": vec,
    }


def cosine(a, b):
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0
