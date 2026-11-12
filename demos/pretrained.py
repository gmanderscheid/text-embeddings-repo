"""J5 lecture 02 - Pretrained embeddings, toward Transformers (no Streamlit here).

- GloVe-50 (lecture 02's glove-wiki-gigaword-50), cut to the 50,000 most frequent words + the hotel corpus by
  data_prep/make_hotel_glove.py: neighbours, analogies, the "capital of" direction.
- all-MiniLM-L6-v2 from the Hugging Face Hub (sentence-transformers): contextual token vectors, one vector per
  sentence, semantic search on the Play catalogue of the notebook, and the attention of the last layer.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from demos.pooling import Glove

ASSETS = Path(__file__).resolve().parent.parent / "assets"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROBES = ["great", "terrible", "boring", "funny", "alien", "murder", "music", "director"]
ANALOGIES = [("king", "woman", "man"), ("paris", "italy", "france"), ("walking", "swam", "walked"), ("bigger", "small", "big")]
PAIRS = [("france", "paris"), ("italy", "rome"), ("germany", "berlin"), ("spain", "madrid"),
         ("japan", "tokyo"), ("russia", "moscow"), ("egypt", "cairo"), ("canada", "ottawa")]
BANK = {"robbery": "The gang plans to rob the bank downtown before dawn.",
        "money": "She deposits the ransom money at the bank.",
        "river": "They have a picnic on the bank of the river."}
STOP = {"a", "an", "the", "of", "to", "and", "in", "on", "for", "his", "her", "their", "who", "with", "by", "is", "us"}


def catalogue():
    return pd.read_csv(ASSETS / "play_catalogue.csv")


# ------------------------------------------------------------------ GloVe
class StaticVectors(Glove):
    "GloVe with the gensim-like queries of the notebook (most_similar excludes the input words)."

    def __init__(self):
        super().__init__()
        self.unit = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def most_similar(self, positive, negative=(), topn=6):
        missing = [w for w in [*positive, *negative] if w not in self.index]
        if missing:
            return missing, []
        q = sum(self.unit[self.index[w]] for w in positive) - sum(self.unit[self.index[w]] for w in negative)
        sims = self.unit @ (q / np.linalg.norm(q))
        for w in [*positive, *negative]:
            sims[self.index[w]] = -2
        top = np.argsort(-sims)[:topn]
        return [], [(str(self.words[i]), float(sims[i])) for i in top]

    def vec(self, w):
        return self.vectors[self.index[w]]


def pca2(X):
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T, vt[:2]


# ------------------------------------------------------------------ MiniLM
def load_encoder():
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(MODEL, device="cpu")
    model = encoder[0].auto_model
    # "eager" attention: same results, but it can return the attention weights (shown on the page)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    else:
        model.config._attn_implementation = "eager"
    return encoder


def token_vectors(encoder, sentence):
    "The tokens of `sentence` (WordPiece, with [CLS]/[SEP]) and one contextual vector per token."
    vecs = encoder.encode(sentence, output_value="token_embeddings", convert_to_numpy=True)
    tokens = encoder.tokenizer.convert_ids_to_tokens(encoder.tokenizer(sentence)["input_ids"])
    return tokens, np.asarray(vecs)


def word_vector(encoder, sentence, word):
    tokens, vecs = token_vectors(encoder, sentence)
    if word not in tokens:
        return None
    v = vecs[tokens.index(word)]
    return v / np.linalg.norm(v)


def attention_from(encoder, sentence, word, layer=-1):
    """How much the token `word` reads from every token of the sentence in one layer (mean over the heads).
    A preview of the next module: this is attention."""
    import torch
    tok = encoder.tokenizer(sentence, return_tensors="pt")
    model = encoder[0].auto_model
    with torch.no_grad():
        out = model(**tok, output_attentions=True)
    att = out.attentions[layer][0].mean(0).numpy()            # (tokens, tokens)
    tokens = encoder.tokenizer.convert_ids_to_tokens(tok["input_ids"][0])
    i = tokens.index(word)
    return tokens, att[i]


def word_attention(encoder, sentence, word, layer=-1):
    """Attention of `word` over the *other words* of the sentence in one layer (mean over heads), renormalised:
    [CLS], [SEP], punctuation and the word itself are left out ([CLS] alone takes ~40% of the weight, a known
    "attention sink" that carries no meaning)."""
    tokens, w = attention_from(encoder, sentence, word, layer)
    keep = [i for i, t in enumerate(tokens) if t not in ("[CLS]", "[SEP]", word) and re.search(r"\w", t)]
    total = w[keep].sum()
    return [(tokens[i], float(w[i] / total)) for i in keep]


def shared_words(a, b):
    words = lambda text: set(re.sub(r"[.,!?]", "", text.lower()).split()) - STOP
    return sorted(words(a) & words(b))


# ------------------------------------------------------------------ what the animation needs
def animation_data(glove, encoder, cat_vectors, cat, query, bank_word="bank", sentences=BANK, analogy=ANALOGIES[0]):
    r = lambda a, n=2: np.round(np.asarray(a, dtype=np.float64), n).tolist()

    # 1. capitals: 2D projection, arrows country -> capital
    words = [w for p in PAIRS for w in p]
    xy, _ = pca2(np.stack([glove.vec(w) for w in words]))
    xy = xy / np.abs(xy).max()

    # 2. analogy a - c + b in a 2D projection fitted on the words involved and the answers
    a, b, c = analogy
    _, best = glove.most_similar([a, b], [c], topn=3)
    aw = [a, c, b] + [w for w, _ in best]
    axy, axes = pca2(np.stack([glove.vec(w) for w in aw]))
    m = np.stack([glove.vec(w) for w in aw]).mean(0)
    proj = lambda v: (v - m) @ axes.T
    target = glove.vec(a) - glove.vec(c) + glove.vec(b)
    scale = np.abs(np.vstack([axy, proj(target)[None]])).max()

    # 3-4. "bank": GloVe row (static) vs MiniLM token vectors (contextual) + attention of the last layer
    static = glove.vec(bank_word) if bank_word in glove.index else np.zeros(50)
    ctx, atts = [], []
    for name, s in sentences.items():
        v = word_vector(encoder, s, bank_word)
        ctx.append(v if v is not None else np.zeros(384))
        if v is not None:
            pairs = word_attention(encoder, s, bank_word)
            atts.append({"tokens": [t for t, _ in pairs], "weights": [round(x, 3) for _, x in pairs]})
        else:
            atts.append({"tokens": [], "weights": []})
    ctx = np.stack(ctx)

    # 5. semantic search: catalogue + query in 2D
    qv = encoder.encode(query, normalize_embeddings=True)
    scores = cat_vectors @ qv
    sxy, _ = pca2(np.vstack([cat_vectors, qv[None]]))
    sxy = sxy / np.abs(sxy).max()
    stars = {w: [[n, round(c, 2)] for n, c in glove.most_similar([w], topn=6)[1]] for w in ["alien", "director", "boring"]}
    return {
        "stars": stars,
        "capitals": {"words": words, "xy": r(xy, 3)},
        "analogy": {"words": aw, "xy": r(axy / scale, 3), "target": r(proj(target) / scale, 3),
                    "best": [[w, round(s, 2)] for w, s in best], "a": a, "b": b, "c": c},
        "bank": {"word": bank_word, "names": list(sentences), "sentences": list(sentences.values()),
                 "static": r(static[:24]), "ctx": r(ctx[:, :24], 3), "cos_ctx": r(ctx @ ctx.T, 3), "att": atts},
        "search": {"titles": cat["title"].tolist(), "plots": cat["plot"].tolist(), "xy": r(sxy, 3),
                   "query": query, "scores": r(scores, 3),
                   "shared": [shared_words(query, p) for p in cat["plot"]]},
    }


# ------------------------------------------------------------------ the "Cosine similarity" page
COSINE_WORDS = ["great", "excellent", "wonderful", "good", "bad", "awful", "terrible", "boring",
                "pizza", "pasta", "bread", "cheese", "king", "queen", "prince", "castle",
                "hotel", "room", "beach", "river", "bank", "money", "film", "movie", "music", "dog", "cat", "car"]
COSINE_PRESETS = [("great", "excellent", "similar"), ("pizza", "pasta", "same topic"),
                  ("good", "bad", "opposite meanings"), ("cheese", "queen", "unrelated")]


def cosine_words_data(glove):
    """The real 50-number GloVe vectors of ~30 words. The animation draws any two of them in the plane they span:
    two vectors always lie in one plane, where their lengths and their angle are exact (no projection error)."""
    return {"words": COSINE_WORDS,
            "vec50": {w: np.round(glove.vec(w).astype(np.float64), 3).tolist() for w in COSINE_WORDS},
            "presets": [list(p) for p in COSINE_PRESETS]}
