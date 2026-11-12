"""Text -> tokens -> ids -> vectors, with three tokenizers (no Streamlit here).

- "words":      the rule of J5 lecture 00: lower-case, keep runs of letters and apostrophes.
                Vocabulary = words seen at least 15 times in the 2,000 nltk movie reviews.
- "characters": lower-case, every character is a token (spaces and punctuation included).
                Vocabulary = characters seen at least 15 times in the same reviews.
- "subwords":   the byte-level BPE tokenizer of GPT-2 (50,257 tokens), via Hugging Face `tokenizers`.

In the two home-made vocabularies, ids 0 and 1 are <pad> and <unk>. They are built once from the
nltk archive and cached in assets/movie_reviews_vocab.json, so the app works offline afterwards.
GPT-2's tokenizer comes from the Hugging Face cache (downloaded on first use).
"""
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/movie_reviews.zip"
CACHE = Path(__file__).resolve().parent.parent / "assets" / "movie_reviews_vocab.json"
PAD, UNK = 0, 1
MIN_COUNT = 15
TOKEN_RE = re.compile(r"[a-z']+")
MODES = ("words", "subwords", "characters")
MAX_TOKENS = {"words": 14, "subwords": 14, "characters": 20}
NEIGHBOURS = 7          # rows of W sent around each looked-up id (what the animated table window shows)


def tokenize(text):
    "Lower-case words made of letters and apostrophes (same rule as the lecture)."
    return TOKEN_RE.findall(text.lower())


def show(token):
    "Make spaces and line breaks visible inside a token."
    return token.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥")


# ------------------------------------------------------------------ vocabularies
def _download_reviews():
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = io.BytesIO(response.read())
    texts = []
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(zf.namelist()):
            parts = Path(name).parts
            if len(parts) == 3 and parts[1] in ("pos", "neg") and name.endswith(".txt"):
                texts.append(zf.read(name).decode("utf-8", errors="ignore"))
    return texts


def load_review_vocabs():
    """The word and character vocabularies of the reviews: lists where list[i] is the token of id i.

    Returns (words, chars, info), info holding the corpus figures shown in the app.
    """
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("min_count") == MIN_COUNT and "chars" in cached:
            return cached["words"], cached["chars"], cached["info"]

    texts = _download_reviews()
    counts = Counter(word for text in texts for word in tokenize(text))
    char_counts = Counter(c for text in texts for c in text.lower())
    words = ["<pad>", "<unk>", *[w for w, c in counts.most_common() if c >= MIN_COUNT]]
    chars = ["<pad>", "<unk>", *[ch for ch, c in char_counts.most_common() if c >= MIN_COUNT]]
    info = {"n_reviews": len(texts), "n_tokens": sum(counts.values()), "n_distinct": len(counts),
            "n_chars": sum(char_counts.values())}

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"min_count": MIN_COUNT, "info": info, "words": words, "chars": chars}))
    return words, chars, info


def load_gpt2():
    "GPT-2's tokenizer and its vocabulary as display strings (vocab[i] = token of id i)."
    from tokenizers import Tokenizer               # imported here: only the subword mode needs it
    tok = Tokenizer.from_pretrained("gpt2")
    size = tok.get_vocab_size()
    decoded = tok.decode_batch([[i] for i in range(size)])
    vocab = [show(d) if d else tok.id_to_token(i) for i, d in enumerate(decoded)]   # '' = <|endoftext|>
    return tok, vocab


def embedding_table(vocab_size, dim, padding=True, seed=0):
    "The weights of a fresh nn.Embedding: random N(0, 1), row PAD = zeros when there is a padding id."
    torch.manual_seed(seed)
    layer = nn.Embedding(vocab_size, dim, padding_idx=PAD if padding else None)
    return layer.weight.detach().numpy().astype(np.float64)


# ------------------------------------------------------------------ tokenizers -> character spans
def _lowered(text):
    # Lower-case char by char so that positions never shift (a few unicode letters grow when lowered)
    return [c.lower() if len(c.lower()) == 1 else c for c in text]


def spans_words(text):
    return [m.span() for m in TOKEN_RE.finditer("".join(_lowered(text)))]


def spans_characters(text):
    return [(i, i + 1) for i in range(len(text))]


def spans_subwords(text, tokenizer):
    enc = tokenizer.encode(text)
    return list(enc.offsets), list(enc.ids)


def split_characters(text, spans, max_tokens, lower):
    """Every character of `text` with the token that owns it (-1 for a dropped separator).

    The text is cut right after its `max_tokens`-th token. Returns (chars, tokens, kept, truncated):
    chars  = [{"c": char, "lower": char after lower-casing, "tok": token index or -1, "k": rank in its token}]
    tokens = [{"start", "end", "n": number of characters it owns}]
    kept   = number of spans kept
    """
    truncated = len(spans) > max_tokens
    spans = spans[:max_tokens]
    if truncated:                                   # cut right after the last kept token
        text = text[:max(end for _, end in spans)]
    lowered = _lowered(text) if lower else list(text)

    owner = [-1] * len(text)
    for t, (start, end) in enumerate(spans):        # a character split in several byte tokens: first one wins
        for i in range(start, end):
            if owner[i] < 0:
                owner[i] = t
    chars, rank = [], Counter()
    for c, low, o in zip(text, lowered, owner):
        chars.append({"c": c, "lower": low, "tok": o, "k": rank[o]})
        rank[o] += 1
    tokens = [{"start": s, "end": e, "n": rank[t]} for t, (s, e) in enumerate(spans)]
    return chars, tokens, truncated


def split_example(text, tokens):
    "The first word cut into several pieces, e.g. '␣z + org + l + ub' (or None)."
    for m in re.finditer(r"\S+", text):
        pieces = [t for t in tokens if t["start"] < m.end() and t["end"] > m.start() - 1]
        alpha = [t for t in pieces if any(ch.isalpha() for ch in t["label"])]
        if len(alpha) >= 2 and len(alpha) == len(pieces):
            return " + ".join(t["label"] for t in alpha)
    return None


# ------------------------------------------------------------------ everything the animation needs
def pipeline(text, mode, dim, vocabs, seed=0):
    """vocabs = {"words": list, "characters": list, "subwords": (tokenizer, list)}; returns a JSON-ready dict."""
    if mode == "subwords":
        tokenizer, vocab = vocabs["subwords"]
        spans, ids = spans_subwords(text, tokenizer)
        lower, padding = False, False
    else:
        vocab = vocabs[mode]
        spans = spans_words(text) if mode == "words" else spans_characters(text)
        lower, padding = True, True
    chars, tokens, truncated = split_characters(text, spans, MAX_TOKENS[mode], lower)

    index = {tok: i for i, tok in enumerate(vocab)}
    for t, tok in enumerate(tokens):
        if mode == "subwords":
            tok["id"] = ids[t]
            tok["label"] = vocab[tok["id"]]
        else:
            piece = "".join(c["lower"] for c in chars[tok["start"]:tok["end"]])
            tok["id"] = index.get(piece, UNK)
            tok["label"] = show(piece)
        tok["unk"] = padding and tok["id"] == UNK

    table = embedding_table(len(vocab), dim, padding, seed)
    needed = {i for tok in tokens for i in range(tok["id"] - NEIGHBOURS, tok["id"] + NEIGHBOURS + 1)}
    needed |= set(range(NEIGHBOURS + 2))                       # the window shown before the first lookup
    rows = {i: table[i].round(2).tolist() for i in sorted(needed) if 0 <= i < len(vocab)}

    return {
        "mode": mode, "chars": chars, "tokens": tokens, "truncated": truncated, "dim": dim,
        "vocab": vocab if mode == "subwords" else [show(t) for t in vocab], "rows": rows, "padding": padding,
        "split_example": split_example(text, tokens) if mode == "subwords" else None,
    }


def count_tokens(text, vocabs):
    "How many tokens each tokenizer makes of the whole text (no truncation)."
    return {"words": len(spans_words(text)), "characters": len(text),
            "subwords": len(vocabs["subwords"][0].encode(text).ids) if "subwords" in vocabs else None}
