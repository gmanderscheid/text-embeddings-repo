"""The three targets of J5 lecture 01 (How embeddings are trained), for the three "Target" demo pages.

Targets 1 and 2 are trained exactly as in the notebook, on the same 2,000 nltk movie reviews:
    1. the label of the text   ReviewClassifier: EmbeddingBag(7000, 64, mean) + Linear(64, 1), BCE, 10 epochs
    2. the missing word        CBOW: Embedding(7000, 64), mean of 6 context words, Linear(64, 7000), CE, 4 epochs
Training CBOW takes 1-2 minutes on a CPU, so both are trained once and saved to assets/embedding_targets.npz,
with snapshots of the table during training (for the animations of the words moving).

    python -m demos.embedding_targets          (from the project folder, rebuilds the files)

"""
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ASSETS = Path(__file__).resolve().parent.parent / "assets"
FILE = ASSETS / "embedding_targets.npz"
PAD, UNK = 0, 1
TOKEN_RE = re.compile(r"[a-z']+")
PROBES = ["great", "terrible", "boring", "funny", "alien", "murder", "music", "director"]
GROUPS = {
    "positive": ["great", "excellent", "wonderful", "brilliant", "superb", "terrific"],
    "negative": ["awful", "boring", "stupid", "worst", "lame", "mess"],
    "sci-fi": ["alien", "aliens", "planet", "space", "earth", "crew"],
    "crime": ["murder", "police", "killer", "detective", "cop", "crime"],
    "music": ["music", "song", "songs", "soundtrack", "musical", "singing"],
}


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# ================================================================== building (run once)
def _reviews():
    import io, urllib.request, zipfile
    texts, labels = [], []
    with urllib.request.urlopen("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/movie_reviews.zip") as r:
        archive = io.BytesIO(r.read())
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(zf.namelist()):
            parts = Path(name).parts
            if len(parts) == 3 and parts[1] in ("pos", "neg") and name.endswith(".txt"):
                texts.append(zf.read(name).decode("utf-8", errors="ignore"))
                labels.append(int(parts[1] == "pos"))
    return texts, labels


def _seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _train_loop(model, train_loader, val_loader, loss_fn, optimizer, epochs, metric, snapshot, snaps_per_epoch=1):
    "The dl_utils.train of the course (keeps the best validation epoch), plus snapshots of the table."
    history, best, best_state = [], float("inf"), None
    snaps = [snapshot()]
    n_batches = len(train_loader)
    marks = {int(n_batches * k / snaps_per_epoch) - 1 for k in range(1, snaps_per_epoch + 1)}
    for epoch in range(epochs):
        model.train()
        total, correct, count = 0.0, 0, 0
        for b, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x); count += len(x)
            if metric:
                correct += ((out > 0).float() == y).sum().item()
            if b in marks:
                snaps.append(snapshot())
        model.eval()
        with torch.no_grad():
            vl, vc, vn = 0.0, 0, 0
            for x, y in val_loader:
                out = model(x)
                vl += loss_fn(out, y).item() * len(x); vn += len(x)
                if metric:
                    vc += ((out > 0).float() == y).sum().item()
        row = {"epoch": epoch + 1, "train_loss": total / count, "val_loss": vl / vn,
               "train_acc": correct / count if metric else np.nan, "val_acc": vc / vn if metric else np.nan}
        history.append(row)
        print(f"epoch {epoch + 1} | train loss {row['train_loss']:.4f} | val loss {row['val_loss']:.4f}"
              + (f" | val acc {row['val_acc']:.1%}" if metric else ""))
        if row["val_loss"] < best:
            best, best_state = row["val_loss"], {k: v.clone() for k, v in model.state_dict().items()}
            best_snap = len(snaps) - 1
    model.load_state_dict(best_state)
    return history, snaps, best_snap


class ReviewClassifier(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, dim, mode="mean", padding_idx=PAD)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.head = nn.Linear(dim, 1)

    def forward(self, ids):
        return self.head(self.embedding(ids))


class CBOW(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, context_ids):
        return self.output(self.embedding(context_ids).mean(dim=1))


def build():
    t0 = time.time()
    texts, labels = _reviews()
    counts = Counter(w for t in texts for w in tokenize(t))
    words = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(6998)]
    vocab = {w: i for i, w in enumerate(words)}
    print(f"{len(texts)} reviews | vocabulary {len(words):,}")

    # ---------------- target 1: the label
    _seed(42)
    MAX_LEN = 1000
    X = torch.full((len(texts), MAX_LEN), PAD, dtype=torch.long)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, UNK) for w in tokenize(t)][:MAX_LEN]
        X[i, :len(ids)] = torch.tensor(ids)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    order = np.random.default_rng(42).permutation(len(texts))
    tr, va = order[:1600], order[1600:]
    train_loader = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[va], y[va]), batch_size=256)
    torch.manual_seed(0)
    clf = ReviewClassifier(len(words), 64)
    hist1, snaps1, best1 = _train_loop(clf, train_loader, val_loader, nn.BCEWithLogitsLoss(),
                                       torch.optim.Adam(clf.parameters(), lr=1e-2), 10, True,
                                       lambda: clf.embedding.weight.detach().clone(), snaps_per_epoch=2)

    # ---------------- target 2: the missing word
    WINDOW = 3
    docs = [[vocab[w] for w in tokenize(t) if w in vocab] for t in texts]
    freq = np.array([counts[words[i]] for i in range(2, len(words))], dtype=np.float64)
    rate = freq / freq.sum()
    keep_prob = np.minimum(1.0, np.sqrt(1e-4 / rate) + 1e-4 / rate)
    rng = np.random.default_rng(0)
    contexts, targets = [], []
    for ids in docs:
        ids = np.array(ids)
        ids = ids[rng.random(len(ids)) < keep_prob[ids - 2]]
        if len(ids) < 2 * WINDOW + 1:
            continue
        win = np.lib.stride_tricks.sliding_window_view(ids, 2 * WINDOW + 1)
        contexts.append(np.delete(win, WINDOW, axis=1)); targets.append(win[:, WINDOW])
    contexts = torch.from_numpy(np.concatenate(contexts)); targets = torch.from_numpy(np.concatenate(targets))
    print(f"{len(targets):,} CBOW examples")
    torch.manual_seed(0)
    cbow = CBOW(len(words), 64)
    n_val = len(targets) // 20
    hist2, snaps2, best2 = _train_loop(cbow, DataLoader(TensorDataset(contexts[n_val:], targets[n_val:]), batch_size=512, shuffle=True),
                                       DataLoader(TensorDataset(contexts[:n_val], targets[:n_val]), batch_size=2048),
                                       nn.CrossEntropyLoss(), torch.optim.Adam(cbow.parameters(), lr=5e-3), 4, False,
                                       lambda: cbow.embedding.weight.detach().clone(), snaps_per_epoch=4)

    # ---------------- the words followed in the animations: probes, groups, the strongest words of each table
    w = clf.head.weight[0].detach()
    scores = (clf.embedding.weight.detach() @ w)
    scores[[PAD, UNK]] = 0
    frequent = np.array([i > UNK and counts[words[i]] >= 50 for i in range(len(words))])
    fs = scores.clone(); fs[~torch.from_numpy(frequent)] = 0
    ranked = fs.argsort().tolist()
    followed = list(dict.fromkeys(PROBES + [x for g in GROUPS.values() for x in g]
                                  + [words[i] for i in ranked[:15] + ranked[-15:]]
                                  + ["film", "movie", "plot", "story", "actor", "good", "bad", "funny", "not", "the"]))
    followed = [x for x in followed if x in vocab]
    fid = [vocab[x] for x in followed]

    # ---------------- example windows for the missing-word animation (validation reviews, unseen by CBOW? no: CBOW
    # used all reviews, as in the notebook; the excerpts are just shown)
    excerpts = []
    for i in [12, 345, 1210, 1788]:
        toks = [x for x in tokenize(texts[order[i]]) if x in vocab]
        start = len(toks) // 3
        excerpts.append(" ".join(toks[start:start + 22]))

    np.savez_compressed(
        FILE, words=np.array(words), counts=np.array([counts[x] if i > 1 else 0 for i, x in enumerate(words)]),
        keep_prob=np.r_[0, 0, keep_prob],
        sup_table=clf.embedding.weight.detach().numpy().astype(np.float16),
        sup_w=w.numpy(), sup_b=clf.head.bias.detach().numpy(),
        sup_hist=np.array([[h["train_loss"], h["val_loss"], h["train_acc"], h["val_acc"]] for h in hist1]),
        sup_snaps=torch.stack(snaps1[:best1 + 1])[:, fid].numpy().astype(np.float16),
        cbow_table=cbow.embedding.weight.detach().numpy().astype(np.float16),
        cbow_out_w=cbow.output.weight.detach().numpy().astype(np.float16),
        cbow_out_b=cbow.output.bias.detach().numpy().astype(np.float16),
        cbow_hist=np.array([[h["train_loss"], h["val_loss"]] for h in hist2]),
        cbow_snaps=torch.stack(snaps2[:best2 + 1])[:, fid].numpy().astype(np.float16),
        followed=np.array(followed), excerpts=np.array(excerpts),
        labels=np.array(labels), val_idx=va,
    )
    print(f"{FILE} written in {time.time() - t0:.0f}s ({FILE.stat().st_size / 1e6:.1f} MB)")


EXCERPTS = ASSETS / "embedding_targets_excerpts.json"


def select_excerpts(n=4, length=22, seed=1):
    """Excerpts where CBOW guesses well (the true word in its top 10 for at least 2 of the 4 windows shown):
    the loss stays high on average, these windows show what the model does when it works. No retraining."""
    import json
    T = Tables()
    texts, _ = _reviews()
    order = np.random.default_rng(7).permutation(len(texts))
    found = []
    for i in order:
        toks = [w for w in tokenize(texts[i]) if w in T.vocab]
        for start in range(0, len(toks) - length, length):
            ex = " ".join(toks[start:start + length])
            T.excerpts = [ex]
            d = missing_animation_data(T, 0, seed)
            good = [w for w in d["windows"] if w["rank"] <= 10 and T.counts[T.vocab[w["target"]]] < 3000]
            if len(d["windows"]) == 4 and len(good) >= 2:
                found.append(ex)
                break
        if len(found) == n:
            break
    EXCERPTS.write_text(json.dumps(found, indent=1))
    print(f"{EXCERPTS}: {len(found)} excerpts")


STORY = ASSETS / "embedding_label_story.json"
STORY_EXAMPLES = [  # real sentences of the corpus (already lower-cased and space-tokenized in nltk's files)
    ("it's dull , perfunctory , uninspired , and boring .", 0),
    ("in my opinion , this idea is absolutely brilliant .", 1),
    ("it is a total waste of time and money .", 0),
]
STORY_TALLY = ["boring", "and", "brilliant", "waste"]
STORY_WORDS = ["brilliant", "masterpiece", "loved", "wonderful", "excellent", "superb", "perfect", "best", "great",
               "boring", "waste", "awful", "worst", "stupid", "bad", "lame", "mess", "ridiculous", "dull",
               "the", "and", "a", "of", "to", "is", "it", "this", "film", "movie", "alien", "music", "director", "plot"]


def build_label_story():
    """Target 1 told in 11 steps: retrains the notebook's ReviewClassifier (same seeds, a few seconds) and keeps
    what the steps show: the random table at the start, one example's real numbers, the push every training
    review gives to a few words, the words moving during training, and the final table."""
    import json
    texts, labels = _reviews()
    counts = Counter(w for t in texts for w in tokenize(t))
    words = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(6998)]
    vocab = {w: i for i, w in enumerate(words)}
    _seed(42)
    MAX_LEN = 1000
    X = torch.full((len(texts), MAX_LEN), PAD, dtype=torch.long)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, UNK) for w in tokenize(t)][:MAX_LEN]
        X[i, :len(ids)] = torch.tensor(ids)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    order = np.random.default_rng(42).permutation(len(texts))
    tr, va = order[:1600], order[1600:]
    torch.manual_seed(0)
    clf = ReviewClassifier(len(words), 64)
    W0, w0, b0 = clf.embedding.weight.detach().clone(), clf.head.weight[0].detach().clone(), float(clf.head.bias.detach())
    hist, snaps, best = _train_loop(clf, DataLoader(TensorDataset(X[tr], y[tr]), batch_size=32, shuffle=True),
                                    DataLoader(TensorDataset(X[va], y[va]), batch_size=256), nn.BCEWithLogitsLoss(),
                                    torch.optim.Adam(clf.parameters(), lr=1e-2), 10, True,
                                    lambda: clf.embedding.weight.detach().clone(), snaps_per_epoch=4)
    W1, w1, b1 = clf.embedding.weight.detach(), clf.head.weight[0].detach(), float(clf.head.bias.detach())
    sig = lambda z: float(1 / (1 + np.exp(-z)))
    r = lambda a, n=3: np.round(np.asarray(a, dtype=np.float64), n).tolist()

    examples = []
    for text, label in STORY_EXAMPLES:
        toks = tokenize(text)
        ids = [vocab.get(w, UNK) for w in toks]
        m0, m1 = W0[ids].mean(0), W1[ids].mean(0)
        z0, z1 = float(m0 @ w0 + b0), float(m1 @ w1 + b1)
        examples.append({"raw": text, "label": label, "tokens": toks, "ids": ids,
                         "counts": [counts[w] for w in toks], "rows0": r(W0[ids][:, :16]), "mean0": r(m0[:16]),
                         "logit0": round(z0, 4), "p0": round(sig(z0), 4), "logit1": round(z1, 3), "p1": round(sig(z1), 4),
                         "grad_row": r(((sig(z0) - label) * w0 / len(ids))[:16], 5)})

    # the push every training review gives to a word, with the random starting model:
    # one SGD step moves the row of `word` by -lr * (p - y) * w / n * count, i.e. along w by (y - p) * count / n
    lens = (X[tr] != PAD).sum(1).float()
    p_init = torch.sigmoid(clf.head.weight.new_tensor(0) + (W0[X[tr]] * (X[tr] != PAD).unsqueeze(-1)).sum(1) / lens[:, None] @ w0 + b0)
    tally = {}
    for wd in STORY_TALLY:
        c = (X[tr] == vocab[wd]).sum(1).float()
        idx = torch.nonzero(c).flatten()
        contrib = ((y[tr, 0] - p_init) * c / lens)[idx]
        tally[wd] = {"contrib": r(contrib * 100, 3), "labels": y[tr, 0][idx].int().tolist(), "n": len(idx),
                     "pos": int(y[tr, 0][idx].sum()), "total": round(float(contrib.sum() * 100), 3)}

    fid = [vocab[w] for w in STORY_WORDS]
    pts = _map(torch.stack(snaps[:best + 1])[:, fid].numpy(), axis_x=w1.numpy())
    groups = ["positive"] * 9 + ["negative"] * 10 + ["neutral"] * 10 + ["topic"] * 4

    freq = [i for i in range(2, len(words)) if counts[words[i]] >= 50]
    F = W1[freq].numpy()
    xy, _ = __import__("demos.pretrained", fromlist=["pca2"]).pca2(F)
    score = (W1[freq] @ w1).numpy()
    if np.corrcoef(xy[:, 0], score)[0, 1] < 0:            # orient PC1 so that positive words are on the right
        xy[:, 0] *= -1
    unit = F / np.linalg.norm(F, axis=1, keepdims=True)
    def nb(w, k=6):
        s_ = unit @ unit[freq.index(vocab[w])]
        s_[freq.index(vocab[w])] = -2
        return [[words[freq[i]], round(float(s_[i]), 2)] for i in np.argsort(-s_)[:k]]
    cosw = lambda T_, a, b: round(float(F_.cosine_similarity(T_[vocab[a]], T_[vocab[b]], dim=0)), 3)
    import torch.nn.functional as F_
    story = {
        "V": len(words), "distinct": len(counts), "max_len": MAX_LEN, "dim": 64, "examples": examples,
        "init_rows": {w: r(W0[vocab[w]][:16]) for w in ["brilliant", "awful", "boring", "and"]},
        "ids": {w: vocab[w] for w in ["brilliant", "awful", "boring", "and"]},
        "cos_init": cosw(W0, "brilliant", "awful"), "cos_final": cosw(W1, "brilliant", "awful"),
        "w0": r(w0[:16]), "b0": round(b0, 4), "tally": tally,
        "map": {"words": STORY_WORDS, "groups": groups, "pts": pts, "epochs": [k / 4 for k in range(best + 1)]},
        # the trained table is almost one-dimensional: PC2 is stretched to be visible (the factor is shown)
        "pca": {"words": [words[i] for i in freq], "xy": r(xy / np.abs(xy).max(axis=0), 3), "score": r(score, 1),
                "stretch": round(float(np.abs(xy[:, 0]).max() / np.abs(xy[:, 1]).max()), 1),
                "var": r(np.var(xy, axis=0) / np.var(F - F.mean(0), axis=0).sum(), 3)},
        "neighbours": {w: nb(w) for w in ["brilliant", "boring", "and", "waste"]},
        "hist": hist,
    }
    STORY.write_text(json.dumps(story))
    print(f"{STORY}: {STORY.stat().st_size / 1e3:.0f} KB")


# ================================================================== loading (the pages)
class Tables:
    def __init__(self):
        z = np.load(FILE)
        self.words = z["words"].tolist()
        self.vocab = {w: i for i, w in enumerate(self.words)}
        self.counts = z["counts"]
        self.keep_prob = z["keep_prob"]
        self.frequent = np.array([i > UNK and c >= 50 for i, c in enumerate(self.counts)])
        self.sup = z["sup_table"].astype(np.float32)
        self.sup_w, self.sup_b = z["sup_w"], float(z["sup_b"][0])
        self.sup_hist = z["sup_hist"]
        self.sup_snaps = z["sup_snaps"].astype(np.float32)
        self.cbow = z["cbow_table"].astype(np.float32)
        self.cbow_out_w = z["cbow_out_w"].astype(np.float32)
        self.cbow_out_b = z["cbow_out_b"].astype(np.float32)
        self.cbow_hist = z["cbow_hist"]
        self.cbow_snaps = z["cbow_snaps"].astype(np.float32)
        self.followed = z["followed"].tolist()
        self.excerpts = z["excerpts"].tolist()
        if EXCERPTS.exists():
            import json
            self.excerpts = json.loads(EXCERPTS.read_text())
        self.labels = z["labels"]
        self.val_idx = z["val_idx"]

    def table(self, which):
        return self.sup if which == "label" else self.cbow

    def neighbours(self, word, which, k=6):
        T = self.table(which)
        V = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        sims = V @ V[self.vocab[word]]
        sims[~self.frequent] = -2
        sims[self.vocab[word]] = -2
        top = np.argsort(-sims)[:k]
        return [(self.words[i], float(sims[i])) for i in top]

    def cosines(self, which, group_words):
        T = self.table(which)[[self.vocab[w] for w in group_words]]
        T = T / np.linalg.norm(T, axis=1, keepdims=True)
        return T @ T.T

    def scores(self):
        "How much each word pushes a review towards positive: its row · the head w."
        s = self.sup @ self.sup_w
        s[[PAD, UNK]] = 0
        return s

    def predict_missing(self, context, k=10):
        "CBOW: the mean of the context rows, the 7,000 scores, the softmax."
        ids = [self.vocab[w] for w in context if w in self.vocab]
        z = self.cbow[ids].mean(0) @ self.cbow_out_w.T + self.cbow_out_b
        p = np.exp(z - z.max()); p /= p.sum()
        return p


# ================================================================== what the animations need
SHOWN = 16                                                   # dims of a row drawn in the animations (of 64)


def _group_of(word):
    return next((g for g, ws in GROUPS.items() if word in ws), "other")


def _map(snaps, axis_x=None):
    """2D positions of the followed words in every snapshot, with axes fixed on the last snapshot:
    x = projection on axis_x (if given) else the 1st principal component, y = the next principal component."""
    final = snaps[-1]
    mean = final.mean(0)
    X = final - mean
    if axis_x is not None:
        ux = axis_x / np.linalg.norm(axis_x)
        R = X - np.outer(X @ ux, ux)
        uy = np.linalg.svd(R, full_matrices=False)[2][0]
    else:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        ux, uy = Vt[0], Vt[1]
    pts = np.stack([np.stack([(S - mean) @ ux, (S - mean) @ uy], 1) for S in snaps])
    pts = pts / np.abs(pts[-1]).max(axis=0)          # each axis scaled to [-1, 1] on the final table
    return np.round(np.clip(pts, -1.3, 1.3).astype(np.float64), 3).tolist()


def missing_animation_data(T, excerpt_index, seed=1):
    toks = T.excerpts[excerpt_index].split()
    keep = np.array([T.keep_prob[T.vocab[w]] for w in toks])
    kept = np.random.default_rng(seed).random(len(toks)) < keep
    kept_toks = [w for w, k in zip(toks, kept) if k]
    windows = []
    for c in range(3, len(kept_toks) - 3)[:4]:
        context = kept_toks[c - 3:c] + kept_toks[c + 1:c + 4]
        probs = T.predict_missing(context)
        target = T.vocab[kept_toks[c]]
        top = np.argsort(-probs)[:8]
        rank = int((probs > probs[target]).sum()) + 1
        windows.append({"center": c, "context": context, "target": kept_toks[c],
                        "top": [[T.words[i], round(float(probs[i]), 4)] for i in top],
                        "p_target": round(float(probs[target]), 5), "rank": rank,
                        "rows": np.round(T.cbow[[T.vocab[w] for w in context]][:, :SHOWN].astype(np.float64), 2).tolist()})
    return {
        "tokens": toks, "keep": np.round(keep, 2).tolist(), "kept": kept.tolist(), "windows": windows,
        "V": len(T.words), "logV": round(float(np.log(len(T.words))), 2),
        "map": {"words": T.followed, "groups": [_group_of(w) for w in T.followed],
                "pts": _map(T.cbow_snaps), "epochs": [k / 4 for k in range(len(T.cbow_snaps))]},
    }


if __name__ == "__main__":
    build()
    select_excerpts()
    build_label_story()
