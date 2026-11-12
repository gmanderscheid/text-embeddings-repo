"""J5 - Text to features: one-hot, counts and TF-IDF on 1,000 hotel reviews, and why they hit a wall.

Every tab answers one question, on the same corpus and with the settings of the sidebar:
one-hot vectors, the long tail of the vocabulary, count vs TF-IDF, over-cleaning, a classifier.
"""
import html
import inspect

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from demos import bag_of_words as bow
from ui import BLUES, COLORS, GRID, MUTED, PANEL, THEME_NAME, figure, page_header, set_plotly_theme, sidebar_section, what_to_look_at

st.set_page_config(page_title="Text to features", page_icon="🧮", layout="wide")
set_plotly_theme()

# theme=None: the charts keep the dark plotly template of ui.py instead of Streamlit's own
CHART = {"theme": None} if "width" in inspect.signature(st.plotly_chart).parameters \
    else {"theme": None, "use_container_width": True}
CLASS_COLORS = {"neutral": COLORS[3], "positive": COLORS[2], "negative": COLORS[0]}
MAX_FEATURES = ["all", 5000, 2000, 1000, 500, 200, 100, 50, 20, 10, 5]


def chart(fig):
    st.plotly_chart(fig, **CHART)


page_header(
    "J5 · Text to features",
    "1,000 TripAdvisor hotel reviews of 50 to 100 words, labelled **neutral** (3 stars), **positive** (4-5) "
    "or **negative** (1-2). Before any classifier can read them, they must become vectors of numbers: "
    "clean the text and choose a vectorizer in the sidebar, then see what each choice costs, tab by tab.",
)


# ------------------------------------------------------------------ cached computations
@st.cache_data
def corpus():
    return bow.load_reviews()


@st.cache_data(show_spinner="Cleaning the reviews (lemmatizing with spaCy takes ~20 s the first time)...")
def cleaned(cleaning):
    return bow.clean_corpus(corpus()["text"], cleaning)


@st.cache_data
def counts(cleaning):
    return bow.word_counts(cleaned(cleaning))


@st.cache_data
def matrix(cleaning, vectorizing):
    return bow.fit_matrix(cleaned(cleaning), vectorizing)


@st.cache_resource
def fitted_vectorizer(cleaning, vectorizing):
    vec = bow.make_vectorizer(vectorizing)
    try:
        vec.fit(cleaned(cleaning))
    except ValueError:
        return None
    return vec


@st.cache_data
def classification(cleaning, vectorizing):
    result = bow.classify(cleaned(cleaning), corpus()["label"].to_numpy(), vectorizing)
    if result:
        result.pop("vectorizer")
    return result


# ------------------------------------------------------------------ sidebar
with sidebar_section("1 · Cleaning"):
    lemmatize = st.checkbox("lemmatize (spaCy)", False,
                            help="each word replaced by its dictionary form, using its part of speech: rooms → room, "
                                 "was / were → be, stayed → stay, n't → not. Applied first, on the raw text.")
    lowercase = st.checkbox("lower-case", True)
    punctuation = st.checkbox("remove punctuation", True, help="“room.” and “room” become the same word")
    digits = st.checkbox("remove digits", False)
    stop_words = st.checkbox("remove English stop words", False,
                             help=f"scikit-learn's list of {len(bow.STOP_WORDS)} words: the, and, was, ... and not, no, never")
    keep_negations = st.checkbox("... but keep negations (not, no, never...)", True, disabled=not stop_words)
cleaning = bow.Cleaning(lowercase, punctuation, digits, stop_words, keep_negations, lemmatize)

with sidebar_section("2 · Vectorizer"):
    kind = st.radio("features", ["count", "tfidf"], horizontal=True,
                    format_func={"count": "CountVectorizer", "tfidf": "TF-IDF"}.get)
    ngram_max = st.radio("n-grams", [1, 2], horizontal=True,
                         format_func={1: "words", 2: "words + pairs of words"}.get)
    min_df = st.slider("min_df: keep a word seen in at least ... reviews", 1, 50, 1)
    max_df = st.slider("max_df: drop a word seen in more than ... of the reviews", 0.05, 1.0, 1.0, 0.05,
                       format="%.2f")
    max_features = st.select_slider("max_features: keep only the ... most frequent words", MAX_FEATURES, "all")
vectorizing = bow.Vectorizing(kind, ngram_max, min_df, max_df, None if max_features == "all" else max_features)

with sidebar_section("3 · Review to inspect"):
    doc_id = st.number_input("review #", 0, 999, 7)

df = corpus()
docs = cleaned(cleaning)
tf, dfreq = counts(cleaning)
X, features = matrix(cleaning, vectorizing)
review, sentiment = df.loc[doc_id, "text"], df.loc[doc_id, "sentiment"]

TABS = ["1 · One-hot vectors", "2 · The long tail", "3 · Count vs TF-IDF", "4 · Over-cleaning", "5 · Classification"]
tabs = st.tabs(TABS)


def review_box(text, label):
    color = CLASS_COLORS[label]
    st.markdown(f"<div style='border-left:4px solid {color};padding:6px 12px;background:{PANEL}'>"
                f"<b style='color:{color}'>review #{doc_id} · {label}</b><br>{html.escape(text)}</div>",
                unsafe_allow_html=True)


# ================================================================== 1 · one-hot
with tabs[0]:
    # the vocabulary the vectorizer keeps: cleaning AND the min_df / max_df / max_features filters of the sidebar
    kept = set() if X is None else {f for f in features if " " not in f}          # words only, not word pairs
    vocab = [w for w, _ in tf.most_common() if w in kept]                         # by frequency
    index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    if V == 0:
        st.error("These filters leave no word at all: loosen min_df, max_df or max_features.")
        st.stop()
    total_tokens = sum(sum(t in kept for t in d.split()) for d in docs)

    st.markdown(f"A **one-hot vector** gives each word of the vocabulary its own axis: the word number *i* is a "
                f"vector of **{V:,}** numbers, all zeros except a single 1 at position *i*. "
                "A review becomes one such row per word. The vocabulary is the vectorizer's: cleaning **and** the "
                "min_df / max_df / max_features filters of the sidebar.")
    c1, c3, c4 = st.columns(3)
    c1.metric("vocabulary V", f"{V:,}", help="words kept by the cleaning and the vectorizer's filters")
    c3.metric("zeros in each one-hot vector", f"{1 - 1 / V:.3%}")
    c4.metric("whole corpus, float32", f"{total_tokens * V * 4 / 1e9:.2f} GB",
              help=f"{total_tokens:,} words × {V:,} × 4 bytes")

    review_box(review, sentiment)

    st.markdown("**Distance between words.** Pick some words: with one-hot vectors, every pair of different words "
                "has a cosine of 0 and a distance of √2. *great* is as far from *excellent* as from *bathroom*.")
    defaults = [w for w in ["great", "excellent", "good", "terrible", "awful", "bathroom"] if w in index]
    picked = st.multiselect("words", vocab, defaults or vocab[:5], max_selections=8)
    if len(picked) >= 2:
        sim = np.eye(len(picked))
        heat = go.Figure(go.Heatmap(z=sim, x=picked, y=picked, colorscale=BLUES, zmin=0, zmax=1, showscale=False,
                                    text=sim, texttemplate="%{text:.0f}"))
        heat.update_layout(template=THEME_NAME, height=90 + 45 * len(picked), margin=dict(l=110),
                           title="cosine similarity between one-hot vectors")
        heat.update_yaxes(autorange="reversed")
        chart(heat)
    what_to_look_at(
        "- How many numbers does the corpus need, and how many of them carry information?\n"
        "- Untick the cleaning options, raise min_df: what happens to V, and to the memory of the corpus?\n"
        "- Is there any pair of words that one-hot vectors consider similar? What would a classifier need?"
    )

# ================================================================== 2 · the long tail
with tabs[1]:
    freqs = np.array(sorted(tf.values(), reverse=True))
    V = len(freqs)
    once = int((freqs == 1).sum())
    one_review = sum(1 for d in dfreq.values() if d == 1)
    top100 = freqs[:100].sum() / freqs.sum()
    raw_V = len(counts(bow.Cleaning(False, False, False, False))[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("vocabulary after cleaning", f"{V:,}", f"{V - raw_V:+,} vs no cleaning at all", delta_color="inverse",
              help="before the min_df / max_df / max_features filters: the curve on the right shows what min_df keeps")
    c2.metric("words seen only once", f"{once / V:.0%}", help=f"{once:,} words")
    c3.metric("words in a single review", f"{one_review / V:.0%}")
    c4.metric("share of the text covered by the 100 most frequent words", f"{top100:.0%}")

    left, right = st.columns(2)
    with left:
        fig = figure(height=400)
        ranks = np.arange(1, V + 1)
        fig.add_trace(go.Scatter(x=ranks, y=freqs, mode="lines", line=dict(color=COLORS[1], width=2.5),
                                 hovertemplate="rank %{x}: %{y} occurrences<extra></extra>"))
        fig.add_vrect(x0=V - once + 1, x1=V, fillcolor=COLORS[0], opacity=.12, line_width=0,
                      annotation_text="seen once", annotation_position="top left")
        fig.update_xaxes(type="log", dtick=1, title="rank of the word (log)")
        fig.update_yaxes(type="log", dtick=1, title="occurrences (log)")
        fig.update_layout(title="Zipf's law: a few words are everywhere, most are rare")
        chart(fig)
    with right:
        thresholds = np.arange(1, 31)
        df_values = np.array(list(dfreq.values()))
        sizes = [(df_values >= m).sum() for m in thresholds]
        fig = figure(height=400)
        fig.add_trace(go.Scatter(x=thresholds, y=sizes, mode="lines+markers", line=dict(color=COLORS[2], width=2.5),
                                 hovertemplate="min_df = %{x}: %{y:,} words<extra></extra>"))
        fig.add_vline(x=min_df, line_dash="dot", line_color=COLORS[0], annotation_text="your min_df")
        fig.update_xaxes(title="min_df (a word must appear in at least ... reviews)")
        fig.update_yaxes(title="words kept", rangemode="tozero")
        fig.update_layout(title="How much vocabulary survives a min_df filter")
        chart(fig)

    rng = np.random.default_rng(doc_id)
    hapax = [w for w, c in tf.items() if c == 1]
    sample = rng.choice(hapax, size=min(40, len(hapax)), replace=False) if hapax else []
    st.markdown("**A random sample of words seen only once** (change the review # for another draw): "
                "typos, names, numbers, glued punctuation... A model cannot learn anything from a word it saw once.")
    st.markdown(" ".join(f"`{html.escape(w)}`" for w in sample))
    what_to_look_at(
        "- What kind of words are in the long tail? Which cleaning options shrink it, and which do not?\n"
        "- Raise min_df: how much of the vocabulary disappears for a small min_df, and what does it remove?\n"
        "- If the top 100 words cover that much of the text, what are they? Are they useful for sentiment?"
    )

# ================================================================== 3 · count vs TF-IDF
with tabs[2]:
    st.markdown(
        "Add up the one-hot rows of a review and you get its **count vector**: one number per word of the vocabulary, "
        "how many times it appears (`CountVectorizer`). **TF-IDF** multiplies each count by "
        "`idf = ln((1 + N) / (1 + df)) + 1`, small for words present in every review, large for rare ones, "
        "then scales each review to length 1 (`TfidfVectorizer`)."
    )
    if X is None:
        st.error("These filters leave no word at all: loosen min_df, max_df or max_features.")
    else:
        dense_gb = X.shape[0] * X.shape[1] * 4 / 1e6
        sparse_mb = (X.nnz * 12) / 1e6
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("matrix shape", f"{X.shape[0]:,} × {X.shape[1]:,}")
        c2.metric("non-zero values", f"{X.nnz:,}")
        c3.metric("zeros", f"{1 - X.nnz / (X.shape[0] * X.shape[1]):.2%}")
        c4.metric("memory as a dense table", f"{dense_gb:.1f} MB",
                  help=f"stored sparse (only the non-zeros, with their positions): {sparse_mb:.2f} MB")

        doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
        top = np.argsort(-doc_freq)[:30]
        rows = list(range(doc_id, min(doc_id + 15, X.shape[0])))
        sub = X[rows][:, top].toarray()
        heat = go.Figure(go.Heatmap(z=sub, x=features[top], y=[f"#{r} {df.loc[r, 'sentiment']}" for r in rows],
                                    colorscale=BLUES, text=sub, texttemplate="%{text:.2g}" if kind == "tfidf" else "%{text}",
                                    hovertemplate="%{y} · %{x}: %{z:.3g}<extra></extra>", showscale=False))
        heat.update_layout(template=THEME_NAME, height=520, margin=dict(l=120),
                           title=f"{'Counts' if kind == 'count' else 'TF-IDF weights'}: 15 reviews × the 30 most widespread words")
        heat.update_yaxes(autorange="reversed")
        heat.update_xaxes(tickangle=-45)
        chart(heat)

        # the same review, both ways
        both = {}
        for k in ["count", "tfidf"]:
            v = fitted_vectorizer(cleaning, bow.Vectorizing(k, ngram_max, min_df, max_df, vectorizing.max_features))
            both[k] = (v.transform([docs[doc_id]]).toarray().ravel(), v.get_feature_names_out())
        review_box(review, sentiment)
        if both["count"][0].any():
            left, right = st.columns(2)
            top_count = set(np.argsort(-both["count"][0])[:15])
            for col, k, color, title in [(left, "count", COLORS[1], "Counts: the most repeated words win"),
                                         (right, "tfidf", COLORS[2], "TF-IDF: the top words of the same review")]:
                weights, names = both[k]
                order = np.argsort(-weights)[:15]
                order = order[weights[order] > 0]
                # on the TF-IDF side, words that were not in the count top 15 are highlighted
                colors = [color if k == "count" or i in top_count else COLORS[3] for i in order]
                fig = figure(height=420)
                fig.add_trace(go.Bar(x=weights[order], y=names[order], orientation="h", marker_color=colors,
                                     hovertemplate="%{y}: %{x:.3g}<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(title=title, margin=dict(l=110))
                col.plotly_chart(fig, **CHART)
            st.caption(f"In orange: words that only reach the top 15 thanks to TF-IDF (rare in the corpus, so more "
                       "telling about this review).")
        else:
            st.warning("This review has no word left with these settings: see tab 4.")
    what_to_look_at(
        "- Which words get the highest counts in a review, and do they say anything about its sentiment?\n"
        "- Which words lose the most weight with TF-IDF, and why?\n"
        "- Switch to “words + pairs of words”: what happens to the shape of the matrix and to its zeros?"
    )

# ================================================================== 4 · over-cleaning
with tabs[3]:
    st.markdown("Each cleaning step shrinks the vocabulary and removes noise, but also removes **information**. "
                "Push the sidebar settings (stop words, min_df, max_df, max_features) and watch what is left.")
    unigrams = None if X is None else {f for f in features if " " not in f}
    STYLE = {"kept": "", "stop word": f"color:{MUTED};text-decoration:line-through",
             "out of vocabulary": f"color:{COLORS[3]};text-decoration:line-through",
             "removed by cleaning": f"color:{COLORS[0]};text-decoration:line-through"}
    def shown(raw, piece):
        if piece is None:
            return html.escape(raw)
        if lemmatize and piece != bow.normalise(raw, bow.Cleaning(lowercase, punctuation, digits)).strip():
            return f"{html.escape(raw)}<span style='color:{COLORS[4]}'>→{html.escape(piece)}</span>"
        return html.escape(piece)
    explained = bow.explain(review, cleaning, unigrams)
    parts = [f"<span style='{STYLE[status]}' title='{status}'>{shown(raw, piece)}</span>"
             for raw, piece, status in explained]
    legend = " · ".join(f"<span style='{s}'>{name}</span>" for name, s in STYLE.items())
    st.markdown(f"<div style='border:1px solid {GRID};border-radius:8px;padding:10px 14px;line-height:1.9'>"
                f"<b style='color:{CLASS_COLORS[sentiment]}'>review #{doc_id} · {sentiment}</b> "
                f"<small>({legend})</small><br>{' '.join(parts)}</div>", unsafe_allow_html=True)
    # the final processed text: only the kept tokens, in order, exactly what the vectorizer counts
    final = [piece for _, piece, status in explained if status == "kept"]
    n_raw = len(review.split())
    pairs_note = " · the word pairs are formed from these words" if ngram_max > 1 else ""
    st.markdown(f"<div style='border:1px solid {GRID};border-left:4px solid {COLORS[4]};border-radius:8px;padding:10px 14px;"
                f"margin-top:8px;background:{PANEL}'><b style='color:{COLORS[4]}'>what the vectorizer reads</b> "
                f"<small style='color:{MUTED}'>({len(final)} tokens kept from {n_raw} words{pairs_note})</small><br>"
                f"<span style='font-family:monospace'>{html.escape(' '.join(final)) or '<i>(empty: nothing left)</i>'}</span></div>",
                unsafe_allow_html=True)

    raw_len = df["text"].str.split().str.len().to_numpy()
    # words of each review still in the vocabulary, and reviews whose vector is all zeros
    kept = np.array([sum(t in unigrams for t in d.split()) for d in docs]) if X is not None else np.zeros(len(df), int)
    empty = np.flatnonzero(X.getnnz(axis=1) == 0) if X is not None else np.arange(len(df))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("words per review, raw", f"{raw_len.mean():.0f}")
    change = kept.mean() / raw_len.mean() - 1
    c2.metric("words per review, kept", f"{kept.mean():.1f}", f"{change:.0%}" if change < -.005 else None)
    c3.metric("reviews with fewer than 5 words left", f"{(kept < 5).sum():,}")
    c4.metric("empty reviews (all zeros)", f"{len(empty):,}")

    left, right = st.columns([3, 2])
    with left:
        fig = figure(height=360)
        fig.add_trace(go.Histogram(x=kept, nbinsx=40, marker_color=COLORS[1]))
        fig.update_xaxes(title="words left per review", rangemode="tozero")
        fig.update_yaxes(title="reviews")
        fig.update_layout(title="What the vectorizer still reads, review by review", bargap=.05)
        chart(fig)
    with right:
        if len(empty):
            st.markdown(f"**{len(empty)} reviews became empty vectors.** The model will give them all the same "
                        "answer, whatever they say. For instance:")
            for i in empty[:3]:
                st.caption(f"#{i} · {df.loc[i, 'sentiment']} · {df.loc[i, 'text'][:220]}...")
        else:
            st.markdown("**No empty review yet.** Try: remove stop words, then max_features = 10 or 5.")

    st.markdown("**Negations.** scikit-learn's English stop-word list contains *not*, *no*, *never*, *nothing*... "
                "For sentiment, removing them can flip the meaning.")
    sentence = st.text_input("try a sentence", "The room was not clean and the staff was never friendly. Nothing to complain about the location though!")
    full_stop = bow.Cleaning(lowercase, punctuation, digits, True, False, lemmatize)
    c1, c2 = st.columns(2)
    c1.markdown("with your cleaning settings")
    c1.code(" ".join(bow.clean_tokens(sentence, cleaning)) or "(empty)", language=None)
    c2.markdown("with the full English stop-word list")
    c2.code(" ".join(bow.clean_tokens(sentence, full_stop)) or "(empty)", language=None)
    with_neg = sum(1 for t in df["text"] if set(bow.normalise(t, cleaning).split()) & bow.NEGATIONS)
    st.caption(f"{with_neg:,} of the 1,000 reviews contain at least one negation word.")
    what_to_look_at(
        "- Which settings make reviews empty first? Are the empty ones a random sample of the classes?\n"
        "- Read the coloured review: are the struck-out words useless for sentiment?\n"
        "- Compare the two versions of the sentence: would a model still know it is negative?"
    )

# ================================================================== 5 · classification
with tabs[4]:
    st.markdown("A logistic regression reads the vectors of 800 reviews (train) and predicts the class of 200 "
                "others (test). The vocabulary is learned on the train reviews only, as it would be in production.")
    res = classification(cleaning, vectorizing)
    if res is None:
        st.error("These filters leave no word at all: loosen min_df, max_df or max_features.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("train accuracy", f"{res['train_acc']:.0%}")
        c2.metric("test accuracy", f"{res['test_acc']:.0%}", f"{res['test_acc'] - res['train_acc']:+.0%} vs train")
        c3.metric("chance level", "33%", help="3 balanced classes")
        c4.metric("features vs train reviews", f"{res['n_features']:,} vs {res['n_train']}",
                  help="more features than examples: the model can memorise the train set"
                  if res["n_features"] > res["n_train"] else None)

        left, right = st.columns([2, 3])
        with left:
            cm = res["confusion"]
            heat = go.Figure(go.Heatmap(z=cm, x=bow.CLASSES, y=bow.CLASSES, colorscale=BLUES, showscale=False,
                                        text=cm, texttemplate="%{text}",
                                        hovertemplate="true %{y}, predicted %{x}: %{z}<extra></extra>"))
            heat.update_layout(template=THEME_NAME, height=380, margin=dict(l=110), title="Confusion matrix (test)")
            heat.update_yaxes(autorange="reversed", title="true class")
            heat.update_xaxes(title="predicted class")
            chart(heat)
        with right:
            fig = figure(height=380)
            for k, name in enumerate(bow.CLASSES):
                top = np.argsort(-res["coef"][k])[:8]
                fig.add_trace(go.Bar(x=res["coef"][k][top], y=[f"{name} · {f}" for f in res["features"][top]],
                                     orientation="h", name=name, marker_color=CLASS_COLORS[name]))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(title="The words that push most toward each class", margin=dict(l=170))
            chart(fig)

        st.markdown("**The same classifier, other choices** (your min_df, max_df, max_features and n-grams):")
        table = []
        for k in ["count", "tfidf"]:
            for name, c in [("no stop-word removal", bow.Cleaning(lowercase, punctuation, digits, False, True, lemmatize)),
                            ("stop words removed", bow.Cleaning(lowercase, punctuation, digits, True, False, lemmatize)),
                            ("stop words removed, negations kept", bow.Cleaning(lowercase, punctuation, digits, True, True, lemmatize)),
                            ("lemmatized" if not lemmatize else "not lemmatized",
                             bow.Cleaning(lowercase, punctuation, digits, stop_words, keep_negations, not lemmatize))]:
                r = classification(c, bow.Vectorizing(k, ngram_max, min_df, max_df, vectorizing.max_features))
                table.append({"features": {"count": "counts", "tfidf": "TF-IDF"}[k], "cleaning": name,
                              "train": r and f"{r['train_acc']:.0%}", "test": r and f"{r['test_acc']:.0%}",
                              "vocabulary": r and f"{r['n_features']:,}"})
        st.dataframe(table, hide_index=True)
    what_to_look_at(
        "- Compare train and test accuracy: what does the gap tell you? Look at the number of features.\n"
        "- Which class is confused the most, and with which one? Read a few neutral reviews: are they easy?\n"
        "- Do the top words of each class make sense? Which ones look like noise?"
    )
