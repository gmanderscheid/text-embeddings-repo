"""J5 lecture 02 - Pretrained embeddings, toward Transformers.

GloVe-50 (static) and all-MiniLM-L6-v2 (contextual, from the Hugging Face Hub), as in the notebook:
neighbours, analogies, one vector per word vs per occurrence, sentence vectors and semantic search.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demos import pretrained as pt
from ui import COLORS, THEME_NAME, embed_animation, figure, page_header, plotly_kwargs, set_plotly_theme, sidebar_section, what_to_look_at

st.set_page_config(page_title="Pretrained embeddings", page_icon="🚀", layout="wide")
set_plotly_theme()
CHART = plotly_kwargs()
BLUE_SCALE = [[0, "#0B0F17"], [1, "#58C4DD"]]

page_header(
    "J5 · Pretrained embeddings, toward Transformers",
    "Nobody trains embeddings from scratch on 2,000 reviews: like the ResNet of J4, we **reuse** vectors trained on "
    "billions of words. **Static** word vectors (GloVe) give one vector per word; **contextual** embeddings from a "
    "small Transformer (MiniLM) give one vector per occurrence, and one per sentence for semantic search.",
)


@st.cache_resource(show_spinner="Loading GloVe...")
def glove():
    return pt.StaticVectors()


@st.cache_resource(show_spinner="Loading all-MiniLM-L6-v2 from the Hugging Face Hub (90 MB the first time)...")
def encoder():
    return pt.load_encoder()


@st.cache_data
def catalogue():
    return pt.catalogue()


@st.cache_resource
def catalogue_vectors():
    return encoder().encode(catalogue()["plot"].tolist(), normalize_embeddings=True)


try:
    enc = encoder()
except Exception as err:
    st.error(f"Could not load the sentence encoder ({err}). `pip install sentence-transformers`, and run once online.")
    st.stop()
g, cat, cv = glove(), catalogue(), catalogue_vectors()

with sidebar_section("The animation"):
    query = st.text_input("search query (step 6)", "aliens trying to contact us")
    analogy = st.selectbox("analogy (step 3)", pt.ANALOGIES, format_func=lambda a: f"{a[0]} − {a[2]} + {a[1]}")

embed_animation("pretrained", pt.animation_data(g, enc, cv, cat, query, analogy=analogy))

tabs = st.tabs(["1 · GloVe neighbours", "2 · Analogies", "3 · Static vs contextual", "4 · Semantic search",
                "5 · Inside the encoder", "6 · The day in one table"])

# ================================================================== 1 · neighbours
with tabs[0]:
    st.dataframe(pd.DataFrame({w: [n for n, _ in g.most_similar([w])[1]] for w in pt.PROBES}).T, hide_index=False)
    word = st.text_input("any word", "apple")
    missing, nb = g.most_similar([word.lower().strip()], topn=10)
    if missing:
        st.warning(f"“{word}” is not among the {len(g.words):,} GloVe words kept here.")
    else:
        fig = figure(height=380)
        fig.add_trace(go.Bar(x=[c for _, c in nb], y=[w for w, _ in nb], orientation="h", marker_color=COLORS[1],
                             hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title=f"GloVe's 10 nearest neighbours of “{word}” (cosine)", margin=dict(l=110))
        st.plotly_chart(fig, **CHART)
    st.caption("Clean neighbours, even for words that were noisy in our 1.3-million-word tables; but the meanings of "
               "Wikipedia and the news (director → executive, apple → iphone, bank → banking).")

# ================================================================== 2 · analogies
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    a = c1.text_input("a", "king"); c = c2.text_input("− c", "man"); b = c3.text_input("+ b", "woman")
    missing, best = g.most_similar([a.lower(), b.lower()], [c.lower()], topn=5)
    if missing:
        st.warning("unknown word(s): " + ", ".join(missing))
    else:
        st.markdown(f"**{a} − {c} + {b} =** " + " · ".join(f"{w} ({s:.2f})" for w, s in best))
    W = [w for p in pt.PAIRS for w in p]
    xy, _ = pt.pca2(np.stack([g.vec(w) for w in W]))
    fig = figure(height=480)
    for k in range(0, len(W), 2):
        fig.add_annotation(x=xy[k + 1, 0], y=xy[k + 1, 1], ax=xy[k, 0], ay=xy[k, 1], xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowcolor=COLORS[1], arrowwidth=1.5)
    fig.add_trace(go.Scatter(x=xy[::2, 0], y=xy[::2, 1], mode="markers+text", text=W[::2], textposition="bottom center",
                             marker=dict(color=COLORS[1], size=10), name="country"))
    fig.add_trace(go.Scatter(x=xy[1::2, 0], y=xy[1::2, 1], mode="markers+text", text=W[1::2], textposition="top center",
                             marker=dict(color=COLORS[0], size=10), name="capital"))
    fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
    fig.update_layout(title="Country → capital in GloVe (2 principal directions of 50): the arrows point roughly the same way")
    st.plotly_chart(fig, **CHART)
    st.caption("Honest version: most_similar excludes the input words; analogies work on famous relations, much less "
               "in general; and the same arithmetic reproduces the stereotypes of the training text.")

# ================================================================== 3 · static vs contextual
with tabs[2]:
    target = st.text_input("ambiguous word", "bank").lower().strip()
    sentences = st.text_area("sentences containing it (one per line)", "\n".join(pt.BANK.values()), height=110).splitlines()
    sentences = [s for s in sentences if s.strip()][:6]
    vecs, names = [], []
    for s in sentences:
        v = pt.word_vector(enc, s, target)
        if v is not None:
            vecs.append(v); names.append(f"sentence {len(names) + 1}")
    if len(vecs) < 2:
        st.info(f"Write at least two sentences where “{target}” is a single token of MiniLM's tokenizer.")
    else:
        V = np.stack(vecs).astype(np.float64)
        st.caption(" · ".join(f"**{n}**: {s}" for n, s in zip(names, [s for s in sentences if pt.word_vector(enc, s, target) is not None])))
        left, right = st.columns(2)
        for col, M, title in [(left, np.ones((len(V), len(V))), "GloVe: one row per word → all 1"),
                              (right, V @ V.T, "MiniLM: one vector per occurrence")]:
            heat = go.Figure(go.Heatmap(z=M, x=names, y=names, zmin=.5, zmax=1, colorscale=BLUE_SCALE, showscale=False,
                                        texttemplate="%{z:.2f}", textfont=dict(size=15)))
            heat.update_layout(template=THEME_NAME, height=380, title=title, margin=dict(l=90))
            heat.update_yaxes(autorange="reversed")
            col.plotly_chart(heat, **CHART)
        layer = st.slider("attention: layer", 1, 6, 6)
        s0 = sentences[0]
        pairs = pt.word_attention(enc, s0, target, layer - 1) if pt.word_vector(enc, s0, target) is not None else []
        if pairs:
            fig = figure(height=320)
            fig.add_trace(go.Bar(x=[t for t, _ in pairs], y=[w for _, w in pairs], marker_color=COLORS[3]))
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(title=f"Where “{target}” reads from in “{s0}” (layer {layer}, mean of 12 heads, words only)")
            st.plotly_chart(fig, **CHART)
            st.caption("A preview of the next module: this weighting is **attention**. [CLS] alone takes ~40% of the "
                       "raw weight in most layers (an “attention sink”): it is left out and the rest renormalised.")

# ================================================================== 4 · semantic search
with tabs[3]:
    q = st.text_input("describe a film", query, key="search_q")
    qv = enc.encode(q, normalize_embeddings=True)
    scores = cv @ qv
    res = cat.assign(score=scores.round(3), shared=[", ".join(pt.shared_words(q, p)) or "none" for p in cat["plot"]])
    st.dataframe(res.sort_values("score", ascending=False)[["title", "score", "shared", "plot"]].rename(
        columns={"shared": "words in common with the query"}), hide_index=True)
    st.markdown("**Four sentences, one matrix:**")
    four = ["The film was a complete waste of time.", "I regret paying for this movie, it was dreadful.",
            "An absolute masterpiece, I was moved to tears.", "The stock market fell sharply on Monday."]
    F = enc.encode(four, normalize_embeddings=True)
    labels = ["waste of time", "regret paying", "masterpiece", "stock market"]
    heat = go.Figure(go.Heatmap(z=(F @ F.T).astype(np.float64), x=labels, y=labels, zmin=0, zmax=1, colorscale=BLUE_SCALE,
                                texttemplate="%{z:.2f}", showscale=False))
    heat.update_layout(template=THEME_NAME, height=420, title="The first two share almost no word and are the closest pair")
    heat.update_yaxes(autorange="reversed")
    st.plotly_chart(heat, **CHART)
    st.caption("Limits: the encoder reads at most 256 tokens (long texts are cut into chunks); embeddings match meaning "
               "but can miss an exact name or code that word matching finds at once: production systems combine both (hybrid search).")

# ================================================================== 5 · inside the encoder
with tabs[4]:
    st.code(str(enc), language=None)
    table = enc[0].auto_model.embeddings.word_embeddings
    c1, c2, c3 = st.columns(3)
    c1.metric("token embedding table", f"{tuple(table.weight.shape)}", help="one row per subword of the tokenizer")
    c2.metric("parameters", f"{sum(p.numel() for p in enc.parameters()) / 1e6:.0f} M")
    c3.metric("max tokens read", enc.max_seq_length)
    text = st.text_input("tokenize", "The unbelievably boring sequel")
    st.markdown(" ".join(f"`{t}`" for t in enc.tokenizer.tokenize(text)))
    st.markdown("1. a **Transformer** (its first layer is an embedding table like ours, then 6 layers that mix the "
                "tokens) → one contextual vector per token; 2. **Pooling** (mean): the masked mean of lecture 00 → one "
                "vector per text; 3. **Normalize**: length 1, so the cosine is a dot product.")

# ================================================================== 6 · recap
with tabs[5]:
    st.markdown(
        "| | Our word2vec | GloVe | Sentence encoder (MiniLM) |\n|---|---|---|---|\n"
        "| Trained on | 1.3 million words | 6 billion tokens | 1 billion+ sentence pairs, on top of a pretrained language model |\n"
        "| Target | the missing word | co-occurrence counts | the partner in the batch (contrastive) |\n"
        "| Unit | word | word | subword, then sentence |\n"
        "| One vector per | word | word | **occurrence** (token) and **text** |\n"
        "| What's inside | an embedding table | an embedding table | an embedding table **+ Transformer layers** + mean pooling |"
    )
    st.info("How does the vector of *bank* look at *river*? That is **attention**: the next module.")

what_to_look_at(
    "- Step 1: which neighbours surprise you? What do they say about the corpus GloVe was trained on?\n"
    "- Step 3: change the analogy in the sidebar. Which ones work, and why only famous relations?\n"
    "- Steps 4-5: why are the three GloVe vectors identical? Which two MiniLM vectors are the closest?\n"
    "- Step 6: write a query that shares no word with the film you want. Then one with an exact name."
)
