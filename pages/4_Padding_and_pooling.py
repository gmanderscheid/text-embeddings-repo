"""J5 - Padding & pooling: from token vectors to one vector per review, then XGBoost.

Top: an animation (components/padding_pooling.html) on sentences typed in the sidebar.
Below: the same 1,000 hotel reviews and the same split as "Text to features", so that bag of words and
pooled GloVe embeddings can be compared with the same models.
"""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from demos import pooling
from ui import COLORS, GRID, PANEL, THEME_NAME, figure, page_header, set_plotly_theme, sidebar_section, what_to_look_at

st.set_page_config(page_title="Padding & pooling", page_icon="🧱", layout="wide")
set_plotly_theme()

TEMPLATE = Path(__file__).resolve().parent.parent / "components" / "padding_pooling.html"
CHART = {"theme": None} if "width" in inspect.signature(st.plotly_chart).parameters \
    else {"theme": None, "use_container_width": True}
CLASS_COLORS = {"neutral": COLORS[3], "positive": COLORS[2], "negative": COLORS[0]}
FAMILY_COLORS = {"one axis per word": COLORS[1], "pooled embeddings": COLORS[3]}
DEFAULT = ("The room was spotless and the staff were lovely\n"
           "Dirty bathroom, rude staff\n"
           "The hotel was okay but the breakfast wasn't great at all")


def chart(fig):
    st.plotly_chart(fig, **CHART)


page_header(
    "J5 · Padding & pooling",
    "Reviews have different lengths, a tensor is a rectangle, and most models want **one vector per text**. "
    "Pad, mask, then pool the word vectors (mean or max), and a classic model like XGBoost can read them. "
    "Below, the same 1,000 hotel reviews as *Text to features*: bag of words vs pooled embeddings.",
)


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading GloVe...")
def glove():
    return pooling.Glove()


@st.cache_data
def reviews():
    return pooling.load_reviews()


@st.cache_resource(show_spinner="Training the models on the 800 train reviews...")
def sentence_models():
    return pooling.fit_sentence_models(glove(), reviews())


@st.cache_data(show_spinner="Training every representation × model on the same split...")
def comparison(max_len, n_train):
    return pd.DataFrame(pooling.compare(glove(), reviews(), max_len, n_train))


@st.cache_data(show_spinner="Learning curves: training on 50 to 800 reviews...")
def curves():
    return pooling.learning_curve(glove(), reviews())


@st.cache_data
def lengths():
    return np.array([len(pooling.tokenize(t)) for t in reviews()["text"]])


# ------------------------------------------------------------------ sidebar
with sidebar_section("Sentences of the animation"):
    text = st.text_area("one sentence per line (up to 4, 10 tokens each)", DEFAULT, height=150)
sentences = [s.strip() for s in text.splitlines() if s.strip()][:4] or ["great hotel"]

with sidebar_section("Experiment on the hotel reviews"):
    max_len = st.slider("max_len: cut every review after ... tokens", 10, 110, 110,
                        help="Longer reviews are truncated before pooling; shorter ones are padded.")
    n_train = st.select_slider("train reviews used", [50, 100, 200, 400, 800], 800)

# ------------------------------------------------------------------ the animation
data = pooling.animation_data(glove(), sentence_models(), sentences)
html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
if hasattr(st, "iframe"):
    st.iframe(html, height=800)
else:
    components.html(html, height=800)

tabs = st.tabs(["1 · The cost of padding", "2 · Bag of words vs pooled embeddings", "3 · Little data", "4 · Your sentences"])

# ================================================================== 1 · padding cost
with tabs[0]:
    n = lengths()
    kept = np.minimum(n, max_len)
    pad_share = 1 - kept.sum() / (len(n) * max_len)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("tokens per review", f"{np.median(n):.0f}", help=f"median; min {n.min()}, max {n.max()}")
    c2.metric("padded tensor", f"(1000, {max_len}, 50)")
    c3.metric("share of the tensor that is padding", f"{pad_share:.0%}")
    c4.metric("reviews truncated", f"{(n > max_len).sum():,}", help="tokens after max_len are lost")
    fig = figure(height=380)
    fig.add_trace(go.Histogram(x=n, nbinsx=40, marker_color=COLORS[1], name="reviews"))
    fig.add_vline(x=max_len, line_dash="dot", line_color=COLORS[0], annotation_text=f"max_len = {max_len}")
    fig.update_xaxes(title="tokens per review")
    fig.update_yaxes(title="reviews")
    fig.update_layout(title="Pad to max_len: every review shorter than the line gets <pad> tokens, every longer one is cut",
                      bargap=.05)
    chart(fig)
    st.markdown("A batch is padded to its longest member. Sorting reviews by length before making batches "
                "(“bucketing”) keeps the padding small; the **mask** makes sure padding never changes the result.")

# ================================================================== 2 · comparison
with tabs[1]:
    res = comparison(max_len if max_len < 110 else None, n_train)
    best_bow = res[res["family"] == "one axis per word"]["test"].max()
    best_emb = res[res["family"] == "pooled embeddings"]["test"].max()
    c1, c2, c3 = st.columns(3)
    c1.metric("best bag of words (test)", f"{best_bow:.0%}", help="~5,000 features, one per word")
    c2.metric("best pooled embeddings (test)", f"{best_emb:.0%}", f"{best_emb - best_bow:+.0%} vs bag of words")
    c3.metric("features", "50 vs ~5,000", help="mean-pooled GloVe vs bag of words")

    res = res.assign(name=res["representation"] + " · " + res["model"])
    fig = figure(height=430)
    fig.add_trace(go.Bar(y=res["name"], x=res["train"], orientation="h", name="train", marker_color=GRID,
                         hovertemplate="%{y}: train %{x:.0%}<extra></extra>"))
    fig.add_trace(go.Bar(y=res["name"], x=res["test"], orientation="h", name="test",
                         marker_color=[FAMILY_COLORS[f] for f in res["family"]],
                         text=[f"{v:.0%}" for v in res["test"]], textposition="outside",
                         hovertemplate="%{y}: test %{x:.0%}<extra></extra>"))
    fig.add_vline(x=1 / 3, line_dash="dot", line_color="#555", annotation_text="chance")
    fig.update_xaxes(range=[0, 1.08], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"Same 1,000 reviews, same split, {n_train} train reviews: accuracy (blue = bag of words, orange = pooled GloVe)",
                      barmode="group", margin=dict(l=330))
    chart(fig)
    st.dataframe(res[["representation", "model", "features", "train", "test", "seconds"]].style.format(
        {"train": "{:.0%}", "test": "{:.0%}", "seconds": "{:.2f}", "features": "{:,}"}), hide_index=True)
    st.markdown(
        "**Reading the result.** With 800 training reviews, the bag of words still wins: each sentiment word gets its "
        "own weight, while averaging 70 static word vectors blurs the few words that carry the opinion (*not*, "
        "*rude*, *lovely* are drowned in *the*, *room*, *was*). Max pooling keeps extremes but loses everything else. "
        "Pooled embeddings are **100× smaller** and know words never seen in training (tab 4), but GloVe was trained "
        "on Wikipedia, not for sentiment: the next step is to learn the embedding **for the task** (lecture 01), or to "
        "use **contextual** embeddings (lecture 02)."
    )

# ================================================================== 3 · learning curve
with tabs[2]:
    lc = curves()
    fig = figure(height=430)
    styles = {"bag of words + logistic regression": (COLORS[1], "solid"),
              "GloVe mean + XGBoost": (COLORS[3], "solid"), "GloVe mean + logistic regression": (COLORS[3], "dot")}
    for name, (color, dash) in styles.items():
        part = lc[lc["model"] == name]
        fig.add_trace(go.Scatter(x=part["n"], y=part["test"], mode="lines+markers", name=name,
                                 line=dict(color=color, width=3, dash=dash),
                                 hovertemplate=f"{name}<br>%{{x}} train reviews: %{{y:.0%}}<extra></extra>"))
    fig.update_xaxes(type="log", tickvals=[50, 100, 200, 400, 800], title="train reviews (log)")
    fig.update_yaxes(tickformat=".0%", title="test accuracy")
    fig.update_layout(title="With few examples, pretrained embeddings start ahead; with more, the bag of words catches up",
                      legend=dict(y=-0.28), margin=dict(b=110))
    chart(fig)
    st.caption("Mean over 3 random subsets of the train reviews for each size; the 200 test reviews never change.")

# ================================================================== 4 · your sentences
with tabs[3]:
    st.markdown("The sentences of the sidebar, read by the three models trained on the 800 train reviews. "
                "Try words the reviews never use (*superb*, *filthy*, *immaculate*...).")
    preds = pooling.predict_sentences(glove(), sentence_models(), sentences)
    for i, s in enumerate(sentences):
        st.markdown(f"<div style='background:{PANEL};border-left:4px solid {COLORS[4]};padding:6px 12px;margin-top:10px'>"
                    f"<b>{s}</b></div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for col, key, title in [(cols[0], "bow", "bag of words + logistic regression"),
                                (cols[1], "mean", "GloVe mean + XGBoost"), (cols[2], "max", "GloVe max + XGBoost")]:
            p = preds[key][i]
            best = int(np.argmax(p))
            col.markdown(f"<small>{title}</small><br><b style='color:{CLASS_COLORS[pooling.CLASSES[best]]}'>"
                         f"{pooling.CLASSES[best]}</b> · " + " · ".join(f"{c} {v:.0%}" for c, v in zip(pooling.CLASSES, p)),
                         unsafe_allow_html=True)
        if preds["unseen"][i]:
            st.caption("words the bag of words never saw in training (silently ignored): " + ", ".join(preds["unseen"][i]))

what_to_look_at(
    "- In the animation, switch mean ↔ max: which numbers change, and which word wins each row in max?\n"
    "- Why must the pads be excluded from a mean, and set to −∞ for a max? What happens with zeros?\n"
    "- Tab 2: 50 features vs 5,000. Why does the bag of words still win here, and what would make embeddings win?\n"
    "- Tab 4: find a sentence the bag of words gets wrong and the embeddings get right, and the opposite."
)
