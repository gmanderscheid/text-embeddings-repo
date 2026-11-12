"""J5 lecture 01 - target #1: the label of the text (supervised).

The ReviewClassifier of the notebook (EmbeddingBag mean + Linear(64, 1), BCE), trained on the 2,000 movie
reviews; its table is loaded from assets/embedding_targets.npz (python -m demos.embedding_targets).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demos import embedding_targets as et
from ui import (BLUES, COLORS, THEME_NAME, embed_animation, figure, page_header, plotly_kwargs, set_plotly_theme,
                sidebar_section, what_to_look_at)

st.set_page_config(page_title="Target 1 · the label", page_icon="🏷️", layout="wide")
set_plotly_theme()
CHART = plotly_kwargs()
RDBU = [[0, "#58C4DD"], [.5, "#0B0F17"], [1, "#FC6255"]]

page_header(
    "J5 · Target 1 · the label of the text",
    "An embedding table has **no target of its own**: it learns whatever the task around it needs. Here the task is "
    "the notebook's sentiment classifier on the 2,000 movie reviews: is this review positive or negative? "
    "The only signal the table ever gets is the gradient of that answer.",
)


@st.cache_resource(show_spinner="Loading the trained tables...")
def tables():
    return et.Tables()


if not et.FILE.exists():
    st.error("assets/embedding_targets.npz is missing: run `python -m demos.embedding_targets` from the project folder (~4 min).")
    st.stop()
T = tables()

@st.cache_data(show_spinner="Retraining the notebook's classifier to record every step (a few seconds, once)...")
def story():
    import json
    if not et.STORY.exists():
        et.build_label_story()
    return json.loads(et.STORY.read_text())


S = story()
with sidebar_section("The review of the animation"):
    example = st.radio("a real sentence of the corpus", range(len(S["examples"])),
                       format_func=lambda i: f"{'👍' if S['examples'][i]['label'] else '👎'} {S['examples'][i]['raw']}")
st.caption("11 steps, every number real: the notebook's vocabulary (7,000 words) and length (1,000 tokens), its random "
           "table at the start, its gradient, and its training run on the 1,600 training reviews.")
embed_animation("target_label", {**S, "example": example}, height=820)

tabs = st.tabs(["1 · What the table learned", "2 · Neighbours", "3 · 30 words, one heat map", "4 · Training"])

with tabs[0]:
    scores = T.scores()
    s = scores.copy()
    s[~T.frequent] = 0
    ranked = np.argsort(s)
    idx = list(ranked[:12]) + list(ranked[-12:])
    df = pd.DataFrame({"word": [T.words[i] for i in idx], "score": [float(scores[i]) for i in idx]}).sort_values("score")
    fig = figure(height=620)
    fig.add_trace(go.Bar(x=df["score"], y=df["word"], orientation="h",
                         marker_color=[COLORS[2] if v > 0 else COLORS[0] for v in df["score"]],
                         hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
    fig.update_layout(title="score = row · w: how much a word pushes a review towards positive (words seen ≥ 50 times)",
                      margin=dict(l=110))
    st.plotly_chart(fig, **CHART)
    st.markdown("The negative side reads like a dictionary of bad reviews. The positive side is noisier: words that "
                "are not positive in themselves but appear mostly in positive reviews of this corpus. "
                "**The table learned correlations with the label, not meanings.**")

with tabs[1]:
    frequent_words = sorted(w for i, w in enumerate(T.words) if T.frequent[i])
    word = st.selectbox("word", frequent_words, index=frequent_words.index("alien"))
    nb = T.neighbours(word, "label", 10)
    fig = figure(height=420)
    fig.add_trace(go.Bar(x=[c for _, c in nb], y=[w for w, _ in nb], orientation="h", marker_color=COLORS[1],
                         hovertemplate="%{y}: cosine %{x:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[-1, 1], title="cosine with " + word)
    fig.update_layout(title=f"The 10 nearest neighbours of “{word}” in the sentiment table", margin=dict(l=110))
    st.plotly_chart(fig, **CHART)
    st.markdown("Try *terrible* or *boring* (clean neighbours: same sentiment), then *alien*, *murder*, *music*, "
                "*director*: their neighbours have nothing to do with their topic, only with their effect on the label.")

with tabs[2]:
    words = [w for g in et.GROUPS.values() for w in g]
    C = T.cosines("label", words)
    heat = go.Figure(go.Heatmap(z=C, x=words, y=words, zmin=-1, zmax=1, colorscale=RDBU,
                                hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    heat.update_layout(template=THEME_NAME, height=640, title="Cosine between 30 words of 5 groups, in the sentiment table",
                       margin=dict(l=100, b=100))
    heat.update_yaxes(autorange="reversed")
    st.plotly_chart(heat, **CHART)
    st.markdown("Almost every cosine is close to +1 or −1: each word was pushed along **one single axis**, the one "
                "that predicts the label. Positive and negative words form two opposite blocks; the topic groups have "
                "no block of their own.")

with tabs[3]:
    h = T.sup_hist
    fig = figure(height=380)
    ep = np.arange(1, len(h) + 1)
    fig.add_trace(go.Scatter(x=ep, y=h[:, 2], name="train accuracy", line=dict(color=COLORS[1], width=3)))
    fig.add_trace(go.Scatter(x=ep, y=h[:, 3], name="validation accuracy", line=dict(color=COLORS[3], width=3)))
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title="epoch")
    fig.update_layout(title="The notebook's training run: train accuracy races to 100%, validation stays near 86%")
    st.plotly_chart(fig, **CHART)
    st.caption("Weights restored from the best validation epoch (4), as `dl_utils.train` does. Same seeds as the notebook.")

what_to_look_at(
    "- Step 5: why does every word of the review get the *same* push? What does it do to “the” or “director”?\n"
    "- Change the label in the toolbar: which way do the words go now?\n"
    "- Step 6 and tab 2: which words end up with a meaningful place, which ones don't? Why?\n"
    "- Would this table be useful to tag the genre of a film?"
)
