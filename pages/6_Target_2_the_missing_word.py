"""J5 lecture 01 - target #2: the missing word (self-supervised, word2vec CBOW).

The CBOW of the notebook (mean of 6 context rows + Linear(64, 7000), cross-entropy), trained on the same
2,000 movie reviews; loaded from assets/embedding_targets.npz (python -m demos.embedding_targets).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demos import embedding_targets as et
from ui import (COLORS, THEME_NAME, embed_animation, figure, page_header, plotly_kwargs, set_plotly_theme,
                sidebar_section, what_to_look_at)

st.set_page_config(page_title="Target 2 · the missing word", page_icon="🕳️", layout="wide")
set_plotly_theme()
CHART = plotly_kwargs()
RDBU = [[0, "#58C4DD"], [.5, "#0B0F17"], [1, "#FC6255"]]

page_header(
    "J5 · Target 2 · the missing word",
    "Labels are expensive, raw text is free. **Hide a word, predict it from its neighbours**: the target comes from "
    "the text itself (self-supervised). The notebook's CBOW, on the same 2,000 reviews, with the same table size.",
)


@st.cache_resource(show_spinner="Loading the trained tables...")
def tables():
    return et.Tables()


if not et.FILE.exists():
    st.error("assets/embedding_targets.npz is missing: run `python -m demos.embedding_targets` from the project folder (~4 min).")
    st.stop()
T = tables()

with sidebar_section("The excerpt of the animation"):
    k = st.radio("review excerpt", range(len(T.excerpts)),
                 format_func=lambda i: " ".join(T.excerpts[i].split()[:6]) + " …")

embed_animation("target_missing", et.missing_animation_data(T, k))

tabs = st.tabs(["1 · Fill in the blank", "2 · Same words, two tables", "3 · Two heat maps", "4 · Training"])

with tabs[0]:
    st.markdown("Write a context with `___` for the hidden word: CBOW averages the rows of the words around it "
                "(3 on each side, as in training) and scores the 7,000 words.")
    sentence = st.text_input("context", "the ___ rode into town and arrested the outlaw")
    toks = sentence.lower().split()
    if "___" in toks:
        c = toks.index("___")
        context = [w for w in toks[max(0, c - 3):c] + toks[c + 1:c + 4] if w in T.vocab]
        p = T.predict_missing(context)
        top = np.argsort(-p)[:12]
        fig = figure(height=430)
        fig.add_trace(go.Bar(x=p[top], y=[T.words[i] for i in top], orientation="h", marker_color=COLORS[2],
                             hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickformat=".1%")
        fig.update_layout(title=f"CBOW's top guesses · context used: {' '.join(context) or '(no known word)'}",
                          margin=dict(l=110))
        st.plotly_chart(fig, **CHART)
        st.caption("Trained on 1.3 million words only: the guesses are noisy, and frequent words (the, a, his) win often. "
                   "The original word2vec read 100 billion words.")
    else:
        st.info("Put `___` where the hidden word is.")

with tabs[1]:
    rows = [{"word": w, "target = label (sentiment)": ", ".join(x for x, _ in T.neighbours(w, "label", 4)),
             "target = missing word (word2vec)": ", ".join(x for x, _ in T.neighbours(w, "cbow", 4))} for w in et.PROBES]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    pairs = [("good", "bad"), ("great", "awful"), ("funny", "boring")]
    cos = lambda which, a, b: float(T.cosines(which, [a, b])[0, 1])
    st.markdown("**Opposite words, in each table:**")
    st.dataframe(pd.DataFrame([{"pair": f"{a} / {b}", "label table": round(cos("label", a, b), 2),
                                "word2vec table": round(cos("cbow", a, b), 2)} for a, b in pairs]), hide_index=True)
    st.markdown("*alien*, *murder*, *music*, *director* get meaningful neighbours in the word2vec table: it organised "
                "words by **topic and role**. *good* and *bad* are close there: they appear in the same sentences, "
                "“a good film”, “a bad film”.")

with tabs[2]:
    words = [w for g in et.GROUPS.values() for w in g]
    c1, c2 = st.columns(2)
    for col, which, title in [(c1, "label", "target = label"), (c2, "cbow", "target = missing word")]:
        heat = go.Figure(go.Heatmap(z=T.cosines(which, words), x=words, y=words, zmin=-1, zmax=1, colorscale=RDBU,
                                    showscale=which == "cbow", hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
        heat.update_layout(template=THEME_NAME, height=560, title=title, margin=dict(l=90, b=90))
        heat.update_yaxes(autorange="reversed")
        heat.update_xaxes(tickfont=dict(size=9)); heat.update_yaxes(tickfont=dict(size=9))
        col.plotly_chart(heat, **CHART)
    st.success("**Close in an embedding space means “interchangeable for the task the table was trained on.”** "
               "Before using someone else's embeddings, ask what they were trained to predict.")

with tabs[3]:
    h = T.cbow_hist
    fig = figure(height=380)
    ep = np.arange(1, len(h) + 1)
    fig.add_trace(go.Scatter(x=ep, y=h[:, 0], name="train loss", line=dict(color=COLORS[1], width=3)))
    fig.add_trace(go.Scatter(x=ep, y=h[:, 1], name="validation loss", line=dict(color=COLORS[3], width=3)))
    fig.add_hline(y=np.log(len(T.words)), line_dash="dot", line_color="#555", annotation_text="log(7,000): uniform guess")
    fig.update_xaxes(title="epoch")
    fig.update_layout(title="CBOW: cross-entropy over the 7,000 words (best validation: epoch 3)")
    st.plotly_chart(fig, **CHART)

what_to_look_at(
    "- Step 2: why drop most occurrences of “the”? What would the table learn about them otherwise?\n"
    "- Steps 3-4: is the loss low? Does it matter for the table?\n"
    "- Step 6 and tab 3: which groups form a block now? Where did positive vs negative go?\n"
    "- Tab 1: write a sentence where the hidden word is obvious to you. Does CBOW find it?"
)
