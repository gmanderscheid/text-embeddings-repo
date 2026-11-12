"""Home page of the Deep Learning Foundations demos (shown by the trainer, separate from the labs).

Run locally:   streamlit run app.py      (opens on http://localhost:8502)
"""
import streamlit as st

from ui import COLORS, page_header, set_plotly_theme

st.set_page_config(page_title="Deep Learning Foundations demos", page_icon="🎬", layout="wide")
set_plotly_theme()

page_header(
    "Deep Learning Foundations · demos",
    "Interactive demos the trainer shows during the lectures. Pick one in the sidebar, or below.",
)

DEMOS = [
    ("J5", "Text to features", "pages/1_Text_to_features.py",
     "One-hot, counts and TF-IDF on 1,000 hotel reviews: the long tail, over-cleaning, a classifier, the limits."),
    ("J5", "Text to vectors", "pages/2_Text_to_vectors.py",
     "Animated: a sentence is cut into tokens (words, subwords, characters), looked up, turned into embeddings."),
    ("J5", "Cosine similarity", "pages/3_Cosine_similarity.py",
     "Animated, on real word vectors: the angle between two words, only the direction counts, then the 50 numbers."),
    ("J5", "Padding & pooling", "pages/4_Padding_and_pooling.py",
     "Animated: pad, mask, mean/max pool GloVe vectors, then XGBoost vs bag of words on the same hotel reviews."),
    ("J5", "Target 1 · the label", "pages/5_Target_1_the_label.py",
     "Lecture 01: the sentiment classifier's table: one gradient for every word of a review, words split on one axis."),
    ("J5", "Target 2 · the missing word", "pages/6_Target_2_the_missing_word.py",
     "Lecture 01: word2vec CBOW: subsampling, a sliding window, a softmax over 7,000 words, topics gathering."),
    ("J5", "Pretrained embeddings", "pages/7_Pretrained_embeddings.py",
     "Lecture 02: GloVe neighbours and analogies, static vs contextual (MiniLM, attention preview), semantic search."),
]

for day, name, target, blurb in DEMOS:
    col_day, col_text = st.columns([1, 9])
    col_day.markdown(f"<h2 style='color:{COLORS[4]}; margin:0'>{day}</h2>", unsafe_allow_html=True)
    with col_text:
        if hasattr(st, "page_link"):            # Streamlit >= 1.31
            st.page_link(target, label=name, icon="➡️")
        else:
            st.markdown(f"**{name}** — open it from the sidebar")
        st.caption(blurb)
