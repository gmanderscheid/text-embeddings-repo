"""J5 - Cosine similarity, on word vectors: the angle between two words, why only the direction counts, from
+1 to -1, then the real computation on the 50 numbers of two GloVe words.

Everything moves in the browser (components/cosine_similarity.html); the page sends it 16 real GloVe words
(drawn in 2D by PCA), their real 50-number cosines, and the vectors of the preset pairs.
"""
import streamlit as st

from demos.pretrained import StaticVectors, cosine_words_data
from ui import embed_animation, page_header, set_plotly_theme, what_to_look_at

st.set_page_config(page_title="Cosine similarity", page_icon="📐", layout="wide")
set_plotly_theme()

page_header(
    "J5 · Cosine similarity",
    "How do we say that two words are *similar*? By the angle between their vectors. The arrows are real GloVe "
    "word vectors (drawn in 2D): drag a tip onto another word, and step through with **next ▶**.",
)


@st.cache_data(show_spinner="Loading GloVe...")
def data():
    return cosine_words_data(StaticVectors())


embed_animation("cosine_similarity", data(), height=800)

with st.expander("The same thing in PyTorch"):
    st.code('''import torch
import torch.nn.functional as F

great, excellent = glove["great"], glove["excellent"]     # two word vectors of 50 numbers

cos = (great @ excellent) / (great.norm() * excellent.norm())   # the formula of the animation
cos = F.cosine_similarity(great, excellent, dim=0)              # the same, built in

# neighbours of a word: one matrix product against the whole (normalised) table
table = F.normalize(table, dim=1)                         # every row has length 1 ...
scores = table @ F.normalize(great, dim=0)                # ... so a dot product IS the cosine
best = scores.topk(6)''', language="python")

what_to_look_at(
    "- Step 1: drag an arrow from food words to royalty words. How does the angle change? And the real 50-number cosine?\n"
    "- Why can the cosine of the 2D drawing differ from the real one? What is lost when 50 numbers are drawn in 2?\n"
    "- Step 3: a word that appears more often can get a longer vector. Does it change its cosine with other words?\n"
    "- Step 5, good / bad: opposite meanings, cosine 0.80. What did GloVe actually learn?"
)
