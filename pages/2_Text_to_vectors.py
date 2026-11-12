"""J5 - Text to vectors: an animated walk through text -> tokens -> ids -> embeddings (lecture 00).

Three tokenizers to compare on the same sentence: the lecture's words, GPT-2's subwords, characters.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from demos.text_pipeline import MAX_TOKENS, count_tokens, load_gpt2, load_review_vocabs, pipeline
from ui import page_header, set_plotly_theme, sidebar_section, what_to_look_at

st.set_page_config(page_title="Text to vectors", page_icon="🔤", layout="wide")
set_plotly_theme()

TEMPLATE = Path(__file__).resolve().parent.parent / "components" / "text_to_vectors.html"
LABELS = {"words": "words (lecture 00)", "subwords": "subwords (GPT-2 BPE)", "characters": "characters"}

page_header(
    "J5 · Text to vectors",
    "Type a sentence and watch the pipeline at the entrance of every language model: the text is cut into "
    "tokens, each token is looked up in the vocabulary to get an id, and each id selects one row of the "
    "embedding table. Step through it with **next ▶**, or let it run with **auto ▶▶**.",
)


@st.cache_data(show_spinner="Building the vocabularies of the 2,000 movie reviews (first run only)...")
def review_vocabs():
    return load_review_vocabs()


@st.cache_resource(show_spinner="Loading the GPT-2 tokenizer...")
def gpt2():
    return load_gpt2()


with sidebar_section("Your text"):
    text = st.text_area("sentence", "This movie was GREAT, but the zorglub was boring!", height=100)
    mode = st.radio("tokenizer", list(LABELS), format_func=LABELS.get,
                    help="Words: the rule of lecture 00. Subwords: the tokenizer of GPT-2. "
                         "Characters: one token per character.")
    dim = st.slider("embedding dimension d", 2, 12, 8,
                    help="The lecture uses 8. Real models use 300 (GloVe) to several thousands (LLMs).")

try:
    words, chars, info = review_vocabs()
except OSError as err:
    st.error(f"Could not download the movie reviews to build the vocabularies ({err}). "
             "Run the page once with an internet connection: they are then cached in assets/.")
    st.stop()
vocabs = {"words": words, "characters": chars}
try:
    vocabs["subwords"] = gpt2()
except Exception as err:                       # no `tokenizers`, or GPT-2 never downloaded and no network
    if mode == "subwords":
        st.error(f"Could not load the GPT-2 tokenizer ({err}). `pip install tokenizers`, and run once online.")
        st.stop()

data = pipeline(text, mode, dim, vocabs)
if data["truncated"]:
    st.info(f"Long text: only the first {MAX_TOKENS[mode]} tokens are animated.")

# "</" escaped so that a "</script>" typed in the sentence cannot close the script tag
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
html = TEMPLATE.read_text().replace("__DATA__", payload)
if hasattr(st, "iframe"):                       # Streamlit >= 1.52
    st.iframe(html, height=800)
else:
    components.html(html, height=800)

with sidebar_section("Same sentence, three tokenizers"):
    counts = count_tokens(text, vocabs)
    sizes = {"words": len(words), "characters": len(chars),
             "subwords": len(vocabs["subwords"][1]) if "subwords" in vocabs else None}
    st.dataframe(pd.DataFrame(
        [[m, counts[m], f"{sizes[m]:,}" if sizes[m] else "-"] for m in LABELS],
        columns=["", "tokens", "vocab"]), hide_index=True)
    st.caption(f"Words and characters: seen at least 15 times in {info['n_reviews']:,} movie reviews "
               f"(same ids as lecture 00). Subwords: GPT-2's 50,257 tokens, learned by BPE on web text.")

tokens = data["tokens"]
if tokens:
    table = pd.DataFrame(
        [[t["label"], t["id"], *data["rows"][t["id"]]] for t in tokens],
        columns=["token", "id", *[f"d{j}" for j in range(dim)]],
    )
    with st.expander("The same result as a table", expanded=False):
        st.dataframe(table.style.format(precision=2).background_gradient(
            cmap="coolwarm", vmin=-2.2, vmax=2.2, subset=[f"d{j}" for j in range(dim)]),
            hide_index=True)

CODE = {
    "words": f'''import re, torch, torch.nn as nn

def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())          # lower-case, keep runs of letters

ids = torch.tensor([vocab.get(w, UNK) for w in tokenize(text)])  # vocabulary lookup
embedding = nn.Embedding({len(words)}, {dim}, padding_idx=PAD)   # the table W''',
    "characters": f'''import torch, torch.nn as nn

ids = torch.tensor([char_vocab.get(c, UNK) for c in text.lower()])   # one id per character
embedding = nn.Embedding({len(chars)}, {dim}, padding_idx=PAD)          # the table W''',
    "subwords": f'''import torch, torch.nn as nn
from tokenizers import Tokenizer                          # pip install tokenizers

tokenizer = Tokenizer.from_pretrained("gpt2")             # byte-level BPE, 50,257 tokens
ids = torch.tensor(tokenizer.encode(text).ids)            # no lower-casing, no <unk>
embedding = nn.Embedding(50257, {dim})                    # GPT-2 itself uses d = 768''',
}
with st.expander("The same thing in PyTorch"):
    st.code(CODE[mode] + f"\nvectors = embedding(ids)                                  "
            f"# shape (length, dim) = ({len(tokens)}, {dim})", language="python")

what_to_look_at(
    "- Switch the tokenizer on the same sentence: how many tokens with each one, and how big is each vocabulary?\n"
    "- Try a made-up word or a typo with each tokenizer: which one loses it in `<unk>`, which ones keep a trace of it?\n"
    "- With subwords, compare “the”, “ the” and “The”: same id or not?\n"
    "- Write the same word twice: are the two vectors the same? Is that a problem for “bank” (river vs money)?\n"
    "- Change d in the sidebar: what changes in the table W, and what stays the same?"
)
