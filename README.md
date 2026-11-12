# Text embeddings · interactive demos

Seven interactive pages (Streamlit) to explain how text becomes numbers for a neural network: from
bag of words to word embeddings, how embeddings are trained, and pretrained / contextual embeddings.
Animated in the spirit of 3Blue1Brown, on a dark theme, with real data and real models.

| Page | What it shows |
|---|---|
| **Text to features** | One-hot, `CountVectorizer`, TF-IDF on 1,000 hotel reviews: the long tail, over-cleaning (stop words, negations, lemmatization with spaCy), a classifier |
| **Text to vectors** | Animated: a sentence → tokens (words, GPT-2 subwords, characters) → ids → embedding vectors |
| **Cosine similarity** | Animated on real GloVe word vectors: the angle between two words, direction vs length, the 50 numbers |
| **Padding & pooling** | Animated: pad, mask, mean / max pooling, then XGBoost on pooled GloVe vs bag of words |
| **Target 1 · the label** | How an embedding table learns from a sentiment label, in 11 steps, on 2,000 movie reviews |
| **Target 2 · the missing word** | word2vec (CBOW): subsampling, a sliding window, a softmax over 7,000 words |
| **Pretrained embeddings** | GloVe neighbours and analogies, static vs contextual vectors (MiniLM, attention), semantic search |

## Requirements

- **Python 3.10 to 3.13**
- about **2.5 GB** of disk for the virtual environment (PyTorch CPU, spaCy, XGBoost, sentence-transformers…)
- an **internet connection the first time**: two small models are downloaded and cached (see below)

## Install and run

From a terminal (in VS Code: <kbd>Ctrl</kbd>+<kbd>`</kbd>), in this folder:

```bash
# 1. create a virtual environment and activate it
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install the packages (5-10 minutes the first time)
pip install --upgrade pip
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The app starts on **http://localhost:8502** (set in `.streamlit/config.toml`). Stop it with <kbd>Ctrl</kbd>+<kbd>C</kbd>.

Next time, two commands are enough:

```bash
source .venv/bin/activate
streamlit run app.py
```

### Open it inside VS Code

1. <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> → **Simple Browser: Show** → `http://localhost:8502`
   (the app opens in an editor tab, next to the code)
2. So that VS Code uses the same packages: <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> → **Python: Select Interpreter**
   → `.venv/bin/python`

### What happens on the first run

| When | What | Size |
|---|---|---|
| first visit of *Text to vectors* (subwords) | GPT-2's tokenizer, from the Hugging Face Hub | ~1 MB |
| first visit of *Pretrained embeddings* | the sentence encoder `all-MiniLM-L6-v2`, from the Hugging Face Hub | ~90 MB |
| first visit of *Padding & pooling* | trains every model of the comparison (then cached) | ~20 s |
| first tick of *lemmatize* in *Text to features* | spaCy lemmatizes the 1,000 reviews (then cached) | ~20 s |

Everything else (datasets, GloVe vectors, trained tables) is already in `assets/`.

## Project structure

```
app.py                 home page (the list of demos)
pages/                 one file per page (the menu follows the numbers)
components/            the animations: HTML + SVG + JavaScript, no dependency, embedded by the pages
demos/                 the logic, without Streamlit: bag_of_words, text_pipeline, pooling, embedding_targets, pretrained
ui.py                  shared look: header, sidebar sections, dark plotly theme, animation embedding
assets/                data and precomputed models (see below)
data_prep/             the scripts that rebuild the hotel reviews and the GloVe file
.streamlit/config.toml dark theme, port 8502
```

## Rebuilding the assets (optional)

Everything in `assets/` is already built. To rebuild it:

```bash
# the 1,000 hotel reviews (downloads a ~220 MB parquet into the Hugging Face cache)
cd data_prep && python make_hotel_reviews.py && cd ..

# GloVe-50 cut to the corpus + the 50,000 most frequent words (downloads 66 MB into data_prep/raw/)
cd data_prep && python make_hotel_glove.py && cd ..

# the two embedding tables of the "Target" pages, trained on the 2,000 movie reviews (~4 min on a CPU)
python -m demos.embedding_targets
```

## Troubleshooting

- **`streamlit: command not found`**: the virtual environment is not active. Run `source .venv/bin/activate`
  (the prompt then starts with `(.venv)`), or call it directly: `.venv/bin/python -m streamlit run app.py`.
  With conda, run `conda deactivate` first so that conda's `base` does not hide the `.venv`.
- **`Port 8502 is already in use`**: an older server is still running. Stop it (`pkill -f "streamlit run app.py"`)
  or use another port: `streamlit run app.py --server.port 8503`.
- **A change in `demos/` does not show up**: Streamlit reruns the pages when they are saved, not always the modules
  they import. Stop the server and start it again.
- **A page is slow the first time**: see the table above; it is cached afterwards.

## Data and models

| Asset | Source | Licence |
|---|---|---|
| `hotel_reviews.csv` | 1,000 reviews sampled from [jniimi/tripadvisor-review-rating](https://huggingface.co/datasets/jniimi/tripadvisor-review-rating) | Apache-2.0 |
| movie reviews (`movie_reviews_vocab.json`, `embedding_*`) | Pang & Lee *polarity dataset v2.0*, via [nltk_data](https://www.nltk.org/nltk_data/) | see the dataset page |
| `hotel_glove50.npz` | [GloVe](https://nlp.stanford.edu/projects/glove/) `glove-wiki-gigaword-50`, cut down | PDDL (public domain) |
| `play_catalogue.csv` | 12 invented film plots | — |
| GPT-2 tokenizer (downloaded) | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | MIT |
| `all-MiniLM-L6-v2` (downloaded) | [sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 |
| `en_core_web_sm` (pip) | [spaCy](https://spacy.io/models/en) | MIT |
