# 📖 RAG-Based Chatbot for Uber Eats Knowledge Base

A **Retrieval-Augmented Generation (RAG)** chatbot that answers Uber Eats customer and partner queries using a structured knowledge base. It combines **document embeddings**, **FAISS vector search**, and **LLM inference via Ollama**, wrapped in a simple **Streamlit UI**.

---

## 🚀 Features

* 🔍 Retrieve knowledge base articles with FAISS
* 🧠 Generate semantic embeddings with `SentenceTransformers`
* 💬 Context-aware answers using Ollama models (`llama3`, `mistral`, etc.)
* 📑 Knowledge base built from Uber Eats Help Center articles
* 🌐 Streamlit-powered interactive app
* ⚡ Evaluation pipeline for multiple models and metrics

---

## 📂 Project Structure

```
├── app.py                 # Main Streamlit app
├── scripts/
│   ├── normalize.py        # Data cleaning
│   ├── build_index.py      # FAISS index creation
│   └── evaluate.py         # Model evaluation
├── data/
│   ├── raw/                # Raw HTML/Markdown
│   ├── clean/              # Normalized Markdown
│   └── index/              # FAISS index + metadata
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/RAG-Based-Chatbot.git
cd RAG-Based-Chatbot

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull Ollama models
ollama pull llama3
ollama pull mistral
```

---

## ▶️ Usage

### Run Streamlit app

```bash
streamlit run app.py
```

### Example query

> *“What is Uber Eats’ cancellation policy?”*
> The chatbot retrieves policy docs and generates a contextual answer.

---

## 🧪 Evaluation

Run benchmark tests across models:

```bash
python scripts/evaluate.py
```

Metrics: **accuracy, precision, recall, F1**.

---

## 🔑 Environment Variables

Add to `.streamlit/secrets.toml`:

```toml
OLLAMA_HOST="http://localhost:11434"
OLLAMA_AUTH="Basic <your-encoded-credentials>"
```

---

## 📊 Roadmap

* [ ] Deploy on Streamlit Cloud
* [ ] Support for GPT-4o, Gemma, etc.
* [ ] Expand KB with merchant & courier docs
* [ ] Fine-tune retrieval pipeline

---

## 🤝 Contributing

PRs welcome! Please open an issue before major changes.
Follow PEP8 + add docstrings.

---

⚡ **Author:** [Hihsikesh Phukan](https://github.com/s4031214)


