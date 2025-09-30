# 📖 RAG-Based Chatbot for Uber Eats Knowledge Base

A Retrieval-Augmented Generation (RAG) chatbot designed to answer customer and partner queries about Uber Eats using an internal knowledge base. The system combines **document embeddings**, **FAISS vector search**, and **LLM inference via Ollama**, all wrapped in an interactive **Streamlit** interface.

---

## 🚀 Features

* 🔍 **Document Retrieval** with FAISS vector database
* 🧠 **Sentence Embeddings** using `SentenceTransformers`
* 💬 **LLM-powered responses** via Ollama (`llama3`, `mistral`, etc.)
* 📑 Structured knowledge base built from Uber Eats Help Center articles
* 🌐 Streamlit web app with an easy-to-use interface
* ⚡ Configurable evaluation pipeline for testing multiple models

---

## 📂 Project Structure

```
├── app.py                 # Main Streamlit app
├── scripts/               # Data preprocessing & indexing scripts
│   ├── normalize.py
│   ├── build_index.py
│   └── evaluate.py
├── data/
│   ├── raw/               # Original HTML/Markdown articles
│   ├── clean/             # Cleaned/normalized Markdown
│   └── index/             # FAISS index + metadata
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/RAG-Based-Chatbot.git
cd RAG-Based-Chatbot
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama models

```bash
ollama pull llama3
ollama pull mistral
```

---

## ▶️ Usage

### Run the Streamlit app

```bash
streamlit run app.py
```

### Example query

> *“What is Uber Eats’ cancellation policy?”*
> The chatbot will retrieve relevant sections and generate a context-aware response.

---

## 🧪 Evaluation

To benchmark retrieval and model accuracy:

```bash
python scripts/evaluate.py
```

Metrics include **accuracy, precision, recall, F1 score**, and qualitative comparison across models.

---

## 🔑 Environment Variables

Configure secrets (e.g., Ollama host, API keys) in `.streamlit/secrets.toml`:

```toml
OLLAMA_HOST="http://localhost:11434"
OLLAMA_AUTH="Basic <your-encoded-credentials>"
```

---

## 📊 Roadmap

* [ ] Deploy via Streamlit Cloud
* [ ] Add support for more LLMs (Gemma, GPT-4o, etc.)
* [ ] Expand knowledge base with merchant/courier policies
* [ ] Fine-tune retrieval pipeline

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.
Ensure that all scripts follow PEP8 and include docstrings.

---

## 📜 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

⚡ **Author:** [Your Name](https://github.com/<your-username>)
💬 Feel free to open an issue for questions or suggestions!

---

Do you want me to also create a **shorter professional version** (like a one-page summary README) for recruiters/hackathon judges, or should we keep it detailed for developers?


Here’s a solid **README.md** draft tailored for your repo (RAG-Based Chatbot for Uber Eats). You can copy it directly or tweak details like repo name, your username, or screenshots.

---

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

## 📜 License

---

⚡ **Author:** [Hihsikesh Phukan](https://github.com/s4031214)

