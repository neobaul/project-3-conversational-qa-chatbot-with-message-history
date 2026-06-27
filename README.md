# LangChain Pipelines — Query Routing, RAG & Agentic Self-Correction

Three progressive LangChain experiments covering intelligent query routing, vector-based retrieval (RAG), and a production-grade agentic self-correcting RAG engine.

---

## What's Inside

### `getting_started.ipynb` — Intelligent Query Router
A multi-chain LLM pipeline that classifies developer questions and routes them to the right expert. Uses a Pydantic schema (`TechnicalReview`) to extract three structured fields — **category**, **complexity**, and **suggested tools** — from every query. Based on the category, the query is routed to either an architecture expert chain or a general engineering chain, each with its own specialized prompt. Routing is handled at runtime using `RunnableParallel` and `RunnableLambda`.

### `main.ipynb` — Vector Stores & Retrievers (RAG Fundamentals)
A hands-on introduction to RAG. Creates a document collection, embeds it using HuggingFace, stores it in Chroma, and demonstrates similarity search. Wires the retriever into a full RAG chain using Groq's LLaMA model to answer questions grounded in the document collection.

### `Simpleapp.py` — Agentic Self-Correcting RAG
The most advanced file. A production-grade async RAG engine that:
1. Scrapes a live URL and chunks + embeds the content into a FAISS vector store
2. Runs **Maximal Marginal Relevance (MMR)** retrieval for diverse context
3. Spins up a **supervisor grading agent** that evaluates each retrieved chunk in parallel
4. Drops low-quality chunks before passing only verified content to the LLM
5. Streams the final response in real time

---

## Project Structure

```
├── getting_started.ipynb   # Intelligent query router with structured output
├── main.ipynb              # RAG fundamentals: Chroma + HuggingFace + Groq
├── Simpleapp.py            # Agentic self-correcting RAG engine (async + streaming)
├── requirements.txt        # Python dependencies
└── project synopsis.txt    # Project overview
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/neobaul/project-3-conversational-qa-chatbot-with-message-history.git
cd project-3-conversational-qa-chatbot-with-message-history
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install langchain-openai langchain-community faiss-cpu beautifulsoup4
```

### 4. Set up API keys
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here           # for main.ipynb
LANGCHAIN_API_KEY=your_key_here      # optional, for LangSmith tracing
```

### 5. Run the agentic RAG engine
```bash
python Simpleapp.py
```

---

## How `Simpleapp.py` Works

```
URL → WebBaseLoader → Text Splitter → FAISS Vector Store
                                            ↓
                              MMR Retriever (k=4, fetch_k=12)
                                            ↓
                    ┌───────── Supervisor Grading Agent ─────────┐
                    │  Grade each chunk: YES / NO (in parallel)  │
                    └────────────────────────────────────────────┘
                                            ↓
                             Verified chunks only → GPT-4o
                                            ↓
                                  Streamed answer
```

If all chunks are rejected by the grading agent, execution halts gracefully rather than hallucinating an answer.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | GPT-4o / GPT-4o-mini (OpenAI) |
| RAG (notebooks) | Groq LLaMA + Chroma + HuggingFace Embeddings |
| RAG (Simpleapp) | FAISS + OpenAI Embeddings + MMR |
| Web Scraping | LangChain `WebBaseLoader` + BeautifulSoup |
| Orchestration | LangChain LCEL (`RunnableParallel`, `RunnableLambda`) |
| Tracing | LangSmith (optional) |
| Language | Python 3.11 |

---

## Key Concepts Demonstrated

- **LCEL (LangChain Expression Language)** — composing chains with `|` syntax
- **Structured output** — Pydantic schemas with `JsonOutputParser`
- **Dynamic routing** — `RunnableLambda` for runtime chain selection
- **RAG pipeline** — embed → store → retrieve → generate
- **MMR retrieval** — maximizes diversity in retrieved chunks
- **Agentic grading** — LLM-as-supervisor to filter bad context before generation
- **Async streaming** — `asyncio.gather` for parallel grading + `astream` for real-time output
