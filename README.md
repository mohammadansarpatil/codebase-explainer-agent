# Codebase Explainer Agent

Autonomous agent that ingests a GitHub repository, builds a local code search index (RAG), answers questions about the codebase, and generates README-style documentation.

## Features (Planned)
- Clone repo from GitHub URL
- Filter irrelevant files
- Chunk code by functions/classes
- Embed chunks locally (SentenceTransformers)
- Store in FAISS
- Tool-using agent for Q&A (LangGraph)
- Streamlit UI

## Local Setup (WIP)

### 1) Create venv & install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Will be added as the project progresses.
