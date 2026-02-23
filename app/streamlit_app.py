import os
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llm.ollama_client import OllamaClient
from rag.embed import Embedder
from rag.answer import answer_question
from app.pipeline import build_index_for_repo
from app.ui_state import index_dir_for


st.set_page_config(page_title="Codebase Explainer Agent", layout="wide")
st.title("🧠 Codebase Explainer Agent")

# Sidebar config
st.sidebar.header("LLM Settings")
model_name = st.sidebar.text_input("Ollama model", value="llama3.1:8b")
ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")

st.sidebar.header("Repo")
repo_url = st.sidebar.text_input("GitHub Repo URL", placeholder="https://github.com/owner/repo")
update_repo = st.sidebar.checkbox("Pull latest if already cloned", value=False)

tabs = st.tabs(["1) Ingest", "2) Chat"])

# Cache embedder so it loads once
@st.cache_resource
def get_embedder():
    return Embedder()

@st.cache_resource
def get_ollama_client(base_url: str, model: str):
    return OllamaClient(base_url=base_url, model=model)


with tabs[0]:
    st.subheader("Ingest repository and build index")

    if not repo_url.strip():
        st.info("Enter a GitHub repo URL in the sidebar.")
    else:
        index_dir = index_dir_for(repo_url)

        st.write("Index location:", str(index_dir))

        if index_dir.exists():
            st.success("Index already exists ✅ (you can go to Chat tab)")
        else:
            st.warning("No index found yet.")

        if st.button("Build / Rebuild Index"):
            with st.spinner("Cloning + chunking + embedding + indexing..."):
                repo_root, index_dir = build_index_for_repo(repo_url, update=update_repo)
            st.success(f"Index built at {index_dir} ✅")

with tabs[1]:
    st.subheader("Ask questions about the codebase")

    if not repo_url.strip():
        st.info("Enter a GitHub repo URL in the sidebar and ingest it first.")
    else:
        index_dir = index_dir_for(repo_url)
        if not index_dir.exists():
            st.warning("Index not built yet. Go to Ingest tab first.")
        else:
            question = st.text_input("Your question", value="How does HTTP request sending work?")
            if st.button("Answer"):
                embedder = get_embedder()
                ollama = get_ollama_client(ollama_url, model_name)

                repo_id = index_dir.name
                repo_root = Path("data/repos")  # we will infer actual clone path later

                # For now, infer repo root by scanning data/repos for owner__repo folder.
                # (Simple week-1 approach; we can store this mapping cleanly later.)
                # We'll just pick the newest folder if multiple exist.
                repo_candidates = sorted(Path("data/repos").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not repo_candidates:
                    st.error("No cloned repo found in data/repos. Please build index again.")
                else:
                    repo_root = repo_candidates[0]

                with st.spinner("Retrieving + generating answer..."):
                    answer, sources = answer_question(
                        llm_chat=ollama.chat,
                        repo_root=repo_root,
                        index_dir=index_dir,
                        embedder=embedder,
                        question=question,
                    )

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Sources used")
                for s in sources:
                    with st.expander(f"{s.rel_path}:{s.start_line}-{s.end_line} | {s.symbol}"):
                        st.code(s.text)
