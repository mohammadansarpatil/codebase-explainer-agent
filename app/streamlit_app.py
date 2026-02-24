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
from rag.architecture import explain_architecture_with_sources


st.set_page_config(page_title="Codebase Explainer Agent", layout="wide")
st.title("🧠 Codebase Explainer Agent")

# Sidebar config
st.sidebar.header("LLM Settings")
model_name = st.sidebar.text_input("Ollama model", value="llama3.1:8b")
ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")

st.sidebar.header("Repo")
repo_url = st.sidebar.text_input("GitHub Repo URL", placeholder="https://github.com/owner/repo")
update_repo = st.sidebar.checkbox("Pull latest if already cloned", value=False)

tabs = st.tabs(["1) Ingest", "2) Chat", "3) Architecture"])

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

with tabs[2]:
    st.subheader("Explain Architecture")

    if not repo_url.strip():
        st.info("Enter a GitHub repo URL in the sidebar and ingest it first.")
    else:
        index_dir = index_dir_for(repo_url)
        if not index_dir.exists():
            st.warning("Index not built yet. Go to Ingest tab first.")
        else:
            if st.button("Explain Architecture"):
                embedder = get_embedder()  # not used here but ok to keep pattern
                ollama = get_ollama_client(ollama_url, model_name)

                # Find repo_root (same approach you're using today)
                repo_candidates = sorted(Path("data/repos").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not repo_candidates:
                    st.error("No cloned repo found in data/repos. Please build index again.")
                else:
                    repo_root = repo_candidates[0]

                    with st.spinner("Building architecture overview..."):
                        arch_json, catalog = explain_architecture_with_sources(ollama.chat, repo_root)

                        with st.expander("Show raw architecture JSON"):
                            st.json(arch_json)


                    if not isinstance(arch_json, dict):
                        st.error("Architecture output was not a dict. Check terminal logs for raw LLM output.")
                        st.stop()
                    
                    st.markdown("### Architecture Overview")

                    for m in arch_json.get("modules", []):
                        path = m.get("path", "").lstrip("/")
                        st.markdown(f"**Module:** `{path}`")
                        st.write(f"Responsibility: {m.get('responsibility','')}")
                        st.write("Key symbols: " + ", ".join(m.get("key_symbols", [])))
                        st.write("Citations: " + ", ".join([f"[{c}]" for c in m.get("citations", [])]))
                        st.divider()

                    st.markdown("### Sources")
                    for item in catalog:
                        with st.expander(f"[{item['id']}] {item['ref']} | {item['symbol']}"):
                            st.code(item["text"])
