import streamlit as st

st.set_page_config(page_title="Codebase Explainer Agent", layout="wide")

st.title("🧠 Codebase Explainer Agent")
st.write("Day 1: Streamlit UI is running ✅")

repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo")

if st.button("Test Input"):
    if repo_url.strip():
        st.success(f"Got URL: {repo_url}")
    else:
        st.warning("Please enter a repository URL.")
