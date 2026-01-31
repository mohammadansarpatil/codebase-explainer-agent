import streamlit as st
import requests

from app.config import OLLAMA_BASE_URL, DEFAULT_MODEL

st.set_page_config(page_title="Codebase Explainer Agent", layout="wide")

st.title("🧠 Codebase Explainer Agent")
st.write("Day 1: Streamlit UI + Local LLM (Ollama) ✅")

# --- Repo input (we'll use this in Day 2) ---
repo_url = st.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/owner/repo"
)

if st.button("Test Input"):
    if repo_url.strip():
        st.success(f"Got URL: {repo_url}")
    else:
        st.warning("Please enter a repository URL.")

st.divider()

# --- LLM Ping (Day 1 task) ---
st.subheader("Local LLM Test (Ollama)")

model_name = st.text_input("Ollama model name", value=DEFAULT_MODEL)
user_msg = st.text_input("Message to model", value="You are running locally via Ollama. Reply in one sentence: confirm you are reachable and ready to answer codebase questions.")

def ollama_chat(model: str, message: str) -> str:
    """
    Calls Ollama's local chat endpoint and returns the assistant response text.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": False
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

if st.button("Ping Ollama"):
    if not model_name.strip():
        st.error("Please enter a model name (e.g., llama3.1:8b).")
    elif not user_msg.strip():
        st.error("Please enter a message.")
    else:
        try:
            with st.spinner("Calling Ollama locally..."):
                answer = ollama_chat(model_name.strip(), user_msg.strip())
            st.success("Model responded ✅")
            st.text_area("Response", value=answer, height=200)
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to Ollama. Make sure Ollama is running:\n\n"
                "1) In another terminal: `ollama serve`\n"
                "2) Then try again."
            )
        except requests.HTTPError as e:
            st.error(f"Ollama returned an HTTP error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
