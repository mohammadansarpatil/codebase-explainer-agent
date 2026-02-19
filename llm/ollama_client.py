import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, messages, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
