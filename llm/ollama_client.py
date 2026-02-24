import json
import requests


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, format: str | None = None):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # ✅ force single JSON response
        }
        if format:
            payload["format"] = format  # e.g. "json"

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        # ✅ Normal: response is one JSON object
        try:
            data = resp.json()
            return data["message"]["content"]
        except Exception:
            # ✅ Fallback: sometimes Ollama returns NDJSON
            text = resp.text.strip()
            last_obj = None

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    last_obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

            if isinstance(last_obj, dict):
                msg = last_obj.get("message", {})
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]

            # Last resort: return raw body so caller can debug
            return text
