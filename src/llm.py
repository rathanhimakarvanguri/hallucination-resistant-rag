"""Thin client for the local Ollama server (chat + embeddings)."""
import requests

from src import config


def _post(path, payload, timeout=300):
    try:
        r = requests.post(f"{config.OLLAMA_URL}{path}", json=payload, timeout=timeout)
    except requests.ConnectionError:
        raise SystemExit("Cannot reach Ollama at localhost:11434. Open the Ollama app (or run `ollama serve`) and try again.")
    if r.status_code == 404:
        raise SystemExit(f"Ollama model not found. Run: ollama pull {payload.get('model')}")
    r.raise_for_status()
    return r.json()


def embed(texts, batch_size=32):
    """Return one embedding vector per input text."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        data = _post("/api/embed", {"model": config.EMBED_MODEL, "input": batch})
        vectors.extend(data["embeddings"])
    return vectors


def chat(system, user, model=None):
    data = _post("/api/chat", {
        "model": model or config.CHAT_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": config.TEMPERATURE},
    })
    return data["message"]["content"].strip()
