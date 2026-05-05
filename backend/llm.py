from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv


load_dotenv()

DEFAULT_OLLAMA_HOST = "http://ollama:11434"


def ask_ollama(prompt: str, model: str = "mistral") -> str:
    ollama_host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    url = f"{ollama_host}/api/generate"

    try:
        response = httpx.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response")

        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        return "Ollama a repondu, mais aucun texte exploitable n'a ete retourne."
    except httpx.RequestError as exc:
        return (
            "Le service Ollama est indisponible pour le moment. "
            f"Verifiez OLLAMA_HOST ({ollama_host}) et le service Docker. Detail: {exc}"
        )
    except httpx.HTTPStatusError as exc:
        return (
            "Ollama a retourne une erreur HTTP. "
            f"Statut: {exc.response.status_code}. Detail: {exc.response.text[:300]}"
        )
    except ValueError:
        return "Ollama a retourne une reponse non JSON impossible a lire."

