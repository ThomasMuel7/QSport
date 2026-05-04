from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

def get_context(question: str) -> list[dict[str, Any]]:
    # RAG sur Supabase - a connecter quand SUPABASE_URL et SUPABASE_KEY sont configures
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        return []

    try:
        from sentence_transformers import SentenceTransformer
        from supabase import create_client

        client = create_client(url, key)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(question).tolist()

        response = client.rpc(
            "match_documents",
            {"query_embedding": embedding, "match_count": 5}
        ).execute()

        results = []
        for row in (response.data or []):
            results.append({
                "content": row.get("content", ""),
                "source": row.get("source", ""),
                "metadata": row.get("metadata", {}),
            })
        return results

    except Exception as e:
        print(f"[rag] erreur: {e}")
        return []
