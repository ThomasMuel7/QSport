from __future__ import annotations
from typing import Any

SYSTEM_PROMPT = (
    "Tu es QSport, expert en predictions sportives football, tennis et NBA."
)

def _format_context(context: list[dict[str, Any]]) -> str:
    if not context:
        return ""
    lines = []
    for index, item in enumerate(context, start=1):
        source = item.get("source") or "source inconnue"
        content = item.get("content") or ""
        if "fallback" not in source:
            lines.append(f"[{index}] {content}")
    return "\n".join(lines)

def _format_history(history: list) -> str:
    if not history:
        return ""
    formatted = []
    for item in history[-5:]:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content") or ""
            formatted.append(f"{role}: {content}")
    return "\n".join(formatted)

def build_prompt(question: str, context: list[dict], history: list) -> str:
    ctx = _format_context(context)
    hist = _format_history(history)
    
    prompt = f"### Instruction\n{SYSTEM_PROMPT}\n"
    if ctx:
        prompt += f"Donnees disponibles: {ctx}\n"
    if hist:
        prompt += f"Historique: {hist}\n"
    prompt += f"\n### Question\n{question}\n\n### Reponse\n"
    return prompt
