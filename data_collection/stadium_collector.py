import requests
import logging
from datetime import datetime, timezone
from typing import Optional
from data_collection.models import StadiumData
from data_collection.config import BSD_API_TOKEN

BSD_BASE_URL = "https://sports.bzzoiro.com/api/"

SURFACE_MAP = {
    "grass": "natural",
    "natural grass": "natural",
    "hybrid": "hybrid",
    "artificial": "artificial",
    "astroturf": "artificial",
    "synthetic": "artificial",
}

def fetch_stadium_data(match_id: int, venue_id: int, attendance: Optional[int]) -> Optional[StadiumData]:
    """
    Récupère les infos du stade et calcule le taux de remplissage + mapping surface.
    """
    headers = {"Authorization": f"Token {BSD_API_TOKEN}"}
    try:
        url = f"{BSD_BASE_URL}venues/{venue_id}/"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        capacity = data.get("capacity")
        surface = data.get("surface", "").strip().lower()
        mapped_surface = SURFACE_MAP.get(surface)
        if mapped_surface is None:
            logging.warning(f"[stadium_collector] Surface inconnue '{surface}' pour venue {venue_id}, fallback 'natural'")
            mapped_surface = "natural"
        fill_rate = None
        if attendance is not None and capacity:
            try:
                fill_rate = float(attendance) / float(capacity)
                fill_rate = min(max(fill_rate, 0.0), 1.0)
            except Exception:
                fill_rate = None
        return StadiumData(
            match_id=match_id,
            venue_id=venue_id,
            stadium_fill_rate=fill_rate,
            surface_type=mapped_surface,
            collected_at=datetime.now(timezone.utc)
        )
    except Exception as e:
        logging.error(f"[stadium_collector] Erreur collecte stade {venue_id}: {e}")
        return None
