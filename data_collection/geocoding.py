import requests
import logging
import time
from typing import Optional, Tuple

def get_stadium_coordinates(stadium_name: str, city: str, country: str) -> Optional[Tuple[float, float]]:
    """
    Récupère lat/lng d'un stade via Nominatim (OpenStreetMap).
    Retourne (latitude, longitude) ou None si non trouvé.
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "QSport-ML-Project/1.0"}
    # Essai 1 : nom du stade + ville
    params = {"q": f"{stadium_name}, {city}, {country}", "format": "json", "limit": 1}
    try:
        time.sleep(1)  # Nominatim impose 1 req/sec
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
        # Essai 2 : juste ville + country si stade non trouvé
        params["q"] = f"{city}, {country}"
        time.sleep(1)
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        results = resp.json()
        if results:
            logging.warning(f"[geocoding] Stade '{stadium_name}' non trouvé, fallback sur ville '{city}'")
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as e:
        logging.error(f"[geocoding] Erreur geocoding {stadium_name}: {e}")
    return None
