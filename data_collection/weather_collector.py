import requests
from datetime import datetime, timezone
from typing import Optional
import logging
from data_collection.models import WeatherData

def fetch_weather_for_match(match_id: int, latitude: float, longitude: float, match_datetime_utc: datetime, timezone_str: str) -> Optional[WeatherData]:
    """
    Récupère la météo à l'heure du match via Open-Meteo Archive API.
    - match_datetime_utc : datetime du match en UTC
    - timezone_str : ex 'Europe/London'
    """
    date_str = match_datetime_utc.strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,precipitation,windspeed_10m",
        "timezone": timezone_str
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Trouver l'heure locale du match dans la réponse
        hours = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]
        precs = data["hourly"]["precipitation"]
        winds = data["hourly"]["windspeed_10m"]

        # Convertir match_datetime_utc en heure locale
        from_zone = timezone.utc
        import zoneinfo
        to_zone = zoneinfo.ZoneInfo(timezone_str)
        match_local = match_datetime_utc.astimezone(to_zone)
        match_hour_str = match_local.strftime("%Y-%m-%dT%H:00")

        # Chercher l'index de l'heure exacte
        idx = None
        for i, t in enumerate(hours):
            if t.startswith(match_hour_str):
                idx = i
                break
        if idx is None:
            raise ValueError(f"Heure du match {match_hour_str} non trouvée dans la réponse météo.")

        return WeatherData(
            match_id=match_id,
            temperature_c=temps[idx],
            precipitation_mm=precs[idx],
            wind_speed_kmh=winds[idx],
            collected_at=datetime.now(timezone.utc)
        )
    except Exception as e:
        logging.error(f"[weather_collector] Erreur collecte météo: {e}")
        return None
