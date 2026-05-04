# Orchestrateur pipeline de collecte contextuelle

import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Union

from supabase import create_client, Client
from data_collection.config import SUPABASE_URL, SUPABASE_KEY, BSD_API_TOKEN
from data_collection.weather_collector import fetch_weather_for_match
from data_collection.timezone_collector import fetch_timezone_info, get_offset_hours
from data_collection.coach_collector import fetch_coach_days_in_post
from data_collection.referee_collector import fetch_referee_stats
from data_collection.stadium_collector import fetch_stadium_data
from data_collection.geocoding import get_stadium_coordinates

import requests

BSD_BASE_URL = "https://sports.bzzoiro.com/api/"

def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def dataclass_to_dict(obj) -> dict:
    d = obj.__dict__.copy()
    if isinstance(d.get("collected_at"), datetime):
        d["collected_at"] = d["collected_at"].isoformat()
    return d

def fetch_match_details(match_id: int) -> Optional[dict]:
    headers = {"Authorization": f"Token {BSD_API_TOKEN}"}
    url = f"{BSD_BASE_URL}events/{match_id}/"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        event = resp.json()
        # Fallback coordonnées si absentes
        venue = event.get("venue", {})
        if (not venue.get("latitude")) or (not venue.get("longitude")):
            venue_id = venue.get("id")
            if venue_id:
                try:
                    venue_url = f"{BSD_BASE_URL}venues/{venue_id}/"
                    vresp = requests.get(venue_url, headers=headers, timeout=10)
                    vresp.raise_for_status()
                    vdata = vresp.json()
                    event["venue"]["latitude"] = vdata.get("latitude")
                    event["venue"]["longitude"] = vdata.get("longitude")
                except Exception as e:
                    logging.warning(f"[pipeline] Impossible de récupérer les coordonnées du stade {venue_id}: {e}")
        return event
    except Exception as e:
        logging.error(f"[pipeline] Erreur récupération match {match_id}: {e}")
        return None

def upsert_data(table: str, data: dict):
    supabase = get_supabase_client()
    try:
        supabase.table(table).upsert(data).execute()
        logging.info(f"[pipeline] Upsert réussi dans {table} pour match_id={data.get('match_id')}")
    except Exception as e:
        logging.error(f"[pipeline] Erreur upsert {table}: {e}")

def collect_contextual_data(match_id: int) -> Dict[str, Optional[object]]:
    """
    Point d'entrée principal.
    1. Récupère le détail du match depuis BSD API
    2. Lance les collecteurs météo, timezone, arbitre, coach, stade
    3. Retourne un dict contenant toutes les dataclasses
    4. Persiste en base via upsert Supabase
    """
    logging.info(f"[pipeline] Début collecte pour match_id={match_id}")
    match = fetch_match_details(match_id)
    if not match:
        logging.error(f"[pipeline] Impossible de récupérer le match {match_id}")
        return {}

    # Extraction des infos nécessaires
    venue = match.get("venue", {})
    referee = match.get("referee", {})
    home_manager = match.get("home_manager", {})
    away_manager = match.get("away_manager", {})
    attendance = match.get("attendance")
    latitude = venue.get("latitude")
    longitude = venue.get("longitude")
    venue_id = venue.get("id")
    referee_id = referee.get("id")
    home_manager_id = home_manager.get("id")
    away_manager_id = away_manager.get("id")
    event_date_str = match.get("event_date")
    match_datetime_utc = datetime.fromisoformat(event_date_str).astimezone(timezone.utc)

    # Fallback geocoding si latitude/longitude absents
    if not latitude or not longitude:
        coords = get_stadium_coordinates(venue.get("name",""), venue.get("city",""), venue.get("country",""))
        if coords:
            latitude, longitude = coords

    results = {}

    # 1. Timezone
    tz_info = None
    try:
        tz_info = fetch_timezone_info(latitude, longitude, match_datetime_utc)
        results["timezone"] = tz_info
        if tz_info:
            upsert_data("match_timezone", {
                "match_id": match_id,
                "home_timezone_id": tz_info["timeZoneId"],
                "home_timezone_offset_h": get_offset_hours(tz_info),
                "away_travel_timezone_delta": None,  # TODO: calculer depuis le stade habituel de l'équipe visiteuse
                "collected_at": datetime.now(timezone.utc).isoformat()
            })
    except Exception as e:
        logging.error(f"[pipeline] Erreur collecte timezone: {e}")

    # 2. Weather
    try:
        tz_str = tz_info["timeZoneId"] if tz_info else "UTC"
        weather = fetch_weather_for_match(match_id, latitude, longitude, match_datetime_utc, tz_str)
        results["weather"] = weather
        if weather:
            upsert_data("match_weather", dataclass_to_dict(weather))
    except Exception as e:
        logging.error(f"[pipeline] Erreur collecte météo: {e}")

    # 3. Coach
    try:
        coach = fetch_coach_days_in_post(match_id, home_manager_id, away_manager_id, match_datetime_utc)
        results["coach"] = coach
        if coach:
            upsert_data("match_coach", dataclass_to_dict(coach))
    except Exception as e:
        logging.error(f"[pipeline] Erreur collecte coach: {e}")

    # 4. Referee
    try:
        referee_data = fetch_referee_stats(match_id, referee_id)
        results["referee"] = referee_data
        if referee_data:
            upsert_data("match_referee", dataclass_to_dict(referee_data))
    except Exception as e:
        logging.error(f"[pipeline] Erreur collecte arbitre: {e}")

    # 5. Stadium
    try:
        stadium = fetch_stadium_data(match_id, venue_id, attendance)
        results["stadium"] = stadium
        if stadium:
            upsert_data("match_stadium", dataclass_to_dict(stadium))
    except Exception as e:
        logging.error(f"[pipeline] Erreur collecte stade: {e}")

    logging.info(f"[pipeline] Collecte terminée pour match_id={match_id}")
    return results

def collect_batch(match_ids: List[int]) -> Dict[int, Dict[str, Optional[object]]]:
    """
    Mode batch : collecte pour une liste de match_ids
    """
    batch_results = {}
    for match_id in match_ids:
        batch_results[match_id] = collect_contextual_data(match_id)
    return batch_results
