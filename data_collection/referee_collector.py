import requests
import logging
from datetime import datetime, timezone
from typing import Optional
from data_collection.models import RefereeData
from data_collection.config import BSD_API_TOKEN

BSD_BASE_URL = "https://sports.bzzoiro.com/api/"

def fetch_referee_stats(match_id: int, referee_id: int) -> Optional[RefereeData]:
    """
    Calcule les stats arbitre sur les 20 derniers matchs arbitrés.
    Si < 5 matchs historiques, retourne toutes les features à None.
    """
    headers = {"Authorization": f"Token {BSD_API_TOKEN}"}
    try:
        # Récupérer les 20 derniers matchs arbitrés (API paginée, ordering côté serveur)
        events_url = f"{BSD_BASE_URL}events/?referee={referee_id}&ordering=-event_date&page_size=20"
        events_resp = requests.get(events_url, headers=headers, timeout=10)
        events_resp.raise_for_status()
        matches = events_resp.json().get("results", [])
        if len(matches) < 5:
            return RefereeData(
                match_id=match_id,
                referee_id=referee_id,
                referee_avg_yellow_cards=None,
                referee_avg_red_cards=None,
                referee_home_win_pct=None,
                collected_at=datetime.now(timezone.utc)
            )
        yellow_cards = []
        red_cards = []
        home_wins = 0
        total = 0
        for m in matches:
            stats = m.get("stats", {})
            yellow = stats.get("yellow_cards") or m.get("yellow_cards")
            red = stats.get("red_cards") or m.get("red_cards")
            if yellow is not None:
                yellow_cards.append(float(yellow))
            if red is not None:
                red_cards.append(float(red))
            # Victoire domicile
            home_score = m.get("home_score")
            away_score = m.get("away_score")
            if home_score is not None and away_score is not None:
                try:
                    if float(home_score) > float(away_score):
                        home_wins += 1
                    total += 1
                except Exception:
                    pass
        avg_yellow = sum(yellow_cards) / len(yellow_cards) if yellow_cards else None
        avg_red = sum(red_cards) / len(red_cards) if red_cards else None
        home_win_pct = home_wins / total if total > 0 else None
        return RefereeData(
            match_id=match_id,
            referee_id=referee_id,
            referee_avg_yellow_cards=avg_yellow,
            referee_avg_red_cards=avg_red,
            referee_home_win_pct=home_win_pct,
            collected_at=datetime.now(timezone.utc)
        )
    except Exception as e:
        logging.error(f"[referee_collector] Erreur collecte arbitre {referee_id}: {e}")
        return None
