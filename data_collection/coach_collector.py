import requests
import logging
from datetime import datetime, timezone
from typing import Optional
from data_collection.models import CoachData
from data_collection.config import BSD_API_TOKEN

BSD_BASE_URL = "https://sports.bzzoiro.com/api/"

def fetch_coach_days_in_post(match_id: int, home_manager_id: int, away_manager_id: int, match_date: datetime) -> Optional[CoachData]:
    """
    Récupère la durée en poste des coachs (en jours) pour un match donné et d'autres infos contextuelles.
    """
    headers = {"Authorization": f"Token {BSD_API_TOKEN}"}
    def get_info(manager_id):
        info = {
            "days": -1,
            "nationality": None,
            "total_matches_managed": None,
            "win_rate_overall": None
        }
        if not manager_id:
            return info
        try:
            url = f"{BSD_BASE_URL}managers/{manager_id}/"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            date_appointed = data.get("date_appointed")
            if date_appointed:
                try:
                    appointed = datetime.strptime(date_appointed, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    delta = (match_date - appointed).days
                    info["days"] = delta if delta >= 0 else -1
                except Exception:
                    info["days"] = -1
            info["nationality"] = data.get("nationality")
            info["total_matches_managed"] = data.get("total_matches_managed")
            info["win_rate_overall"] = data.get("win_rate_overall")
        except Exception as e:
            logging.error(f"[coach_collector] Erreur collecte coach {manager_id}: {e}")
        return info

    home_info = get_info(home_manager_id)
    away_info = get_info(away_manager_id)
    return CoachData(
        match_id=match_id,
        home_coach_days_in_post=home_info["days"],
        away_coach_days_in_post=away_info["days"],
        home_nationality=home_info["nationality"],
        away_nationality=away_info["nationality"],
        home_total_matches_managed=home_info["total_matches_managed"],
        away_total_matches_managed=away_info["total_matches_managed"],
        home_win_rate_overall=home_info["win_rate_overall"],
        away_win_rate_overall=away_info["win_rate_overall"],
        collected_at=datetime.now(timezone.utc)
    )
