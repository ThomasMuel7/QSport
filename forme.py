from __future__ import annotations

# dataclass: simplifie la définition des clients API (attributs + constructeur).
from dataclasses import dataclass
# datetime/timedelta: gestion des dates de match et des fenêtres temporelles (21 jours).
from datetime import datetime, timedelta
# typing: annotations pour clarifier les structures manipulées.
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo
# re + unicodedata: normalisation robuste des noms (accents, ponctuation, variantes).
import re
import unicodedata
import logging
logging.basicConfig(level=logging.WARNING)

# pandas: structure de sortie demandée (DataFrame).
import pandas as pd
# geopy: géocodage open-source (sans Google).
from geopy.geocoders import Nominatim
# requests: appels HTTP vers BSD.
import requests
# timezonefinder: déduction du fuseau depuis lat/lng.
from timezonefinder import TimezoneFinder


# URL de base de l'API BSD (sports.bzzoiro.com).
BSD_BASE_URL = "https://sports.bzzoiro.com/api"

# Logger module
logger = logging.getLogger(__name__)


# ----------------------------
# Utilitaires généraux
# ----------------------------
def _to_datetime(value: Optional[str]) -> Optional[datetime]:
    """
    Convertit une chaîne ISO 8601 en datetime timezone-aware.
    Retourne None si la valeur est absente ou invalide.
    """
    if not value:
        return None
    iso_value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_value)
    except ValueError:
        return None


def _norm_name(name: str) -> str:
    """
    Normalisation stricte: trim + minuscule + espaces compressés.
    """
    return " ".join((name or "").strip().lower().split())


def _norm_name_loose(name: str) -> str:
    """
    Normalisation souple pour matching fuzzy:
    - supprime les accents,
    - supprime la ponctuation,
    - conserve lettres/chiffres/espaces.
    """
    base = _norm_name(name)
    base = unicodedata.normalize("NFKD", base)
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    base = re.sub(r"[^a-z0-9 ]+", " ", base)
    return " ".join(base.split())


def _haversine_km(origin: Tuple[float, float], destination: Tuple[float, float]) -> float:
    """
    Distance orthodromique ("à vol d'oiseau") en kilomètres entre deux points lat/lng.
    """
    from math import asin, cos, radians, sin, sqrt

    lat1, lon1 = origin
    lat2, lon2 = destination
    r_lat1, r_lon1 = radians(lat1), radians(lon1)
    r_lat2, r_lon2 = radians(lat2), radians(lon2)
    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1
    a = sin(dlat / 2) ** 2 + cos(r_lat1) * cos(r_lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371.0088 * c


# ----------------------------
# Client BSD (données football)
# ----------------------------
@dataclass
class BSDClient:
    """
    Client HTTP pour l'API BSD.
    - authentification via Token,
    - pagination gérée automatiquement.
    """

    api_key: str
    timeout: int = 30
    base_url: str = BSD_BASE_URL

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {self.api_key}",
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        GET simple vers BSD + gestion des erreurs HTTP.
        """
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self.session.get(url, params=params or {}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _get_paginated(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Récupère toutes les pages:
        - events/fixtures/matches: pagination limit/offset,
        - autres endpoints: pagination page.
        """
        params = dict(params or {})
        results: List[Dict[str, Any]] = []

        if path.strip("/") in {"events", "fixtures", "matches"}:
            limit = int(params.pop("limit", 200))
            offset = int(params.pop("offset", 0))

            while True:
                batch_params = {**params, "limit": limit, "offset": offset}
                payload = self._get(path, batch_params)
                batch = payload.get("results", [])
                results.extend(batch)
                if not payload.get("next") or not batch:
                    break
                offset += limit
            return results

        page = int(params.pop("page", 1))
        while True:
            payload = self._get(path, {**params, "page": page})
            batch = payload.get("results", [])
            results.extend(batch)
            if not payload.get("next") or not batch:
                break
            page += 1
        return results

    def get_event_detail(self, event_id: int, full: bool = True, tz: str = "UTC") -> Dict[str, Any]:
        """
        Détail complet d'un match.
        """
        return self._get(f"events/{event_id}/", {"full": str(full).lower(), "tz": tz})

    def get_player_stats_for_event(self, event_id: int, tz: str = "UTC") -> List[Dict[str, Any]]:
        """
        Stats joueurs du match (souvent vides avant le coup d'envoi).
        """
        return self._get_paginated("player-stats/", {"event": event_id, "tz": tz})

    def get_player_history(self, player_id: int, tz: str = "UTC") -> List[Dict[str, Any]]:
        """
        Historique de stats d'un joueur (tous matchs disponibles).
        """
        return self._get_paginated("player-stats/", {"player": player_id, "tz": tz})

    def get_team_events_until(self, team_id: int, target_date: datetime, tz: str = "UTC") -> List[Dict[str, Any]]:
        """
        Matchs terminés d'une équipe jusqu'à la date ciblée.
        Utilisé pour calculer les jours de repos.
        """
        date_to = target_date.date().isoformat()
        return self._get_paginated(
            "events/",
            {
                "team_id": team_id,
                "date_to": date_to,
                "status": "finished",
                "tz": tz,
                "limit": 200,
            },
        )

    def get_predicted_lineup(self, event_id: int) -> Optional[Dict[str, Any]]:
        """
        Lineup prédite (endpoint beta) pour les matchs à venir.
        404 => pas de lineup prédite disponible.
        """
        try:
            return self._get(f"predicted-lineup/{event_id}/")
        except requests.HTTPError as exc:
            resp = exc.response
            status_code = resp.status_code if resp is not None else None
            # Pas de lineup prédite
            if status_code == 404:
                return None
            # Cas fréquent: 400 Bad Request -> log du contenu pour diagnostic et continuer
            if status_code == 400 and resp is not None:
                try:
                    content = resp.json()
                except Exception:
                    content = resp.text
                logger.warning(
                    "predicted-lineup 400 pour event %s: %s",
                    event_id,
                    content,
                )
                return None
            # sinon ré-élever
            raise

    def get_team_players(self, team_id: int) -> List[Dict[str, Any]]:
        """
        Effectif d'une équipe, utile pour résoudre les ids joueurs quand
        les player-stats du match n'existent pas encore.
        """
        return self._get_paginated("players/", {"team": team_id})


# ----------------------------
# Service géographique 
# ----------------------------
@dataclass
class GeoService:
    """
    Service local pour:
    - géocoder un stade/ville/pays avec Nominatim (OpenStreetMap),
    - déduire le fuseau avec timezonefinder,
    - calculer l'offset UTC via zoneinfo à la date du match.
    """

    timeout: int = 20
    user_agent: str = "qsport-fatigue-metrics"

    def __post_init__(self) -> None:
        self.geolocator = Nominatim(user_agent=self.user_agent, timeout=self.timeout)
        self.tz_finder = TimezoneFinder()
        self._geocode_cache: Dict[str, Optional[Tuple[float, float]]] = {}
        self._tz_name_cache: Dict[str, Optional[str]] = {}

    def geocode(self, query: str) -> Optional[Tuple[float, float]]:
        """
        Convertit une requête texte en coordonnées lat/lng.
        """
        key = query.strip().lower()
        if key in self._geocode_cache:
            return self._geocode_cache[key]

        try:
            location = self.geolocator.geocode(query)
        except Exception:  # noqa: BLE001
            location = None

        if not location:
            self._geocode_cache[key] = None
            return None

        coords = (float(location.latitude), float(location.longitude))
        self._geocode_cache[key] = coords
        return coords

    def timezone_name(self, lat: float, lng: float) -> Optional[str]:
        """
        Renvoie le nom IANA du fuseau (ex: Europe/Paris) depuis lat/lng.
        """
        cache_key = f"{lat:.6f},{lng:.6f}"
        if cache_key in self._tz_name_cache:
            return self._tz_name_cache[cache_key]

        tz_name = self.tz_finder.timezone_at(lat=lat, lng=lng)
        self._tz_name_cache[cache_key] = tz_name
        return tz_name

    def timezone_offset_seconds(self, lat: float, lng: float, match_dt: datetime) -> Optional[int]:
        """
        Offset UTC (en secondes) d'un point géographique au moment du match.
        """
        tz_name = self.timezone_name(lat, lng)
        if not tz_name:
            return None

        try:
            tz = ZoneInfo(tz_name)
        except Exception:  # noqa: BLE001
            return None

        local_dt = match_dt.astimezone(tz)
        offset = local_dt.utcoffset()
        if offset is None:
            return None
        return int(offset.total_seconds())


# ----------------------------
# Fonctions métier (fatigue)
# ----------------------------
def _latest_previous_match_date(events: Iterable[Dict[str, Any]], before_dt: datetime) -> Optional[datetime]:
    """
    Renvoie la date du dernier match strictement antérieur à before_dt.
    """
    candidates: List[datetime] = []
    for event in events:
        event_dt = _to_datetime(event.get("event_date"))
        if event_dt and event_dt < before_dt:
            candidates.append(event_dt)
    return max(candidates) if candidates else None


def _extract_starters_by_side(event_detail: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extrait les titulaires home/away depuis le champ lineups.
    Supporte deux formats possibles renvoyés par BSD:
    - liste [{player_name, is_home, ...}],
    - dict {home: {players: [...]}, away: {players: [...]}}.
    """
    lineups = event_detail.get("lineups") or []

    if isinstance(lineups, list):
        home_names = [
            row.get("player_name", "")
            for row in lineups
            if isinstance(row, dict) and row.get("is_home") is True
        ]
        away_names = [
            row.get("player_name", "")
            for row in lineups
            if isinstance(row, dict) and row.get("is_home") is False
        ]
        return {"home": [n for n in home_names if n][:11], "away": [n for n in away_names if n][:11]}

    if isinstance(lineups, dict):
        home_players = ((lineups.get("home") or {}).get("players") or [])
        away_players = ((lineups.get("away") or {}).get("players") or [])
        home_names = [row.get("name", "") for row in home_players if isinstance(row, dict)]
        away_names = [row.get("name", "") for row in away_players if isinstance(row, dict)]
        return {"home": [n for n in home_names if n][:11], "away": [n for n in away_names if n][:11]}

    return {"home": [], "away": []}


def _extract_predicted_starters(predicted_lineup: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extrait les titulaires prédits home/away depuis /predicted-lineup/{event_id}/.
    """
    if not predicted_lineup:
        return {"home": [], "away": []}

    lineups = predicted_lineup.get("lineups") or {}
    home_starters = lineups.get("home", {}).get("starters", []) or []
    away_starters = lineups.get("away", {}).get("starters", []) or []

    home_names = [row.get("name", "") for row in home_starters][:11]
    away_names = [row.get("name", "") for row in away_starters][:11]
    return {"home": home_names, "away": away_names}


def _map_starter_names_to_player_ids(starter_names: List[str], event_player_stats: List[Dict[str, Any]]) -> List[int]:
    """
    Résout les ids joueurs à partir des player-stats du match.
    Fonctionne surtout pour les matchs déjà joués/en cours.
    """
    mapping: Dict[str, int] = {}
    for row in event_player_stats:
        player = row.get("player") or {}
        player_id = player.get("id")
        for key_name in (player.get("name", ""), player.get("short_name", "")):
            normalized = _norm_name(key_name)
            if normalized and player_id:
                mapping[normalized] = int(player_id)

    player_ids: List[int] = []
    for name in starter_names:
        player_id = mapping.get(_norm_name(name))
        if player_id:
            player_ids.append(player_id)
    return player_ids


def _best_match_player_id(starter_name: str, candidates: List[Dict[str, Any]]) -> Optional[int]:
    """
    Matching fuzzy: choisit l'id joueur le plus plausible dans un effectif.
    Heuristique: recouvrement de tokens + bonus sur le nom de famille.
    """
    starter_norm = _norm_name_loose(starter_name)
    if not starter_norm:
        return None

    starter_tokens = set(starter_norm.split())
    starter_last = starter_norm.split()[-1] if starter_norm.split() else ""

    best_id: Optional[int] = None
    best_score = -1
    for candidate in candidates:
        candidate_id = candidate.get("id")
        if not candidate_id:
            continue

        candidate_names = [candidate.get("name", ""), candidate.get("short_name", "")]
        cand_norms = [_norm_name_loose(x) for x in candidate_names if x]
        if not cand_norms:
            continue

        if starter_norm in cand_norms:
            return int(candidate_id)

        cand_token_sets = [set(norm.split()) for norm in cand_norms if norm]
        cand_last_names = [norm.split()[-1] for norm in cand_norms if norm and norm.split()]

        last_name_bonus = 2 if starter_last and starter_last in cand_last_names else 0
        overlap = max((len(starter_tokens & token_set) for token_set in cand_token_sets), default=0)
        score = overlap + last_name_bonus

        if score > best_score:
            best_score = score
            best_id = int(candidate_id)

    return best_id if best_score >= 2 else None


def _map_starter_names_to_team_player_ids(starter_names: List[str], team_players: List[Dict[str, Any]]) -> List[int]:
    """
    Fallback pour les matchs à venir:
    - on mappe les noms des titulaires sur l'effectif de l'équipe.
    """
    if not starter_names or not team_players:
        return []

    lookup: Dict[str, int] = {}
    for player in team_players:
        player_id = player.get("id")
        if not player_id:
            continue
        for key_name in (player.get("name", ""), player.get("short_name", "")):
            strict = _norm_name(key_name)
            loose = _norm_name_loose(key_name)
            if strict:
                lookup[strict] = int(player_id)
            if loose:
                lookup[loose] = int(player_id)

    ids: List[int] = []
    for starter in starter_names:
        strict = _norm_name(starter)
        loose = _norm_name_loose(starter)
        player_id = lookup.get(strict) or lookup.get(loose)
        if player_id is None:
            player_id = _best_match_player_id(starter, team_players)
        if player_id is not None and player_id not in ids:
            ids.append(player_id)

    return ids


def _minutes_in_window(history: Iterable[Dict[str, Any]], window_start: datetime, window_end: datetime) -> float:
    """
    Somme des minutes jouées dans [window_start, window_end[.
    """
    total = 0.0
    for row in history:
        event_info = row.get("event") or {}
        event_dt = _to_datetime(event_info.get("event_date"))
        if not event_dt:
            continue
        if window_start <= event_dt < window_end:
            total += float(row.get("minutes_played") or 0.0)
    return total


def _team_recent_load(
    client: BSDClient,
    starter_player_ids: List[int],
    match_datetime: datetime,
    days: int = 21,
    tz: str = "UTC",
) -> Optional[float]:
    """
    Charge moyenne des titulaires:
    moyenne des minutes sur les X derniers jours (par défaut 21).
    """
    if not starter_player_ids:
        return None

    window_start = match_datetime - timedelta(days=days)
    totals = []
    for player_id in starter_player_ids:
        history = client.get_player_history(player_id, tz=tz)
        totals.append(_minutes_in_window(history, window_start=window_start, window_end=match_datetime))

    if not totals:
        return None
    return float(sum(totals) / len(totals))


def _build_venue_query(venue: Dict[str, Any]) -> Optional[str]:
    """
    Construit une requête géocodable depuis un objet venue BSD.
    """
    if not venue:
        return None
    parts = [
        (venue.get("name") or "").strip(),
        (venue.get("city") or "").strip(),
        (venue.get("country") or "").strip(),
    ]
    parts = [p for p in parts if p]
    if not parts:
        return None
    return ", ".join(parts)


def _build_venue_query_candidates(venue: Dict[str, Any]) -> List[str]:
    """
    Construit une liste de requêtes décroissantes pour augmenter le taux de géocodage:
    1) nom du stade + ville + pays
    2) ville + pays
    3) ville
    4) pays
    """
    if not venue:
        return []

    name = (venue.get("name") or "").strip()
    city = (venue.get("city") or "").strip()
    country = (venue.get("country") or "").strip()

    candidates: List[str] = []
    if name and city and country:
        candidates.append(f"{name}, {city}, {country}")
    if city and country:
        candidates.append(f"{city}, {country}")
    if city:
        candidates.append(city)
    if country:
        candidates.append(country)

    # Déduplication tout en gardant l'ordre.
    seen = set()
    ordered: List[str] = []
    for query in candidates:
        norm = query.lower().strip()
        if norm and norm not in seen:
            seen.add(norm)
            ordered.append(query)
    return ordered


def _geocode_with_fallback(geo_service: GeoService, venue: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Essaye plusieurs variantes de requêtes pour trouver des coordonnées.
    """
    for query in _build_venue_query_candidates(venue):
        coords = geo_service.geocode(query)
        if coords:
            return coords
    return None


def _compute_travel_and_timezone_metrics(
    geo_service: Optional[GeoService],
    away_home_venue: Dict[str, Any],
    match_venue: Dict[str, Any],
    match_dt: datetime,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calcule:
    - deplacement_km: distance entre le stade domicile de l'équipe away et le stade du match,
    - decalage_horaire_heures: différence absolue d'offset UTC entre ces deux points au moment du match.

    Si le service géographique n'est pas disponible, retourne (None, None).
    """
    if geo_service is None:
        return None, None

    if not away_home_venue or not match_venue:
        return None, None

    away_coords = _geocode_with_fallback(geo_service, away_home_venue)
    match_coords = _geocode_with_fallback(geo_service, match_venue)
    if not away_coords or not match_coords:
        return None, None

    distance_km = _haversine_km(away_coords, match_coords)

    away_offset_sec = geo_service.timezone_offset_seconds(away_coords[0], away_coords[1], match_dt)
    match_offset_sec = geo_service.timezone_offset_seconds(match_coords[0], match_coords[1], match_dt)

    if away_offset_sec is None or match_offset_sec is None:
        return float(distance_km), None

    tz_diff_hours = abs(match_offset_sec - away_offset_sec) / 3600.0
    return float(distance_km), float(tz_diff_hours)


def compute_fatigue_for_match(
    client: BSDClient,
    event_id: int,
    tz: str = "UTC",
    geo_service: Optional[GeoService] = None,
) -> Dict[str, Any]:
    """
    Calcule toutes les métriques pour un match:
    - repos (home/away + diff),
    - charge des titulaires sur 21 jours,
    - déplacement + décalage horaire (si service géographique disponible).
    """
    event = client.get_event_detail(event_id=event_id, full=True, tz=tz)

    event_dt = _to_datetime(event.get("event_date"))
    if not event_dt:
        raise ValueError(f"No valid event_date for event_id={event_id}")

    home_obj = event.get("home_team_obj") or {}
    away_obj = event.get("away_team_obj") or {}
    home_team_id = home_obj.get("id")
    away_team_id = away_obj.get("id")
    if not home_team_id or not away_team_id:
        raise ValueError(f"Missing home/away team ids for event_id={event_id}")

    # 1) Repos: dernier match joué avant le match courant.
    home_events = client.get_team_events_until(team_id=int(home_team_id), target_date=event_dt, tz=tz)
    away_events = client.get_team_events_until(team_id=int(away_team_id), target_date=event_dt, tz=tz)

    home_last_match = _latest_previous_match_date(home_events, before_dt=event_dt)
    away_last_match = _latest_previous_match_date(away_events, before_dt=event_dt)

    home_rest_days = (event_dt - home_last_match).days if home_last_match else None
    away_rest_days = (event_dt - away_last_match).days if away_last_match else None
    rest_days_diff_a_minus_b = (
        float(home_rest_days - away_rest_days)
        if home_rest_days is not None and away_rest_days is not None
        else None
    )

    # 2) T titulaires: lineups officielles sinon lineups prédites.
    starters = _extract_starters_by_side(event)
    if not starters["home"] or not starters["away"]:
        predicted = client.get_predicted_lineup(event_id=event_id)
        predicted_starters = _extract_predicted_starters(predicted)
        starters = {
            "home": starters["home"] or predicted_starters["home"],
            "away": starters["away"] or predicted_starters["away"],
        }

    # 3) Résolution des ids joueurs:
    # - d'abord via player-stats du match,
    # - fallback via l'effectif de l'équipe (utile pour match futur).
    event_player_stats = client.get_player_stats_for_event(event_id=event_id, tz=tz)
    home_starters_ids = _map_starter_names_to_player_ids(starters["home"], event_player_stats)
    away_starters_ids = _map_starter_names_to_player_ids(starters["away"], event_player_stats)

    if len(home_starters_ids) < 11 and home_team_id:
        home_players = client.get_team_players(int(home_team_id))
        home_starters_ids = _map_starter_names_to_team_player_ids(starters["home"], home_players)
    if len(away_starters_ids) < 11 and away_team_id:
        away_players = client.get_team_players(int(away_team_id))
        away_starters_ids = _map_starter_names_to_team_player_ids(starters["away"], away_players)

    # 4) Charge 21 jours pour chaque équipe.
    home_load_21d = _team_recent_load(client, home_starters_ids, event_dt, days=21, tz=tz)
    away_load_21d = _team_recent_load(client, away_starters_ids, event_dt, days=21, tz=tz)
    load_diff_a_minus_b = (
        float(home_load_21d - away_load_21d)
        if home_load_21d is not None and away_load_21d is not None
        else None
    )

    # 5) Déplacement et décalage horaire:
    # - départ = stade domicile équipe away,
    # - arrivée = stade du match (ou stade home si absent).
    away_home_venue = (away_obj.get("venue") or {}) if isinstance(away_obj, dict) else {}
    match_venue = event.get("venue") or home_obj.get("venue") or {}
    deplacement_km, decalage_horaire_heures = _compute_travel_and_timezone_metrics(
        geo_service=geo_service,
        away_home_venue=away_home_venue,
        match_venue=match_venue,
        match_dt=event_dt,
    )

    return {
        "event_id": event.get("id"),
        "event_date": event.get("event_date"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "rest_days_diff_a_minus_b": rest_days_diff_a_minus_b,
        "home_starters_count": len(home_starters_ids),
        "away_starters_count": len(away_starters_ids),
        "home_load_minutes_21d": home_load_21d,
        "away_load_minutes_21d": away_load_21d,
        "load_minutes_diff_a_minus_b": load_diff_a_minus_b,
        "deplacement_km": deplacement_km,
        "decalage_horaire_heures": decalage_horaire_heures,
    }


def compute_fatigue_dataframe(
    api_key: str,
    matches_df: pd.DataFrame,
    event_id_column: str = "BSD_EVENT_ID",
    tz: str = "UTC",
    output_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calcule les métriques de fatigue pour chaque match du DataFrame.

    Retourne un DataFrame contenant uniquement les colonnes de fatigue,
    indexé par event_id. Peut être sauvegardé dans un CSV dédié.
    """
    if event_id_column not in matches_df.columns:
        raise ValueError(f"La colonne {event_id_column} est absente du DataFrame.")

    client = BSDClient(api_key=api_key)
    geo_service = GeoService()

    event_ids = matches_df[event_id_column].tolist()
    total = len(event_ids)
    results = []

    print(f"Calcul fatigue pour {total} matchs...")
    for i, event_id_value in enumerate(event_ids, start=1):
        if pd.isna(event_id_value):
            results.append({"event_id": None, "error": f"Missing {event_id_column}"})
        else:
            try:
                results.append(compute_fatigue_for_match(
                    client=client,
                    event_id=int(event_id_value),
                    tz=tz,
                    geo_service=geo_service,
                ))
            except requests.HTTPError as exc:
                results.append({"event_id": int(event_id_value), "error": f"HTTPError: {exc}"})
                print(f"  [{i}/{total}] ERREUR HTTP event {int(event_id_value)}: {exc}")
            except Exception as exc:  # pylint: disable=broad-except
                results.append({"event_id": int(event_id_value), "error": f"Error: {exc}"})
                print(f"  [{i}/{total}] ERREUR event {int(event_id_value)}: {exc}")

        if i % 50 == 0 or i == total:
            print(f"  {i}/{total} matchs traités")

    fatigue_df = pd.DataFrame(results)

    errors = fatigue_df['error'].notna().sum() if 'error' in fatigue_df.columns else 0
    print(f"Terminé — {total - errors}/{total} ok, {errors} erreurs")

    if output_csv_path:
        fatigue_df.to_csv(output_csv_path, index=False)
        print(f"Sauvegardé dans {output_csv_path}")

    return fatigue_df



def simple_compute_fatigue(
    api_key: str,
    matches_df: pd.DataFrame,
    output_csv_path: str = "data/fatigue.csv",
) -> pd.DataFrame:
    """
    Interface simplifiée pour calculer les métriques de fatigue.
    
    Utilise par défaut la colonne 'id' du DataFrame et timezone UTC.
    
    Args:
        api_key: Clé API BSD
        matches_df: DataFrame contenant au minimum une colonne 'id'
        output_csv_path: Chemin de sauvegarde du CSV (par défaut: data/fatigue.csv)
    
    Returns:
        DataFrame avec les métriques de fatigue
    
    Example:
        >>> fatigue = simple_compute_fatigue(TOKEN, pl_events)
    """
    return compute_fatigue_dataframe(
        api_key=api_key,
        matches_df=matches_df,
        event_id_column="id",
        tz="UTC",
        output_csv_path=output_csv_path,
    )


if __name__ == "__main__":

    # 1) Mets ta clé BSD ici.
    API_KEY = "c9d89a0da7aa0729ca0f7d96ec1940b59875375f"
    # 2) Crée un DataFrame d'entrée avec autant de lignes que nécessaire.
    input_matches_df = pd.DataFrame(
        {
            "MATCH_LABEL": [
                "OL vs Rennes",
                "Tondela vs FC alverca",
                "Osasuna vs Betis",
            ],
            "BSD_EVENT_ID": [2048, 580, 997],
        }
    )

    # 3) Calcule les métriques de fatigue et sauvegarde dans fatigue.csv.
    fatigue_df = compute_fatigue_dataframe(
        api_key=API_KEY,
        matches_df=input_matches_df,
        event_id_column="BSD_EVENT_ID",
        tz="UTC",
        output_csv_path="fatigue.csv",
    )
    print(fatigue_df)
