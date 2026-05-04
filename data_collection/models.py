from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class WeatherData:
    match_id: int
    temperature_c: float
    precipitation_mm: float
    wind_speed_kmh: float
    collected_at: datetime

@dataclass
class CoachData:
    match_id: int
    home_coach_days_in_post: int       # -1 si inconnu
    away_coach_days_in_post: int       # -1 si inconnu
    collected_at: datetime
    home_nationality: Optional[str] = None
    away_nationality: Optional[str] = None
    home_total_matches_managed: Optional[int] = None
    away_total_matches_managed: Optional[int] = None
    home_win_rate_overall: Optional[float] = None
    away_win_rate_overall: Optional[float] = None

@dataclass
class RefereeData:
    match_id: int
    referee_id: Optional[int]
    collected_at: datetime
    referee_avg_yellow_cards: Optional[float] = None
    referee_avg_red_cards: Optional[float] = None
    referee_home_win_pct: Optional[float] = None
    referee_avg_penalties_per_match: Optional[float] = None
    referee_avg_fouls_per_match: Optional[float] = None
    referee_draw_pct: Optional[float] = None

@dataclass
class StadiumData:
    match_id: int
    venue_id: int
    collected_at: datetime
    stadium_fill_rate: Optional[float] = None
    surface_type: str = "natural"                  # "natural" | "hybrid" | "artificial"
    altitude: Optional[float] = None
    city: Optional[str] = None
    country: Optional[str] = None

@dataclass
class TimezoneData:
    match_id: int
    home_timezone_id: str             # ex: "Europe/London"
    home_timezone_offset_h: float
    away_travel_timezone_delta: float
    collected_at: datetime
