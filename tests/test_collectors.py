import pytest
from datetime import datetime
from data_collection.models import WeatherData, CoachData, RefereeData, StadiumData

def test_weather_collector_returns_valid_data():
    # Match Anfield (lat=53.4308, lng=-2.9608) le 2024-11-02 à 15h00
    # Vérifie que temperature_c est entre -20 et 50
    # Vérifie que precipitation_mm >= 0
    # Vérifie que wind_speed_kmh >= 0
    pass

def test_surface_mapping_covers_known_values():
    # Vérifie que "grass", "hybrid", "artificial" sont tous mappés
    pass

def test_coach_days_in_post_calculation():
    # Avec date_appointed="2024-01-01" et match_date="2024-06-01" → 152 jours
    pass

def test_referee_fallback_to_none_when_insufficient_history():
    # Si < 5 matchs historiques → toutes les features arbitre = None
    pass
