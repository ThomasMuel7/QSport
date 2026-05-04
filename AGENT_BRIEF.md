# Q-Sport — Brief Agent : Collecte & Pipeline de Données Contextuelles

> **Tu es un agent de développement Python.**  
> Tu pars de zéro. Tu ne connais rien du projet.  
> Lis ce brief en entier avant d'écrire la moindre ligne de code.

---

## 1. Contexte du projet

Q-Sport est un pipeline de machine learning hybride (classique + quantique) qui prédit l'issue de matchs de football. Ton rôle est de construire **le module de collecte et de préparation des données contextuelles** : météo, arbitre, coach, stade, et fuseaux horaires.

Ces données seront stockées dans une base **Supabase (PostgreSQL)** et consommées par le pipeline ML principal.

---

## 2. Ce que tu dois produire

Un module Python structuré dans `data_collection/` avec :

```
data_collection/
├── __init__.py
├── weather_collector.py      # Météo via open-meteo
├── timezone_collector.py     # Décalage horaire via Google Timezone API
├── stadium_collector.py      # Stade : surface + remplissage via BSD API
├── referee_collector.py      # Arbitre : sévérité + biais local via BSD API
├── coach_collector.py        # Coach : durée en poste via BSD API
├── pipeline.py               # Orchestrateur : appelle tout et écrit en BDD
└── models.py                 # Dataclasses Python pour chaque entité
```

---

## 3. Sources de données — APIs autorisées

### 3.1 API Football principale — BSD (Bzzoiro Sports Data)
- **URL base** : `https://sports.bzzoiro.com/api/`
- **Auth** : header `Authorization: Token YOUR_BSD_TOKEN`
- **Gratuit, sans rate limit, sans carte bancaire**
- **Documentation** : `https://sports.bzzoiro.com/docs/`

Endpoints utiles :
| Endpoint | Ce qu'il retourne |
|---|---|
| `GET /api/events/{id}/` | Détail d'un match : arbitre, stade, équipes, lineups, unavailable_players |
| `GET /api/events/?league=&season=` | Liste des matchs |
| `GET /api/referees/{id}/` | Profil d'un arbitre |
| `GET /api/managers/{id}/` | Profil d'un coach (date de prise en poste = `date_appointed`) |
| `GET /api/venues/{id}/` | Données du stade : capacité, surface, latitude, longitude |

Champs importants dans `/api/events/{id}/` :
```json
{
  "id": 580,
  "home_team": "Liverpool",
  "away_team": "Arsenal",
  "event_date": "2026-02-10T17:30:00+0400",
  "venue": { "id": 12, "name": "Anfield", "capacity": 53394, "surface": "grass", "latitude": 53.4308, "longitude": -2.9608 },
  "referee": { "id": 7, "name": "Michael Oliver" },
  "home_manager": { "id": 42, "name": "Arne Slot", "date_appointed": "2024-06-01" },
  "away_manager": { "id": 55, "name": "Mikel Arteta", "date_appointed": "2019-12-20" },
  "attendance": 52500,
  "unavailable_players": { "home": [...], "away": [...] }
}
```

### 3.2 Météo historique — Open-Meteo
- **URL** : `https://api.open-meteo.com/v1/forecast` (futur) ou `https://archive-api.open-meteo.com/v1/archive` (historique)
- **Gratuit, aucune clé requise**
- **Docs** : `https://open-meteo.com/en/docs`

Paramètres à utiliser :
```
latitude=53.4308
longitude=-2.9608
start_date=2026-02-10
end_date=2026-02-10
hourly=temperature_2m,precipitation,windspeed_10m
timezone=Europe/London
```

Extraire la valeur à l'heure du match (ex: 17h00 locale).

Champs à récupérer :
- `temperature_2m` → température en °C à l'heure du match
- `precipitation` → précipitations en mm/h
- `windspeed_10m` → vitesse du vent en km/h

### 3.3 Fuseaux horaires — Google Timezone API
- **URL** : `https://maps.googleapis.com/maps/api/timezone/json`
- **Auth** : paramètre `key=YOUR_GOOGLE_API_KEY`
- **Docs** : `https://developers.google.com/maps/documentation/timezone`

Paramètres :
```
location=53.4308,-2.9608
timestamp=1739203800   (Unix timestamp du match)
key=YOUR_GOOGLE_KEY
```

Réponse utile :
```json
{
  "rawOffset": 0,
  "dstOffset": 3600,
  "timeZoneId": "Europe/London"
}
```

Utiliser `timeZoneId` pour convertir la date du match en heure locale (utile pour Open-Meteo).

---

## 4. Features à calculer — définitions exactes

### 4.1 Météo (`WeatherData`)
| Feature | Type | Calcul |
|---|---|---|
| `temperature_c` | float | Valeur `temperature_2m` à l'heure du match |
| `precipitation_mm` | float | Valeur `precipitation` à l'heure du match |
| `wind_speed_kmh` | float | Valeur `windspeed_10m` à l'heure du match |

### 4.2 Coach (`CoachData`)
| Feature | Type | Calcul |
|---|---|---|
| `home_coach_days_in_post` | int | `match_date - home_manager.date_appointed` en jours |
| `away_coach_days_in_post` | int | `match_date - away_manager.date_appointed` en jours |

**Règle métier** : Si `date_appointed` est null → mettre -1 (valeur sentinelle, le modèle ML ignore ces cas).

### 4.3 Arbitre (`RefereeData`)
Ces stats doivent être calculées sur l'**historique des matchs passés** de l'arbitre (pas le match courant).

| Feature | Type | Calcul |
|---|---|---|
| `referee_avg_yellow_cards` | float | Moyenne cartons jaunes distribués par match sur les 20 derniers matchs arbitrés |
| `referee_avg_red_cards` | float | Idem pour cartons rouges |
| `referee_home_win_pct` | float | % victoires domicile parmi les 20 derniers matchs arbitrés (entre 0.0 et 1.0) |

**Règle métier** : Si moins de 5 matchs historiques disponibles → mettre `None` / `NULL` pour toutes les features arbitre.

### 4.4 Stade (`StadiumData`)
| Feature | Type | Calcul |
|---|---|---|
| `stadium_fill_rate` | float | `attendance / venue.capacity` (entre 0.0 et 1.0). Si `attendance` null → `None` |
| `surface_type` | str | Enum : `"natural"`, `"hybrid"`, `"artificial"`. Mapper depuis le champ `surface` de l'API |

Mapping surface :
```python
SURFACE_MAP = {
    "grass": "natural",
    "natural grass": "natural",
    "hybrid": "hybrid",
    "artificial": "artificial",
    "astroturf": "artificial",
    "synthetic": "artificial",
}
```

### 4.5 Décalage horaire (`TimezoneData`)
| Feature | Type | Calcul |
|---|---|---|
| `home_timezone_offset_h` | float | Offset en heures du stade domicile par rapport à UTC |
| `away_travel_timezone_delta` | float | Différence d'offset entre le timezone du stade et le timezone du pays de l'équipe visiteuse (approximé par son stade habituel) |

---

## 5. Dataclasses attendues (`models.py`)

```python
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

@dataclass
class RefereeData:
    match_id: int
    referee_id: Optional[int]
    referee_avg_yellow_cards: Optional[float]
    referee_avg_red_cards: Optional[float]
    referee_home_win_pct: Optional[float]
    collected_at: datetime

@dataclass
class StadiumData:
    match_id: int
    venue_id: int
    stadium_fill_rate: Optional[float]
    surface_type: str                  # "natural" | "hybrid" | "artificial"
    collected_at: datetime

@dataclass
class TimezoneData:
    match_id: int
    home_timezone_id: str             # ex: "Europe/London"
    home_timezone_offset_h: float
    away_travel_timezone_delta: float
    collected_at: datetime
```

---

## 6. Structure du pipeline (`pipeline.py`)

```python
def collect_contextual_data(match_id: int) -> dict:
    """
    Point d'entrée principal.
    1. Récupère le détail du match depuis BSD API
    2. Lance les collecteurs météo, timezone, arbitre, coach, stade
    3. Retourne un dict contenant toutes les dataclasses
    4. Persiste en base via upsert Supabase
    """
```

Le pipeline doit :
- Fonctionner en mode **batch** (liste de match_ids) et en mode **single** (un match_id)
- Logger chaque étape avec `logging` (niveau INFO par défaut)
- Gérer les erreurs par collecteur indépendamment : si la météo échoue, les autres continuent
- Écrire les résultats dans Supabase via `supabase-py`

---

## 7. Gestion des clés API (`config.py`)

Toutes les clés doivent être chargées depuis les variables d'environnement, jamais hardcodées.

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

BSD_API_TOKEN = os.getenv("BSD_API_TOKEN")           # Token BSD bzzoiro
GOOGLE_TIMEZONE_KEY = os.getenv("GOOGLE_TIMEZONE_KEY") # Clé Google Maps
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
```

Fichier `.env.example` à créer :
```
BSD_API_TOKEN=your_token_here
GOOGLE_TIMEZONE_KEY=your_google_key_here
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

---

## 8. Dépendances Python requises

```txt
# requirements.txt
requests==2.31.0
python-dotenv==1.0.0
supabase==2.3.0
```

---

## 9. Tables Supabase à créer

Exécuter ces migrations SQL dans Supabase avant de lancer le pipeline.

```sql
-- Météo
CREATE TABLE IF NOT EXISTS match_weather (
    match_id        INTEGER PRIMARY KEY,
    temperature_c   FLOAT,
    precipitation_mm FLOAT,
    wind_speed_kmh  FLOAT,
    collected_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Coach
CREATE TABLE IF NOT EXISTS match_coach (
    match_id                    INTEGER PRIMARY KEY,
    home_coach_days_in_post     INTEGER,
    away_coach_days_in_post     INTEGER,
    collected_at                TIMESTAMPTZ DEFAULT NOW()
);

-- Arbitre
CREATE TABLE IF NOT EXISTS match_referee (
    match_id                    INTEGER PRIMARY KEY,
    referee_id                  INTEGER,
    referee_avg_yellow_cards    FLOAT,
    referee_avg_red_cards       FLOAT,
    referee_home_win_pct        FLOAT,
    collected_at                TIMESTAMPTZ DEFAULT NOW()
);

-- Stade
CREATE TABLE IF NOT EXISTS match_stadium (
    match_id            INTEGER PRIMARY KEY,
    venue_id            INTEGER,
    stadium_fill_rate   FLOAT,
    surface_type        TEXT CHECK (surface_type IN ('natural','hybrid','artificial')),
    collected_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Timezone
CREATE TABLE IF NOT EXISTS match_timezone (
    match_id                    INTEGER PRIMARY KEY,
    home_timezone_id            TEXT,
    home_timezone_offset_h      FLOAT,
    away_travel_timezone_delta  FLOAT,
    collected_at                TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 10. Tests à écrire

Créer `tests/test_collectors.py` avec au minimum :

```python
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
```

Lancer avec : `pytest tests/ -v`

---

## 11. Checklist de livraison

Avant de dire que c'est terminé, vérifie :

- [ ] `python -c "from data_collection.pipeline import collect_contextual_data"` ne lève aucune erreur
- [ ] `.env.example` présent à la racine
- [ ] `requirements.txt` à jour
- [ ] Tous les tests passent (`pytest tests/ -v`)
- [ ] Aucun token ou clé API hardcodé dans le code
- [ ] Les `None` sont bien gérés pour les features arbitre (< 5 matchs historiques)
- [ ] Le pipeline ne crashe pas si un collecteur échoue (try/except par collecteur)
- [ ] Les migrations SQL sont dans `db/migrations/001_contextual_data.sql`

---

## 12. Ce que tu NE dois PAS faire

- Ne pas créer de base SQLite locale — tout va dans Supabase
- Ne pas utiliser API-Football (api-sports.io) — BSD est suffisant et gratuit
- Ne pas scraper de sites web — uniquement des APIs documentées
- Ne pas calculer l'ELO, le xG, l'ACWR, la composition probable — ce sont d'autres modules
- Ne pas créer d'interface utilisateur

---

*Brief rédigé par l'orchestrateur du projet Q-Sport — toute question doit remonter au chef de projet (Thomas Muel).*
