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
    match_id                        INTEGER PRIMARY KEY,
    home_coach_days_in_post         INTEGER,
    away_coach_days_in_post         INTEGER,
    home_nationality                TEXT,
    away_nationality                TEXT,
    home_total_matches_managed      INTEGER,
    away_total_matches_managed      INTEGER,
    home_win_rate_overall           FLOAT,
    away_win_rate_overall           FLOAT,
    collected_at                    TIMESTAMPTZ DEFAULT NOW()
);

-- Arbitre
CREATE TABLE IF NOT EXISTS match_referee (
    match_id                        INTEGER PRIMARY KEY,
    referee_id                      INTEGER,
    referee_avg_yellow_cards        FLOAT,
    referee_avg_red_cards           FLOAT,
    referee_home_win_pct            FLOAT,
    referee_avg_penalties_per_match FLOAT,
    referee_avg_fouls_per_match     FLOAT,
    referee_draw_pct                FLOAT,
    collected_at                    TIMESTAMPTZ DEFAULT NOW()
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
