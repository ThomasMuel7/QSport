from langchain.tools import tool
from datetime import date
import csv
import os

_TEAM_MAPPING = {}
_mapping_path = os.path.join(os.path.dirname(__file__), 'teams_mapping.csv')
try:
    with open(_mapping_path) as f:
        for row in csv.DictReader(f):
            _TEAM_MAPPING[row['alias'].lower()] = row['official']
except Exception as e:
    print(f'[tools] Mapping equipes non charge: {e}')

def resolve_team_name(name: str) -> str:
    return _TEAM_MAPPING.get(name.strip().lower(), name.strip())

try:
    from nba_prediction import NBAPredictor
    _nba_predictor = NBAPredictor()
except Exception as e:
    _nba_predictor = None
    print(f"[tools] NBA predictor non disponible: {e}")

try:
    from foot_prediction import predire as predire_foot
except Exception as e:
    predire_foot = None
    print(f"[tools] Foot predictor non disponible: {e}")

try:
    from tennis_prediction import predict_match as predire_tennis
except Exception as e:
    predire_tennis = None
    print(f"[tools] Tennis predictor non disponible: {e}")

@tool
def predict_nba(teams: str) -> str:
    """Predit le resultat d'un match NBA. Format: 'TEAM1 vs TEAM2' avec trigrammes NBA (ex: LAL vs GSW)."""
    try:
        parts = [t.strip().strip("'\"") for t in teams.split("vs")]
        if len(parts) != 2:
            return "Format invalide. Utilise: LAL vs GSW"
        home, away = parts[0].strip(), parts[1].strip()
        if _nba_predictor is None:
            return "Modele NBA non disponible."
        result = _nba_predictor.predict_game(
            home, away,
            model="xgboost",
            absent_home=None,
            absent_away=None
        )
        return (
            f"Vainqueur predit : {result['winner']}. "
            f"{result['home_team']} : {result['p_home_win']:.0%} de chance de gagner. "
            f"{result['away_team']} : {result['p_away_win']:.0%} de chance de gagner. "
            f"Ecart predit : {result['point_diff_pred']:+.1f} points."
        )
    except Exception as e:
        return f"Prediction NBA indisponible: {e}"

@tool
def predict_foot(teams: str) -> str:
    """Predit le resultat d'un match de football (Ligue 1, Premier League, Bundesliga, Serie A, La Liga).
    Format: 'EQUIPE1 vs EQUIPE2' avec les noms complets (ex: 'Paris Saint-Germain vs Marseille')."""
    if predire_foot is None:
        return "Modele football non disponible."
    try:
        parts = [t.strip().strip("'\"") for t in teams.split("vs")]
        if len(parts) != 2:
            return "Format invalide. Utilise: 'Paris Saint-Germain vs Marseille'"
        home, away = resolve_team_name(parts[0].strip()), resolve_team_name(parts[1].strip())

        result = predire_foot(home, away)

        if result is None:
            return f"Prediction impossible pour {home} vs {away}."

        date_str = f" (match prevu le {result['date']})" if result.get("date") else ""
        prob_home = result.get(home, result.get(list(result.keys())[1], 0))
        prob_away = result.get(away, result.get(list(result.keys())[2], 0))
        prob_nul  = result.get("nul", 0)
        return (
            f"Prediction{date_str} : "
            f"{home} gagne : {prob_home:.1f}%, "
            f"Match nul : {prob_nul:.1f}%, "
            f"{away} gagne : {prob_away:.1f}%."
        )
    except Exception as e:
        return f"Prediction football indisponible: {e}"
    
@tool
def predict_tennis(players: str) -> str:
    """Predit le resultat d'un match de tennis ATP.
    Format: 'JOUEUR1 vs JOUEUR2' avec noms complets (ex: 'Jannik Sinner vs Carlos Alcaraz').
    Surface optionnelle a la fin: 'Jannik Sinner vs Carlos Alcaraz sur Clay'."""
    if predire_tennis is None:
        return "Modele tennis non disponible."
    try:
        surface = "Hard"
        input_str = players
        for surf_kw, surf_val in [("clay", "Clay"), ("terre", "Clay"), ("gazon", "Grass"),
                                   ("grass", "Grass"), ("hard", "Hard"), ("dur", "Hard")]:
            if f" sur {surf_kw}" in players.lower() or f" on {surf_kw}" in players.lower():
                surface = surf_val
                input_str = players.lower().replace(f" sur {surf_kw}", "").replace(f" on {surf_kw}", "")
                input_str = players[:len(input_str)]
                break

        parts = [t.strip().strip("'\"") for t in input_str.split("vs")]
        if len(parts) != 2:
            return "Format invalide. Utilise: 'Jannik Sinner vs Carlos Alcaraz'"
        player1, player2 = parts[0].strip(), parts[1].strip()

        today = date.today().strftime("%Y-%m-%d")
        result = predire_tennis(player1, player2, match_date=today, surface=surface)

        if result is None:
            return f"Donnees insuffisantes pour predire {player1} vs {player2}. Verifie les noms exacts des joueurs."

        winner     = result["predicted_winner"]
        p1_prob    = result["p1_prob"] * 100
        p2_prob    = result["p2_prob"] * 100
        confidence = result["confidence"] * 100

        return (
            f"Vainqueur predit : {winner} (confiance {confidence:.0f}%). "
            f"{player1} : {p1_prob:.0f}% de chance de gagner. "
            f"{player2} : {p2_prob:.0f}% de chance de gagner. "
            f"Surface : {surface}."
        )
    except Exception as e:
        return f"Prediction tennis indisponible: {e}"

if __name__ == "__main__":
    print(predict_nba("LAL vs GSW"))
    print(predict_foot("Paris Saint-Germain vs Marseille"))
    print(predict_tennis("Carlos Alcaraz vs Jannik Sinner sur Clay"))