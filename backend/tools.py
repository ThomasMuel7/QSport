from langchain.tools import tool
import os

try:
    from nba_prediction import NBAPredictor
    _nba_predictor = NBAPredictor()
except Exception as e:
    _nba_predictor = None
    print(f"[tools] NBA predictor non disponible: {e}")

try:
    from odds import get_match_odds, SPORTS_MAPPING
except Exception as e:
    get_match_odds = None
    print(f"[tools] Odds non disponible: {e}")

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

            # Cotes NBA
        cotes_str = ""
        if get_match_odds:
            cotes = get_match_odds("basketball_nba", result['home_team'], result['away_team'])
            if cotes and "error" not in cotes[0] and "message" not in cotes[0]:
                c = cotes[0]
                cotes_str = (
                    f" Cotes : {c.get('Cote Home (1)', 'N/A')} (domicile) "
                    f"/ {c.get('Cote Away (2)', 'N/A')} (exterieur)."
                )

        return (
            f"Vainqueur predit : {result['winner']}. "
            f"{result['home_team']} : {result['p_home_win']:.0%} de chance de gagner. "
            f"{result['away_team']} : {result['p_away_win']:.0%} de chance de gagner. "
            f"Ecart predit : {result['point_diff_pred']:+.1f} points."
            f"{cotes_str}"
        )
    except Exception as e:
        return f"Prediction NBA indisponible: {e}"

@tool
def predict_foot(teams: str) -> str:
    """Predit le resultat d'un match de football. Format: 'EQUIPE1 vs EQUIPE2'."""
    return f"Prediction foot en cours d'integration pour {teams}."

@tool
def predict_tennis(players: str) -> str:
    """Predit le resultat d'un match de tennis. Format: 'JOUEUR1 vs JOUEUR2'."""
    return f"Prediction tennis en cours d'integration pour {players}."
