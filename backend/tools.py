from langchain.tools import tool
import sys, os
sys.path.insert(0, "/app")

try:
    from nba_prediction import NBAPredictor
    _nba_predictor = NBAPredictor()
except Exception as e:
    _nba_predictor = None
    print(f"[tools] NBA predictor non disponible: {e}")

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
        return f"{result['winner']} gagne avec {result['confidence']:.0%} de confiance."
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
