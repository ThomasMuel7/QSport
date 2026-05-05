"""
nba_predictor.py — Q-Sport NBA Prediction Engine
=================================================
Fichier centralisé pour la prédiction de matchs NBA.
Utilisable directement par un agent LLM ou en ligne de commande.

Les blessés sont récupérés AUTOMATIQUEMENT depuis ESPN —
aucune information manuelle n'est nécessaire.

Usage agent :
    from nba_predictor import NBAPredictor
    predictor = NBAPredictor()
    predictor.predict_game_auto("LAL", "GSW")
    predictor.predict_game_auto("OKC", "LAL", is_playoffs=1)
    predictor.compare_models("BOS", "MIA")

Usage ligne de commande :
    python nba_predictor.py --home LAL --away GSW
    python nba_predictor.py --home OKC --away LAL --playoffs --model all
    python nba_predictor.py --home LAL --away GSW --refresh
"""

import time
import pickle
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Chemins ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

# ── Mapping trigrammes → noms ───────────────────────────────────────────────
TEAM_MAP = {
    "LAL": "Lakers",      "GSW": "Warriors",     "BOS": "Celtics",
    "MIA": "Heat",        "PHX": "Suns",         "DEN": "Nuggets",
    "MIL": "Bucks",       "PHI": "76ers",        "NYK": "Knicks",
    "DAL": "Mavericks",   "MEM": "Grizzlies",    "NOP": "Pelicans",
    "SAC": "Kings",       "MIN": "Timberwolves", "OKC": "Thunder",
    "LAC": "Clippers",    "POR": "Blazers",      "UTA": "Jazz",
    "CHI": "Bulls",       "CLE": "Cavaliers",    "DET": "Pistons",
    "IND": "Pacers",      "ATL": "Hawks",        "CHA": "Hornets",
    "ORL": "Magic",       "WAS": "Wizards",      "BKN": "Nets",
    "TOR": "Raptors",     "HOU": "Rockets",      "SAS": "Spurs",
}
NAME_TO_TRI = {v.lower(): k for k, v in TEAM_MAP.items()}


# ── Architecture QNN ────────────────────────────────────────────────────────
N_QUBITS = 8
N_LAYERS = 3

_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_dev, interface="torch")
def _quantum_circuit(features, weights):
    qml.AngleEmbedding(features, wires=range(N_QUBITS), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


class HybridQNNv2(nn.Module):
    """Modèle hybride classique-quantique : Dense(Tanh) → QNN → Sigmoid."""

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.classical = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Tanh(),
        )
        self.weights = nn.Parameter(
            torch.tensor(
                np.random.uniform(0, np.pi, (n_layers, n_qubits)),
                dtype=torch.float32,
            )
        )
        self.scaling = nn.Parameter(torch.ones(1))
        self.bias    = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_prep = self.classical(x) * np.pi
        out = torch.stack([
            _quantum_circuit(x_prep[i], self.weights)
            for i in range(x.shape[0])
        ]).float()
        return torch.sigmoid(self.scaling * out + self.bias)


# ── Moteur de prédiction ────────────────────────────────────────────────────
class NBAPredictor:
    """Moteur de prédiction NBA — charge tous les modèles une seule fois."""

    def __init__(self, data_dir: str | Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self._load_artifacts()
        self._load_team_stats()

    # ── Chargement ──────────────────────────────────────────────────────────

    def _load_artifacts(self) -> None:
        """Charge les modèles, scalers et listes de features."""
        log.info("Chargement des modèles...")

        with open(self.data_dir / "nba_xgb_clf.pkl", "rb") as f:
            self.xgb_clf = pickle.load(f)
        with open(self.data_dir / "nba_xgb_reg.pkl", "rb") as f:
            self.xgb_reg = pickle.load(f)
        with open(self.data_dir / "nba_features_xgb.pkl", "rb") as f:
            self.features_xgb = pickle.load(f)
        with open(self.data_dir / "nba_features_qnn.pkl", "rb") as f:
            self.features_qnn = pickle.load(f)
        with open(self.data_dir / "nba_scaler_xgb.pkl", "rb") as f:
            self.scaler_xgb = pickle.load(f)
        with open(self.data_dir / "nba_scaler_qnn.pkl", "rb") as f:
            self.scaler_qnn = pickle.load(f)

        n = len(self.features_qnn)

        class _ClassicNN(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n, 32), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(16, 1), nn.Sigmoid(),
                )
            def forward(self, x):
                return self.net(x).squeeze()

        self.nn_model = _ClassicNN(n)
        self.nn_model.load_state_dict(
            torch.load(self.data_dir / "nba_nn_classic_v2.pt", weights_only=True)
        )
        self.nn_model.eval()

        self.qnn_model = HybridQNNv2()
        self.qnn_model.load_state_dict(
            torch.load(self.data_dir / "nba_qnn_v2_weights.pt", weights_only=True)
        )
        self.qnn_model.eval()

        log.info("Modèles chargés ✓  (XGBoost · NN · QNN)")

    def _load_team_stats(self) -> None:
        """Charge les stats actuelles des équipes."""
        self.team_stats   = pd.read_csv(self.data_dir / "nba_team_stats_current.csv")
        self.team_recent  = pd.read_csv(self.data_dir / "nba_team_stats_recent.csv")
        self.player_stats = pd.read_csv(self.data_dir / "nba_player_stats_current.csv")
        log.info(f"Stats équipes chargées ✓  ({len(self.team_stats)} équipes)")

    # ── Résolution des noms ──────────────────────────────────────────────────

    @staticmethod
    def resolve_team(name: str) -> str:
        """Convertit un trigramme ou nom partiel en nom complet."""
        name = name.strip()
        if name.upper() in TEAM_MAP:
            return TEAM_MAP[name.upper()]
        for val in TEAM_MAP.values():
            if name.lower() in val.lower():
                return val
        raise ValueError(
            f"Équipe '{name}' non reconnue. "
            f"Utilisez un trigramme (ex: LAL) ou un nom partiel (ex: Lakers)."
        )

    def _get_stats(self, team_name: str) -> pd.Series:
        """Récupère les stats d'une équipe par nom."""
        row = self.team_stats[
            self.team_stats["TEAM_NAME"].str.contains(team_name, case=False)
        ]
        if row.empty:
            raise ValueError(f"Statistiques introuvables pour '{team_name}'.")
        return row.iloc[0]

    # ── Construction du vecteur de features ─────────────────────────────────

    def _build_feature_vector(
        self,
        home_name: str,
        away_name: str,
        home_rest_days: int = 2,
        away_rest_days: int = 2,
        home_is_b2b: int = 0,
        away_is_b2b: int = 0,
        is_playoffs: int = 0,
        absent_home: list[str] | None = None,
        absent_away: list[str] | None = None,
    ) -> dict:
        """Construit le vecteur de features à partir des stats équipes."""
        home = self._get_stats(home_name)
        away = self._get_stats(away_name)

        def absence_penalty(absent_players: list[str] | None) -> float:
            if not absent_players:
                return 0.0
            penalty = 0.0
            for player_name in absent_players:
                match = self.player_stats[
                    self.player_stats["PLAYER_NAME"].str.contains(
                        player_name, case=False, na=False
                    )
                ]
                if not match.empty:
                    min_pg = match.iloc[0].get("MIN", 0)
                    penalty += min_pg * 0.15
            return penalty

        home_net = home["NET_RATING"] - absence_penalty(absent_home)
        away_net = away["NET_RATING"] - absence_penalty(absent_away)

        return {
            "is_playoffs":                               is_playoffs,
            "diff_roll_estimatedNetRating_10":           home_net - away_net,
            "diff_roll_estimatedOffensiveRating_10":     home["OFF_RATING"] - away["OFF_RATING"],
            "diff_roll_estimatedDefensiveRating_10":     home["DEF_RATING"] - away["DEF_RATING"],
            "diff_roll_effectiveFieldGoalPercentage_10": home["EFG_PCT"]    - away["EFG_PCT"],
            "diff_roll_defensiveReboundPercentage_10":   home["DREB_PCT"]   - away["DREB_PCT"],
            "home_roll_estimatedNetRating_10":           home_net,
            "away_roll_estimatedNetRating_10":           away_net,
            "home_roll_pts_for_10":                      home["OFF_RATING"],
            "away_roll_pts_for_10":                      away["OFF_RATING"],
            "home_roll_pts_against_10":                  home["DEF_RATING"],
            "away_roll_pts_against_10":                  away["DEF_RATING"],
            "scoring_momentum":                          home["OFF_RATING"] - away["OFF_RATING"],
            "home_rest_days":                            home_rest_days,
            "away_rest_days":                            away_rest_days,
            "home_is_b2b":                               home_is_b2b,
            "away_is_b2b":                               away_is_b2b,
            "rest_diff":                                 home_rest_days - away_rest_days,
            "b2b_advantage":                             away_is_b2b - home_is_b2b,
        }

    # ── Prédiction d'un match ────────────────────────────────────────────────

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        home_rest_days: int = 2,
        away_rest_days: int = 2,
        home_is_b2b: int = 0,
        away_is_b2b: int = 0,
        is_playoffs: int = 0,
        absent_home: list[str] | None = None,
        absent_away: list[str] | None = None,
        model: str = "qnn",
    ) -> dict:
        """
        Prédit le résultat d'un match NBA.
        Paramètres : home_team/away_team (trigramme ou nom partiel), model (xgboost|nn|qnn).
        """
        home_name = self.resolve_team(home_team)
        away_name = self.resolve_team(away_team)

        stat_map = self._build_feature_vector(
            home_name, away_name,
            home_rest_days, away_rest_days,
            home_is_b2b, away_is_b2b,
            is_playoffs, absent_home, absent_away,
        )

        if model == "xgboost":
            x = np.array([stat_map.get(f, 0) for f in self.features_xgb]).reshape(1, -1)
            p_home     = float(self.xgb_clf.predict_proba(self.scaler_xgb.transform(x))[0, 1])
            point_diff = float(self.xgb_reg.predict(self.scaler_xgb.transform(x))[0])

        elif model == "nn":
            x   = np.array([stat_map.get(f, 0) for f in self.features_qnn]).reshape(1, -1)
            x_t = torch.tensor(
                np.clip(self.scaler_qnn.transform(x), -np.pi, np.pi),
                dtype=torch.float32,
            )
            with torch.no_grad():
                p_home = float(self.nn_model(x_t).item())
            point_diff = (p_home - 0.5) * 20

        elif model == "qnn":
            x        = np.array([stat_map.get(f, 0) for f in self.features_qnn]).reshape(1, -1)
            x_scaled = np.clip(self.scaler_qnn.transform(x), -np.pi, np.pi)[:, :N_QUBITS]
            x_t      = torch.tensor(x_scaled, dtype=torch.float32)
            with torch.no_grad():
                p_home = float(self.qnn_model(x_t).item())
            point_diff = (p_home - 0.5) * 20

        else:
            raise ValueError(f"Modèle '{model}' inconnu. Choisir parmi : xgboost, nn, qnn")

        p_away = 1.0 - p_home
        winner = home_name if p_home >= 0.5 else away_name
        conf   = max(p_home, p_away)

        result = {
            "home_team":       home_name,
            "away_team":       away_name,
            "p_home_win":      round(p_home, 3),
            "p_away_win":      round(p_away, 3),
            "winner":          winner,
            "confidence":      round(conf, 3),
            "point_diff_pred": round(point_diff, 1),
            "model":           model,
            "context": {
                "home_is_b2b": bool(home_is_b2b),
                "away_is_b2b": bool(away_is_b2b),
                "is_playoffs": bool(is_playoffs),
                "absent_home": absent_home or [],
                "absent_away": absent_away or [],
            },
            "odds": self.get_odds(home_team, away_team),
        }

        print(self.format_result_for_llm(result))
        return result

    # ── Cotes bookmakers ─────────────────────────────────────────────────────

    def get_odds(self, home_team: str, away_team: str) -> dict:
        """
        Récupère les cotes bookmakers via ESPN scoreboard.
        Retourne les cotes décimales (format européen).
        Ex: 2.25 signifie que 1€ misé rapporte 2.25€.
        Fonctionne depuis Docker (pas de blocage contrairement à NBA.com).
        """
        import requests

        url  = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        odds = {"home_decimal": None, "away_decimal": None, "source": None}

        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data      = resp.json()
            home_name = self.resolve_team(home_team)
            away_name = self.resolve_team(away_team)

            for event in data.get("events", []):
                competitors = event.get("competitions", [{}])[0].get("competitors", [])
                teams = [c["team"]["displayName"] for c in competitors]
                home_match = any(home_name.lower() in t.lower() for t in teams)
                away_match = any(away_name.lower() in t.lower() for t in teams)

                if home_match and away_match:
                    comp = event["competitions"][0]
                    for odds_data in comp.get("odds", []):
                        home_ml = odds_data.get("homeTeamOdds", {}).get("moneyLine")
                        away_ml = odds_data.get("awayTeamOdds", {}).get("moneyLine")
                        if home_ml and away_ml:
                            def ml_to_decimal(ml: float) -> float:
                                """Convertit moneyline américain en cote décimale européenne."""
                                if ml < 0:
                                    return round(1 + 100 / abs(ml), 2)
                                else:
                                    return round(1 + ml / 100, 2)
                            odds["home_decimal"] = ml_to_decimal(home_ml)
                            odds["away_decimal"] = ml_to_decimal(away_ml)
                            odds["source"]       = odds_data.get(
                                "provider", {}
                            ).get("name", "Bookmaker")
                    break

        except Exception as e:
            log.warning(f"Cotes ESPN indisponibles : {e}")

        return odds

    # ── Formatage pour le LLM ────────────────────────────────────────────────

    @staticmethod
    def format_result_for_llm(r: dict) -> str:
        """
        Formate le résultat de manière concise pour le LLM.

        Exemple de sortie :
            OKC vs Boston Celtics :
            Thunder gagne avec 68% de confiance, écart prédit : +7 pts
            Cotes ESPN : Thunder @ 1.56 / Celtics @ 2.45
        """
        winner = r["winner"]
        loser  = r["away_team"] if r["winner"] == r["home_team"] else r["home_team"]
        diff   = r["point_diff_pred"]

        lines = [
            f"{r['home_team']} vs {r['away_team']} :",
            f"{winner} gagne avec {r['confidence']:.0%} de confiance, "
            f"écart prédit : {diff:+.0f} pts",
        ]

        odds = r.get("odds", {})
        if odds.get("home_decimal") is not None:
            if r["winner"] == r["home_team"]:
                cote_win  = odds["home_decimal"]
                cote_lose = odds["away_decimal"]
            else:
                cote_win  = odds["away_decimal"]
                cote_lose = odds["home_decimal"]
            src = odds.get("source", "Bookmakers")
            lines.append(
                f"Cotes {src} : {winner} @ {cote_win} / {loser} @ {cote_lose}"
            )
        else:
            lines.append("Cotes bookmakers : non disponibles pour ce match")

        return "\n".join(lines)

    # ── Comparaison des 3 modèles ────────────────────────────────────────────

    def compare_models(self, home_team: str, away_team: str, **kwargs) -> dict:
        """Prédit avec les 3 modèles et retourne un consensus."""
        results = {}
        for m in ["xgboost", "nn", "qnn"]:
            results[m] = self.predict_game(home_team, away_team, model=m, **kwargs)

        p_consensus = float(np.mean([results[m]["p_home_win"] for m in results]))
        home_name   = results["xgboost"]["home_team"]
        away_name   = results["xgboost"]["away_team"]
        winner      = home_name if p_consensus >= 0.5 else away_name

        print(f"\n📊 CONSENSUS ({home_name} vs {away_name})")
        print(f"   XGBoost : {results['xgboost']['p_home_win']:.1%}")
        print(f"   NN      : {results['nn']['p_home_win']:.1%}")
        print(f"   QNN     : {results['qnn']['p_home_win']:.1%}")
        print(f"   Moyenne : {p_consensus:.1%}  → {winner}")

        return {
            "models": results,
            "consensus": {
                "p_home": round(p_consensus, 3),
                "p_away": round(1 - p_consensus, 3),
                "winner": winner,
            },
        }

    # ── Rapport de blessures ESPN ────────────────────────────────────────────

    def get_injury_report(self) -> pd.DataFrame:
        """
        Récupère le rapport de blessures NBA via ESPN.
        Fonctionne depuis Docker (pas de blocage 403).
        """
        import requests

        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()

            rows = []
            for team_data in data.get("injuries", []):
                team_name = team_data.get("team", {}).get("displayName", "")
                for injury in team_data.get("injuries", []):
                    athlete = injury.get("athlete", {})
                    rows.append({
                        "player_name": athlete.get("displayName", ""),
                        "team":        team_name,
                        "status":      injury.get("status", ""),
                        "reason":      injury.get("details", {}).get("type", ""),
                    })

            df = pd.DataFrame(rows)
            log.info(f"Injury report ESPN chargé ✓  ({len(df)} joueurs concernés)")
            return df

        except Exception as e:
            log.warning(f"ESPN injury report indisponible : {e}")
            path = self.data_dir / "nba_injury_report.csv"
            if path.exists():
                return pd.read_csv(path)
            return pd.DataFrame(columns=["player_name", "team", "status", "reason"])

    def get_absent_players(
        self,
        team_name: str,
        injury_df: pd.DataFrame | None = None,
    ) -> list[str]:
        """Retourne les joueurs OUT ou DOUBTFUL d'une équipe."""
        if injury_df is None:
            injury_df = self.get_injury_report()
        if injury_df.empty:
            return []
        team_injuries = injury_df[
            injury_df["team"].str.contains(team_name, case=False, na=False)
        ]
        return team_injuries[
            team_injuries["status"].str.upper().isin(["OUT", "DOUBTFUL"])
        ]["player_name"].tolist()

    # ── Méthode principale pour l'agent ─────────────────────────────────────

    def predict_game_auto(
        self,
        home_team: str,
        away_team: str,
        home_rest_days: int = 2,
        away_rest_days: int = 2,
        home_is_b2b: int = 0,
        away_is_b2b: int = 0,
        is_playoffs: int = 0,
        model: str = "qnn",
    ) -> dict:
        """
        Méthode principale pour l'agent LLM.
        Prédit un match en récupérant AUTOMATIQUEMENT les blessés depuis ESPN.

        Usage :
            predictor.predict_game_auto("LAL", "GSW")
            predictor.predict_game_auto("OKC", "LAL", is_playoffs=1)
        """
        home_name = self.resolve_team(home_team)
        away_name = self.resolve_team(away_team)

        injury_df   = self.get_injury_report()
        absent_home = self.get_absent_players(home_name, injury_df)
        absent_away = self.get_absent_players(away_name, injury_df)

        if absent_home:
            log.info(f"Absents {home_name} : {', '.join(absent_home)}")
        if absent_away:
            log.info(f"Absents {away_name} : {', '.join(absent_away)}")

        return self.predict_game(
            home_team, away_team,
            home_rest_days=home_rest_days,
            away_rest_days=away_rest_days,
            home_is_b2b=home_is_b2b,
            away_is_b2b=away_is_b2b,
            is_playoffs=is_playoffs,
            absent_home=absent_home or None,
            absent_away=absent_away or None,
            model=model,
        )

    # ── Rafraîchissement des stats ───────────────────────────────────────────

    def refresh_team_stats(self, season: str = "2025-26") -> None:
        """Rafraîchit les stats des équipes depuis l'API NBA."""
        try:
            from nba_api.stats.endpoints import leaguedashteamstats
            log.info("Rafraîchissement des stats depuis l'API NBA...")
            dash = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
            )
            time.sleep(0.6)
            self.team_stats = dash.get_data_frames()[0]
            self.team_stats.to_csv(
                self.data_dir / "nba_team_stats_current.csv", index=False
            )
            log.info(f"Stats rafraîchies ✓  ({len(self.team_stats)} équipes)")
        except Exception as e:
            log.warning(f"Impossible de rafraîchir les stats : {e}")


# ── Interface ligne de commande ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q-Sport NBA Predictor")
    parser.add_argument("--home",     required=True, help="Équipe domicile (ex: LAL)")
    parser.add_argument("--away",     required=True, help="Équipe extérieure (ex: GSW)")
    parser.add_argument("--model",    default="qnn", choices=["xgboost", "nn", "qnn", "all"])
    parser.add_argument("--home-b2b", action="store_true", help="Domicile en back-to-back")
    parser.add_argument("--away-b2b", action="store_true", help="Extérieur en back-to-back")
    parser.add_argument("--playoffs", action="store_true", help="Match de playoffs")
    parser.add_argument("--refresh",  action="store_true", help="Rafraîchir les stats NBA")
    args = parser.parse_args()

    predictor = NBAPredictor()

    if args.refresh:
        predictor.refresh_team_stats()

    kwargs = dict(
        home_is_b2b=int(args.home_b2b),
        away_is_b2b=int(args.away_b2b),
        is_playoffs=int(args.playoffs),
    )

    if args.model == "all":
        injury_df   = predictor.get_injury_report()
        home_name   = predictor.resolve_team(args.home)
        away_name   = predictor.resolve_team(args.away)
        absent_home = predictor.get_absent_players(home_name, injury_df)
        absent_away = predictor.get_absent_players(away_name, injury_df)
        predictor.compare_models(
            args.home, args.away,
            absent_home=absent_home or None,
            absent_away=absent_away or None,
            **kwargs,
        )
    else:
        predictor.predict_game_auto(args.home, args.away, model=args.model, **kwargs)


if __name__ == "__main__":
    main()