# ============================================================
# ÉTAPE 5 : PRÉDICTION DE MATCHS ATP — VERSION FINALE
# ============================================================
#
# Deux modes :
#   1. predict_match()   — prédiction manuelle (2 noms + date + contexte)
#   2. predict_ongoing() — prédiction sur tous les matchs ongoing
#
# Nouveautés vs version précédente :
#   ✅ match_date obligatoire → fatigue calculée à la bonne date
#   ✅ Nouvelles features (rank_ratio, momentum, serve_dom...)
#   ✅ get_player_stats() avec date pour éviter le leakage
#   ✅ FEATURE_COLS aligné avec le modèle entraîné
# ============================================================

from datetime import date

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from paths import data_path, model_path

# ─────────────────────────────────────────────────────────────
# A. CHARGEMENT
# ─────────────────────────────────────────────────────────────

# Redéfinition de PipelineWrapper (nécessaire si le modèle est un RF wrappé)
from sklearn.pipeline import Pipeline

class PipelineWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    def predict(self, X):
        return self.pipeline.predict(X)
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

# ── Chargement du meilleur modèle ──
model_data = joblib.load(model_path("tennis", "best_model.pkl"))
model      = model_data["model"]
model_name = model_data["model_name"]

# ── Chargement des données ──
feat_df  = pd.read_csv(data_path("tennis", "atp_features_clean_final.csv"))
df_orig  = pd.read_csv(data_path("tennis", "atp_clean.csv"), parse_dates=["tourney_date"])
df_orig  = df_orig.sort_values("tourney_date").reset_index(drop=True)
ongoing  = pd.read_csv(data_path("tennis", "ongoing_tourneys.csv"))

# ── Préparation de feat_clean ──
feat_clean = feat_df.copy()
feat_clean = feat_clean.sort_values("year").reset_index(drop=True)

# Ajout de la colonne p1_name pour la recherche par nom
# (doit correspondre à la façon dont elle a été créée dans le feature engineering)
if "p1_name" not in feat_clean.columns:
    # Fallback si p1_name absent du CSV
    player_names_feat = []
    for i, row_orig in df_orig.iterrows():
        player_names_feat.append(row_orig["winner_name"])
        player_names_feat.append(row_orig["loser_name"])
    feat_clean["player_name"] = player_names_feat
    USE_PLAYER_NAME = False
else:
    USE_PLAYER_NAME = True
    print("   → Utilisation de p1_name (colonne existante)")


# ── FEATURE_COLS : liste des features attendues par le modèle ──
# Exclure les colonnes non-features
FEATURE_COLS = [c for c in feat_clean.columns
                if c not in ["label", "year", "diff_last20_won",
                             "p1_name", "p2_name", "player_name"]]

# Ajouter les nouvelles features dérivées créées lors de l'entraînement optimisé
# (elles ne sont pas dans feat_clean mais dans le modèle entraîné)
NEW_FEATURES = [
    "rank_ratio",
    "p1_momentum", "p2_momentum", "diff_momentum",
    "p1_serve_dom", "p2_serve_dom", "diff_serve_dom",
    "diff_bp_pressure",
]
# On les ajoute seulement si le modèle les connaît
if hasattr(model_data, "get") and "feature_names" in model_data:
    model_features = model_data["feature_names"]
    FEATURE_COLS = model_features  # utiliser exactement les features du modèle
else:
    for f in NEW_FEATURES:
        if f not in FEATURE_COLS:
            FEATURE_COLS.append(f)

N_MATCHES = len(df_orig)


# ─────────────────────────────────────────────────────────────
# B. ENCODAGES CONTEXTUELS
# ─────────────────────────────────────────────────────────────

ROUND_ORDER = {"R128":1,"R64":2,"R32":3,"R16":4,"QF":5,"SF":6,"F":7,"RR":3,"BR":6}
LEVEL_ORDER = {"G":5,"M":4,"F":4,"A":3,"D":2,"C":1}
HAND_MAP    = {"R":1,"L":-1,"U":0}
INDOOR_MAP  = {"Y":1,"N":0,"Unknown":0}

# ─────────────────────────────────────────────────────────────
# C. FONCTION : RÉCUPÉRER LES STATS D'UN JOUEUR
# ─────────────────────────────────────────────────────────────

def get_player_stats(player_name: str, match_date: pd.Timestamp) -> dict:
    """
    Récupère les dernières stats rolling d'un joueur AVANT une date donnée.

    Paramètres :
        player_name : nom exact du joueur
        match_date  : date du match à prédire (pd.Timestamp)
                      → seuls les matchs AVANT cette date sont utilisés

    Retourne :
        dict avec toutes les stats p1_* du joueur + infos de debug
    """
    mask    = (df_orig["winner_name"] == player_name) | \
              (df_orig["loser_name"]  == player_name)

    # Uniquement les matchs AVANT la date du match à prédire (no leakage)
    matches = df_orig[mask & (df_orig["tourney_date"] < match_date)] \
                .sort_values("tourney_date")

    if len(matches) == 0:
        return None

    last          = matches.iloc[-1]
    is_winner     = last["winner_name"] == player_name
    label_val     = 1 if is_winner else 0
    expected_rank = float(last["winner_rank"] if is_winner else last["loser_rank"])

    # ── Recherche de la ligne dans feat_clean ──
    if USE_PLAYER_NAME:
        candidates = feat_clean[feat_clean["p1_name"] == player_name]
    else:
        candidates = feat_clean[feat_clean["player_name"] == player_name]

    candidates = candidates[candidates["label"] == label_val]
    exact      = candidates[candidates["p1_rank"] == expected_rank]
    feat_row   = exact.iloc[-1] if len(exact) > 0 else candidates.iloc[-1]

    # Extraire toutes les stats p1_*
    stats = {col.replace("p1_", ""): feat_row[col]
             for col in feat_clean.columns if col.startswith("p1_")
             and col not in ["p1_name"]}

    # ── Fatigue RÉELLE calculée par rapport à la date du match ──
    # Nombre de matchs dans les 14 jours avant la date du match
    fatigue_window = matches[
        matches["tourney_date"] >= match_date - pd.Timedelta(days=14)
    ]
    stats["fatigue_14d"]     = len(fatigue_window)
    stats["days_since_last"] = (match_date - last["tourney_date"]).days

    # Infos de debug
    stats["_last_match_date"] = last["tourney_date"].date()
    stats["_last_opponent"]   = last["loser_name"] if is_winner else last["winner_name"]
    stats["_last_result"]     = "Victoire" if is_winner else "Défaite"

    return stats


# ─────────────────────────────────────────────────────────────
# D. FONCTION : CONSTRUIRE LES FEATURES D'UN MATCH
# ─────────────────────────────────────────────────────────────

def build_match_features(p1_stats: dict, p2_stats: dict,
                          surface: str, tourney_level: str,
                          round_str: str, best_of: int,
                          indoor: str, month: int,
                          h2h_n: int = 0,
                          h2h_win_rate_p1: float = np.nan,
                          h2h_win_rate_surf: float = np.nan) -> pd.DataFrame:
    """
    Construit la ligne de features complète pour un match donné.
    Inclut les nouvelles features dérivées de l'entraînement optimisé.
    """
    row = {}

    # ── Features statiques ──
    row["p1_rank"] = p1_stats.get("rank", 999)
    row["p1_age"]  = p1_stats.get("age",  np.nan)
    row["p1_ht"]   = p1_stats.get("ht",   np.nan)
    row["p2_rank"] = p2_stats.get("rank", 999)
    row["p2_age"]  = p2_stats.get("age",  np.nan)
    row["p2_ht"]   = p2_stats.get("ht",   np.nan)

    # ── Différentiels statiques ──
    row["diff_rank"]     = row["p1_rank"]    - row["p2_rank"]
    row["diff_rank_pts"] = p1_stats.get("rank_pts", 0) - p2_stats.get("rank_pts", 0)
    row["diff_seed"]     = p1_stats.get("seed", 0)     - p2_stats.get("seed", 0)
    row["diff_age"]      = row["p1_age"]     - row["p2_age"]
    row["diff_ht"]       = row["p1_ht"]      - row["p2_ht"]
    row["same_hand"]     = int(p1_stats.get("hand", 0) == p2_stats.get("hand", 0))

    # ── Rolling features P1 et P2 ──
    rolling_stats = [
        "last5_won", "last10_won", "last20_won",
        "last10_ace_rate", "last10_df_rate", "last10_fs_in_pct",
        "last10_fs_won_pct", "last10_ss_won_pct", "last10_bp_saved_pct",
        "last10_n_matches",
        "last5_win_rate_Hard", "last5_win_rate_Clay",
        "last10_win_rate_Hard", "last10_win_rate_Clay",
        "last20_win_rate_Hard", "last20_win_rate_Clay",
        "fatigue_14d", "days_since_last",
    ]
    for stat in rolling_stats:
        row[f"p1_{stat}"] = p1_stats.get(stat, np.nan)
        row[f"p2_{stat}"] = p2_stats.get(stat, np.nan)

    # ── Différentiels rolling ──
    diff_stats = [
        "last5_won", "last5_ace_rate", "last5_df_rate", "last5_fs_in_pct",
        "last5_fs_won_pct", "last5_ss_won_pct", "last5_bp_saved_pct",
        "last10_won", "last10_ace_rate", "last10_df_rate", "last10_fs_in_pct",
        "last10_fs_won_pct", "last10_ss_won_pct", "last10_bp_saved_pct",
        "last10_n_matches",
        "last20_ace_rate", "last20_df_rate", "last20_fs_in_pct",
        "last20_fs_won_pct", "last20_ss_won_pct", "last20_bp_saved_pct",
        "last5_win_rate_Hard", "last5_win_rate_Clay",
        "last10_win_rate_Hard", "last10_win_rate_Clay",
        "last20_win_rate_Hard", "last20_win_rate_Clay",
        "fatigue_14d", "days_since_last",
    ]
    for stat in diff_stats:
        v1 = p1_stats.get(stat, np.nan)
        v2 = p2_stats.get(stat, np.nan)
        try:
            row[f"diff_{stat}"] = v1 - v2 if (not np.isnan(float(v1)) and
                                               not np.isnan(float(v2))) else np.nan
        except (TypeError, ValueError):
            row[f"diff_{stat}"] = np.nan

    # ── H2H ──
    row["h2h_n"]             = h2h_n
    row["h2h_win_rate_p1"]   = h2h_win_rate_p1
    row["h2h_win_rate_surf"] = h2h_win_rate_surf

    # ── Contexte match ──
    row["round_num"]         = ROUND_ORDER.get(round_str, 3)
    row["tourney_level_num"] = LEVEL_ORDER.get(tourney_level, 3)
    row["indoor_enc"]        = INDOOR_MAP.get(str(indoor), 0)
    row["month"]             = month
    row["best_of"]           = best_of
    row["surf_Clay"]         = int(surface == "Clay")
    row["surf_Grass"]        = int(surface == "Grass")
    row["surf_Hard"]         = int(surface == "Hard")

    # ── Nouvelles features dérivées (entraînement optimisé) ──
    # Ces features ont été ajoutées lors de l'étape d'optimisation
    # et doivent être recalculées ici pour correspondre au modèle

    # 1. Ratio de classement
    r1 = row["p1_rank"]
    r2 = row["p2_rank"]
    row["rank_ratio"] = float(np.clip(r1 / r2, 0.01, 100)) if r2 > 0 else np.nan

    # 2. Momentum (tendance récente vs long terme)
    p1_l5  = row.get("p1_last5_won",  np.nan)
    p1_l20 = row.get("p1_last20_won", np.nan)
    p2_l5  = row.get("p2_last5_won",  np.nan)
    p2_l20 = row.get("p2_last20_won", np.nan)

    try:
        row["p1_momentum"]   = float(p1_l5) - float(p1_l20)
    except (TypeError, ValueError):
        row["p1_momentum"]   = np.nan
    try:
        row["p2_momentum"]   = float(p2_l5) - float(p2_l20)
    except (TypeError, ValueError):
        row["p2_momentum"]   = np.nan
    try:
        row["diff_momentum"] = row["p1_momentum"] - row["p2_momentum"]
    except (TypeError, ValueError):
        row["diff_momentum"] = np.nan

    # 3. Service dominance (ace_rate + fs_won_pct - df_rate)
    def serve_dom(stats, prefix):
        a = stats.get(f"{prefix}last10_ace_rate",    np.nan)
        f = stats.get(f"{prefix}last10_fs_won_pct",  np.nan)
        d = stats.get(f"{prefix}last10_df_rate",     np.nan)
        try:
            return float(a) + float(f) - float(d)
        except (TypeError, ValueError):
            return np.nan

    row["p1_serve_dom"]   = serve_dom(p1_stats, "")
    row["p2_serve_dom"]   = serve_dom(p2_stats, "")
    try:
        row["diff_serve_dom"] = row["p1_serve_dom"] - row["p2_serve_dom"]
    except (TypeError, ValueError):
        row["diff_serve_dom"] = np.nan

    # 4. Pression sous break
    bp  = row.get("diff_last10_bp_saved_pct", np.nan)
    fsw = row.get("diff_last10_fs_won_pct",   np.nan)
    try:
        row["diff_bp_pressure"] = float(bp) - float(fsw)
    except (TypeError, ValueError):
        row["diff_bp_pressure"] = np.nan

    # ── Alignement strict sur les features du modèle ──
    X = pd.DataFrame([row])
    # Ajouter les colonnes manquantes avec NaN
    for col in FEATURE_COLS:
        if col not in X.columns:
            X[col] = np.nan
    # Sélectionner uniquement les features dans le bon ordre
    X = X[FEATURE_COLS]
    return X


# ─────────────────────────────────────────────────────────────
# E. FONCTION : H2H HISTORIQUE
# ─────────────────────────────────────────────────────────────

def get_h2h(p1_name: str, p2_name: str, surface: str = None) -> tuple:
    """
    Calcule le H2H historique entre deux joueurs.
    Retourne (n_matchs, win_rate_p1_global, win_rate_p1_surface).
    """
    mask = (
        ((df_orig["winner_name"] == p1_name) & (df_orig["loser_name"] == p2_name)) |
        ((df_orig["winner_name"] == p2_name) & (df_orig["loser_name"] == p1_name))
    )
    h2h = df_orig[mask]
    n   = len(h2h)

    if n == 0:
        return 0, np.nan, np.nan

    p1_wins  = (h2h["winner_name"] == p1_name).sum()
    win_rate = p1_wins / n

    if surface:
        h2h_s     = h2h[h2h["surface"] == surface]
        n_s       = len(h2h_s)
        p1_wins_s = (h2h_s["winner_name"] == p1_name).sum()
        win_rate_s = p1_wins_s / n_s if n_s > 0 else np.nan
    else:
        win_rate_s = np.nan

    return n, win_rate, win_rate_s


# ─────────────────────────────────────────────────────────────
# F. PRÉDICTION SIMPLE — deux joueurs + date + contexte
# ─────────────────────────────────────────────────────────────

def predict_match(player1: str, player2: str,
                  match_date: str,
                  surface: str       = "Hard",
                  tourney_level: str = "M",
                  round_str: str     = "R32",
                  best_of: int       = 3,
                  indoor: str        = "N"):
    """
    Prédit le vainqueur d'un match entre deux joueurs.

    Paramètres :
        player1, player2  : noms exacts des joueurs (casse importante) (Prénom Nom)
        match_date        : date du match "YYYY-MM-DD" (obligatoire)
        surface           : "Hard", "Clay", "Grass"
        tourney_level     : "G" (GC), "M" (Masters), "A" (500/250),
                            "D" (Challenger), "C" (ITF)
        round_str         : "R128","R64","R32","R16","QF","SF","F"
        best_of           : 3 ou 5 (5 uniquement en Grand Chelem)
        indoor            : "Y" ou "N"

    Exemple :
        predict_match("Jannik Sinner", "Carlos Alcaraz",
                      match_date="2026-05-05",
                      surface="Clay", tourney_level="M",
                      round_str="SF", best_of=3)
    """
    date  = pd.Timestamp(match_date)
    month = date.month

    # ── Récupération des stats ──
    p1_stats = get_player_stats(player1, date)
    p2_stats = get_player_stats(player2, date)

    if p1_stats is None:
        return None
    if p2_stats is None:
        return None

    # ── H2H ──
    h2h_n, h2h_wr, h2h_wr_s = get_h2h(player1, player2, surface)

    # ── Construction des features ──
    X = build_match_features(
        p1_stats, p2_stats,
        surface=surface, tourney_level=tourney_level,
        round_str=round_str, best_of=best_of,
        indoor=indoor, month=month,
        h2h_n=h2h_n, h2h_win_rate_p1=h2h_wr,
        h2h_win_rate_surf=h2h_wr_s
    )

    # ── Prédiction ──
    proba       = model.predict_proba(X)[0]
    p1_win_prob = proba[1]
    p2_win_prob = proba[0]
    winner      = player1 if p1_win_prob > 0.5 else player2
    confidence  = max(p1_win_prob, p2_win_prob)

    # ── Affichage ──
    bar_len = 40
    p1_bar  = int(p1_win_prob * bar_len)
    p2_bar  = bar_len - p1_bar


    return {
        "player1"          : player1,
        "player2"          : player2,
        "p1_prob"          : p1_win_prob,
        "p2_prob"          : p2_win_prob,
        "predicted_winner" : winner,
        "confidence"       : confidence,
    }

if __name__ == "__main__":
    # Exemple de test
    result = predict_match(
        player1="Carlos Alcaraz",
        player2="Jannik Sinner",
        match_date=date.today().strftime("%Y-%m-%d"),
    )
    print(result)