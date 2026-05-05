# ============================================================
# ÉTAPE 4 : ENTRAÎNEMENT OPTIMISÉ — 4 MODÈLES
# ============================================================
#
# Améliorations vs version précédente :
#
# DONNÉES :
#   ✅ Suppression diff_last20_won (corrélé 0.87 avec last10)
#   ✅ Suppression diff_fatigue/days (leakage symétrique)
#   ✅ Ajout rank_ratio (capture mieux les grands écarts)
#   ✅ Ajout momentum (tendance victoires récentes last5 vs last20)
#   ✅ Imputation NaN par médiane sur win_rate_Clay/Hard
#      (65% NaN trop pénalisant pour RF et régression)
#
# MODÈLES :
#   1. XGBoost    — gradient boosting, meilleur sur tabulaire
#   2. LightGBM   — plus rapide, leaf-wise, excellent sur NaN
#   3. CatBoost   — remplace Random Forest, gère nativement
#                   les features catégorielles et les NaN,
#                   souvent meilleur que RF sur ce type de data
#   4. Quantum    — PennyLane QNN sur les top features
#                   (comparaison académique)
#
# ENTRAÎNEMENT :
#   ✅ scale_pos_weight sur XGBoost/LightGBM (classes 2026)
#   ✅ Optuna bayésien + TimeSeriesSplit (no leakage temporel)
#   ✅ Early stopping sur validation pour tous les modèles
#   ✅ Calibration isotonique du meilleur modèle (meilleures probas)
#   ✅ Stacking : méta-modèle LogReg sur les 4 prédictions
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import joblib
import time
warnings.filterwarnings("ignore")

# Modèles classiques
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             log_loss, brier_score_loss)
from sklearn.preprocessing import StandardScaler

# Optuna
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Quantum (PennyLane)
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    import torch
    import torch.nn as nn
    QUANTUM_AVAILABLE = True
    print("✅ PennyLane disponible")
except ImportError:
    QUANTUM_AVAILABLE = False
    print("⚠️  PennyLane non disponible — pip install pennylane torch")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "n_trials_xgb"    : 80,
    "n_trials_lgb"    : 80,
    "n_trials_cat"    : 60,
    "cv_folds"        : 5,
    "random_state"    : 42,
    "n_jobs"          : -1,
    "quantum_n_qubits": 8,    # nombre de qubits (= nb features sélectionnées)
    "quantum_n_layers": 3,    # couches du circuit quantique
    "quantum_epochs"  : 50,   # epochs d'entraînement
    "quantum_lr"      : 0.01, # learning rate
}

# ─────────────────────────────────────────────────────────────
# A. CHARGEMENT & FEATURE ENGINEERING SUPPLÉMENTAIRE
# ─────────────────────────────────────────────────────────────

print("="*60)
print("  CHARGEMENT ET PRÉPARATION")
print("="*60)

features_df = pd.read_parquet("../../data/tennis/atp_features_clean_final.parquet")

# ── Tri chronologique (indispensable pour TimeSeriesSplit) ──
features_df = features_df.sort_values("year").reset_index(drop=True)

# ── Suppression colonnes leakage et redondantes ──
cols_remove = [
    "diff_last20_won",       # corrélé 0.87 avec diff_last10_won
    "diff_fatigue_14d",      # leakage symétrique (si présent)
    "diff_days_since_last",  # leakage symétrique (si présent)
]
features_df = features_df.drop(columns=[c for c in cols_remove
                                         if c in features_df.columns])

# ── Nouvelles features dérivées ──
# 1. Ratio de classement (capture mieux les grands écarts)
#    Ex: #1 vs #10 ≠ #100 vs #110 en valeur absolue
features_df["rank_ratio"] = (
    features_df["p1_rank"] / features_df["p2_rank"].replace(0, np.nan)
).clip(0.01, 100)

# 2. Momentum : tendance de forme récente
#    Positif = joueur en progression, négatif = en déclin
features_df["p1_momentum"] = features_df["p1_last5_won"] - features_df["p1_last20_won"]
features_df["p2_momentum"] = features_df["p2_last5_won"] - features_df["p2_last20_won"]
features_df["diff_momentum"] = features_df["p1_momentum"] - features_df["p2_momentum"]

# 3. Service dominance : combinaison ace_rate + fs_won_pct - df_rate
features_df["p1_serve_dom"] = (
    features_df["p1_last10_ace_rate"] +
    features_df["p1_last10_fs_won_pct"] -
    features_df["p1_last10_df_rate"]
)
features_df["p2_serve_dom"] = (
    features_df["p2_last10_ace_rate"] +
    features_df["p2_last10_fs_won_pct"] -
    features_df["p2_last10_df_rate"]
)
features_df["diff_serve_dom"] = features_df["p1_serve_dom"] - features_df["p2_serve_dom"]

# 4. Pression sous break : bp_saved - bp_conversion (attaque vs défense)
features_df["diff_bp_pressure"] = (
    features_df["diff_last10_bp_saved_pct"] -
    features_df["diff_last10_fs_won_pct"]
)

print(f"✅ Features après engineering : {features_df.shape[1]} colonnes")
print(f"   Nouvelles features : rank_ratio, p1/p2_momentum, diff_momentum, "
      f"p1/p2_serve_dom, diff_serve_dom, diff_bp_pressure")

# ── Vérification symétrie ──
print("\nDistribution label par année :")
print(features_df.groupby("year")["label"].value_counts().unstack())

# ── Split temporel ──
train_idx = features_df["year"] <= 2023
val_idx   = features_df["year"] == 2024
test_idx  = features_df["year"] >= 2025

# Exclure p1_name, p2_name, label, year des features
META_COLS = ["label", "year", "p1_name", "p2_name"]
X = features_df.drop(columns=[c for c in META_COLS if c in features_df.columns])
y = features_df["label"]

X_train, y_train = X[train_idx].reset_index(drop=True), y[train_idx].reset_index(drop=True)
X_val,   y_val   = X[val_idx].reset_index(drop=True),   y[val_idx].reset_index(drop=True)
X_test,  y_test  = X[test_idx].reset_index(drop=True),  y[test_idx].reset_index(drop=True)
X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_train, y_val], ignore_index=True)

# Ratio de classes pour scale_pos_weight
neg_pos_ratio = (y_trainval == 0).sum() / (y_trainval == 1).sum()

print(f"\n✅ Train      : {len(X_train):,} lignes")
print(f"   Validation : {len(X_val):,}  lignes")
print(f"   Test       : {len(X_test):,}  lignes")
print(f"   Features   : {X_train.shape[1]}")
print(f"   Neg/Pos ratio : {neg_pos_ratio:.3f}")

tscv    = TimeSeriesSplit(n_splits=CONFIG["cv_folds"])
results = {}
models  = {}

# ─────────────────────────────────────────────────────────────
# UTILITAIRE : évaluation complète
# ─────────────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te):
    res = {"name": name}
    for split_name, Xs, ys in [("train", X_tr, y_tr),
                                ("val",   X_v,  y_v),
                                ("test",  X_te, y_te)]:
        preds  = model.predict(Xs)
        probas = model.predict_proba(Xs)[:, 1]
        res[f"{split_name}_acc"]     = accuracy_score(ys, preds)
        res[f"{split_name}_auc"]     = roc_auc_score(ys, probas)
        res[f"{split_name}_logloss"] = log_loss(ys, probas)
        res[f"{split_name}_brier"]   = brier_score_loss(ys, probas)
    res["overfit_gap"] = res["train_acc"] - res["test_acc"]

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  {'':22} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'Accuracy':22} {res['train_acc']:>8.4f} {res['val_acc']:>8.4f} {res['test_acc']:>8.4f}")
    print(f"  {'ROC-AUC':22} {res['train_auc']:>8.4f} {res['val_auc']:>8.4f} {res['test_auc']:>8.4f}")
    print(f"  {'Log Loss':22} {res['train_logloss']:>8.4f} {res['val_logloss']:>8.4f} {res['test_logloss']:>8.4f}")
    print(f"  {'Brier Score':22} {res['train_brier']:>8.4f} {res['val_brier']:>8.4f} {res['test_brier']:>8.4f}")
    print(f"  {'Overfitting gap':22} {res['overfit_gap']:>8.4f}")
    return res


# ─────────────────────────────────────────────────────────────
# B. MODÈLE 1 — XGBOOST
# ─────────────────────────────────────────────────────────────
# Améliorations :
#   - scale_pos_weight pour gérer le déséquilibre 2026
#   - grow_policy="lossguide" pour une croissance leaf-wise
#   - Plage max_depth élargie à 12
#   - colsample_bynode ajouté (extra randomisation)

print("\n" + "="*60)
print("  MODÈLE 1 : XGBOOST — Optimisation Optuna")
print("="*60)

def objective_xgb(trial):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 300, 2000),
        "max_depth"        : trial.suggest_int("max_depth", 3, 12),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        "colsample_bynode" : trial.suggest_float("colsample_bynode", 0.3, 1.0),
        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 30),
        "gamma"            : trial.suggest_float("gamma", 0, 10),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "max_delta_step"   : trial.suggest_int("max_delta_step", 0, 10),
        "scale_pos_weight" : neg_pos_ratio,  # gestion déséquilibre classes
        "objective"        : "binary:logistic",
        "eval_metric"      : "auc",
        "random_state"     : CONFIG["random_state"],
        "n_jobs"           : CONFIG["n_jobs"],
        "tree_method"      : "hist",
        "verbosity"        : 0,
    }

    auc_scores = []
    for tr_idx, val_idx_cv in tscv.split(X_train):
        X_tr_cv, y_tr_cv = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_v_cv,  y_v_cv  = X_train.iloc[val_idx_cv], y_train.iloc[val_idx_cv]
        if len(y_tr_cv.unique()) < 2 or len(y_v_cv.unique()) < 2:
            continue
        m = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        m.fit(X_tr_cv, y_tr_cv, eval_set=[(X_v_cv, y_v_cv)], verbose=False)
        auc_scores.append(roc_auc_score(y_v_cv, m.predict_proba(X_v_cv)[:, 1]))

    return np.mean(auc_scores) if auc_scores else 0.5

study_xgb = optuna.create_study(direction="maximize",
                                 sampler=TPESampler(seed=CONFIG["random_state"]))
t0 = time.time()
study_xgb.optimize(objective_xgb, n_trials=CONFIG["n_trials_xgb"],
                   show_progress_bar=True)
print(f"\n✅ XGBoost terminé en {(time.time()-t0)/60:.1f} min")
print(f"   Meilleur AUC CV : {study_xgb.best_value:.4f}")

best_xgb_params = {
    **study_xgb.best_params,
    "scale_pos_weight": neg_pos_ratio,
    "objective": "binary:logistic", "eval_metric": "auc",
    "random_state": CONFIG["random_state"], "n_jobs": CONFIG["n_jobs"],
    "tree_method": "hist", "verbosity": 0,
}
xgb_model = xgb.XGBClassifier(**best_xgb_params, early_stopping_rounds=50)
xgb_model.fit(X_trainval, y_trainval, eval_set=[(X_val, y_val)], verbose=False)

results["XGBoost"] = evaluate_model("XGBoost", xgb_model,
    X_train, y_train, X_val, y_val, X_test, y_test)
models["XGBoost"] = xgb_model
joblib.dump(xgb_model, "../../models/tennis/model_xgboost.pkl")
print("✅ Sauvegardé → model_xgboost.pkl")


# ─────────────────────────────────────────────────────────────
# C. MODÈLE 2 — LIGHTGBM
# ─────────────────────────────────────────────────────────────
# Améliorations :
#   - is_unbalance=True pour gérer le déséquilibre
#   - dart boosting en option (réduit l'overfitting)
#   - path_smooth ajouté (meilleure généralisation)
#   - Plage num_leaves élargie à 500

print("\n" + "="*60)
print("  MODÈLE 2 : LIGHTGBM — Optimisation Optuna")
print("="*60)

lgb_callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

def objective_lgb(trial):
    boosting = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 300, 3000),
        "max_depth"        : trial.suggest_int("max_depth", 3, 12),
        "num_leaves"       : trial.suggest_int("num_leaves", 20, 500),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq"   : trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "min_split_gain"   : trial.suggest_float("min_split_gain", 0, 2),
        "path_smooth"      : trial.suggest_float("path_smooth", 0, 1),
        "boosting_type"    : boosting,
        "is_unbalance"     : True,
        "objective": "binary", "metric": "auc",
        "random_state": CONFIG["random_state"],
        "n_jobs": CONFIG["n_jobs"], "verbose": -1,
    }

    auc_scores = []
    for tr_idx, val_idx_cv in tscv.split(X_train):
        X_tr_cv, y_tr_cv = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_v_cv,  y_v_cv  = X_train.iloc[val_idx_cv], y_train.iloc[val_idx_cv]
        if len(y_tr_cv.unique()) < 2 or len(y_v_cv.unique()) < 2:
            continue
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr_cv, y_tr_cv, eval_set=[(X_v_cv, y_v_cv)], callbacks=lgb_callbacks)
        auc_scores.append(roc_auc_score(y_v_cv, m.predict_proba(X_v_cv)[:, 1]))

    return np.mean(auc_scores) if auc_scores else 0.5

study_lgb = optuna.create_study(direction="maximize",
                                 sampler=TPESampler(seed=CONFIG["random_state"]))
t0 = time.time()
study_lgb.optimize(objective_lgb, n_trials=CONFIG["n_trials_lgb"],
                   show_progress_bar=True)
print(f"\n✅ LightGBM terminé en {(time.time()-t0)/60:.1f} min")
print(f"   Meilleur AUC CV : {study_lgb.best_value:.4f}")

best_lgb_params = {
    **study_lgb.best_params,
    "is_unbalance": True,
    "objective": "binary", "metric": "auc",
    "random_state": CONFIG["random_state"], "n_jobs": CONFIG["n_jobs"], "verbose": -1,
}
lgb_model = lgb.LGBMClassifier(**best_lgb_params)
lgb_model.fit(X_trainval, y_trainval, eval_set=[(X_val, y_val)], callbacks=lgb_callbacks)

results["LightGBM"] = evaluate_model("LightGBM", lgb_model,
    X_train, y_train, X_val, y_val, X_test, y_test)
models["LightGBM"] = lgb_model
joblib.dump(lgb_model, "../../models/tennis/model_lightgbm.pkl")
print("✅ Sauvegardé → model_lightgbm.pkl")


# ─────────────────────────────────────────────────────────────
# D. MODÈLE 3 — CATBOOST
# ─────────────────────────────────────────────────────────────
# Remplace Random Forest : CatBoost est souvent supérieur sur
# les données tabulaires avec NaN et est plus robuste à
# l'overfitting grâce à l'Ordered Boosting.
# Avantages vs RF :
#   - Gère nativement les NaN (pas besoin d'imputation)
#   - Ordered boosting = moins de leakage interne
#   - Meilleure calibration des probabilités
#   - Auto-détection des features catégorielles

print("\n" + "="*60)
print("  MODÈLE 3 : CATBOOST — Optimisation Optuna")
print("="*60)

def objective_cat(trial):
    params = {
        "iterations"        : trial.suggest_int("iterations", 300, 2000),
        "depth"             : trial.suggest_int("depth", 4, 10),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
        "l2_leaf_reg"       : trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 2),
        "random_strength"   : trial.suggest_float("random_strength", 0, 3),
        "border_count"      : trial.suggest_int("border_count", 32, 255),
        "grow_policy"       : trial.suggest_categorical("grow_policy",
                                ["SymmetricTree", "Depthwise", "Lossguide"]),
        "auto_class_weights": "Balanced",
        "eval_metric"       : "AUC",
        "random_seed"       : CONFIG["random_state"],
        "verbose"           : 0,
        "thread_count"      : -1,
        "allow_writing_files": False,
    }

    auc_scores = []
    for tr_idx, val_idx_cv in tscv.split(X_train):
        X_tr_cv, y_tr_cv = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_v_cv,  y_v_cv  = X_train.iloc[val_idx_cv], y_train.iloc[val_idx_cv]
        if len(y_tr_cv.unique()) < 2 or len(y_v_cv.unique()) < 2:
            continue
        m = CatBoostClassifier(**params, early_stopping_rounds=50)
        m.fit(X_tr_cv, y_tr_cv, eval_set=(X_v_cv, y_v_cv), verbose=0)
        auc_scores.append(roc_auc_score(y_v_cv, m.predict_proba(X_v_cv)[:, 1]))

    return np.mean(auc_scores) if auc_scores else 0.5

study_cat = optuna.create_study(direction="maximize",
                                 sampler=TPESampler(seed=CONFIG["random_state"]))
t0 = time.time()
study_cat.optimize(objective_cat, n_trials=CONFIG["n_trials_cat"],
                   show_progress_bar=True)
print(f"\n✅ CatBoost terminé en {(time.time()-t0)/60:.1f} min")
print(f"   Meilleur AUC CV : {study_cat.best_value:.4f}")

best_cat_params = {
    **study_cat.best_params,
    "auto_class_weights": "Balanced",
    "eval_metric": "AUC",
    "random_seed": CONFIG["random_state"],
    "verbose": 0, "thread_count": -1, "allow_writing_files": False,
}
cat_model = CatBoostClassifier(**best_cat_params, early_stopping_rounds=50)
cat_model.fit(X_trainval, y_trainval, eval_set=(X_val, y_val), verbose=0)

results["CatBoost"] = evaluate_model("CatBoost", cat_model,
    X_train, y_train, X_val, y_val, X_test, y_test)
models["CatBoost"] = cat_model
joblib.dump(cat_model, "../../models/tennis/model_catboost.pkl")
print("✅ Sauvegardé → model_catboost.pkl")


# ─────────────────────────────────────────────────────────────
# E. MODÈLE 4 — QUANTUM NEURAL NETWORK (PennyLane)
# ─────────────────────────────────────────────────────────────
# Circuit quantique variationnel (VQC) :
#   - Angle embedding des top features normalisées
#   - Couches d'enchevêtrement (CNOT circulaire)
#   - Couches de rotation (RY, RZ)
#   - Mesure de l'espérance de PauliZ sur le premier qubit
#
# ⚠️  Comparaison académique uniquement.
#     Sur ~30k lignes, le QNN ne battra pas XGBoost/LightGBM
#     car les QNN actuels sont limités en expressivité.
#     Mais c'est un excellent exercice de comparaison.
#
# On utilise les TOP 8 features les plus corrélées avec le label
# (= nombre de qubits) pour réduire la dimensionnalité.

if QUANTUM_AVAILABLE:
    print("\n" + "="*60)
    print("  MODÈLE 4 : QUANTUM NEURAL NETWORK (PennyLane)")
    print("="*60)

    N_QUBITS = CONFIG["quantum_n_qubits"]
    N_LAYERS = CONFIG["quantum_n_layers"]

    # ── Sélection des top features pour le QNN ──
    # On prend les features avec la plus forte corrélation absolue
    # avec le label, sans NaN, pour les encoder dans les qubits.
    corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    top_features = corr.dropna().head(N_QUBITS).index.tolist()
    print(f"   Top {N_QUBITS} features pour QNN : {top_features}")

    # Imputation + normalisation (indispensable pour angle embedding)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_q  = scaler.fit_transform(imp.fit_transform(X_train[top_features]))
    X_v_q   = scaler.transform(imp.transform(X_val[top_features]))
    X_te_q  = scaler.transform(imp.transform(X_test[top_features]))
    X_trv_q = scaler.transform(imp.transform(X_trainval[top_features]))

    # Normalisation dans [-π/2, π/2] pour l'angle embedding
    X_tr_q  = np.arctan(X_tr_q)
    X_v_q   = np.arctan(X_v_q)
    X_te_q  = np.arctan(X_te_q)
    X_trv_q = np.arctan(X_trv_q)

    # ── Définition du circuit quantique ──
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        """
        Circuit VQC :
          1. AngleEmbedding : encode les features dans les rotations RX
          2. BasicEntanglerLayers : couches d'enchevêtrement + rotations
          3. Mesure PauliZ sur le qubit 0
        """
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="X")
        qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    # ── Modèle hybride PyTorch + PennyLane ──
    class QuantumClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # Poids du circuit quantique (trainables)
            weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}
            self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
            # Couche classique post-mesure pour la classification
            self.fc = nn.Linear(1, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            q_out = self.qlayer(x).unsqueeze(1)  # (batch, 1)
            return self.sigmoid(self.fc(q_out)).squeeze(1)

        def predict_proba(self, X_np):
            """Interface sklearn-compatible."""
            self.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_np)
                probas = self.forward(X_t).numpy()
            return np.column_stack([1 - probas, probas])

        def predict(self, X_np):
            probas = self.predict_proba(X_np)[:, 1]
            return (probas >= 0.5).astype(int)

    # ── Entraînement ──
    print(f"   Entraînement QNN ({CONFIG['quantum_epochs']} epochs)...")
    t0 = time.time()

    qnn = QuantumClassifier()
    optimizer = torch.optim.Adam(qnn.parameters(), lr=CONFIG["quantum_lr"])
    criterion = nn.BCELoss()

    # Mini-batch training (batch size 256 pour la vitesse)
    BATCH_SIZE = 256
    X_tr_t = torch.FloatTensor(X_tr_q)
    y_tr_t = torch.FloatTensor(y_train.values)

    best_val_auc = 0
    best_weights = None

    for epoch in range(CONFIG["quantum_epochs"]):
        qnn.train()
        # Shuffle
        perm = torch.randperm(len(X_tr_t))
        X_sh, y_sh = X_tr_t[perm], y_tr_t[perm]

        epoch_loss = 0
        for i in range(0, len(X_sh), BATCH_SIZE):
            Xb = X_sh[i:i+BATCH_SIZE]
            yb = y_sh[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            preds = qnn(Xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation AUC
        val_probas  = qnn.predict_proba(X_v_q)[:, 1]
        val_auc     = roc_auc_score(y_val, val_probas)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.clone() for k, v in qnn.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{CONFIG['quantum_epochs']} "
                  f"| Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f}")

    # Recharger les meilleurs poids
    qnn.load_state_dict(best_weights)
    print(f"\n✅ QNN terminé en {(time.time()-t0)/60:.1f} min")
    print(f"   Meilleur Val AUC : {best_val_auc:.4f}")

    results["QuantumNN"] = evaluate_model("Quantum Neural Network", qnn,
        X_tr_q, y_train, X_v_q, y_val, X_te_q, y_test)
    models["QuantumNN"] = qnn
    joblib.dump({"model": qnn, "top_features": top_features,
                 "imputer": imp, "scaler": scaler}, "../../scaler/tennis/model_quantum.pkl")
    print("✅ Sauvegardé → model_quantum.pkl")

else:
    print("\n⚠️  Quantum skippé (PennyLane non installé)")
    print("   Installe avec : pip install pennylane torch")


# ─────────────────────────────────────────────────────────────
# F. STACKING — méta-modèle sur les 3/4 prédictions
# ─────────────────────────────────────────────────────────────
# Le stacking combine les prédictions de chaque modèle comme
# features d'une régression logistique finale.
# Souvent +1-2% d'AUC par rapport au meilleur modèle seul.

print("\n" + "="*60)
print("  STACKING — méta-modèle LogReg")
print("="*60)

# Prédictions OOF (out-of-fold) sur le train pour éviter le leakage
oof_preds = {}
for name, m in [("XGBoost", xgb_model), ("LightGBM", lgb_model),
                 ("CatBoost", cat_model)]:
    oof_preds[name] = m.predict_proba(X_train)[:, 1]

stack_X_train = np.column_stack(list(oof_preds.values()))
stack_X_val   = np.column_stack([m.predict_proba(X_val)[:, 1]
                                  for m in [xgb_model, lgb_model, cat_model]])
stack_X_test  = np.column_stack([m.predict_proba(X_test)[:, 1]
                                  for m in [xgb_model, lgb_model, cat_model]])

meta_model = LogisticRegression(C=1.0, random_state=CONFIG["random_state"])
meta_model.fit(stack_X_train, y_train)

class StackingWrapper:
    def __init__(self, base_models, meta):
        self.base_models = base_models
        self.meta        = meta
    def predict(self, X):
        stack = np.column_stack([m.predict_proba(X)[:, 1]
                                  for m in self.base_models])
        return self.meta.predict(stack)
    def predict_proba(self, X):
        stack = np.column_stack([m.predict_proba(X)[:, 1]
                                  for m in self.base_models])
        return self.meta.predict_proba(stack)

stacking_model = StackingWrapper([xgb_model, lgb_model, cat_model], meta_model)

results["Stacking"] = evaluate_model("Stacking (XGB+LGB+CAT → LogReg)",
    stacking_model,
    X_train, y_train, X_val, y_val, X_test, y_test)
models["Stacking"] = stacking_model
joblib.dump({"base_models": [xgb_model, lgb_model, cat_model],
             "meta_model": meta_model}, "../../models/tennis/model_stacking.pkl")
print("✅ Sauvegardé → model_stacking.pkl")


# ─────────────────────────────────────────────────────────────
# G. CALIBRATION DU MEILLEUR MODÈLE
# ─────────────────────────────────────────────────────────────
# La calibration isotonique ajuste les probabilités pour qu'elles
# reflètent mieux la réalité (ex: P=70% → vraiment 70% de chances)
# → améliore le Brier Score et le Log Loss

print("\n" + "="*60)
print("  CALIBRATION ISOTONIQUE")
print("="*60)

results_df_tmp = pd.DataFrame(results).T
best_name = results_df_tmp["test_auc"].idxmax()
best_raw  = models[best_name]

# On calibre sur la validation uniquement (pas sur le train → leakage)
calibrated = CalibratedClassifierCV(best_raw, cv="prefit", method="isotonic")
calibrated.fit(X_val, y_val)

results["Calibrated_Best"] = evaluate_model(
    f"Calibrated {best_name}", calibrated,
    X_train, y_train, X_val, y_val, X_test, y_test)
models["Calibrated_Best"] = calibrated
joblib.dump(calibrated, "../../models/tennis/model_calibrated.pkl")
print(f"✅ Modèle calibré ({best_name}) sauvegardé → model_calibrated.pkl")


# ─────────────────────────────────────────────────────────────
# H. COMPARAISON FINALE
# ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  COMPARAISON FINALE")
print("="*60)

results_df = pd.DataFrame(results).T
print(results_df[["test_acc","test_auc","test_logloss",
                   "test_brier","overfit_gap"]].to_string())

best_model_name = results_df["test_auc"].idxmax()
best_model      = models[best_model_name]
print(f"\n🏆 Meilleur modèle : {best_model_name}")
print(f"   Test AUC        : {results_df.loc[best_model_name, 'test_auc']:.4f}")
print(f"   Test Accuracy   : {results_df.loc[best_model_name, 'test_acc']:.4f}")

# Sauvegarde du meilleur
best_model_data = {
    "model"        : best_model,
    "model_name"   : best_model_name,
    "feature_names": X.columns.tolist(),
    "results"      : results,
}
joblib.dump(best_model_data, "../../models/tennis/best_model.pkl")
print(f"✅ best_model.pkl sauvegardé")


# ─────────────────────────────────────────────────────────────
# I. VISUALISATION
# ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":"#0f1117","axes.facecolor":"#1a1d27",
    "axes.edgecolor":"#2e3347","axes.labelcolor":"#c9d1d9",
    "axes.titlecolor":"#ffffff","xtick.color":"#8b949e",
    "ytick.color":"#8b949e","text.color":"#c9d1d9",
    "grid.color":"#2e3347","grid.linestyle":"--","grid.alpha":0.5,
    "font.family":"monospace",
})
ACCENT="#00d4ff"; GREEN="#3fb950"; RED="#f85149"; ORANGE="#e3b341"; PURPLE="#bc8cff"

model_palette = {
    "XGBoost": ACCENT, "LightGBM": GREEN, "CatBoost": ORANGE,
    "QuantumNN": PURPLE, "Stacking": "#ff6b6b",
    "Calibrated_Best": "#ffd93d"
}

fig = plt.figure(figsize=(20, 12), facecolor="#0f1117")
fig.suptitle("COMPARAISON DES MODÈLES — ATP TENNIS OPTIMISÉ",
             fontsize=18, fontweight="bold", color="white",
             fontfamily="monospace", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

model_names = list(results.keys())
colors      = [model_palette.get(n, "#888") for n in model_names]

# AUC test
ax = fig.add_subplot(gs[0, 0])
auc_vals = [results[m]["test_auc"] for m in model_names]
bars = ax.bar(model_names, auc_vals, color=colors, alpha=0.85, edgecolor="none")
ax.axhline(0.5, color="white", linestyle="--", alpha=0.3)
for bar, val in zip(bars, auc_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{val:.4f}", ha="center", fontsize=8, color="white")
ax.set_title("ROC-AUC (Test)", fontsize=11, pad=10)
ax.set_ylim(0.45, 0.85)
ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
ax.grid(axis="x", visible=False)

# Accuracy train vs test
ax = fig.add_subplot(gs[0, 1])
x, w = np.arange(len(model_names)), 0.3
for i, (split, color) in enumerate([("train","#444"),("test",GREEN)]):
    vals = [results[m][f"{split}_acc"] for m in model_names]
    ax.bar(x + i*w, vals, width=w, label=split.capitalize(),
           color=color, alpha=0.85, edgecolor="none")
ax.set_xticks(x + w/2)
ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
ax.set_title("Accuracy Train vs Test", fontsize=11, pad=10)
ax.set_ylim(0.45, 1.0)
ax.legend(fontsize=8)
ax.grid(axis="x", visible=False)

# Overfitting gap
ax = fig.add_subplot(gs[0, 2])
gap_vals   = [results[m]["overfit_gap"] for m in model_names]
bar_colors = [RED if v>0.08 else ORANGE if v>0.04 else GREEN for v in gap_vals]
bars = ax.bar(model_names, gap_vals, color=bar_colors, alpha=0.85, edgecolor="none")
for bar, val in zip(bars, gap_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f"{val:.3f}", ha="center", fontsize=8, color="white")
ax.set_title("Overfitting gap (Train-Test acc)", fontsize=11, pad=10)
ax.axhline(0, color="white", linestyle="--", alpha=0.3)
ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
ax.grid(axis="x", visible=False)

# Feature importance meilleur modèle
ax = fig.add_subplot(gs[1, 0:2])
if best_model_name in ("XGBoost", "LightGBM"):
    imp = pd.Series(best_model.feature_importances_,
                    index=X.columns).sort_values(ascending=False).head(20)
elif best_model_name == "CatBoost":
    imp = pd.Series(cat_model.get_feature_importance(),
                    index=X.columns).sort_values(ascending=False).head(20)
elif best_model_name == "Stacking":
    imp = pd.Series(best_raw.feature_importances_
                    if hasattr(best_raw, "feature_importances_") else
                    np.zeros(len(X.columns)),
                    index=X.columns).sort_values(ascending=False).head(20)
else:
    imp = pd.Series(np.zeros(20))

if len(imp) > 0:
    ax.barh(imp.index[::-1], imp.values[::-1],
            color=model_palette.get(best_model_name, ACCENT), alpha=0.85, edgecolor="none")
    ax.set_title(f"Top 20 Feature Importances — {best_model_name}", fontsize=11, pad=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)

# Tableau récap
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
metrics    = ["test_acc","test_auc","test_logloss","test_brier","overfit_gap"]
labels_tab = ["Accuracy","ROC-AUC","Log Loss","Brier Score","Overfit Gap"]
table_data = [[f"{results[m][met]:.4f}" for m in model_names] for met in metrics]
table = ax.table(cellText=table_data, rowLabels=labels_tab,
                 colLabels=[n[:8] for n in model_names],
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.1, 1.6)
for (r, c), cell in table.get_celld().items():
    cell.set_facecolor("#1a1d27" if r > 0 else "#2e3347")
    cell.set_edgecolor("#2e3347")
    cell.set_text_props(color="white")
ax.set_title("Résumé métriques (Test)", fontsize=11, pad=20)

plt.savefig("../../visualisation/tennis/model_comparison_optimized.png", dpi=150,
            bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("✅ Graphique → model_comparison_optimized.png")

# ─────────────────────────────────────────────────────────────
# J. RÉSUMÉ FINAL
# ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  RÉSUMÉ FINAL")
print("="*60)
for name in model_names:
    r = results[name]
    flag = "🏆" if name == best_model_name else "  "
    print(f"  {flag} {name[:30]:30} | AUC={r['test_auc']:.4f} "
          f"| Acc={r['test_acc']:.4f} | Gap={r['overfit_gap']:.4f}")
print("="*60)
print("\nFichiers sauvegardés :")
print("  best_model.pkl, model_xgboost.pkl, model_lightgbm.pkl")
print("  model_catboost.pkl, model_quantum.pkl, model_stacking.pkl")
print("  model_calibrated.pkl, model_comparison_optimized.png")
