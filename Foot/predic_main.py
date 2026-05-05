import pandas as pd
import numpy as np
import re
import pickle
import requests
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

# =========================================================
# 1. MOTEUR DE FEATURE ENGINEERING (RECONSTRUCTION)
# =========================================================

MAJOR_LEAGUES = ["FL1", "PL", "BL1", "SA", "PD"]
MAJOR_LEAGUE_NAMES = {
    "FL1": "Ligue 1",
    "PL": "Premier League",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "PD": "La Liga",
}

FOLDER_MODEL = 'model'
FOLDER_MODEL_SIMPLE = 'model_simple'


def normaliser_nom_equipe(value):
    """Normalise un nom d'équipe pour la comparaison."""
    text = str(value).strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parser_dates_melangees(values):
    """Parse des dates mixtes sans warning, en gardant le day-first pour les formats avec /."""
    series = pd.Series(values)
    iso_mask = series.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}($|[ T])")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(series.loc[iso_mask], errors="coerce", format="mixed")
    if (~iso_mask).any():
        parsed.loc[~iso_mask] = pd.to_datetime(series.loc[~iso_mask], errors="coerce", dayfirst=True, format="mixed")

    return parsed

def obtenir_meteo_match(lat, lon, date_match, time_match):
    """Récupère les prévisions météo pour un match futur (jusqu'à 16 jours)."""
    if pd.isna(lat) or pd.isna(lon):
        return {}

    # L'URL change pour le "forecast"
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_match,
        "end_date": date_match,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m",
        "timezone": "Europe/London"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Lève une erreur si le status n'est pas 200
    except requests.exceptions.RequestException as e:
        print(f"[Météo] Erreur API le {date_match}: {e}")
        return {}

    data = response.json()
    
    if "hourly" not in data:
        print(f"[Météo] Pas de données disponibles pour le {date_match} (trop loin dans le futur ?)")
        return {}

    df_meteo = pd.DataFrame(data["hourly"])
    df_meteo["time"] = pd.to_datetime(df_meteo["time"])

    # On définit la fenêtre du match
    kickoff = pd.to_datetime(f"{date_match} {time_match}")
    full_time = kickoff + pd.Timedelta(hours=2)
    
    # Filtrage
    df_match = df_meteo[(df_meteo["time"] >= kickoff) & (df_meteo["time"] <= full_time)]

    if df_match.empty:
        # Petit fallback : si le match est pile à la limite de la prévision
        return {}

    return {
        "Temp_Moy_C": round(df_match["temperature_2m"].mean(), 1),
        "Humidite_Moy_%": round(df_match["relative_humidity_2m"].mean(), 1),
        "Pluie_Tot_mm": round(df_match["precipitation"].sum(), 1),
        "Neige_Tot_cm": round(df_match["snowfall"].sum(), 1),
        "Vent_Moy_kmh": round(df_match["wind_speed_10m"].mean(), 1),
        "Rafale_Max_kmh": round(df_match["wind_gusts_10m"].max(), 1),
    }

def preparer_match_futur(home_team, away_team, match_date_str, match_heure_str, referee, df_brut):
    """
    Recalcule intégralement l'environnement d'avant-match :
    Elo, Pi-Rating, Forme (L5, L3), et refait les divisions de la saison (_pm).
    """
    df = df_brut.copy()
    df['Date'] = parser_dates_melangees(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # -- 1. MACHINE À REMONTER LE TEMPS (ELO, PI, FORME) --
    teams = pd.concat([df['Home'], df['Away']]).unique()
    team_stats = {team: {
        'last_date': pd.NaT, 'pts_history': [], 'gd_history': [], 
        'xg_diff_history': [], 'elo': 1500.0, 'pi_home': 0.0, 'pi_away': 0.0          
    } for team in teams}
    
    h2h_stats = {} 
    K_ELO, LAMBDA_PI = 20, 0.1
    
    for idx, row in df.iterrows():
        home, away = row['Home'], row['Away']
        score_str = str(row['Score'])
        
        if pd.notna(row['Score']) and ('-' in score_str or '–' in score_str):
            try:
                buts = re.split(r'[-–]', score_str)
                hg, ag = float(buts[0].strip()), float(buts[1].strip())
                gd = hg - ag
                
                # H2H et Points
                if hg > ag:
                    home_pts, away_pts, h2h_result, s_home, s_away = 3, 0, 1, 1, 0
                elif hg == ag:
                    home_pts, away_pts, h2h_result, s_home, s_away = 1, 1, 0, 0.5, 0.5
                else:
                    home_pts, away_pts, h2h_result, s_home, s_away = 0, 3, 0, 0, 1
                    
                team_stats[home]['pts_history'].append(home_pts)
                team_stats[away]['pts_history'].append(away_pts)
                team_stats[home]['gd_history'].append(gd)
                team_stats[away]['gd_history'].append(-gd)

                home_xg = float(row['Domicile_xG_Pour_xG_home']) if pd.notna(row['Domicile_xG_Pour_xG_home']) else 0.0
                away_xg = float(row['Exterieur_xG_Pour_xG_away']) if pd.notna(row['Exterieur_xG_Pour_xG_away']) else 0.0
                xg_diff = home_xg - away_xg
    
                team_stats[home]['xg_diff_history'].append(xg_diff)
                team_stats[away]['xg_diff_history'].append(-xg_diff)
                
                # ELO
                exp_home = 1 / (1 + 10 ** ((team_stats[away]['elo'] - team_stats[home]['elo']) / 400))
                exp_away = 1 / (1 + 10 ** ((team_stats[home]['elo'] - team_stats[away]['elo']) / 400))
                team_stats[home]['elo'] += K_ELO * (s_home - exp_home)
                team_stats[away]['elo'] += K_ELO * (s_away - exp_away)
                
                # PI-RATING
                expected_gd = team_stats[home]['pi_home'] - team_stats[away]['pi_away']
                pi_error = gd - expected_gd
                team_stats[home]['pi_home'] += LAMBDA_PI * pi_error
                team_stats[away]['pi_away'] -= LAMBDA_PI * pi_error

                # H2H et Dates
                h2h_key = (home, away)
                if h2h_key not in h2h_stats: h2h_stats[h2h_key] = []
                h2h_stats[h2h_key].append(h2h_result)
                team_stats[home]['last_date'] = row['Date']
                team_stats[away]['last_date'] = row['Date']
                
            except ValueError:
                pass 

    # -- 2. CONSTRUCTION DU VECTEUR DU MATCH FUTUR --
    
    input_data = {}
    
    # Intégration des dynamiques calculées (L'état des équipes AUJOURD'HUI)
    input_data['Home_Elo'] = team_stats[home_team]['elo']
    input_data['Away_Elo'] = team_stats[away_team]['elo']
    input_data['Home_Pi_Rating'] = team_stats[home_team]['pi_home']
    input_data['Away_Pi_Rating'] = team_stats[away_team]['pi_away']
    
    home_last_d = team_stats[home_team]['last_date']
    away_last_d = team_stats[away_team]['last_date']

    if match_date_str :
        match_date = pd.to_datetime(match_date_str)
        # en loyenne un match par semaine
        input_data['Home_Rest_Days'] = (match_date - home_last_d).days if pd.notna(home_last_d) else 7
        input_data['Away_Rest_Days'] = (match_date - away_last_d).days if pd.notna(away_last_d) else 7
        input_data['Date'] = match_date
        input_data['Time'] = match_heure_str
    
    input_data['Home_Pts_L5'] = sum(team_stats[home_team]['pts_history'][-5:])
    input_data['Away_Pts_L5'] = sum(team_stats[away_team]['pts_history'][-5:])
    input_data['Home_GD_L3'] = sum(team_stats[home_team]['gd_history'][-3:])
    input_data['Away_GD_L3'] = sum(team_stats[away_team]['gd_history'][-3:])
    input_data['Home_xG_Diff_L3'] = np.mean(team_stats[home_team]['xg_diff_history'][-3:]) if team_stats[home_team]['xg_diff_history'] else 0.0
    input_data['Away_xG_Diff_L3'] = np.mean(team_stats[away_team]['xg_diff_history'][-3:]) if team_stats[away_team]['xg_diff_history'] else 0.0
    
    h2h_key_futur = (home_team, away_team)
    input_data['H2H_Home_WinRate_L5'] = np.mean(h2h_stats[h2h_key_futur][-5:]) if h2h_key_futur in h2h_stats else 0.0

    # -- 3. RÉCUPÉRATION DES STATS BRUTES ET CALCUL DES MOYENNES (_pm) --
    # On identifie la saison actuelle pour compter les matchs comme dans la cellule 98
    saison_actuelle = df['Season'].iloc[-1]
    nb_matchs_home = len(df[(df['Home'] == home_team) & (df['Season'] == saison_actuelle)])
    nb_matchs_away = len(df[(df['Away'] == away_team) & (df['Season'] == saison_actuelle)])
    
    last_home_row = df[df['Home'] == home_team].iloc[-1] if not df[df['Home'] == home_team].empty else pd.Series()
    last_away_row = df[df['Away'] == away_team].iloc[-1] if not df[df['Away'] == away_team].empty else pd.Series()

    # Liste des attributs racines à diviser (Cellule 98)
    attributs_racine = [
        'xG_Pour_xG', 'xG_Pour_Buts*', 'xG_Pour_Tirs',
        'xG_Contre_xG', 'xG_Contre_Buts*', 'xG_Contre_Tirs',
        'TypesButs_Pour_Dans Le Jeu', 'TypesButs_Pour_Contre-Attaque',
        'TypesButs_Pour_Coup de Pied Arrêté', 'TypesButs_Pour_Penalty', 'TypesButs_Pour_Contre son Camp',
        'TypesButs_Contre_Dans Le Jeu', 'TypesButs_Contre_Contre-Attaque',
        'TypesButs_Contre_Coup de Pied Arrêté', 'TypesButs_Contre_Penalty', 'TypesButs_Contre_Contre son Camp'
    ]
    
    # On intègre d'abord tout, puis on calcule les ratios
    for col in last_home_row.index:
        input_data[col] = last_home_row[col]

    # On ajoute les colonnes de l'équipe extérieure sans écraser
    # les clés déjà présentes (préférer les valeurs domicile si conflit).
    for col in last_away_row.index:
        if col not in input_data:
            input_data[col] = last_away_row[col]

    for racine in attributs_racine:
        col_h, col_a = f"Domicile_{racine}_home", f"Exterieur_{racine}_away"
        if col_h in input_data and nb_matchs_home > 0:
            input_data[f"{col_h}_pm"] = input_data[col_h] / nb_matchs_home
        if col_a in input_data and nb_matchs_away > 0:
            input_data[f"{col_a}_pm"] = input_data[col_a] / nb_matchs_away

    # -- 4. NETTOYAGE FINAL (MÊME LOGIQUE QUE LE NOTEBOOK) --

    #On save juste le stade qui sera utile plus tard
    venue = input_data["Venue"]

    # Suppression des familles granulaires (Cellule 91)[cite: 2]
    mots_cles_a_supprimer = ['DirectionsTirs', 'CotésUtilisés', 'Detailed_Assists', 'CausesCartons']
    # Suppression des colonnes brutes spécifiques (Cellule 96 et 105)[cite: 2]
    cols_to_drop = [
        'Exterieur_General_Buts_away', 'Exterieur_General_Cartons_J_away', 
        'Exterieur_General_Cartons_R_away', 'Exterieur_xG_Pour_xGDiff_away', 
        'Exterieur_xG_Contre_xGDiff_away', 'Domicile_General_Buts_home', 
        'Domicile_General_Cartons_J_home', 'Domicile_General_Cartons_R_home', 
        'Domicile_xG_Pour_xGDiff_home', 'Domicile_xG_Contre_xGDiff_home',
        'Unnamed: 0', 'Score', 'Neige_Tot_cm', 'Referee', 'Time', 'Venue', 
        'Home_Manager', 'Away_Manager', 'Home_Goals', 'Away_Goals',

        #Infos relatives à la météo à considérer ssi on a une date
        'Temp_Moy_C',
        'Humidite_Moy_%',
        'Pluie_Tot_mm',
        'Vent_Moy_kmh',
        'Rafale_Max_kmh',

        #Infos relatives à l'arbitre : 
        'avg_yellow_per_match',
        'avg_red_per_match',
        'avg_goals_per_match',
        'avg_fouls_per_match'
    ]
    # On ajoute aussi les racines brutes qui ont été transformées en _pm 
    for racine in attributs_racine:
        cols_to_drop.append(f"Domicile_{racine}_home")
        cols_to_drop.append(f"Exterieur_{racine}_away")


    final_keys = list(input_data.keys())
    for key in final_keys:
        if any(mot in key for mot in mots_cles_a_supprimer) or key in cols_to_drop:
            input_data.pop(key, None)


    # -- 5. MÉTÉO ET CONTEXTE DU MATCH --
    if match_date_str and referee:
        df_stades = pd.read_csv("data/stades.csv")

        map_lat = df_stades.set_index("name")["lat"].to_dict()
        map_lon = df_stades.set_index("name")["lon"].to_dict()

        lat = map_lat.get(venue)
        lon = map_lon.get(venue)
        meteo = obtenir_meteo_match(lat, lon, match_date_str, match_heure_str)

        input_data['Temp_Moy_C'] = meteo["Temp_Moy_C"] if meteo else df["Temp_Moy_C"].mean()
        input_data['Humidite_Moy_%'] = meteo["Humidite_Moy_%"] if meteo else df["Humidite_Moy_%"].mean()
        input_data['Pluie_Tot_mm'] = meteo["Pluie_Tot_mm"] if meteo else df["Pluie_Tot_mm"].mean()
        input_data['Vent_Moy_kmh'] = meteo["Vent_Moy_kmh"] if meteo else df["Vent_Moy_kmh"].mean()
        input_data['Rafale_Max_kmh'] = meteo["Rafale_Max_kmh"] if meteo else df["Rafale_Max_kmh"].mean()

        #Infos relatives à l'arbitre : 
        df_referees = pd.read_csv("data/referees.csv")

        input_data['avg_yellow_per_match'] = df['avg_yellow_per_match'].mean()
        input_data['avg_red_per_match']    = df['avg_red_per_match'].mean()
        input_data['avg_goals_per_match']  = df['avg_goals_per_match'].mean()
        input_data['avg_fouls_per_match']  = df['avg_fouls_per_match'].mean()

    input_data['Home'] = home_team
    input_data['Away'] = away_team

    return pd.DataFrame([input_data])

# =========================================================
# 2. SOUS-FONCTIONS SPÉCIFIQUES À CHAQUE MODÈLE
# =========================================================

def predire_xgboost(match_df, is_simple):
    """Logique de prédiction spécifique à XGBoost"""
    print(" Utilisation du modèle : XGBoost")
    
    folder = "model_simple" if is_simple else "model" 
    
    with open(folder + '/xgb.pkl', 'rb') as f:
        model = pickle.load(f)
        
    colonnes_attendues = model.feature_names_in_ 
    donnees_alignees = {col: match_df.iloc[0][col] if col in match_df.columns else 0.0 for col in colonnes_attendues}
    match_final = pd.DataFrame([donnees_alignees])
    
    feature_types = getattr(model, 'feature_types', None)
    for i, col in enumerate(colonnes_attendues):
        if feature_types and feature_types[i] == 'c':
            match_final[col] = match_final[col].astype("category")
        else:
            match_final[col] = pd.to_numeric(match_final[col], errors='coerce').fillna(0.0)

    try:
        probabilites = model.predict_proba(match_final)[0]
    except Exception:
        import xgboost as xgb
        dmatrix = xgb.DMatrix(match_final, enable_categorical=True)
        probabilites = model.get_booster().predict(dmatrix)[0]
        
    return probabilites, model.classes_


def predire_random_forest(match_df, is_simple):
    print(" Utilisation du modèle : Random Forest")

    folder = "model_simple" if is_simple else "model" 
    
    with open(folder + '/rf.pkl', 'rb') as f:
        model = pickle.load(f)
        
    colonnes_attendues = model.feature_names_in_ 
    donnees_alignees = {col: match_df.iloc[0][col] if col in match_df.columns else 0.0 for col in colonnes_attendues}
    match_final = pd.DataFrame([donnees_alignees])
    
    # On force tout en numérique (le modèle n'a été entraîné que sur du numérique)
    for col in match_final.columns:
        match_final[col] = pd.to_numeric(match_final[col], errors='coerce').fillna(0.0)

    probabilites = model.predict_proba(match_final)[0]
    return probabilites, model.classes_

def predire_catboost(match_df, is_simple):
    from catboost import CatBoostClassifier
    print(" Utilisation du modèle : CatBoost")

    folder = "model_simple" if is_simple else "model" 
    
    model = CatBoostClassifier()
    model.load_model(folder + '/catboost.cbm')
    
    colonnes_attendues = model.feature_names_ 
    donnees_alignees = {col: match_df.iloc[0][col] if col in match_df.columns else np.nan for col in colonnes_attendues}
    match_final = pd.DataFrame([donnees_alignees])
    
    # Remplir les NaN pour les catégorielles comme dans le notebook
    cat_features = model.get_cat_feature_indices()
    cat_feature_names = [colonnes_attendues[i] for i in cat_features]
    match_final[cat_feature_names] = match_final[cat_feature_names].fillna('Missing')
    
    # Numériques
    num_features = [col for col in colonnes_attendues if col not in cat_feature_names]
    match_final[num_features] = match_final[num_features].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    probabilites = model.predict_proba(match_final)[0]
    return probabilites, model.classes_

def predire_qnn(match_df, is_simple):
    print(" ⚛️ Utilisation du modèle : QNN (Quantum Neural Network)")

    # Importer localement les dépendances lourdes pour éviter des conflits
    # de bibliothèques natives lors du chargement d'autres modèles (XGBoost)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pennylane as qml

    # ARCHITECTURE QNN (définie localement)
    N_QUBITS  = 7
    QNN_REPS  = 3
    N_CLASSES = 3
    dev = qml.device('lightning.qubit', wires=N_QUBITS)

    @qml.qnode(dev, interface='torch', diff_method='best')
    def qnode(x, theta):
        qml.templates.AngleEmbedding(x, wires=range(N_QUBITS))
        qml.templates.StronglyEntanglingLayers(theta, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_CLASSES)]

    class QNodeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.randn(QNN_REPS, N_QUBITS, 3) * 0.1)

        def forward(self, x):
            x = x.float()
            if x.ndim == 1: x = x.unsqueeze(0)
            outs = []
            for i in range(x.shape[0]):
                ev = qnode(x[i], self.theta)
                outs.append(torch.stack(ev))
            return torch.stack(outs)

    class HybridQNN(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.pre_net = nn.Sequential(
                nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, N_QUBITS), nn.Tanh(),
            )
            self.q_layer  = QNodeLayer()
            self.post_net = nn.Sequential(nn.Linear(N_CLASSES, N_CLASSES))

        def forward(self, x):
            x = self.pre_net(x)
            x = self.q_layer(x)
            return self.post_net(x)
        
    folder = "model_simple" if is_simple else "model" 

    # 1. Chargement des métadonnées (Scaler, Encodeur, Colonnes)
    with open(folder + '/qnn_cols.pkl', 'rb') as f: num_cols = pickle.load(f)
    with open(folder + '/qnn_scaler.pkl', 'rb') as f: scaler_q = pickle.load(f)
    with open(folder + '/qnn_le.pkl', 'rb') as f: le = pickle.load(f)

    # 2. Alignement des données du match avec les colonnes de l'entraînement
    donnees_alignees = {col: match_df.iloc[0][col] if col in match_df.columns else 0.0 for col in num_cols}
    match_final = pd.DataFrame([donnees_alignees])[num_cols]

    # 3. Standardisation
    X_raw = match_final.fillna(0).values
    X_scaled = scaler_q.transform(X_raw)
    X_tensor = torch.from_numpy(X_scaled).float()

    # 4. Chargement et inférence du modèle
    model = HybridQNN(n_features=len(num_cols)).float()
    model.load_state_dict(torch.load(folder + '/quantum.pth', map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        # Convertir les logits bruts en probabilités (softmax)
        probabilites = F.softmax(logits, dim=1).numpy()[0]

    return probabilites, le.classes_

# =========================================================
# 3. DISPATCHER & EXÉCUTION
# =========================================================

def faire_un_pronostic(df_match, modele_choisi="XGBoost", is_simple=False):
    
    try:
        if modele_choisi.lower() == "xgboost":
            probabilites, classes = predire_xgboost(df_match, is_simple)
        elif modele_choisi.lower() == "random forest":
            probabilites, classes = predire_random_forest(df_match, is_simple)
        elif modele_choisi.lower() == "catboost":
            probabilites, classes = predire_catboost(df_match, is_simple)
        elif modele_choisi.lower() == "qnn":
            probabilites, classes = predire_qnn(df_match, is_simple) 
        else:
            raise ValueError(f"Le modèle '{modele_choisi}' n'est pas reconnu.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier du modèle {modele_choisi} est introuvable dans 'model/'.")
        return None, None
    
    return classes, probabilites

def normaliser_nom_equipe(value):
    """Normalise un nom d'équipe pour la comparaison."""
    text = str(value).strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def rechercher_prochain_match_api(team_home, team_away, api_key):
    """
    Recherche le match dans les 30 prochains jours avec filtrage de dates.
    """
    maintenant = datetime.now()
    date_debut = maintenant.strftime('%Y-%m-%d')
    date_fin = (maintenant + timedelta(days=30)).strftime('%Y-%m-%d')

    headers = {'X-Auth-Token': api_key}
    t1 = normaliser_nom_equipe(team_home)
    t2 = normaliser_nom_equipe(team_away)

    for competition_code in MAJOR_LEAGUES:
        url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches"
        params = {
            'status': 'SCHEDULED',
            'dateFrom': date_debut,
            'dateTo': date_fin,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 429:
                print("Limite crédits football-data atteinte, réessaie plus tard.")
                return None, None, None, None, None
            response.raise_for_status()
            matches = response.json().get('matches', [])

        except Exception as e:
            print(f"Erreur API sur {competition_code} : {e}")
            continue

        for match in sorted(matches, key=lambda item: item.get('utcDate', '')):
            home_api = normaliser_nom_equipe(match.get('homeTeam', {}).get('name', ''))
            away_api = normaliser_nom_equipe(match.get('awayTeam', {}).get('name', ''))

            if (t1 in home_api and t2 in away_api):
                date_match = match['utcDate'].split('T')[0]
                heure_match = match['utcDate'].split('T')[1][:5]
                real_home = match['homeTeam']['name']
                real_away = match['awayTeam']['name']
                referees = [ref.get('name') for ref in match.get('referees', [])]
                referee = None
                if len(referees) > 0 :
                    referee = referees[0]
                competition_name = MAJOR_LEAGUE_NAMES.get(competition_code, competition_code)

                print(f" Match trouvé : {real_home} vs {real_away} ({competition_name})")
                print(f" Date : {date_match} à {heure_match} UTC")

                return date_match, heure_match, referee

    print(f"Aucun match programmé pour {team_home} vs {team_away} dans les 5 championnats majeurs.")
    return None, None, None

def predire(team_home, team_away):
    """
    Les noms des équipes doivent être exactement ceux présents dans le fichier CSV processed.csv
    """
    load_dotenv()
    API_KEY = os.getenv("API_KEY_FOOTBALL_DATA")
    
    # Chargement des données etc
    df_historique = pd.read_csv("data/processed.csv", low_memory=False)

    equipe_a = team_home
    equipe_b = team_away
    is_simple = False

    date_api, heure_api, referee = rechercher_prochain_match_api(equipe_a, equipe_b, API_KEY)

    if not date_api:
        # On essaie dans l'autre sens
        date_api, heure_api, referee = rechercher_prochain_match_api(equipe_b, equipe_a, API_KEY)
        temp = equipe_b
        equipe_b = equipe_a
        equipe_a = temp

    if not referee:
        #Alors le match n'a pas été trouvé
        #On interroge le modèle sans rien : ni de données d'arbitre, ni de date : le mode simple
        is_simple = True
        # On refixe domicile/away comme la demande d'origine
        equipe_a = team_home
        equipe_b = team_away

    # On va itérer sur tous les modèles puis prendre la moyenne des probas
    modeles = ["xgboost", "random forest", "catboost"]

    res = {
        0 : 0,
        1 : 0,
        2 : 0
    }

    match_a_predire = preparer_match_futur(equipe_a, equipe_b, date_api, heure_api, referee, df_historique)
    
    for modele in modeles:
        classes, probabilites = faire_un_pronostic(match_a_predire, modele_choisi=modele, is_simple=is_simple)
        for cls, proba in zip(classes, probabilites):
            res[cls] += proba

    nbModele = len(modeles)
    for key in res:
        res[key] /= nbModele
        res[key] = round(res[key] * 100, 2)

    final = {
        "date" : date_api,
        equipe_a : res[0],
        equipe_b : res[1],
        "nul" : res[2]
    }

    json_string = json.dumps(final, indent=4, ensure_ascii=False)

    print(json_string)
    return final

    
if __name__ == "__main__":

    equipe_a = "Brest"
    equipe_b = "Paris Saint-Germain"
    predire(equipe_a, equipe_b)