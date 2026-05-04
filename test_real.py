import os
from dotenv import load_dotenv
load_dotenv()
import requests
import logging
from data_collection.config import BSD_API_TOKEN
from data_collection.pipeline import collect_contextual_data
import data_collection.pipeline as p

# Désactive l'upsert Supabase pour ce test
p.upsert_data = lambda table, data: None

logging.basicConfig(level=logging.INFO)

token = os.getenv("BSD_API_TOKEN")
if token:
    print(f"BSD_API_TOKEN chargé: {token[:10]}...{'*'*10}")
else:
    print("BSD_API_TOKEN non trouvé dans l'environnement !")

BSD_BASE_URL = "https://sports.bzzoiro.com/api/"

# 1. Récupérer un match terminé (filtrage API ou fallback Python)
headers = {"Authorization": f"Token {BSD_API_TOKEN}"}
url = f"{BSD_BASE_URL}events/?status=finished&date_from=2026-01-01&date_to=2026-04-01"
resp = requests.get(url, headers=headers, timeout=10)
if resp.status_code == 200:
    data = resp.json()
    match = None
    for m in data.get("results", []):
        if m.get("home_score") is not None and m.get("away_score") is not None:
            match = m
            break
    if not match:
        # Fallback : requête sans status, filtrage côté Python
        url2 = f"{BSD_BASE_URL}events/?date_from=2026-01-01&date_to=2026-01-31"
        resp2 = requests.get(url2, headers=headers, timeout=10)
        resp2.raise_for_status()
        data2 = resp2.json()
        for m in data2.get("results", []):
            if m.get("home_score") is not None and m.get("away_score") is not None:
                match = m
                break
    if not match:
        print("Aucun match terminé trouvé dans la période demandée.")
        exit(1)
else:
    print(f"Erreur API: {resp.status_code}")
    exit(1)

match_id = match["id"]
home_team = match.get("home_team")
away_team = match.get("away_team")
print(f"Test sur match_id={match_id} : {home_team} vs {away_team}")

# 2. Collecte contextuelle (sans upsert)
results = collect_contextual_data(match_id)

# 3. Affichage détaillé
for key, val in results.items():
    print(f"\n--- {key.upper()} ---")
    if val is None:
        print("Aucune donnée collectée (None)")
    elif hasattr(val, '__dict__'):
        for k, v in val.__dict__.items():
            print(f"{k}: {v}")
    elif isinstance(val, dict):
        for k, v in val.items():
            print(f"{k}: {v}")
    else:
        print(val)

# 4. Résumé final
print("\nRésumé :")
for key in ["weather", "coach", "referee", "stadium", "timezone"]:
    status = results.get(key)
    check = "✅" if status else "❌"
    print(f"{check} {key}")
print(f"Match testé : {match_id} | {home_team} vs {away_team}")
