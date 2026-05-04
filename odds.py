import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# Dictionnaire de mapping mis à jour avec les clés exactes de ton JSON
SPORTS_MAPPING = {
    "nba": "basketball_nba",
    "premier_league": "soccer_epl",
    "ligue_1": "soccer_france_ligue_one",
    "bundesliga": "soccer_germany_bundesliga",
    "serie_a": "soccer_italy_serie_a",
    "la_liga": "soccer_spain_la_liga",
    "champions_league": "soccer_uefa_champs_league", # Ajout bonus issu de ta liste
}

def get_match_odds(sport_key: str, team_a: str, team_b: str = "") -> list[dict]:
    """
    Outil pour récupérer les cotes (h2h) d'un match spécifique.
    Idéal pour un Agent IA. Retourne une liste de dictionnaires.
    
    :param sport_key: Clé du sport (ex: 'basketball_nba', 'soccer_epl').
    :param team_a: Nom de l'équipe à domicile (ou une partie du nom).
    :param team_b: Nom de l'équipe à l'extérieur (ou une partie du nom).
    """

    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h'
    response = requests.get(url)
    
    if response.status_code != 200:
        return [{"error": f"Erreur API ({response.status_code}): {response.text}"}]
        
    data = response.json()
    
    # Savoir si on doit chercher une cote pour le match nul
    is_soccer = "soccer" in sport_key
    
    df = _flatten_odds(data, is_soccer)
    
    if df.empty:
        return [{"message": f"Aucune cote disponible actuellement pour la compétition {sport_key}."}]

    # Filtrage selon les équipes demandées par l'agent
    mask = df['Match'].str.contains(team_a, case=False, na=False)
    if team_b:
        mask = mask & df['Match'].str.contains(team_b, case=False, na=False)
        
    match_interet = df[mask]
    
    if match_interet.empty:
        return [{"message": f"Aucun match trouvé pour cet/ces équipe(s)."}]
        
    # On retourne les données sous forme de dictionnaire (parfait pour l'IA)
    return match_interet.to_dict(orient='records')

def _flatten_odds(data: list, is_soccer: bool) -> pd.DataFrame:
    rows = []
    for match in data:
        home_team = match.get('home_team')
        away_team = match.get('away_team')
        time = match.get('commence_time')
        
        for bookmaker in match.get('bookmakers', []):
            bookie_name = bookmaker['title']
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    # Création d'un dictionnaire Nom -> Prix
                    odds = {outcome['name']: outcome['price'] for outcome in market['outcomes']}
                    
                    row_data = {
                        'Match': f"{home_team} vs {away_team}",
                        'Date': time,
                        'Bookmaker': bookie_name,
                        'Cote Home (1)': odds.get(home_team),
                        'Cote Away (2)': odds.get(away_team)
                    }
                    
                    # Si c'est du foot, on ajoute la colonne Draw (X)
                    if is_soccer:
                        row_data['Cote Draw (X)'] = odds.get('Draw')
                        
                    rows.append(row_data)
                    
    return pd.DataFrame(rows)

def _get_tennis_keys() -> list[dict]:
    url = f'https://api.the-odds-api.com/v4/sports/?apiKey={ODDS_API_KEY}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # On filtre pour ne renvoyer que le tennis à l'agent pour ne pas surcharger son contexte
        active_tennis = [{"name" : sport['title'], "key": sport['key']} for sport in data if 'tennis' in sport['group'].lower()]
        return active_tennis
    return [{"error": "Impossible de récupérer les sports."}]

def show_comp_keys() -> list[dict]:
    """
    Outil pour l'IA : Permet de récupérer la liste de toutes les compétitions actives en ce moment.
    L'agent doit utiliser cette liste pour trouver la bonne 'key' d'une compétition.
    """
    return _get_tennis_keys() + [{"name": name, "key": key} for name, key in SPORTS_MAPPING.items()]