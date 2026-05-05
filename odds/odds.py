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
    """Récupère les cotes head-to-head (H2H) pour un ou plusieurs bookmakers.

        Usage attendu pour un agent IA
        - Entrée (args):
            - ``sport_key`` (str): clé de la compétition telle que renvoyée par l'API
                (ex: "basketball_nba", "soccer_france_ligue_one"). L'agent doit
                appeler `show_comp_keys()` pour découvrir les clés actives.
            - ``team_a`` (str): chaîne partielle ou entière correspondant à l'équipe
                domicile recherchée (ex: "Paris SG", "Lakers"). Recherche insensible
                à la casse.
            - ``team_b`` (str, optionnel): chaîne partielle pour l'équipe extérieure.

        Comportement
        - Interroge l'API The Odds API en mode ``markets=h2h`` et transforme la
            réponse en une liste de dictionnaires, un par bookmaker/match.
        - Filtre les résultats pour ne garder que les lignes contenant ``team_a``
            (et ``team_b`` si fourni) dans la colonne "Match".

        Sortie (retour): ``list[dict]`` — chaque dictionnaire contient au minimum :
            - ``Match`` (str): "Home vs Away" (ex: "Paris SG vs Lyon").
            - ``Date`` (str): timestamp de début du match tel que fourni par l'API.
            - ``Bookmaker`` (str): nom du bookmaker.
            - ``Cote Home (1)`` (float|None): cote pour l'équipe à domicile.
            - ``Cote Away (2)`` (float|None): cote pour l'équipe extérieure.
            - ``Cote Draw (X)`` (float|None): présente seulement si le sport gère le
                match nul (p.ex. football).

        En cas d'erreur ou d'absence de données, la fonction renvoie une liste
        contenant un dictionnaire d'erreur/message, par exemple :
            - ``[{"error": "Erreur API (401): ..."}]``
            - ``[{"message": "Aucun match trouvé pour cet/ces équipe(s)."}]``

        Remarques pour l'agent
        - L'agent doit normaliser ou fuzzy-matcher les noms d'équipe avant appel
            si nécessaire (la fonction utilise une recherche simple par substring).
        - Ne pas partager la clé API dans les logs. Si ``ODDS_API_KEY`` est vide,
            la requête retournera une erreur de l'API.
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
        active_tennis = [{"name" : sport['title'], "key": sport['key']} for sport in data if 'tennis' in sport['group'].lower()]
        return active_tennis
    return [{"error": "Impossible de récupérer les sports."}]

def show_comp_keys() -> list[dict]:
    """Renvoie une liste de compétitions utiles pour l'agent.

        But
        - Fournir à l'agent un sous-ensemble exploitable des sports/compétitions
            et leurs clés (`key`) à utiliser dans ``get_match_odds``.

        Format de retour
        - ``list[dict]`` avec des éléments de la forme :
                - ``{"name": <str>, "key": <str>}``
            ou, en cas d'erreur :
                - ``[{"error": "Impossible de récupérer les sports."}]``

        Exemple
        - ``[{"name": "ATP Men", "key": "tennis_atp"}, {"name": "NBA", "key": "basketball_nba"}]``

        Remarques
        - Cette fonction combine les clés locales (``SPORTS_MAPPING``) et les
            compétitions de tennis actives récupérées via l'API.
    """
    return _get_tennis_keys() + [{"name": name, "key": key} for name, key in SPORTS_MAPPING.items()]