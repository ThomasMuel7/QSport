import os
import time
import pandas as pd
from io import StringIO
from functools import reduce
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURATION PARAMÉTRABLE DES TABLEAUX ---
# tab_btn : sélecteur de l'onglet
# table_id : ID html de la table
# home_btn / away_btn : sélecteurs Domicile/Extérieur
# sub_types : (Optionnel) Pour les boutons internes type xG (Pour/Contre)
TABS_CONFIG = {
    "General": {
        "tab_btn": "#stage-team-stats-options > li:nth-child(1) > a",
        "table_id": "statistics-team-table-summary",
        "home_btn": "#statistics-team-mini-filter-summary > div > #field > dd:nth-child(2) > a",
        "away_btn": "#statistics-team-mini-filter-summary > div > #field > dd:nth-child(3) > a",
    },
    "Defense": {
        "tab_btn": "#stage-team-stats-options > li:nth-child(2) > a",
        "table_id": "statistics-team-table-defensive",
        "home_btn": "#statistics-team-mini-filter-defensive > div > #field > dd:nth-child(2) > a",
        "away_btn": "#statistics-team-mini-filter-defensive > div > #field > dd:nth-child(3) > a",
    },
    "Attaque": {
        "tab_btn": "#stage-team-stats-options > li:nth-child(3) > a",
        "table_id": "statistics-team-table-offensive",
        "home_btn": "#statistics-team-mini-filter-offensive > div > #field > dd:nth-child(2) > a",
        "away_btn": "#statistics-team-mini-filter-offensive > div > #field > dd:nth-child(3) > a",
    },
    "xG": {
        "tab_btn": "//a[contains(text(), 'xG') or contains(text(), 'Expected Goals')]", 
        "table_id": "statistics-team-table-xg",
        "home_btn": "#statistics-team-mini-filter-xg > div > #field > dd:nth-child(2) > a",
        "away_btn": "#statistics-team-mini-filter-xg > div > #field > dd:nth-child(3) > a",
        "sub_types": {
            "Pour": "//a[contains(text(), 'Pour')]",
            "Contre": "//a[contains(text(), 'Contre') or contains(text(), 'Against')]"
        }
    },
    "TypesButs" : {
        "tab_btn": "//a[contains(text(), 'Types de Buts')]", 
        "table_id": "stage-goals-grid",
        "home_btn": "#stage-goals-filter-field > dl > dd:nth-child(3) > a",
        "away_btn": "#stage-goals-filter-field > dl > dd:nth-child(4) > a",
        "sub_types": {
            "Pour": "#stage-goals-filter-against > dl > dd:nth-child(2) > a",
            "Contre": "#stage-goals-filter-against > dl > dd:nth-child(3) > a"
        }
    },
    "TypesPasses" : {
        "tab_btn": "//a[contains(text(), 'Types de Passes')]", 
        "table_id": "stage-passes-grid",
        "home_btn": "#stage-passes-filter-field > dl > dd:nth-child(3) > a",
        "away_btn": "#stage-passes-filter-field > dl > dd:nth-child(4) > a",
        "sub_types": {
            "Pour": "#stage-passes-filter-against > dl > dd:nth-child(2) > a",
            "Contre": "#stage-passes-filter-against > dl > dd:nth-child(3) > a"
        }
    },
    "CausesCartons" : {
        "tab_btn": "//a[contains(text(), 'Causes des Cartons')]", 
        "table_id": "stage-cards-grid",
        "home_btn": "#stage-cards-filter-field > dl > dd:nth-child(3) > a",
        "away_btn": "#stage-cards-filter-field > dl > dd:nth-child(4) > a"
    },
    "CotésUtilisés" : {
        "tab_btn": "#stage-pitch-stats-options > li:nth-child(1) > a", 
        "table_id": "stage-touch-channels-grid",
        "home_btn": "#stage-touch-channels-filter-field > dl > dd:nth-child(3) > a",
        "away_btn": "#stage-touch-channels-filter-field > dl > dd:nth-child(4) > a"
    },
    "DirectionsTirs" : {
        "tab_btn": "//a[contains(text(), 'Directions des Tirs')]", 
        "table_id": "stage-attempt-directions-grid",
        "home_btn": "#stage-attempt-directions-filter-field > dl > dd:nth-child(3) > a",
        "away_btn": "#stage-attempt-directions-filter-field > dl > dd:nth-child(4) > a",
        "sub_types": {
            "Pour": "#stage-attempt-directions-filter-against > dl > dd:nth-child(2) > a",
            "Contre": "#stage-attempt-directions-filter-against > dl > dd:nth-child(3) > a"
        }
    },
    "ZoneTirs" : {
        "tab_btn": "//a[contains(text(), 'Zones de Tirs')]", 
        "table_id": "stage-attempt-zones-grid",
        "home_btn": "#stage-attempt-zones-filter-field > dl > dd:nth-child(3) > a",
        "away_btn": "#stage-attempt-zones-filter-field > dl > dd:nth-child(4) > a",
        "sub_types": {
            "Pour": "#stage-attempt-zones-filter-against > dl > dd:nth-child(2) > a",
            "Contre": "#stage-attempt-zones-filter-against > dl > dd:nth-child(3) > a"
        }
    },
    "ZoneAction" : {
        "tab_btn": "#stage-pitch-stats-options > li:nth-child(4) > a", 
        "table_id": "stage-touch-zones-grid",
        "home_btn": "#stage-touch-zones-filter-field > dl > dd:nth-child(3) > a",
        "away_btn": "#stage-touch-zones-filter-field > dl > dd:nth-child(4) > a"
    },
}

# --- STRUCTURE DES DONNÉES ---
data_structure = {
    "Ligue_1": {
        "region": 74, "tournament": 22, "slug": "france-ligue-1",
        "seasons": {
#            "2025-2026": {"s": 10792, "st": 24609},
            "2024-2025": {"s": 10329, "st": 23414},
#            "2023-2024": {"s": 9635, "st": 22105},
#            "2022-2023": {"s": 9129, "st": 21037},
#            "2021-2022": {"s": 8671, "st": 19866}
        }
    },
    # "Premier_League": {
    #     "region": 252, "tournament": 2, "slug": "angleterre-premier-league",
    #     "seasons": {
    #         "2025-2026": {"s": 10743, "st": 24533},
    #         "2024-2025": {"s": 10316, "st": 23400},
    #         "2023-2024": {"s": 9618, "st": 22076},
    #         "2022-2023": {"s": 9075, "st": 20934},
    #         "2021-2022": {"s": 8618, "st": 19793}
    #     }
    # },
    # "La_Liga": {
    #     "region": 206, "tournament": 4, "slug": "espagne-la-liga",
    #     "seasons": {
    #         "2025-2026": {"s": 10787, "st": 24594},
    #         "2024-2025": {"s": 10317, "st": 23401},
    #         "2023-2024": {"s": 9682, "st": 22176},
    #         "2022-2023": {"s": 9149, "st": 21073},
    #         "2021-2022": {"s": 8681, "st": 19895}
    #     }
    # },
    # "Bundesliga": {
    #     "region": 81, "tournament": 3, "slug": "allemagne-bundesliga",
    #     "seasons": {
    #         "2025-2026": {"s": 10720, "st": 24478},
    #         "2024-2025": {"s": 10365, "st": 23471},
    #         "2023-2024": {"s": 9649, "st": 22128},
    #         "2022-2023": {"s": 9120, "st": 21026},
    #         "2021-2022": {"s": 8667, "st": 19862}
    #     }
    # },
    # "Serie_A": {
    #     "region": 108, "tournament": 5, "slug": "italie-serie-a",
    #     "seasons": {
    #         "2025-2026": {"s": 10732, "st": 24500},
    #         "2024-2025": {"s": 10375, "st": 23490},
    #         "2023-2024": {"s": 9659, "st": 22143},
    #         "2022-2023": {"s": 9159, "st": 21087},
    #         "2021-2022": {"s": 8735, "st": 19982}
    #     }
    # }
}

DETAILED_CATEGORIES = [
    "shots", "goals", "passes", "key-passes", "assists",
    "dribbles", "tackles", "interception", "clearances", 
    "blocks", "offsides", "fouls", "cards", "saves", "possession-loss", "aerial"
]

def clean_and_parse_html(html_str, prefix):
    """Nettoie et préfixe les colonnes pour permettre un merge massif sans collision."""
    html_cleaned = html_str.replace('</span><span class="red-card-box">', '</span>-<span class="red-card-box">')
    df = pd.read_html(StringIO(html_cleaned))[0]
    
    if 'Discipline' in df.columns:
        df[['Cartons_J', 'Cartons_R']] = df['Discipline'].str.split('-', expand=True)
        df = df.drop(columns=['Discipline'])
    
    df['Équipe'] = df['Équipe'].str.replace(r'^\d+\.\s', '', regex=True)
    
    # Préfixage (ex: Domicile_General_Buts)
    cols = [f"{prefix}_{col}" if col != 'Équipe' else col for col in df.columns]
    df.columns = cols
    return df

def get_generic_tabs(driver, wait):
    all_tab_dfs = []
    
    for tab_name, cfg in TABS_CONFIG.items():
        print(f"      Traitement onglet : {tab_name}")
        
        # 1. Clic sur l'onglet (CSS ou XPath)
        selector_type = By.XPATH if cfg["tab_btn"].startswith("//") else By.CSS_SELECTOR
        tab_btn = wait.until(EC.element_to_be_clickable((selector_type, cfg["tab_btn"])))
        driver.execute_script("arguments[0].click();", tab_btn)
        time.sleep(5)

        # 2. Utilisation des sélecteurs Domicile / Extérieur de la structure
        modes = {
            "Domicile": cfg["home_btn"],
            "Exterieur": cfg["away_btn"]
        }
        
        for mode_label, mode_selector in modes.items():
            print(f"         Passage en mode : {mode_label}")
            mode_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, mode_selector)))
            driver.execute_script("arguments[0].click();", mode_btn)
            time.sleep(3)

            # Cas particulier xG avec Pour/Contre
            if "sub_types" in cfg:
                for sub_label, sub_selec in cfg["sub_types"].items():
                    print(f"            Sous-onglet : {sub_label}")
                    try:
                        s_type = By.XPATH if sub_selec.startswith("//") else By.CSS_SELECTOR
                        # On attend que le bouton soit présent SANS faire crasher tout le script s'il manque
                        sub_btn = WebDriverWait(driver, 15).until(EC.presence_of_element_located((s_type, sub_selec)))
                                
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", sub_btn)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", sub_btn)
                        time.sleep(5) # Petit temps de synchro
                                
                        table_html = driver.find_element(By.ID, cfg["table_id"]).get_attribute('outerHTML')
                        all_tab_dfs.append(clean_and_parse_html(table_html, f"{mode_label}_{tab_name}_{sub_label}"))
                    except Exception as sub_e:
                        print(f"              Saut de {sub_label} (Bouton non trouvé ou table absente)")
            else:
                # Cas General, Defense, Attaque
                html = driver.find_element(By.ID, cfg["table_id"]).get_attribute('outerHTML')
                all_tab_dfs.append(clean_and_parse_html(html, f"{mode_label}_{tab_name}"))
                
    return all_tab_dfs

def get_detailed_data(driver, wait):
    """
    On utilise les sélecteurs Domicile/Extérieur du General par défaut 
    pour naviguer dans l'onglet Detailed.
    """
    print("      Traitement onglet : Détaillé")
    dfs = []
    
    detailed_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#stage-team-stats-options > li:nth-child(5) > a")))
    driver.execute_script("arguments[0].click();", detailed_btn)
    time.sleep(4)

    home_sel = "#home"
    away_sel = "#away"
    modes = {"Domicile": home_sel, "Exterieur": away_sel}

    for mode_label, mode_selector in modes.items():
        m_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, mode_selector)))
        driver.execute_script("arguments[0].click();", m_btn)
        time.sleep(3)

        for cat in DETAILED_CATEGORIES:
            dropdown = Select(wait.until(EC.presence_of_element_located((By.ID, "category"))))
            dropdown.select_by_value(cat)
            time.sleep(4)
            
            html = driver.find_element(By.ID, "statistics-team-table-detailed").get_attribute('outerHTML')
            dfs.append(clean_and_parse_html(html, f"{mode_label}_Detailed_{cat.capitalize()}"))
            
    return dfs

def scrap_league_season(driver, url, league_name, season_name):
    wait = WebDriverWait(driver, 30)
    driver.get(url)
    
    # Cookies
    try:
        cookie_btn = wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Tout accepter')]")))
        driver.execute_script("arguments[0].click();", cookie_btn)
    except: pass

    dataframes = []
    
    # 1. Récupération des onglets pilotés par TABS_CONFIG
    dataframes.extend(get_generic_tabs(driver, wait))
    
    # 2. Récupération du Détaillé
    dataframes.extend(get_detailed_data(driver, wait))

    # Fusion de tous les tableaux collectés (Merge horizontal massif)
    if dataframes:
        print(f"--- Fusion finale pour {league_name} {season_name} ---")
        # On fusionne tout sur la colonne 'Équipe'
        df_final = reduce(lambda left, right: pd.merge(left, right, on='Équipe', how='outer'), dataframes)

        df_final = df_final.copy()

        # Nettoyage des doublons de colonnes si besoin
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]

        df_final['Championnat'] = league_name
        df_final['Saison'] = season_name

        folder = f"data/foot/{league_name}"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{season_name.replace('/', '_')}_complet.csv"
        df_final.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"  Fichier saison généré : {filename}")


        # Retourne le DataFrame pour collecte globale
        #return df_final

    return None

# --- EXECUTION ---
options = Options()
options.add_argument("--start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
combined_dfs = []

try:
    for league, config in data_structure.items():
        try:
            for season, ids in config["seasons"].items():
                url = (f"https://fr.whoscored.com/regions/{config['region']}/tournaments/{config['tournament']}/"
                       f"seasons/{ids['s']}/stages/{ids['st']}/teamstatistics/{config['slug']}-{season.replace('_', '-')}")
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                #df_season = scrap_league_season(driver, url, league, season)
                scrap_league_season(driver, url, league, season)
                #if df_season is not None:
                #    combined_dfs.append(df_season)
        finally:
            driver.quit() 
            time.sleep(5)
finally:
    driver.quit()

if combined_dfs:
    df_all = pd.concat(combined_dfs, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    filename_all = "data/foot/all_leagues_seasons_complet.csv"
    df_all.to_csv(filename_all, index=False, encoding='utf-8-sig')
    print(f"  Fichier combiné généré : {filename_all}")
else:
    print("Aucun tableau collecté. Aucun fichier combiné créé.")