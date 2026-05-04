import pandas as pd
import os
from pathlib import Path
import re

def merge_csvs_from_leagues():
    """
    Récupère et fusionne tous les fichiers CSV des dossiers La_Liga, Ligue_1 et Premier_League.
    Retourne un DataFrame fusionné et l'exporte en CSV.
    """
    # Définir les dossiers des ligues
    leagues_folders = ['data/Ligue_1', 'data/Premier_League', 'data/Bundesliga', 'data/La_liga', 'data/Serie_A']
    ranking_columns = [
        'Domicile_TypesButs_Pour_R',
        'Domicile_TypesButs_Contre_R',
        'Exterieur_TypesButs_Pour_R',
        'Exterieur_TypesButs_Contre_R',
        'Domicile_TypesPasses_Pour_R',
        'Domicile_TypesPasses_Contre_R',
        'Exterieur_TypesPasses_Pour_R',
        'Exterieur_TypesPasses_Contre_R',
        'Domicile_CausesCartons_R',
        'Exterieur_CausesCartons_R',
        'Domicile_CotésUtilisés_R',
        'Exterieur_CotésUtilisés_R',
        'Domicile_DirectionsTirs_Pour_R',
        'Domicile_DirectionsTirs_Contre_R',
        'Exterieur_DirectionsTirs_Pour_R',
        'Exterieur_DirectionsTirs_Contre_R',
        'Domicile_ZoneTirs_Pour_R',
        'Domicile_ZoneTirs_Contre_R',
        'Exterieur_ZoneTirs_Pour_R',
        'Exterieur_ZoneTirs_Contre_R',
        'Domicile_ZoneAction_R',
        'Exterieur_ZoneAction_R',
    ]

    # Vérifier le répertoire de travail courant
    base_path = Path.cwd()
    
    dataframes = []
    
    def _is_percent_col_by_header(col_name: str) -> bool:
        s = col_name.lower()
        tokens = ['%', 'pct', 'percent', 'pourcent', 'pourcentage']
        return any(t in s for t in tokens)

    def _normalize_percent_series(s: pd.Series) -> pd.Series:
        # Work on a copy
        s2 = s.astype(str).str.strip()
        # Replace comma with dot for decimals
        s2 = s2.str.replace(',', '.', regex=False)
        # Remove percentage sign and whitespace
        has_percent = s2.str.contains('%', na=False)
        # Remove % symbol
        s_clean = s2.str.replace('%', '', regex=False)
        # Convert to numeric where possible
        num = pd.to_numeric(s_clean, errors='coerce')
        # If originally had '%' then divide by 100
        if has_percent.any():
            num = num / 100.0
        else:
            # If numeric and values seem in 0-100 range, scale to 0-1
            if pd.notna(num).any():
                mx = num.max()
                mn = num.min()
                if mx is not None and mx > 1.001:
                    num = num / 100.0
        # Arrondir pour éviter les artefacts flottants (3 décimales)
        try:
            num = num.round(3)
        except Exception:
            pass
        return num

    def _detect_percent_columns(df: pd.DataFrame):
        header_cols = []
        cell_cols = []
        for col in df.columns:
            if _is_percent_col_by_header(col):
                header_cols.append(col)
                continue
            # check sample values for '%' sign
            try:
                sample = df[col].dropna().astype(str).head(2000)
                if sample.str.contains('%', na=False).any():
                    cell_cols.append(col)
            except Exception:
                continue
        return list(set(header_cols + cell_cols))

    # Parcourir chaque dossier de ligue
    for folder in leagues_folders:
        folder_path = base_path / folder
        
        if not folder_path.exists():
            print(f"   Le dossier {folder} n'existe pas.")
            continue
        
        print(f"📂 Lecture des fichiers dans {folder}...")
        
        # Trouver tous les fichiers CSV dans le dossier
        csv_files = list(folder_path.glob('*.csv'))
        
        if not csv_files:
            print(f"      Aucun fichier CSV trouvé dans {folder}")
            continue
        
        # Charger et ajouter chaque CSV au DataFrame
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Normaliser les colonnes en pourcentage en valeurs entre 0 et 1
                pct_cols = _detect_percent_columns(df)
                if pct_cols:
                    print(f"     Normalisation des colonnes % pour {csv_file.name}: {pct_cols}")
                    for c in pct_cols:
                        try:
                            df[c] = _normalize_percent_series(df[c])
                        except Exception as e:
                            print(f"       Erreur normalisation colonne {c}: {e}")

                dataframes.append(df)
                print(f"     {csv_file.name} chargé ({len(df)} lignes)")
            except Exception as e:
                print(f"     Erreur lors du chargement de {csv_file.name}: {e}")
    
    if not dataframes:
        print("  Aucun fichier CSV n'a pu être chargé.")
        return None
    
    # Fusionner tous les DataFrames
    print("\n  Fusion des fichiers...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"  Fusion terminée: {len(merged_df)} lignes au total")
    print(f"  Colonnes: {list(merged_df.columns)}")

    cols_to_drop = [col for col in ranking_columns if col in merged_df.columns]
    if cols_to_drop:
        merged_df = merged_df.drop(columns=cols_to_drop)
        print(f"  Colonnes de classement supprimées: {cols_to_drop}")
    
    # Exporter le résultat
    output_path = base_path / 'data' / 'whoScored.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"\n  Fichier fusionné exporté: {output_path}")
    
    return merged_df


if __name__ == "__main__":
    df = merge_csvs_from_leagues()
    if df is not None:
        print(f"\nAperçu des données:\n{df.head()}")
