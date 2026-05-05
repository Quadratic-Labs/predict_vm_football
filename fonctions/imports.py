# Fichier utile pour l'import des données issues de plateformes différentes

import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from git import Repo
import pandas as pd
import json
from pathlib import Path
import pyarrow
import soccerdata as sd
import locale

def download_kaggle_dataset(dataset_query, destination_folder):
    """
    Authentifie l'utilisateur et télécharge un dataset Kaggle
    """
    # On charge son identifiant et une clé Kaggle dans le but de s'identifier et utiliser
    # l'API Kaggle
    load_dotenv(dotenv_path="../.env")

    # On définit les variables d'environnement attendues par l'API Kaggle
    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_API_TOKEN') 

    # On se connecte à l'API Kaggle
    api = KaggleApi()

    # Création du dossier si nécessaire
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    # On télécharge les données
    print(f"Téléchargement de {dataset_query} vers {destination_folder}...")
    api.dataset_download_files(
        dataset_query, 
        path=destination_folder, 
        unzip=True
    )
    print("Téléchargement correctement effectué !")



def download_github_dataset(repo_url, data_path):
    """
    Authentifie l'utilisateur et télécharge un dataset Github
    """

    # Cas 1 : Premier téléchargement des données
    if not os.path.exists(data_path):
        print("Premier téléchargement du dataset...")
        Repo.clone_from(repo_url, data_path)
        print("Téléchargement terminé !")
    # Cas 2 : Le dossier existe, on met à jour
    else:
        print("Le dossier existe déjà. Vérification des mises à jour...")
        repo = Repo(data_path)
        origin = repo.remotes.origin

        # On récupère les changements sans tout écraser
        info = origin.pull()

        if info[0].flags & info[0].HEAD_UPTODATE:
            print("Les données sont déjà à jour !")
        else:
            print("Nouvelles données récupérées avec succès.")



def download_football_data_datasets(seasons, leagues) :
    """
    Télécharge les datasets du site football-data.co.uk
    """

    # On crée un dataframe vide
    dfs = []

    # On boucle les données de chaque championnat pour obtenir toutes les données
    for league in leagues:
        for season in seasons :
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            data = pd.read_csv(url)
            data["league"] = league
            dfs.append(data)
    
    # On concatène les données de tous les championnats
    full_data = pd.concat(dfs, ignore_index=True)

    # On définit le chemin vers le dossier data
    output_path = "../data/football_data.csv"

    # On enregistre le fichier à cet endroit
    full_data.to_csv(output_path, index=False)
    print("Téléchargement terminé !")





def compile_statsbomb_to_feather(json_folder_path, output_folder_path, output_name, record_path=None, meta=None, columns_to_keep=None, recursive=False):
    """
    Compile des fichiers JSON StatsBomb (recherche récursive dans les sous-dossiers).
    """
    json_folder = Path(json_folder_path)
    output_folder = Path(output_folder_path)
    
    if recursive:
        json_files = list(json_folder.rglob("*.json"))
    else:
        json_files = list(json_folder.glob("*.json"))
    
    if not json_files:
        msg = "dans les sous-dossiers" if recursive else "à la racine"
        print(f"Aucun fichier JSON trouvé {msg} dans {json_folder_path}")
        return
    
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{output_name}.feather"

    all_dfs = []
    print(f"Traitement de {len(json_files)} fichiers trouvés dans {json_folder.name}...")

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.json_normalize(data, record_path=record_path, meta=meta, errors='ignore')
        
        if 'match_id' not in df.columns:
            df['match_id'] = file_path.stem

        if columns_to_keep:
            cols = [c for c in columns_to_keep if c in df.columns]
            df = df[cols].copy()
        
        all_dfs.append(df)

    print(f"Fusion et sauvegarde...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.columns = final_df.columns.astype(str)
    
    final_df.to_feather(output_path)
    print(f"Terminé ! Fichier : {output_path.name} ({len(final_df)} lignes)")




locale.setlocale(locale.LC_TIME, 'English_United States.1252')

def get_understat_xg(start_season, end_season, leagues=None):
    """
    Récupère les stats xG/xA depuis Understat via soccerdata.
    Colonnes disponibles : games, goals, shots, time, xG, assists, xA,
                           key_passes, yellow, red, position, npg, npxG, xGChain, xGBuildup
    """
    if leagues is None:
        leagues = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1',
                   'GER-Bundesliga', 'ITA-Serie A']

    # Understat utilise l'année de début de saison (ex: 2020 pour 2020-21)
    seasons_list = list(range(start_season, end_season + 1))

    print(f"Understat | Ligues : {leagues} | Saisons : {seasons_list}")

    try:
        understat = sd.Understat(leagues=leagues, seasons=seasons_list)
        df = understat.read_player_season_stats()

        if df is not None and not df.empty:
            df = df.reset_index()
            print(f"✓ {len(df)} lignes récupérées")
            print(f"Colonnes : {df.columns.tolist()}")
            return df
        else:
            print("✗ DataFrame vide")
            return pd.DataFrame()

    except Exception as e:
        print(f"--- ERREUR ---\n{e}")
        return pd.DataFrame()


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns]
    return df


def get_players_advanced_stats_range(start_season, end_season, leagues=['FRA-Ligue 1']):

    stat_types = [
        'standard',
        'keeper',
        'shooting',   # Contient Expected_xG, Expected_npxG, Expected_xA...
        'playing_time',
        'misc'
    ]

    seasons_list = []
    for year in range(start_season, end_season + 1):
        next_year = str(year + 1)[-2:]
        seasons_list.append(f"{str(year)[-2:]}-{next_year}")

    print(f"Ligues : {leagues} | Saisons : {seasons_list}")

    try:
        fbref = sd.FBref(leagues=leagues, seasons=seasons_list)
        all_stats = {}

        for stat in stat_types:
            print(f"  → Extraction : {stat}...")
            try:
                df = fbref.read_player_season_stats(stat_type=stat)
                if df is not None and not df.empty:
                    df = flatten_columns(df.reset_index())
                    all_stats[stat] = df

                    # Affiche les colonnes xG si présentes
                    xg_cols = [c for c in df.columns if 'xG' in c or 'xA' in c or 'npxG' in c or 'Expected' in c]
                    if xg_cols:
                        print(f"     ✓ {len(df)} lignes | Colonnes xG trouvées : {xg_cols}")
                    else:
                        print(f"     ✓ {len(df)} lignes | (pas de xG dans ce stat_type)")
                else:
                    print(f"     ✗ Vide")
            except Exception as e:
                print(f"     ✗ Erreur sur '{stat}' : {e}")

        if not all_stats:
            return pd.DataFrame()

        # Clés de fusion
        merge_keys = ['player', 'team', 'season', 'league']
        merge_keys = [k for k in merge_keys if k in all_stats['standard'].columns]

        df_final = all_stats['standard']

        for stat, df in all_stats.items():
            if stat == 'standard':
                continue
            cols_to_keep = merge_keys + [c for c in df.columns if c not in df_final.columns]
            df_final = df_final.merge(df[cols_to_keep], on=merge_keys, how='left')
            print(f"  → Fusion '{stat}' : {df_final.shape[1]} colonnes")

        # Résumé des colonnes xG dans le dataset final
        xg_final = [c for c in df_final.columns if any(x in c for x in ['xG', 'xA', 'npxG', 'Expected'])]
        print(f"\nColonnes xG dans le dataset final : {xg_final}")
        print(f"Dataset final : {df_final.shape[0]} lignes x {df_final.shape[1]} colonnes")
        return df_final

    except Exception as e:
        print(f"\n--- ERREUR CRITIQUE ---\nDétails : {e}")
        return pd.DataFrame()
