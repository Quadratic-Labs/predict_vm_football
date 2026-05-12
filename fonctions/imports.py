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
import requests
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor

def download_kaggle_dataset(dataset_query, destination_folder):
    """
    Authentifie l'utilisateur via l'API Kaggle et télécharge un jeu de données complet.

    Cette fonction automatise la récupération de données depuis Kaggle en chargeant les 
    identifiants de sécurité depuis un fichier d'environnement (.env). 
    Elle s'occupe de la création du répertoire de destination, du téléchargement et de 
    l'extraction automatique des fichiers.

    arguments:
        dataset_query (str): L'identifiant du dataset sur Kaggle.
        destination_folder (str): Le chemin du répertoire local où les fichiers seront enregistrés.

    Returns:
        None: La fonction affiche l'état d'avancement dans la console et enregistre les 
            fichiers directement sur le disque.
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
    Clone un jeu de données hébergé sur un dépôt GitHub.

    Cette fonction gère le cycle de vie du dataset local : elle effectue un 'clone' complet 
    si le dossier est absent, ou un 'pull' (mise à jour) si le dépôt existe déjà. Cela 
    permet de garantir que l'utilisateur travaille toujours avec la version la plus 
    récente des données sans avoir à supprimer et retélécharger manuellement le projet.

    arguments:
        repo_url (str): L'URL distante du dépôt GitHub (HTTPS ou SSH).
        data_path (str): Le chemin du répertoire local où le dataset doit être stocké.

    returns:
        None: La fonction gère les opérations Git en arrière-plan et affiche l'état 
            de la synchronisation (création, mise à jour ou déjà à jour).
    """

    # Premier téléchargement des données
    if not os.path.exists(data_path):
        print("Premier téléchargement du dataset...")
        Repo.clone_from(repo_url, data_path)
        print("Téléchargement terminé !")
    # Le dossier existe, on met à jour
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



def download_football_data_datasets(annee_debut, annee_fin, leagues) :
    """
    Télécharge, agrège et archive localement les données de plusieurs ligues et saisons.

    Cette fonction automatise la récupération des fichiers CSV distants pour une liste de 
    championnats et de périodes définie. Elle centralise les données dans un seul dataframe 
    en ajoutant une colonne de référence pour la ligue, puis exporte le résultat consolidé 
    au format CSV pour une utilisation hors ligne.

    arguments:
        seasons (list): Liste des codes de saisons (ex: ["2324", "2425"]).
        leagues (list): Liste des codes de championnats (ex: ["E0", "F1"]).

    returns:
        None: La fonction consolide les données en mémoire, les enregistre dans le dossier 
            '/data' et confirme la fin de l'opération dans la console.
    """
    # Génération automatique de la liste des saisons
    seasons = [f"{str(annee)[2:]}{str(annee+1)[2:]}" for annee in range(annee_debut, annee_fin)]


    mapping = {
        "GB1": "E0",
        "ES1": "SP1",
        "FR1": "F1",
        "L1": "D1",
        "IT1": "I1"
    }

    leagues = [
        mapping[league]
        for league in leagues
    ]


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






def players_filtered(players_path, appearances_path, annee_debut, annee_fin, leagues):
    """
    Filtre le fichier des joueurs pour ne conserver que les joueurs ayant évolué dans l'un
    des championnats du Big 5 sur la période étudiée.

    arguments:
        players_path (str) : Chemin vers le fichier CSV contenant les informations joueurs.
        appearances_path (str) : Chemin vers le fichier CSV contenant les apparitions en match.
        annee_debut (int) : Première saison incluse dans l'étude.
        annee_fin (int) : Dernière saison incluse dans l'étude.
        leagues (list) : Liste des identifiants de compétitions correspondant au Big 5.

    returns:
        None
            Le fichier est écrasé avec les données filtrées.
    """

    df_players = pd.read_csv(players_path)
    df_app = pd.read_csv(appearances_path)

    # Filtre temporel
    df_players = df_players[
        (df_players["last_season"] >= annee_debut) &
        (df_players["last_season"] <= annee_fin +1)
    ]

    # Joueurs Big 5
    big5_players = set(
        df_app[
            df_app["competition_id"].isin(leagues)
        ]["player_id"].unique()
    )

    # Filtre final
    df_players = df_players[
        df_players["player_id"].isin(big5_players)
    ]

    df_players.to_csv(players_path, index=False)

    print(
        f"players.csv filtré : "
        f"{len(df_players)} joueurs conservés."
    )


def valuations_filtered(valuations_path, appearances_path, annee_debut, annee_fin, leagues):
    """
    Filtre le fichier des valeurs marchandes des joueurs pour ne conserver que les
    valorisations appartenant à la période étudiée et aux périodes durant lesquelles
    les joueurs évoluent dans le Big 5.

    arguments:
        valuations_path (str) : Chemin vers le fichier CSV contenant les valorisations des
            joueurs.
        appearances_path (str) : Chemin vers le fichier CSV contenant les apparitions en match.
        annee_debut (int) : Première saison incluse dans l'étude.
        annee_fin (int) : Dernière saison incluse dans l'étude.
        leagues (list) : Liste des identifiants de compétitions correspondant au Big 5.

    returns:
        None
            Le fichier est écrasé avec les données filtrées.
    """

    date_debut = pd.Timestamp(
        year=annee_debut - 1,
        month=7,
        day=1
    )

    date_fin = pd.Timestamp(
        year=annee_fin + 1,
        month=6,
        day=30
    )


    # Lecture
    df_val = pd.read_csv(valuations_path)
    df_app = pd.read_csv(appearances_path)

    # Dates
    df_val["date"] = pd.to_datetime(df_val["date"])
    df_app["date"] = pd.to_datetime(df_app["date"])

    # Filtre temporel
    df_val = df_val[
        (df_val["date"] >= date_debut)
        & (df_val["date"] <= date_fin)
    ]

    # Apparitions Big 5
    df_big5 = df_app[
        df_app["competition_id"].isin(leagues)
    ]

    # Intervalles par joueur
    intervals = (
        df_big5
        .groupby("player_id")["date"]
        .agg(["min", "max"])
        .reset_index()
        .rename(
            columns={
                "min": "start_big5",
                "max": "end_big5"
            }
        )
    )

    # Ajout des intervalles
    df_val = df_val.merge(
        intervals,
        on="player_id",
        how="inner"
    )

    # Filtre sur l'intervalle
    df_val = df_val[
        (df_val["date"] >= df_val["start_big5"])
        & (df_val["date"] <= df_val["end_big5"])
    ]

    # Nettoyage
    df_val = df_val.drop(
        columns=["start_big5", "end_big5"]
    )

    # Sauvegarde
    df_val.to_csv(
        valuations_path,
        index=False
    )

    print(
        f"player_valuations.csv filtré : "
        f"{len(df_val)} lignes conservées."
    )




def compile_statsbomb_to_feather(json_folder_path, output_folder_path, output_name,
                                 record_path=None, meta=None, columns_to_keep=None, recursive=False):
    """
    Compile et normalise des fichiers JSON StatsBomb en un fichier feather unique.

    Cette fonction parcourt un répertoire (de manière récursive ou non) pour extraire les 
    données JSON souvent imbriquées de StatsBomb. Elle utilise 'json_normalize' pour 
    aplatir les structures complexes, injecte les identifiants de match manquants à partir 
    des noms de fichiers et optimise le stockage final au format feather pour des lectures 
    ultra-rapides lors des analyses futures.

    arguments:
        json_folder_path (str/Path): Chemin du dossier contenant les fichiers JSON.
        output_folder_path (str/Path): Chemin où sauvegarder le fichier compilé.
        output_name (str): Nom du fichier de sortie.
        record_path (str/list, optional): Chemin vers les données imbriquées dans le JSON.
        meta (list, optional): Liste des champs de métadonnées à inclure lors de la normalisation.
        columns_to_keep (list, optional): Liste restreinte de colonnes à conserver pour réduire 
                                        le poids du fichier.
        recursive (bool): Si True, cherche également dans tous les sous-dossiers. Par défaut False.

    returns:
        None: La fonction sauvegarde le DataFrame consolidé sur le disque et affiche 
            un résumé du traitement.
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

def get_understat_xg(start_season, end_season, leagues):
    """
    Récupère les statistiques avancées (xG, xA) des joueurs via le scraper Understat de Soccerdata.

    Cette fonction extrait les données de performance par saison pour les joueurs évoluant 
    dans les ligues majeures européennes. Elle permet d'accéder à des métriques avancées ainsi que
    les contributions à la création d'occasions (xGChain, xGBuildup).

    arguments:
        start_season (int): Année de début de la fenêtre (ex: 2020 pour la saison 2020-2021).
        end_season (int): Année de fin de la fenêtre (incluse).
        leagues (list, optional): Liste des championnats au format soccerdata. 
                                Par défaut : les "Big Five" européens.

    returns:
        pd.DataFrame: Un DataFrame indexé par joueur et saison contenant les statistiques 
                    détaillées. Retourne un DataFrame vide en cas d'échec ou d'absence de données.
    """
    mapping = {
        "GB1": "ENG-Premier League",
        "ES1": "ESP-La Liga",
        "FR1": "FRA-Ligue 1",
        "L1": "GER-Bundesliga",
        "IT1": "ITA-Serie A"
    }

    leagues = [
        mapping[league]
        for league in leagues
    ]

    # Understat utilise l'année de début de saison (ex: 2020 pour 2020-21)
    seasons_list = list(range(start_season, end_season + 1))

    print(f"Understat | Ligues : {leagues} | Saisons : {seasons_list}")

    try:
        understat = sd.Understat(leagues=leagues, seasons=seasons_list)
        df = understat.read_player_season_stats()

        if df is not None and not df.empty:
            df = df.reset_index()
            print(f"{len(df)} lignes récupérées")
            print(f"Colonnes : {df.columns.tolist()}")
            return df
        else:
            print("DataFrame vide")
            return pd.DataFrame()

    except Exception as e:
        print(f"--- ERREUR ---\n{e}")
        return pd.DataFrame()


def flatten_columns(df):
    """
    Aplatit les colonnes d'un dataframe.

    Cette fonction fusionne les différents niveaux d'index en une seule chaîne de caractères
    en utilisant un underscore '_' comme séparateur, facilitant ainsi l'accès aux données et
    l'exportation.

    arguments:
        df : Le dataframe.

    returns:
        dataframe: Le dataframe avec des colonnes renommées de manière linéaire 
                    (ex: ('score', 'mean') devient 'score_mean').
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns]
    return df


def get_players_advanced_stats_range(start_season, end_season, leagues):
    """
    Récupère et fusionne l'intégralité des statistiques détaillées des joueurs via FBref.

    Cette fonction extrait plusieurs types de rapports (standard, gardiens, tirs, temps de jeu,
    divers) sur une période donnée. Elle traite le formatage spécifique des saisons de FBref,
    aplatit les structures multi-indexées et réalise une jointure pour créer un profil de joueur
    incluant toutes les métriques.

    arguments:
        start_season (int): Année civile de début (ex: 2022).
        end_season (int): Année civile de fin (incluse).
        leagues (list): Liste des championnats au format soccerdata. 
                        Par défaut : ['FRA-Ligue 1'].

    returns:
        dataframe: Un dataset consolidé où chaque ligne représente un joueur par saison/équipe, 
                    regroupant l'ensemble des colonnes techniques extraites.
    """

    mapping = {
        "GB1": "ENG-Premier League",
        "ES1": "ESP-La Liga",
        "FR1": "FRA-Ligue 1",
        "L1": "GER-Bundesliga",
        "IT1": "ITA-Serie A"
    }

    leagues = [
        mapping[league]
        for league in leagues
    ]


    stat_types = [
        'standard',
        'keeper',
        'shooting',
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
            print(f"  Extraction : {stat}...")
            try:
                df = fbref.read_player_season_stats(stat_type=stat)
                if df is not None and not df.empty:
                    df = flatten_columns(df.reset_index())
                    all_stats[stat] = df

                    # Affiche les colonnes xG si présentes
                    xg_cols = [c for c in df.columns if 'xG' in c or 'xA' in c or 'npxG' in c or 'Expected' in c]
                    if xg_cols:
                        print(f"     {len(df)} lignes | Colonnes xG trouvées : {xg_cols}")
                    else:
                        print(f"     {len(df)} lignes | (pas de xG dans ce stat_type)")
                else:
                    print(f"     Vide")
            except Exception as e:
                print(f"     Erreur sur '{stat}' : {e}")

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
            print(f"  Fusion '{stat}' : {df_final.shape[1]} colonnes")

        # Résumé des colonnes xG dans le dataset final
        xg_final = [c for c in df_final.columns if any(x in c for x in ['xG', 'xA', 'npxG', 'Expected'])]
        print(f"\nColonnes xG dans le dataset final : {xg_final}")
        print(f"Dataset final : {df_final.shape[0]} lignes x {df_final.shape[1]} colonnes")
        return df_final

    except Exception as e:
        print(f"\n--- ERREUR CRITIQUE ---\nDétails : {e}")
        return pd.DataFrame()



def merge_fbref_understat(df_fbref, df_understat, output_path=None):
    """
    Nettoie, dédoublonne et fusionne les données FBref et Understat.
    
    arguments:
        df_fbref : Dataset de base (FBref)
        df_understat : Dataset contenant les xG (Understat)
        output_path (str, optional): Chemin pour sauvegarder le CSV final.
        
    returns:
        dataframe: Le dataset fusionné.
    """
    # Dédoublonnage d'Understat
    # On garde les colonnes utiles + les clés de jointure
    keys = ['player', 'team', 'league', 'season']
    cols_to_keep = keys + ['xg', 'xa', 'np_xg', 'xg_chain', 'xg_buildup']
    
    df_xg_clean = df_understat.drop_duplicates(subset=keys).copy()
    
    print(f"Dédoublonnage Understat : {len(df_understat)} -> {len(df_xg_clean)} lignes.")

    # Harmonisation des types pour les colonnes clés
    # On s'assure que la saison est en entier et les chaînes en minuscules/sans espaces
    for df in [df_fbref, df_xg_clean]:
        df['season'] = df['season'].astype(int)
        for col in ['player', 'team', 'league']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    # Fusion
    df_merged = df_fbref.merge(
        df_xg_clean[cols_to_keep],
        on=keys,
        how='left'
    )

    # Statistiques de contrôle
    coverage = df_merged['xg'].notna().mean()
    print(f"Fusion terminée : {df_merged.shape[0]} lignes et {df_merged.shape[1]} colonnes.")
    print(f"Taux de correspondance xG (Coverage) : {coverage:.1%}")

    # Sauvegarde
    if output_path:
        df_merged.to_csv(output_path, index=False, sep=',', encoding='utf-8-sig')
        print(f"Fichier sauvegardé sous : {output_path}")

    return df_merged



# Pour le chargement des données de blessures Transfermarkt

class TransfermarktArchiveScraper:
    def __init__(self, max_workers=10):
        self.base_url = "https://www.transfermarkt.fr"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.session = requests.Session()
        self.max_workers = max_workers
        self.dict_d1_annuel = {}
        self.annee_limite_basse = 2020 # Valeur par défaut

    def get_soup(self, url):
        time.sleep(0.3)
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
        except:
            return None

    def get_league_clubs(self, league_url):
        soup = self.get_soup(league_url)
        if not soup: return []
        table = soup.find("table", class_="items")
        if not table: return []
        return [self.base_url + a['href'] for a in table.select("td.zentriert.no-border-rechts a")]

    def get_club_players(self, club_url):
        soup = self.get_soup(club_url)
        if not soup: return []
        links = soup.select("td.hauptlink a")
        return list(set([self.base_url + l['href'] for l in links if 'profil/spieler' in l['href']]))

    def process_player(self, player_url):
        injury_url = player_url.replace("profil", "verletzungen")
        soup = self.get_soup(injury_url)
        if not soup: return []

        player_name = soup.find("h1").text.strip() if soup.find("h1") else "Inconnu"
        table = soup.find("table", class_="items")
        if not table or not table.find("tbody"): return []

        injuries = []
        for row in table.find("tbody").find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 5:
                saison_str = cols[0].text.strip()
                try:
                    annee_cle = 2000 + int(saison_str.split('/')[0])
                except: continue

                if annee_cle in self.dict_d1_annuel:
                    nom_club = "Non spécifié"
                    for img in row.find_all("img"):
                        if img.has_attr('alt') and "spieler" not in img.get('src', ''):
                            nom_club = img['alt']
                            break
                    
                    if any(nom_club.lower() in d.lower() for d in self.dict_d1_annuel[annee_cle]):
                        injuries.append({
                            "Nom": player_name,
                            "Club_Moment_Blessure": nom_club,
                            "Saison": saison_str,
                            "Blessure": cols[1].text.strip(),
                            "Debut": cols[2].text.strip(),
                            "Fin": cols[3].text.strip(),
                            "Jours": cols[4].text.strip(),
                            "Matchs_Manques": cols[-1].text.strip()
                        })
                
                # Utilisation de la limite basse dynamique pour arrêter le scan
                if "/" in saison_str and int(saison_str.split('/')[0]) < (self.annee_limite_basse - 2000):
                    break
        return injuries



def run_full_injury_scraping(annee_debut, annee_fin, leagues, output_file, max_threads=12):
    """
    Orchestre le scraping complet des blessures pour les ligues et années données.
    """
    # Initialisation du scraper
    scraper = TransfermarktArchiveScraper(max_workers=max_threads)
    scraper.annee_limite_basse = annee_debut
    annees = list(range(annee_debut, annee_fin + 1))
    all_player_urls = set()

    # Étape 1 : Cartographie et collecte
    print(f"Étape 1 : Cartographie historique ({annee_debut}-{annee_fin})")
    for annee in annees:
        scraper.dict_d1_annuel[annee] = []
        print(f"  Analyse de la saison {annee}/{annee+1}")
        
        for code in leagues:
            url_ligue = f"https://www.transfermarkt.fr/ligue-1/startseite/wettbewerb/{code}/plus/?saison_id={annee}"
            club_urls = scraper.get_league_clubs(url_ligue)
            
            for c_url in club_urls:
                nom_clean = c_url.split('/')[3].replace('-', ' ').title()
                scraper.dict_d1_annuel[annee].append(nom_clean)
                
                players = scraper.get_club_players(c_url)
                all_player_urls.update(players)

    print(f"Étape 2 : Scraping des blessures ({len(all_player_urls)} joueurs)")
    final_data = []
    with ThreadPoolExecutor(max_workers=scraper.max_workers) as executor:
        results = list(executor.map(scraper.process_player, list(all_player_urls)))

    for res in results:
        final_data.extend(res)

    # Étape 3 : Sauvegarde
    print("Étape 3 : Sauvegarde des données")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(final_data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"Terminé ! {len(df)} lignes enregistrées dans : {output_file}")
    return df