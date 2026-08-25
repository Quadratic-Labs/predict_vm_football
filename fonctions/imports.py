# Fichier utile pour l'import des données issues de plateformes différentes

import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from git import Repo
import pandas as pd
import numpy as np
import json
from pathlib import Path
import soccerdata as sd
import locale
import requests
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Désactive la propagation des erreurs internes du module logging
logging.raiseExceptions = False


# Import des données Transfermarkt

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

    returns:
        None: La fonction affiche l'état d'avancement dans la console et enregistre les 
              fichiers.
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


def players_filtered(players_path, appearances_path, annee_debut, annee_fin, ligues):
    """
    Filtre le fichier des joueurs pour ne conserver que les joueurs ayant évolué dans l'un
    des championnats du Big 5 sur la période étudiée.

    arguments:
        players_path (str) : Chemin vers le fichier CSV contenant les informations joueurs.
        appearances_path (str) : Chemin vers le fichier CSV contenant les apparitions en match.
        annee_debut (int) : Première saison incluse dans l'étude.
        annee_fin (int) : Dernière saison incluse dans l'étude.
        ligues (list) : Liste des identifiants de compétitions correspondant au Big 5.

    returns:
        None : Le fichier est écrasé avec les données filtrées.
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
            df_app["competition_id"].isin(ligues)
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


def valuations_filtered(valuations_path, appearances_path, annee_debut, annee_fin, ligues):
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
        ligues (list) : Liste des identifiants de compétitions correspondant au Big 5.

    returns:
        None : Le fichier est écrasé avec les données filtrées.
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
        df_app["competition_id"].isin(ligues)
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


# Import des données Soccerdata (Understat et FBref)

locale.setlocale(locale.LC_TIME, 'English_United States.1252')

def get_understat_data(start_season, end_season, ligues):
    """
    Récupère les statistiques avancées (xG, xA) des joueurs via le scraper Understat de Soccerdata.

    Cette fonction extrait les données de performance par saison pour les joueurs évoluant 
    dans les ligues majeures européennes.

    arguments:
        start_season (int): Année de début de la fenêtre (ex: 2020 pour la saison 2020-2021).
        end_season (int): Année de fin de la fenêtre.
        ligues (list, optional): Liste des championnats au format Soccerdata. 
                                Par défaut : le "Big Five" européens.

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

    ligues = [
        mapping[ligue]
        for ligue in ligues
    ]

    # Understat utilise l'année de début de saison (ex: 2020 pour 2020-21)
    seasons_list = list(range(start_season, end_season + 1))

    print(f"Récupération des données Understat pour les ligues : {ligues} et saisons : {seasons_list}")

    try:
        understat = sd.Understat(ligues=ligues, seasons=seasons_list)
        df = understat.read_player_season_stats()

        if df is not None and not df.empty:
            df = df.reset_index()
            print(f"{len(df)} lignes récupérées")
            return df
        else:
            print("Dataframe vide")
            return pd.DataFrame()

    except Exception as e:
        print(f"Erreur\n{e}")
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


def get_fbref_data(start_season, end_season, ligues):
    """
    Récupère et fusionne l'intégralité des statistiques détaillées des joueurs via FBref.

    Cette fonction extrait plusieurs types de rapports (standard, gardiens, tirs, temps de jeu,
    divers) sur une période donnée. Elle traite le formatage spécifique des saisons de FBref,
    aplatit les structures multi-indexées et réalise une jointure pour créer un profil de joueur
    incluant toutes les métriques.

    arguments:
        start_season (int): Année civile de début (ex: 2022).
        end_season (int): Année civile de fin (incluse).
        ligues (list): Liste des championnats au format Soccerdata.

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

    ligues = [
        mapping[ligue]
        for ligue in ligues
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

    print(f"Récupération des données FBref pour les ligues : {ligues} et saisons : {seasons_list}")

    try:
        fbref = sd.FBref(ligues=ligues, seasons=seasons_list)
        all_stats = {}

        for stat in stat_types:
            print(f"  Extraction : {stat}...")
            try:
                df = fbref.read_player_season_stats(stat_type=stat)
                if df is not None and not df.empty:
                    df = flatten_columns(df.reset_index())
                    all_stats[stat] = df
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
        print(f"Format du dataset final : {df_final.shape[0]} lignes et {df_final.shape[1]} colonnes")
        return df_final

    except Exception as e:
        print(f"\nErreur critique\nDétails : {e}")
        return pd.DataFrame()


def merge_fbref_understat(df_fbref, df_understat, output_path=None):
    """
    Nettoie, dédoublonne et fusionne les données FBref et Understat.
    
    arguments:
        df_fbref : Dataset FBref
        df_understat : Dataset Understat
        output_path (str, optional): Chemin pour sauvegarder le CSV final.
        
    returns:
        dataframe: Le dataset fusionné.
    """
    # Dédoublonnage d'Understat
    # On garde les colonnes utiles + les clés de jointure
    keys = ['player', 'team', 'league', 'season']
    cols_to_keep = keys + ['xg', 'xa', 'np_xg', 'xg_chain', 'xg_buildup']
    
    df_xg_clean = df_understat.drop_duplicates(subset=keys).copy()

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

    # Sauvegarde
    if output_path:
        df_merged.to_csv(output_path, index=False, sep=',', encoding='utf-8-sig')
        print(f"Fichier sauvegardé sous : {output_path}")

    return df_merged


# Import des données de blessures Transfermarkt

class TransfermarktInjuryScraper:

    # Slug pour construire les URLs correctement
    LEAGUE_SLUGS = {
        "FR1": "ligue-1",
        "GB1": "premier-league",
        "L1":  "bundesliga",
        "IT1": "serie-a",
        "ES1": "laliga",
    }

    def __init__(self, max_workers=10):
        self.base_url = "https://www.transfermarkt.com"
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        self.session = requests.Session()
        self.max_workers = max_workers

    def _get_soup(self, url: str, retries: int = 5):
        for attempt in range(retries):
            try:
                r = self.session.get(url, headers=self.headers, timeout=15)
                if r.status_code == 200:
                    return BeautifulSoup(r.content, "html.parser")
                elif r.status_code == 429:
                    wait = 10 + attempt * 10  # 10s, 20s, 30s...
                    print(f"      Rate limited (429), attente {wait}s...")
                    time.sleep(wait)
                else:
                    time.sleep(2 + attempt * 2)
            except Exception as e:
                time.sleep(2 + attempt * 2)
        return None
    
    def _get_soup_slow(self, url: str, retries: int = 5):
        """Version lente."""
        for attempt in range(retries):
            time.sleep(2 + attempt * 3)
            try:
                r = self.session.get(url, headers=self.headers, timeout=15)
                if r.status_code == 200:
                    return BeautifulSoup(r.content, "html.parser")
                elif r.status_code == 429:
                    wait = 15 + attempt * 15
                    print(f"      Rate limited (429), attente {wait}s...")
                    time.sleep(wait)
            except Exception as e:
                print(f"      Erreur réseau (tentative {attempt+1}): {e}")
        return None

    def get_clubs_for_season(self, league_code: str, season_year: int) -> list:
        slug = self.LEAGUE_SLUGS.get(league_code, league_code.lower())
        url = (
            f"{self.base_url}/{slug}/startseite/wettbewerb"
            f"/{league_code}/plus/?saison_id={season_year}"
        )
        soup = self._get_soup_slow(url)
        
        # DEBUG
        if not soup:
            print(f"      [{league_code}] Pas de réponse HTTP pour : {url}")
            return []

        table = soup.find("table", class_="items")
        if not table:
            # Affiche les 500 premiers caractères du HTML reçu pour diagnostiquer
            print(f"         [{league_code}] Table 'items' introuvable. URL: {url}")
            print(f"         HTML reçu (extrait) : {str(soup)[:500]}")
            return []

        clubs = []
        for a in table.select("td.hauptlink a"):
            href = a.get("href", "")
            if "/startseite/verein/" in href:
                club_id = href.split("/verein/")[1].split("/")[0]
                clubs.append({
                    "name":     a.text.strip(),
                    "club_id":  club_id,
                    "slug":     href.split("/")[1],
                    "season":   season_year,
                    "league":   league_code,
                })

        # DEBUG
        if not clubs:
            print(f"      [{league_code}] Table trouvée mais aucun lien td.hauptlink. URL: {url}")
            # Affiche les liens trouvés dans la table pour voir la vraie structure
            all_links = table.select("a")[:5]
            for lnk in all_links:
                print(f"         Lien trouvé : {lnk.get('href', '')} | texte: {lnk.text.strip()}")

        return clubs

    def get_players_for_club_season(self, club_slug: str, club_id: str, season_year: int) -> list:
        url = (
            f"{self.base_url}/{club_slug}/kader/verein/{club_id}"
            f"/saison_id/{season_year}/plus/1"
        )
        soup = self._get_soup_slow(url)
        if not soup:
            return []

        players = []
        seen = set()

        for a in soup.select("table.items td.hauptlink a"):
            href = a.get("href", "")
            if "/profil/spieler/" not in href:
                continue
            player_id = href.split("/spieler/")[1].split("/")[0]
            if player_id in seen:
                continue
            seen.add(player_id)

            slug = href.split("/")[1]
            players.append({
                "player_id":   player_id,
                "player_slug": slug,
                "name":        a.text.strip(),
                "injury_url":  (
                    f"{self.base_url}/{slug}"
                    f"/verletzungen/spieler/{player_id}"
                ),
            })
        return players

    def get_player_injuries(self, player: dict) -> list:
        soup = self._get_soup(player["injury_url"])
        if not soup:
            return []

        table = soup.find("table", class_="items")
        if not table or not table.find("tbody"):
            return []

        injuries = []
        for row in table.find("tbody").find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 6:
                continue

            # Club au moment de la blessure : image dans la dernière colonne avant les stats
            club_moment = "Non spécifié"
            for img in row.find_all("img"):
                src = img.get("src", "")
                alt = img.get("alt", "")
                # Les logos de clubs ont une URL différente des photos joueurs
                if "vereins" in src or ("wappen" in src):
                    club_moment = alt
                    break
                # Fallback : toute image qui n'est pas un joueur
                if alt and "spieler" not in src and "player" not in src:
                    club_moment = alt
                    break

            injuries.append({
                "player_id":        player["player_id"],
                "Nom":              player["name"],
                "Club_Blessure":    club_moment,
                "Saison":           cols[0].text.strip(),
                "Blessure":         cols[1].text.strip(),
                "Debut":            cols[2].text.strip(),
                "Fin":              cols[3].text.strip(),
                "Jours":            cols[4].text.strip(),
                "Matchs_Manques":   cols[5].text.strip(),
            })
        return injuries


def scraping_blessures_transfermarkt(annee_debut: int, annee_fin: int, output_file: str, ligues=None,
    max_threads: int = 10) -> pd.DataFrame:

    if ligues is None:
        ligues = {"FR1", "GB1", "L1", "IT1", "ES1"}
    if isinstance(ligues, dict):
        league_codes = set(ligues.values())
    else:
        league_codes = set(ligues)

    scraper = TransfermarktInjuryScraper(max_workers=max_threads)
    annees  = range(annee_debut, annee_fin + 1)

    print(f"Cartographie clubs et joueurs ({annee_debut} à {annee_fin})")

    all_players: dict = {}
    valid_pairs: set  = set()

    for annee in annees:
        saison_str = f"{annee % 100:02d}/{(annee + 1) % 100:02d}"
        print(f"\n  Saison {annee}/{annee+1}")

        for code in league_codes:
            clubs = scraper.get_clubs_for_season(code, annee)
            print(f"    [{code}] {len(clubs)} clubs trouvés")

            for club in clubs:
                players = scraper.get_players_for_club_season(
                    club["slug"], club["club_id"], annee
                )
                for p in players:
                    pid = p["player_id"]
                    if pid not in all_players:
                        all_players[pid] = p
                    valid_pairs.add((pid, saison_str))

    print(f"\n  Cartographie terminée : {len(all_players)} joueurs uniques et {len(valid_pairs)} paires joueur/saison")


    print(f"Scraping des blessures de {len(all_players)} joueurs")

    with ThreadPoolExecutor(max_workers=scraper.max_workers) as executor:
        results = list(executor.map(scraper.get_player_injuries, list(all_players.values())))

    all_injuries = [inj for res in results for inj in res]
    print(f"{len(all_injuries)} blessures brutes récupérées")

    # Étape 4 : filtrage sur les saisons D1
    def saison_courte(s: str) -> str:
        parts = s.split("/")
        if len(parts) == 2:
            return f"{int(parts[0]) % 100:02d}/{parts[1][-2:]}"
        return s

    filtered = [
        inj for inj in all_injuries
        if (inj["player_id"], saison_courte(inj["Saison"])) in valid_pairs
    ]
    print(f"{len(filtered)} blessures dans les championnats du Big 5 après filtrage sur les saisons")


    print(f"Sauvegardes à l'emplacement {output_file}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # On crée le dataframe
    df = pd.DataFrame(filtered)
    
    # Si la colonne player_id existe, on la réordonne pour la placer tout au début
    if "player_id" in df.columns:
        cols = ["player_id"] + [col for col in df.columns if col != "player_id"]
        df = df[cols]
    
    # Temporairement, on remplace "Non spécifié" par NaN
    df['Club_Blessure'] = df['Club_Blessure'].replace('Non spécifié', np.nan)

    # On trie le DataFrame pour mettre les vrais clubs en premier au sein de chaque groupe.
    # Ainsi, les valeurs valides se retrouveront au-dessus des NaN.
    df = df.sort_values(by=['player_id', 'Saison', 'Club_Blessure'], na_position='last')

    # On propage le vrai club vers le bas et vers le haut au sein de chaque 
    # groupe {Joueur + Saison}
    df['Club_Blessure'] = df.groupby(['player_id', 'Saison'])['Club_Blessure'].transform(lambda x: x.ffill().bfill())

    # S'il reste des joueurs qui n'avaient aucun autre club cette saison-là, on remet
    # "Non spécifié" pour ne pas laisser de NaN
    df['Club_Blessure'] = df['Club_Blessure'].fillna('Non spécifié')

    # On remet le dataframe dans son ordre initial
    df = df.sort_index()
        
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"  {len(df)} lignes enregistrées.")
    return df


# Import des données football-data.co.uk

def download_football_data_datasets(annee_debut: int, annee_fin: int, ligues: list):
    """
    Télécharge, agrège et archive localement les données de plusieurs ligues et saisons.
    """
    # Génération automatique de la liste des saisons
    seasons = [f"{str(annee)[2:]}{str(annee+1)[2:]}" for annee in range(annee_debut, annee_fin + 1)]

    mapping = {
        "GB1": "E0",
        "ES1": "SP1",
        "FR1": "F1",
        "L1": "D1",
        "IT1": "I1"
    }

    ligues = [
        mapping[league]
        for league in ligues
    ]

    # On crée une liste vide pour stocker les DataFrames
    dfs = []

    # On boucle les données de chaque championnat pour obtenir toutes les données
    for league in ligues:
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            
            try:
                # On lit le CSV normalement
                data = pd.read_csv(url)

                # On ajoute une colonne pour indiquer le championnat
                data = data.copy()
                data["league"] = league
                
                dfs.append(data)
                print(f"Téléchargement réussi pour {league} saison {season} : {len(data)} lignes")
            except Exception as e:
                print(f"Impossible de télécharger {league} pour la saison {season} : {e}")
    
    if not dfs:
        print("Aucune donnée n'a pu être téléchargée.")
        return

    # On concatène les données de tous les championnats
    full_data = pd.concat(dfs, ignore_index=True)

    # On définit le chemin vers le dossier data
    output_path = "../data/football_data.csv"

    # On enregistre le fichier à cet endroit
    full_data.to_csv(output_path, index=False)
    print("Téléchargement terminé !")



# Import des données du classement FIFA

def extraire_classement_fin_saison(
    df_fifa: pd.DataFrame, nombre_equipes: int = 10, annee_debut: int = 2020, annee_fin: int = None
):
    """Prend en paramètre un DataFrame brut du classement FIFA, calcule les fins de saisons 
    (septembre à août) entre l'année de début et l'année de fin spécifiées, et retourne le Top X mondial
    de fin de saison.
    """
    # Copie locale pour éviter de modifier le DataFrame d'origine
    df = df_fifa.copy()

    # Conversion de la date
    df["rank_date"] = pd.to_datetime(df["rank_date"])

    date_debut_limite = f"{annee_debut}-09-01"
    df = df[df["rank_date"] >= date_debut_limite].copy()

    if annee_fin is not None:
        date_fin_limite = f"{int(annee_fin) + 1}-08-31"
        df = df[df["rank_date"] <= date_fin_limite].copy()

    # Fonction interne pour attribuer la bonne saison
    def determiner_saison(date):
        if date.month >= 9:
            return f"{date.year}-{date.year + 1}"
        else:
            return f"{date.year - 1}-{date.year}"

    # Application des calculs de saisons
    df["saison"] = df["rank_date"].apply(determiner_saison)
    df["season_year"] = df["rank_date"].apply(
        lambda d: d.year if d.month >= 9 else d.year - 1
    )

    # Tri par date pour mettre la plus récente d'une saison à la fin
    df = df.sort_values(by="rank_date")

    # On regroupe par saison et par pays, et on garde la dernière ligne disponible
    df_fin_saison = df.drop_duplicates(
        subset=["saison", "country_full"], keep="last"
    ).copy()

    # Tri final pour une lecture propre
    df_fin_saison = df_fin_saison.sort_values(by=["season_year", "rank"])

    # Filtre dynamique du nombre d'équipes
    df_fin_saison = df_fin_saison[df_fin_saison["rank"] <= nombre_equipes]

    df_fin_saison.to_csv(r'..\data\classement_fifa\fifa_ranking_fin_saison.csv', index=False, sep=',', encoding='utf-8-sig')

    return df_fin_saison





# ANNEXES

# Import des données Statsbomb
    
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
