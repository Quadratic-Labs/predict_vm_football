import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from git import Repo
import pandas as pd

# On charge son identifiant et une clé Kaggle dans le but de s'identifier et utiliser
# l'API Kaggle
load_dotenv(dotenv_path="../.env")

# On définit les variables d'environnement attendues par l'API Kaggle
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_API_TOKEN') 

# On se connecte à l'API Kaggle
api = KaggleApi()

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
