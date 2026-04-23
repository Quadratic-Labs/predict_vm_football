import os
import pandas as pd
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import json
import requests
from datetime import datetime




def nombre_NA_par_fichier(path):

    # On cherche le dossier dans lequel sont les bases de données
    folder = Path(path)

    # On crée un dataframe vide pour ensuite produire un résumé des données
    summary = []

    # On regarde le nombre de lignes, colonnes et NA par fichier csv

    for file in folder.glob("*.csv"):
        df = pd.read_csv(file)

        summary.append({
            "fichier": file.name,
            "lignes": len(df),
            "colonnes": len(df.columns),
            "NA": df.isna().sum().sum()
        })

    print(pd.DataFrame(summary))



