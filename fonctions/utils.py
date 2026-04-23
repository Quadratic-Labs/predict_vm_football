import os
import pandas as pd
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import json
import requests
from datetime import datetime
import numpy as np



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





import pandas as pd
from pathlib import Path

def analyze_missing_data(folder_path, threshold=10):
    """
    Analyse les fichiers CSV et FEATHER d'un dossier.
    Identifie les colonnes ayant un taux de NA supérieur au seuil.
    """
    folder = Path(folder_path)
    summary = []
    total_cols = {}

    # 1. On cherche les deux extensions
    files = list(folder.glob("*.csv")) + list(folder.glob("*.feather"))
    
    if not files:
        print(f"Aucun fichier CSV ou Feather trouvé dans {folder_path}")
        return None

    for file in files:
        # 2. Lecture adaptée selon l'extension
        if file.suffix == '.csv':
            df = pd.read_csv(file)
        elif file.suffix == '.feather':
            df = pd.read_feather(file)
        
        total_rows = len(df)
        total_cols[file.name] = len(df.columns)
        
        # 3. Calcul des NA
        for col in df.columns:
            na_count = df[col].isna().sum()
            na_rate = (na_count / total_rows) * 100 if total_rows > 0 else 0
            
            if na_rate > threshold:
                summary.append({
                    "fichier": file.name,
                    "format": file.suffix.replace('.', ''),
                    "colonne": col,
                    "lignes": total_rows,
                    "NA": na_count,
                    "taux_NA_%": round(na_rate, 2)
                })

    if not summary:
        print(f"Aucune colonne n'a plus de {threshold}% de NA dans les {len(files)} fichiers.")
        return None

    # 4. Traitement des résultats
    result = pd.DataFrame(summary)
    
    print(f"\nSynthèse (> {threshold}% de NA) :")
    count_per_file = result.groupby("fichier")["colonne"].count()
    for fichier, count in count_per_file.items():
        print(f"  - {fichier} : {count}/{total_cols[fichier]}")

    # Tri et affichage détaillé
    result_sorted = result.sort_values(by=["fichier", "taux_NA_%"], ascending=[True, False])
    for fichier, df_file in result_sorted.groupby("fichier"):
        fmt = df_file['format'].iloc[0]
        print(f"\nFichier: {fichier} [Format: {fmt.upper()}]")
        print(df_file[["colonne", "lignes", "NA", "taux_NA_%"]].to_string(index=False))

    return result_sorted






def analyze_fuzzy_duplicates(folder_path, similarity_threshold=0.6):
    """
    Analyse les doublons partiels dans les fichiers CSV et Feather.
    Un doublon est détecté si deux lignes partagent au moins 'similarity_threshold' % de colonnes identiques.
    """
    folder = Path(folder_path)
    summary_duplicates = []

    files = list(folder.glob("*.csv")) + list(folder.glob("*.feather"))
    
    if not files:
        print(f"Aucun fichier trouvé dans {folder_path}")
        return None

    for file in files:
        # Lecture
        if file.suffix == '.csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_feather(file)
            
        total_rows = len(df)
        if total_rows <= 1:
            summary_duplicates.append({
                "fichier": file.name, "lignes": total_rows, "doublons_partiels": 0, "taux_%": 0
            })
            continue

        # --- Logique de détection de doublons partiels ---
        # On convertit le DF en matrice de valeurs
        values = df.values
        n_rows, n_cols = values.shape
        
        # Seuil de colonnes identiques nécessaires
        required_matches = int(similarity_threshold * n_cols)
        
        duplicates_count = 0
        is_duplicate = np.zeros(n_rows, dtype=bool)

        # Comparaison ligne à ligne optimisée (attention : lent sur de très gros fichiers)
        for i in range(n_rows):
            if is_duplicate[i]: continue
            
            # On compare la ligne i avec toutes les lignes suivantes
            # On compte combien de colonnes sont égales (en gérant les NaN)
            matches = np.sum(values[i] == values[i+1:], axis=1)
            
            # On identifie les indices des lignes qui dépassent le seuil
            dup_indices = np.where(matches >= required_matches)[0] + (i + 1)
            
            if len(dup_indices) > 0:
                is_duplicate[dup_indices] = True
        
        n_duplicates = np.sum(is_duplicate)
        dup_rate = (n_duplicates / total_rows) * 100
        
        summary_duplicates.append({
            "fichier": file.name,
            "lignes": total_rows,
            "doublons_partiels": n_duplicates,
            "taux_doublons_%": round(dup_rate, 2)
        })

    result_dup = pd.DataFrame(summary_duplicates).sort_values(by="taux_doublons_%", ascending=False)
    
    print(f"\nAnalyse des doublons partiels (Seuil: {similarity_threshold*100}%) :\n")
    print(result_dup.to_string(index=False))
    
    return result_dup