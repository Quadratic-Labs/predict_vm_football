import os
import pandas as pd
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
import json
import requests
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



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














def analyze_fuzzy_duplicates_fast(folder_path, threshold=0.8, window=5):
    """
    Détecte les doublons partiels rapidement.
    Compare chaque ligne avec les 'window' lignes suivantes après tri.
    """
    folder = Path(folder_path)
    files = list(folder.glob("*.csv")) + list(folder.glob("*.feather"))
    
    if not files:
        print(f"Aucun fichier trouvé dans {folder_path}")
        return

    print(f"\nAnalyse floue rapide (Seuil: {threshold*100}%, Voisinage: {window})")
    print("-" * 75)
    print(f"{'Fichier':<35} | {'Lignes':<8} | {'Doublons':<8} | {'Taux %':<8}")
    print("-" * 75)

    for file in files:
        df = pd.read_csv(file) if file.suffix == '.csv' else pd.read_feather(file)
        
        if len(df) <= 1:
            continue

        # On trie pour rapprocher les doublons potentiels
        # (On prend les 3 premières colonnes pour le tri par défaut)
        df = df.sort_values(by=list(df.columns[:3])).reset_index(drop=True)
        
        values = df.values
        n_rows, n_cols = values.shape
        req_matches = int(threshold * n_cols)
        is_duplicate = np.zeros(n_rows, dtype=bool)

        # On ne compare que dans un petit voisinage (window)
        for i in range(n_rows):
            if is_duplicate[i]: continue
            
            # Comparaison limitée aux 'window' lignes suivantes
            end_idx = min(i + 1 + window, n_rows)
            if i + 1 >= end_idx: continue
            
            matches = np.sum(values[i] == values[i+1 : end_idx], axis=1)
            dup_indices = np.where(matches >= req_matches)[0] + (i + 1)
            
            if len(dup_indices) > 0:
                is_duplicate[dup_indices] = True
        
        n_dup = np.sum(is_duplicate)
        rate = (n_dup / n_rows) * 100
        print(f"{file.name:<35} | {n_rows:<8} | {n_dup:<8} | {rate:>6.2f}%")

    print("-" * 75)








def analyze_temporal_coverage(folder_path):
    """
    Analyse les colonnes de type 'date' pour chaque fichier CSV et Feather.
    Affiche la date de début (min) et de fin (max) pour chaque colonne identifiée.
    """
    folder = Path(folder_path)
    files = list(folder.glob("*.csv")) + list(folder.glob("*.feather"))
    
    if not files:
        print(f"Aucun fichier trouvé dans {folder_path}")
        return

    print(f"\nCouverture temporelle des datasets")
    print("-" * 60)

    for file in files:
        # Lecture adaptée
        if file.suffix == '.csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_feather(file)
            
        # Détection des colonnes contenant "date" dans leur nom
        date_cols = [col for col in df.columns if "date" in col.lower()]
        
        if not date_cols:
            continue
            
        print(f"\n{file.name}")
        
        for col in date_cols:
            # Conversion en datetime (coerce transforme les erreurs en NaT)
            # On utilise copy() pour éviter les warnings de Pandas
            temp_dates = pd.to_datetime(df[col], errors="coerce")
            
            # Suppression des NaT pour le calcul
            temp_dates = temp_dates.dropna()
            
            if not temp_dates.empty:
                min_date = temp_dates.min().strftime('%d/%m/%Y')
                max_date = temp_dates.max().strftime('%d/%m/%Y')
                print(f"   • {col:<20} : {min_date} ➔ {max_date}")
            else:
                print(f"   • {col:<20} : Aucune date valide trouvée")

    print("-" * 60)








def analyze_seasonal_coverage(folder_path):
    """
    Analyse la répartition des observations par saison (Juillet à Juin).
    Compatible CSV et Feather.
    """
    folder = Path(folder_path)
    files = list(folder.glob("*.csv")) + list(folder.glob("*.feather"))
    
    if not files:
        print(f"Aucun fichier trouvé dans {folder_path}")
        return

    print(f"\nRépartition par Saisons (01/07 au 30/06)")
    print("-" * 60)

    for file in files:
        # Lecture
        df = pd.read_csv(file) if file.suffix == '.csv' else pd.read_feather(file)
        
        # Identification des colonnes dates
        date_cols = [col for col in df.columns if "date" in col.lower()]
        
        if not date_cols:
            continue
            
        print(f"\nFichier: {file.name}")
        
        for col in date_cols:
            # Conversion en datetime
            dates = pd.to_datetime(df[col], errors="coerce").dropna()
            
            if dates.empty:
                continue

            # LOGIQUE DE SAISON : 
            # Si le mois est >= 7 (Juillet), la saison est Année / Année+1
            # Si le mois est < 7 (Janv-Juin), la saison est Année-1 / Année
            def get_season(d):
                year = d.year
                if d.month >= 7:
                    return f"{year}/{year+1}"
                else:
                    return f"{year-1}/{year}"

            # Application de la fonction
            seasons = dates.apply(get_season)
            
            # Affichage des résultats triés par saison
            counts = seasons.value_counts().sort_index()
            print(f"  Colonne: {col}")
            for season, count in counts.items():
                print(f"    • {season} : {count} observations")

    print("-" * 60)







def get_kaggle_dataset_last_update(dataset_query, env_path="../.env"):
    """
    Se connecte à l'API Kaggle et récupère la date de dernière mise à jour 
    d'un dataset spécifique passé en argument.
    """
    # 1. Chargement des identifiants
    load_dotenv(dotenv_path=env_path)
    
    username = os.getenv('KAGGLE_USERNAME')
    api_token = os.getenv('KAGGLE_API_TOKEN')

    if not username or not api_token:
        print("Erreur : Identifiants Kaggle manquants dans le fichier .env")
        return None

    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_API_TOKEN'] = api_token

    # 2. Authentification
    try:
        api = KaggleApi()
    except Exception as e:
        print(f"Erreur d'authentification Kaggle : {e}")
        return None

    # 3. Recherche du dataset exact
    # On utilise l'API pour l'affichage de la fiche du dataset
    try:
        datasets = api.dataset_list(search=dataset_query)
        
        for ds in datasets:
            if ds.ref == dataset_query:
                # Récupération de la date (gestion des noms d'attributs)
                date_maj = getattr(ds, 'lastUpdated', None) or getattr(ds, 'last_updated', "Date inconnue")
                
                print(f"Dataset trouvé : {ds.ref}")
                print(f"Dernière mise à jour : {date_maj}")
                return date_maj

        print(f"Dataset '{dataset_query}' non trouvé sur Kaggle.")
        return None
        
    except Exception as e:
        print(f"Erreur lors de la recherche du dataset : {e}")
        return None











def plot_transfermarkt_quality(df_players, df_clubs=None):
    df_players = df_players.copy()
    
    # 1. Calcul de l'âge
    df_players['date_of_birth'] = pd.to_datetime(df_players['date_of_birth'], errors='coerce')
    df_players['age'] = pd.Timestamp.now().year - df_players['date_of_birth'].dt.year

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3)

    # --- GRAPH 1 & 2 : Valeurs ---
    sns.histplot(df_players['market_value_in_eur'].dropna(), kde=True, ax=axes[0, 0], color='blue')
    sns.histplot(np.log1p(df_players['market_value_in_eur'].dropna()), kde=True, ax=axes[0, 1], color='green')
    axes[0, 0].set_title("Distribution Valeur Marchande")
    axes[0, 1].set_title("Distribution (Log Scale)")

    # --- GRAPH 3 : Âge ---
    sns.histplot(df_players['age'].dropna(), bins=20, kde=True, ax=axes[1, 0], color='orange')
    axes[1, 0].set_title("Répartition par Âge")



    # --- GRAPH 4 : Boxplot Big 5 (version finale propre) ---
    if df_clubs is not None:
        
        # 🔹 Identification des colonnes
        col_club_p = 'current_club_id' if 'current_club_id' in df_players.columns else 'club_id'
        col_club_c = 'club_id'
        col_comp = [c for c in df_clubs.columns if 'competition' in c and 'id' in c][0]
        
        # 🔹 Merge joueurs + clubs
        temp_df = df_players.merge(
            df_clubs[[col_club_c, col_comp]], 
            left_on=col_club_p, 
            right_on=col_club_c, 
            how='left'
        )
        
        # 🔹 IDs du Big 5
        big5_ids = ['GB1', 'ES1', 'L1', 'IT1', 'FR1']
        
        # 🔹 Filtrage Big 5
        df_big5 = temp_df[temp_df[col_comp].isin(big5_ids)].copy()
        
        # 🔹 Mapping IDs → noms lisibles
        league_names = {
            'GB1': 'Premier League',
            'ES1': 'La Liga',
            'L1': 'Bundesliga',
            'IT1': 'Serie A',
            'FR1': 'Ligue 1'
        }
        
        df_big5['league_name'] = df_big5[col_comp].map(league_names)
        
        # 🔹 Ordre logique des ligues
        order = [
            'Premier League',
            'La Liga',
            'Bundesliga',
            'Serie A',
            'Ligue 1'
        ]
        
        # 🔹 Plot
        sns.boxplot(
            data=df_big5,
            x='league_name',
            y='market_value_in_eur',
            ax=axes[1, 1],
            order=order
        )
        
        axes[1, 1].set_title("Prix par ligue (Big 5)")
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xlabel("Ligue")
        axes[1, 1].set_ylabel("Valeur marchande (€)")
        
        plt.setp(axes[1, 1].get_xticklabels(), rotation=45)





def check_all_missing_values(df, title="Valeurs manquantes par colonne"):
    """
    Affiche le % de valeurs manquantes pour toutes les colonnes du DataFrame
    qui contiennent au moins un NaN.
    """
    # Calcul du % de manquants pour chaque colonne
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    # On ne garde que les colonnes qui ont des manquants (> 0)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if missing_pct.empty:
        print(f"✅ Aucune valeur manquante détectée dans le dataset : {title}")
        return

    plt.figure(figsize=(15, 6))
    sns.barplot(x=missing_pct.index, y=missing_pct.values, palette="Reds_r")
    
    plt.title(f"{title} (%)", fontsize=14)
    plt.ylabel("% de NaN")
    plt.xticks(rotation=45, ha='right')
    
    # Seuil d'alerte à 20%
    plt.axhline(y=20, color='black', linestyle='--', label="Seuil critique (20%)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()