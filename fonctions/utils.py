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