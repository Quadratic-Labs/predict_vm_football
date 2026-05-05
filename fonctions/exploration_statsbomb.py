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
import matplotlib.patches as mpatches
import unicodedata




def audit_statsbomb_data(df_matches, df_lineups):
    """
    Analyse exploratoire des données StatsBomb : NA, doublons et cohérence de liaison.
    """
    # 1. Analyse des doublons
    match_dups = df_matches.duplicated(subset=['match_id']).sum()
    # Pour les lineups, un doublon est un même joueur deux fois dans le même match
    lineup_dups = df_lineups.duplicated(subset=['match_id', 'player_id']).sum()
    
    # 2. Analyse des NA (Focus sur les colonnes critiques)
    na_matches = df_matches[['match_id', 'home_team.home_team_name', 'away_team.away_team_name']].isnull().sum()
    na_lineups = df_lineups[['match_id', 'player_id', 'player_name']].isnull().sum()
    
    # 3. Vérification de la liaison (Merge Check)
    matches_in_lineups = df_lineups['match_id'].unique()
    missing_matches = df_matches[~df_matches['match_id'].isin(matches_in_lineups)]
    
    # --- AFFICHAGE ---
    print(f"AUDIT STATSBOMB")
    print(f"-------------------------------------------")
    print(f"• Matchs   : {len(df_matches)} lignes | Doublons : {match_dups}")
    print(f"• Lineups  : {len(df_lineups)} lignes | Doublons : {lineup_dups}")
    print(f"• Intégrité: {len(missing_matches)} match(s) sans composition d'équipe")
    print(f"-------------------------------------------")
    
    if not na_matches.any() and not na_lineups.any():
        print("Aucune valeur manquante sur les identifiants clés.")
    else:
        print("Colonnes avec NA détectées :")
        print(na_matches[na_matches > 0])
        print(na_lineups[na_lineups > 0])
        
    print(f"\nTop 5 Compétitions (StatsBomb) :")
    print(df_matches['competition.competition_name'].value_counts().head(5))
    print("-" * 43)

    return {"missing_ids": missing_matches['match_id'].tolist()}



def check_statsbomb_temporal_coverage(df_matches):
    """
    Analyse la fenêtre temporelle des matchs StatsBomb à partir du 1er juillet 2020.
    """
    if 'match_date' not in df_matches.columns:
        print("Colonne 'match_date' introuvable.")
        return

    # 1. Conversion en datetime
    df_matches['match_date'] = pd.to_datetime(df_matches['match_date'], errors='coerce')
    
    # 2. Filtrage à partir du 1er juillet 2020
    df_filtered = df_matches[df_matches['match_date'] >= '2020-07-01'].copy()
    
    # 3. Calculs sur les données filtrées
    dates = df_filtered['match_date']
    
    if dates.empty:
        print("Aucun match trouvé après le 01/07/2020.")
        return None, None

    start_date = dates.min()
    end_date = dates.max()
    delta = (end_date - start_date).days
    
    print(f"COUVERTURE TEMPORELLE STATSBOMB (Depuis 01/07/2020)")
    print(f"-------------------------------------------")
    print(f"• Premier match : {start_date.strftime('%d/%m/%Y')}")
    print(f"• Dernier match  : {end_date.strftime('%d/%m/%Y')}")
    print(f"• Étendue        : {delta} jours (env. {round(delta/365, 1)} ans)")
    print(f"• Nb de matchs   : {len(df_filtered)}")
    print(f"-------------------------------------------")
    
    # Répartition par saison sur le périmètre filtré
    if 'season.season_name' in df_filtered.columns:
        print("\nMatchs par Saison :")
        print(df_filtered['season.season_name'].value_counts().sort_index(ascending=False).head(5))
        
    return start_date, end_date




def plot_statsbomb_distributions(df_matches):
    """
    Affiche la répartition des matchs par compétition et la distribution des buts.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Graphique 1 : Top Compétitions
    sns.countplot(data=df_matches, y='competition.competition_name', 
                  order=df_matches['competition.competition_name'].value_counts().index, 
                  ax=axes[0], palette='viridis')
    axes[0].set_title('Répartition des Matchs par Compétition')
    axes[0].set_xlabel('Nombre de Matchs')

    # Graphique 2 : Distribution des buts (Home vs Away)
    goals_df = df_matches[['home_score', 'away_score']].melt()
    sns.histplot(data=goals_df, x='value', hue='variable', kde=True, 
                 element="step", ax=axes[1], palette='magma')
    axes[1].set_title('Distribution des Buts (Domicile vs Extérieur)')
    axes[1].set_xlabel('Buts marqués')

    plt.tight_layout()
    plt.show()