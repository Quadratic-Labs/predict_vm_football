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
    Réalise une analyse exploratoire des données StatsBomb.

    Cette fonction vérifie la qualité du jeu de données en identifiant les doublons d'identifiants, 
    en mesurant la complétude des colonnes critiques et en validant l'intégrité relationnelle 
    (s'assurer que chaque match possède bien une composition d'équipe associée). Elle fournit 
    également un aperçu rapide du volume de données par compétition.

    arguments:
        df_matches : Le dataset des matchs StatsBomb.
        df_lineups : Le dataset des compositions d'équipe.

    returns:
        dict: Un dictionnaire contenant la liste des 'match_id' orphelins (matchs présents 
            dans le calendrier mais n'ayant aucune donnée de lineup).
    """
    # Analyse des doublons
    match_dups = df_matches.duplicated(subset=['match_id']).sum()
    # Pour les lineups, un doublon est un même joueur deux fois dans le même match
    lineup_dups = df_lineups.duplicated(subset=['match_id', 'player_id']).sum()
    
    # Analyse des NA
    na_matches = df_matches[['match_id', 'home_team.home_team_name', 'away_team.away_team_name']].isnull().sum()
    na_lineups = df_lineups[['match_id', 'player_id', 'player_name']].isnull().sum()
    
    # Vérification de la liaison
    matches_in_lineups = df_lineups['match_id'].unique()
    missing_matches = df_matches[~df_matches['match_id'].isin(matches_in_lineups)]
    
   
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
    Analyse et valide la fenêtre temporelle des matchs StatsBomb à partir du 1er juillet 2020.

    Cette fonction convertit les dates de matchs au format datetime, filtre le dataset pour 
    ne conserver que les rencontres récentes et calcule l'étendue chronologique de la base de
    données. 

    arguments:
        df_matches : Le dataset des matchs StatsBomb contenant au minimum la colonne 'match_date'.

    returns:
        tuple: Un tuple contenant (start_date, end_date) sous forme d'objets Timestamp.
            Renvoie (None, None) si aucun match ne correspond aux critères de filtrage.
    """
    if 'match_date' not in df_matches.columns:
        print("Colonne 'match_date' introuvable.")
        return

    # Conversion en datetime
    df_matches['match_date'] = pd.to_datetime(df_matches['match_date'], errors='coerce')
    
    # Filtrage à partir du 1er juillet 2020
    df_filtered = df_matches[df_matches['match_date'] >= '2020-07-01'].copy()
    
    # Calculs sur les données filtrées
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
    Génère une analyse graphique de la répartition des matchs et de la dynamique des scores.

    Cette fonction crée un diagramme en barres horizontal pour comparer le volume de données
    par compétition, et un histogramme superposé avec une estimation de la densité
    pour analyser la distribution des buts marqués à domicile versus à l'extérieur.

    arguments:
        df_matches: Le dataset des matchs StatsBomb contenant les colonnes 
                                'competition.competition_name', 'home_score' et 'away_score'.

    returns:
        None: La fonction affiche directement une figure composée de deux graphiques 
            complémentaires (Seaborn/Matplotlib).
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Graphique 1 : Top Compétitions
    sns.countplot(data=df_matches, y='competition.competition_name', 
                  order=df_matches['competition.competition_name'].value_counts().index, 
                  ax=axes[0], palette='viridis')
    axes[0].set_title('Répartition des Matchs par Compétition')
    axes[0].set_xlabel('Nombre de Matchs')

    # Graphique 2 : Distribution des buts
    goals_df = df_matches[['home_score', 'away_score']].melt()
    sns.histplot(data=goals_df, x='value', hue='variable', kde=True, 
                 element="step", ax=axes[1], palette='magma')
    axes[1].set_title('Distribution des Buts (Domicile vs Extérieur)')
    axes[1].set_xlabel('Buts marqués')

    plt.tight_layout()
    plt.show()