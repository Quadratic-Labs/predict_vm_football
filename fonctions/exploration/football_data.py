import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import json
import requests
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import unicodedata




def check_football_data_updates(season="2526"):
    """
    Vérifie la fraîcheur des données sur football-data.co.uk pour les cinq grands championnats
    européens.

    Cette fonction interroge les en-têtes HTTP (méthode HEAD) des fichiers CSV distants pour 
    extraire la date de dernière modification sans télécharger l'intégralité des données. 
    Elle permet de savoir rapidement si de nouveaux résultats de matchs ont été publiés 
    pour la saison spécifiée.

    arguments:
        season (str): Le code de la saison au format "AABB" (ex: "2526" pour 2025-2026). 
                    Par défaut : "2526".

    Returns:
        dataframe: Un tableau récapitulatif contenant le nom de la compétition, son code 
                    et la date (ou l'état) de la dernière mise à jour serveur.
    """
    # Les ligues à analyser
    leagues = {
        "E0": "Premier League",
        "SP1": "Liga",
        "I1": "Serie A",
        "F1": "Ligue 1",
        "D1": "Bundesliga"
    }

    updates = []
    
    print(f"Vérification des mises à jour (Saison {season})...\n")

    for code, name in leagues.items():
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        
        try:
            # On utilise HEAD pour ne pas télécharger tout le fichier, juste les infos
            response = requests.head(url, timeout=10)
            last_mod = response.headers.get('Last-Modified', "Indisponible")
            
            updates.append({
                "Compétition": name,
                "Code": code,
                "Dernière mise à jour": last_mod
            })
        except Exception as e:
            updates.append({
                "Compétition": name,
                "Code": code,
                "Dernière mise à jour": f"Erreur: {str(e)}"
            })

    # Retourne un dataframe pour un affichage propre
    df_updates = pd.DataFrame(updates)
    return df_updates




def process_football_data(df_raw):
    """
    Filtre et nettoie le dataset football-data pour ne conserver que les variables essentielles.

    Cette fonction effectue une sélection stratégique parmi les nombreuses colonnes du 
    dataset d'origine pour isoler les informations de match, les statistiques de jeu et les cotes
    de paris. Elle gère également la conversion temporelle et supprime les lignes incomplètes
    pour garantir un signal propre pour l'analyse.

    arguments:
        df_raw : Le dataset brut chargé directement depuis football-data.co.uk.

    returns:
        dataframe: Un dataframe optimisé contenant uniquement les colonnes critiques 
                    (Match, Stats, Cotes) avec des types de données normalisés.
    """
    # Sélection des colonnes
    cols_match = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    cols_stats = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    cols_odds  = ['B365H', 'B365D', 'B365A', 'AvgH', 'AvgD', 'AvgA']
    
    # Élagage pour éviter la fragmentation
    df = df_raw[cols_match + cols_stats + cols_odds].copy()
    
    # Nettoyage
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.dropna(subset=['FTHG', 'FTAG'])
    

    print(f"Audit Football-Data :")
    print(f"- Nombre de colonnes initiales : {df_raw.shape[1]}")
    print(f"- Nombre de colonnes après élagage : {df.shape[1]}")
    print(f"- Matchs validés : {len(df)}")
    print("-" * 30)
    
    return df



def audit_data_quality(df, subset_duplicates=['Date', 'HomeTeam', 'AwayTeam']):
    """
    Effectue un audit ciblé de la qualité des données en isolant les valeurs manquantes
    et les doublons.

    Cette fonction permet de diagnostiquer rapidement l'état d'un dataset en se concentrant 
    uniquement sur les variables problématiques (colonnes avec des NaN). Elle vérifie 
    également l'existence de doublons basés sur un sous-ensemble de colonnes stratégiques 
    (par défaut : la date et les équipes).

    arguments:
        df : Le dataframe à auditer.
        subset_duplicates (list): Liste des colonnes à utiliser pour identifier les doublons 
                                potentiels. Par défaut : ['Date', 'HomeTeam', 'AwayTeam'].

    returns:
        pd.DataFrame: Un résumé contenant uniquement les colonnes présentant des valeurs 
                    manquantes et leur décompte respectif.
    """
    # Analyse des doublons
    dup_count = df.duplicated(subset=subset_duplicates).sum()
    
    # Analyse des NA par colonne
    na_summary = df.isnull().sum()
    na_only = na_summary[na_summary > 0].reset_index()
    na_only.columns = ['Variable', 'Nombre de NA']
    

    print(f"Analyse qualité :")
    print(f"-------------------------------")
    print(f"• Doublons détectés : {dup_count}")
    print(f"-------------------------------")
    
    if len(na_only) > 0:
        print("Colonnes avec données manquantes :")
        # On affiche le tableau proprement
        print(na_only.to_string(index=False))
    else:
        print("Aucune valeur manquante détectée dans le dataset.")
        
    print("-" * 31)
    
    return na_only




def audit_football_data_coherence(df):
    """
    Effectue un audit de cohérence logique et métier sur le dataset football-data.

    Cette fonction vérifie la validité des données en croisant différentes variables.
    Elle contrôle que les scores sont physiquement possibles (non négatifs), 
    que le résultat final déclaré concorde avec le décompte des buts, que les 
    cotes de paris sont réalistes et que chaque rencontre est unique dans le dataset.

    arguments:
        df : Le DataFrame contenant les données de matchs, incluant les scores, 
                        le résultat final et les cotes.

    returns:
        dict: Un dictionnaire synthétisant le nombre d'anomalies détectées par catégorie 
            (erreurs de score, de résultat, de cotes et doublons).
    """
    # Vérification des scores négatifs
    invalid_scores = df[(df['FTHG'] < 0) | (df['FTAG'] < 0)]
    
    # Vérification de la cohérence des résultats
    # On s'assure que le résultat 'H', 'D', 'A' correspond mathématiquement aux buts
    home_win_error = df[(df['FTR'] == 'H') & (df['FTHG'] <= df['FTAG'])]
    away_win_error = df[(df['FTR'] == 'A') & (df['FTAG'] <= df['FTHG'])]
    draw_error = df[(df['FTR'] == 'D') & (df['FTHG'] != df['FTAG'])]
    
    total_res_errors = len(home_win_error) + len(away_win_error) + len(draw_error)

    # Audit des Cotes (Recherche de valeurs aberrantes < 1.0)
    # On vérifie uniquement sur les colonnes de cotes moyennes présentes
    odd_cols = ['AvgH', 'AvgD', 'AvgA']
    present_odd_cols = [c for c in odd_cols if c in df.columns]
    
    odd_error_count = 0
    if present_odd_cols:
        odd_error = df[(df[present_odd_cols] < 1).any(axis=1)]
        odd_error_count = len(odd_error)

    # Vérification des Doublons
    # On utilise les colonnes d'origine ou les clés si elles existent
    dup_cols = ['Date', 'HomeTeam', 'AwayTeam']
    match_duplicates = df.duplicated(subset=dup_cols).sum()

  
    print(f"Audit de cohérence terminé :")
    print(f"-------------------------------------------")
    print(f"• Erreurs de score (négatifs)  : {len(invalid_scores)}")
    print(f"• Erreurs de résultat (FTR)    : {total_res_errors}")
    print(f"• Anomalies sur les cotes      : {odd_error_count}")
    print(f"• Matchs en double             : {match_duplicates}")
    print(f"-------------------------------------------")
    
    if (len(invalid_scores) + total_res_errors + odd_error_count + match_duplicates) == 0:
        print("Intégrité des données : PARFAITE")
    else:
        print("Des anomalies ont été détectées. Vérifiez votre source.")

    return {
        "score_errors": len(invalid_scores),
        "result_errors": total_res_errors,
        "odd_errors": odd_error_count,
        "duplicates": match_duplicates
    }





def check_temporal_coverage(df, date_col):
    """
    Calcule et affiche la fenêtre chronologique couverte par une colonne de date spécifique.

    Cette fonction permet de valider l'étendue temporelle d'un dataset en identifiant les 
    bornes minimales et maximales.

    arguments:
        df : Le dataframe contenant les données temporelles.
        date_col (str): Le nom de la colonne de date à analyser.

    returns:
        tuple: Un tuple contenant (min_date, max_date) sous forme d'objets Timestamp, 
            permettant d'utiliser ces bornes pour des filtrages ultérieurs.
        None: Si la colonne spécifiée est absente du DataFrame.
    """
    if date_col not in df.columns:
        print(f"La colonne '{date_col}' est absente du DataFrame.")
        return None

    # Conversion temporaire pour s'assurer du format datetime
    temp_dates = pd.to_datetime(df[date_col], errors='coerce')
    
    min_date = temp_dates.min()
    max_date = temp_dates.max()
    duration = (max_date - min_date).days

    print(f"Couverture temporelle [{date_col}] :")
    print(f"  • Début : {min_date.strftime('%d/%m/%Y')}")
    print(f"  • Fin   : {max_date.strftime('%d/%m/%Y')}")
    print(f"  • Durée : {duration} jours")
    print("-" * 35)
    
    return min_date, max_date






def analyze_league_distribution(df, league_col='Div'):
    """
    Normalise les identifiants des ligues et analyse la distribution des matchs par compétition.

    Cette fonction remplace les codes des championnats par leurs noms complets pour rendre
    les analyses et les graphiques plus lisibles. 
    Elle fournit ensuite un décompte précis du nombre de rencontres par ligue pour 
    vérifier l'équilibre du dataset.

    arguments:
        df : Le dataset contenant les données de football-data.
        league_col (str): Le nom de la colonne contenant les codes de ligues. 
                        Par défaut : 'Div'.

    returns:
        dataframe: Une copie du dataframe original avec les noms de ligues 
                    explicites dans la colonne cible.
    """
    # Mapping officiel des 5 grands championnats
    mapping = {
        "E0": "Premier League",
        "SP1": "Liga",
        "I1": "Serie A",
        "F1": "Ligue 1",
        "D1": "Bundesliga"
    }
    
    # Copie pour éviter de modifier le dataframe original par référence
    df_mapped = df.copy()
    
    # Remplacement des noms
    df_mapped[league_col] = df_mapped[league_col].replace(mapping)
    
    # Calcul et affichage
    counts = df_mapped[league_col].value_counts()
    
    print("Répartition par Compétition :")
    print("-" * 30)
    print(counts.to_string())
    print("-" * 30)
    print(f"Total Matchs : {len(df_mapped)}")
    
    return df_mapped