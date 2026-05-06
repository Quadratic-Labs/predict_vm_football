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


def get_kaggle_dataset_last_update(dataset_query, env_path="../.env"):
    """
    Se connecte à l'API Kaggle pour récupérer la date de dernière mise à jour d'un dataset.

    Cette fonction authentifie l'utilisateur via les variables d'environnement chargées 
    depuis un fichier .env, recherche le dataset spécifié et extrait sa date de 
    dernière modification.

    arguments :
        dataset_query (str): L'identifiant complet du dataset sur Kaggle
        env_path (str): Le chemin vers le fichier .env contenant KAGGLE_USERNAME et KAGGLE_API_TOKEN. 
                        Par défaut : "../.env".

    returns :
        datetime/str: La date de dernière mise à jour si le dataset est trouvé.
        None: Si les identifiants sont manquants, si le dataset n'existe pas ou en cas d'erreur.
    """
    load_dotenv(dotenv_path=env_path)

    username = os.getenv("KAGGLE_USERNAME")
    api_token = os.getenv("KAGGLE_API_TOKEN")

    if not username or not api_token:
        print("Erreur : Identifiants Kaggle manquants dans le fichier .env")
        return None

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = api_token

    try:
        api = KaggleApi()

        datasets = api.dataset_list(search=dataset_query)

        for ds in datasets:
            if ds.ref == dataset_query:

                date_maj = (
                    getattr(ds, "lastUpdated", None)
                    or getattr(ds, "last_updated", None)
                )

                print(f"Dataset trouvé : {ds.ref}")
                print(f"Dernière mise à jour : {date_maj}")

                return date_maj

        print(f"Dataset '{dataset_query}' non trouvé sur Kaggle.")
        return None

    except Exception as e:
        print(f"Erreur lors de la recherche du dataset : {e}")
        return None





def process_fbref_performance(df_fbref, min_minutes_played=450):
    """
    Analyse la fiabilité statistique et filtre le dataset FBref par temps de jeu et poste.
    
    Args:
        df_fbref (pd.DataFrame): Dataset brut FBref.
        min_minutes_played (int): Seuil de minutes pour la représentativité.
        
    Returns:
        tuple: (df_field_players, df_keepers)
    """
    
    df_val = df_fbref[df_fbref['Min'] > 0].copy()
    
    if 'Gls' in df_val.columns:
        # Calcul de la métrique de contrôle (Gls/90)
        df_val['Gls_90'] = (df_val['Gls'] / df_val['Min']) * 90
        
        # Calcul de la déviation standard par tranches de 100 min
        df_val['Min_Group'] = (df_val['Min'] // 100) * 100
        stability_check = df_val.groupby('Min_Group')['Gls_90'].std()

        # Visualisation de la stabilisation
        plt.figure(figsize=(10, 5))
        plt.plot(stability_check.index, stability_check.values, marker='o', color='darkblue', linewidth=2)
        plt.axvline(x=min_minutes_played, color='red', linestyle='--', label=f'Seuil : {min_minutes_played} min')
        plt.title("Validation Statistique : Stabilisation de la Variance")
        plt.xlabel("Minutes jouées (Tranches de 100 min)")
        plt.ylabel("Écart-type des Buts/90 (Volatilité)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        # Calcul de la réduction du bruit pour le log
        std_low = df_val[df_val['Min'] < min_minutes_played]['Gls_90'].std()
        std_high = df_val[df_val['Min'] >= min_minutes_played]['Gls_90'].std()
        noise_red = ((std_low - std_high) / std_low) * 100
        print(f"Preuve mathématique : Réduction du bruit de {noise_red:.1f}%")


    # Filtrage de représentativité
    df_rep = df_fbref[df_fbref['Min'] >= min_minutes_played].copy()
    
    # Séparation des joueurs de champ / gardiens
    df_field = df_rep[~df_rep['Pos'].str.contains('GK', na=False)].copy()
    df_keepers = df_rep[df_rep['Pos'].str.contains('GK', na=False)].copy()
    

    print(f"FILTRAGE TERMINÉ")
    print(f"- Joueurs initiaux : {len(df_fbref)}")
    print(f"- Joueurs conservés : {len(df_rep)} (Soit {len(df_field)} joueurs de champ)")
    
    return df_field, df_keepers





def prune_fbref_columns(df_input, threshold=0.80, verbose=True):
    """
    Analyse la fiabilité statistique et filtre le dataset FBref par temps de jeu et par poste.

    Cette fonction effectue une validation mathématique de la représentativité des données en 
    visualisant la stabilisation de la variance des buts par 90 minutes. Elle 
    sépare ensuite les joueurs de champ des gardiens de but après avoir appliqué un seuil 
    de temps de jeu minimal.

    arguments:
        df_fbref: Le dataset brut issu de FBref contenant les statistiques de performance.
        min_minutes_played (int): Le seuil minimal de minutes jouées pour considérer qu'un profil est 
                                statistiquement représentatif. Par défaut : 450.

    returns:
        tuple: Un tuple contenant deux DataFrames :
            - df_field : Les joueurs de champ ayant dépassé le seuil de minutes.
            - df_keepers : Les gardiens de but ayant dépassé le seuil de minutes.
    """
    df_work = df_input.copy()
    initial_cols = df_work.shape[1]

    # Gestion des Valeurs Manquantes
    null_counts = df_work.isnull().sum() / len(df_work) * 100
    null_counts_filtered = null_counts[null_counts > 0].sort_values(ascending=False)

    if verbose and not null_counts_filtered.empty:
        plt.figure(figsize=(12, 5))
        null_counts_filtered.head(25).plot(kind='bar', color='salmon')
        plt.axhline(y=threshold*100, color='black', linestyle='--', label=f'Seuil {int(threshold*100)}%')
        plt.title("Analyse des données manquantes (Top 25 variables)")
        plt.ylabel("% de NA")
        plt.legend()
        plt.tight_layout()
        plt.show()

    limit = len(df_work) * threshold




def resolve_player_duplicates(df_input, player_col='Player', metric_col='Min'):
    """
    Gère les doublons de joueurs au sein d'une même saison, généralement causés par des transferts.

    Lorsqu'un joueur évolue dans plusieurs clubs au cours d'une même saison, FBref génère 
    souvent une ligne par club. Cette fonction permet de ne conserver qu'une seule entrée par 
    joueur en sélectionnant la ligne où son temps de jeu est le plus élevé.

    arguments:
        df_input: Le dataset contenant les joueurs en doublon.
        player_col (str): Le nom de la colonne identifiant les joueurs. Par défaut : 'Player'.
        metric_col (str): La colonne utilisée pour décider quelle ligne conserver (la valeur 
                        maximale sera gardée). Par défaut : 'Min'.

    returns:
        dataframe: Un DataFrame où chaque joueur apparaît de manière unique.
    """
    df = df_input.copy()
    
    duplicate_counts = df[player_col].value_counts()
    players_with_dup = duplicate_counts[duplicate_counts > 1].index.tolist()
    nb_duplicates_initial = len(df) - df[player_col].nunique()

    if players_with_dup:
        print(f"Joueurs avec plusieurs lignes (transferts/doublons) : {len(players_with_dup)}")
    
    # Tri par joueur puis par la métrique (Minutes) décroissante
    # On garde la ligne "first" qui sera celle avec le maximum de minutes
    df_unique = df.sort_values(by=[player_col, metric_col], ascending=[True, False])
    df_unique = df_unique.drop_duplicates(subset=[player_col], keep='first')
    
    print(f"Bilan de l'unicité :")
    print(f"- Lignes traitées : {len(df)}")
    print(f"- Lignes après dédoublonnage : {len(df_unique)}")
    print(f"- 'Doublons de transfert' supprimés : {nb_duplicates_initial}")
    print("-" * 40)
    
    return df_unique





def normalize_fbref_formats(df_input):
    """
    Normalise les formats de données FBref et traite les valeurs manquantes par type de colonne.

    Cette fonction effectue un nettoyage technique. Elle convertit le format 
    d'âge spécifique à FBref en valeur numérique, l'imputation des données numériques 
    manquantes par zéro (fréquent pour les statistiques de performance non réalisées), 
    et le marquage des données textuelles manquantes comme inconnues.

    arguments:
        df_input: Le DataFrame brut contenant les données extraites de FBref.

    returns:
        dataframe: Le DataFrame nettoyé avec les types de colonnes harmonisés et 
                    aucune valeur manquante.
    """
    df = df_input.copy()
    
    # Conversion de l'âge
    if 'Age' in df.columns:
        if df['Age'].dtype == 'object':
            # On split sur le tiret et on prend le premier élément
            df['Age'] = df['Age'].str.split('-').str[0]
            # Conversion en float
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Imputation des colonnes numériques
    # On remplace les valeurs manquantes par 0
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Imputation des colonnes non-numériques
    # On remplace les valeurs manquantes par "Unkown"
    non_num_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    df[non_num_cols] = df[non_num_cols].fillna("Unknown")
    
    print(f"Normalisation des formats terminée.")
    print(f"- Colonnes numériques imputées (0) : {len(num_cols)}")
    print(f"- Colonnes texte imputées (Unknown) : {len(non_num_cols)}")
    print(f"- Valeurs manquantes totales restantes : {df.isnull().sum().sum()}")
    
    return df




def audit_fbref_outliers(df, metrics=['Gls', 'Ast', 'Sh/90', 'Age'], top_n=5):
    """
    Identifie et visualise les valeurs aberrantes pour des métriques de performance clés.

    Cette fonction génère une grille de boxplots pour analyser la distribution statistique des
    variables sélectionnées. Elle permet de détecter visuellement les performances
    exceptionnelles ou les erreurs de données, et liste explicitement les 
    joueurs affichant les valeurs les plus élevées pour la métrique principale.

    arguments:
        df: Le dataset contenant les statistiques des joueurs.
        metrics (list): Liste des colonnes à analyser graphiquement (max 4 pour l'affichage). 
                        Par défaut : ['Gls', 'Ast', 'Sh/90', 'Age'].
        top_n (int): Nombre de joueurs extrêmes à afficher dans le bilan textuel. 
                    Par défaut : 5.

    returns:
        None: La fonction affiche directement les graphiques et les tableaux de données.
    """

    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            plt.subplot(2, 2, i+1)
            sns.boxplot(x=df[metric], color='skyblue', fliersize=5)
            plt.title(f'Distribution de : {metric}', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
        else:
            print(f"La métrique '{metric}' est absente du DataFrame.")

    plt.tight_layout()
    plt.show()

    # Identification des cas extrêmes (sur la première métrique, souvent les buts)
    primary_metric = metrics[0]
    if primary_metric in df.columns:
        print(f"Top {top_n} 'Outliers' Performance ({primary_metric}) :")
        top_players = df[['Player', 'Squad', primary_metric, 'Min']].sort_values(by=primary_metric, ascending=False).head(top_n)
        print(top_players)
        print("-" * 40)
    
    return
