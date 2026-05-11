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




def plot_transfermarkt_quality(df_players, df_clubs=None):
    """
    Génère un diagnostic visuel de la qualité et de la distribution des données Transfermarkt.

    Cette fonction crée une grille de visualisations permettant d'analyser la cohérence des 
    données de valeur marchande et la démographie des joueurs. Si un référentiel de clubs
    est fourni, elle génère également une comparaison du marché financier entre les
    championnats du Big 5 européen.

    arguments:
        df_players : Le dataset des joueurs contenant au minimum les colonnes 
                                'market_value_in_eur' et 'date_of_birth'.
        df_clubs : Le dataset des clubs pour l'analyse par ligue. 
                    Nécessaire pour afficher la comparaison du Big 5. 
                    Par défaut : None.

    returns:
        None: La fonction affiche directement une figure composée de 4 graphiques.
    """
    df_players = df_players.copy()
    
    # Calcul de l'âge
    df_players['date_of_birth'] = pd.to_datetime(df_players['date_of_birth'], errors='coerce')
    df_players['age'] = pd.Timestamp.now().year - df_players['date_of_birth'].dt.year

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3)

    # Graphiques 1 et 2 : valeurs marchandes
    sns.histplot(df_players['market_value_in_eur'].dropna(), kde=True, ax=axes[0, 0], color='blue')
    sns.histplot(np.log1p(df_players['market_value_in_eur'].dropna()), kde=True, ax=axes[0, 1], color='green')
    axes[0, 0].set_title("Distribution Valeur Marchande")
    axes[0, 1].set_title("Distribution (Log Scale)")

    # Graphique 3 : âge
    sns.histplot(df_players['age'].dropna(), bins=20, kde=True, ax=axes[1, 0], color='orange')
    axes[1, 0].set_title("Répartition par Âge")

    # Graphique ' : boxplots
    if df_clubs is not None:
        
        # Identification des colonnes
        col_club_p = 'current_club_id' if 'current_club_id' in df_players.columns else 'club_id'
        col_club_c = 'club_id'
        col_comp = [c for c in df_clubs.columns if 'competition' in c and 'id' in c][0]
        
        # Merge joueurs + clubs
        temp_df = df_players.merge(
            df_clubs[[col_club_c, col_comp]], 
            left_on=col_club_p, 
            right_on=col_club_c, 
            how='left'
        )
        
        # identifiants du Big 5
        big5_ids = ['GB1', 'ES1', 'L1', 'IT1', 'FR1']
        
        # Filtrage Big 5
        df_big5 = temp_df[temp_df[col_comp].isin(big5_ids)].copy()
        
        
        league_names = {
            'GB1': 'Premier League',
            'ES1': 'La Liga',
            'L1': 'Bundesliga',
            'IT1': 'Serie A',
            'FR1': 'Ligue 1'
        }
        
        df_big5['league_name'] = df_big5[col_comp].map(league_names)
        
        # Ordre logique des ligues
        order = [
            'Premier League',
            'La Liga',
            'Bundesliga',
            'Serie A',
            'Ligue 1'
        ]
        

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
    Identifie et visualise le pourcentage de valeurs manquantes pour chaque colonne du DataFrame.

    Cette fonction calcule le taux de complétude des données et génère un graphique à barres 
    présentant uniquement les colonnes contenant des valeurs nulles. Elle inclut un 
    seuil critique visuel à 20%.

    arguments:
        df : Le DataFrame à analyser.
        title (str): Le titre personnalisé pour le graphique. 
                    Par défaut : "Valeurs manquantes par colonne".

    returns:
        None: La fonction affiche un message texte si aucune valeur manquante n'est trouvée, 
            ou un graphique le cas échéant.
    """
    # Calcul du % de manquants pour chaque colonne
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    # On ne garde que les colonnes qui ont des manquants (> 0)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if missing_pct.empty:
        print(f"Aucune valeur manquante détectée dans le dataset : {title}")
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



def check_player_duplicates(df_players):
    """
    Identifie les doublons potentiels de joueurs en croisant le nom et la date de naissance.

    Cette fonction effectue une vérification de l'intégrité du dataset en isolant les joueurs 
    qui partagent exactement le même nom et la même date de naissance.

    arguments:
        df_players : Le dataset des joueurs contenant au minimum les colonnes 
                            'name' et 'date_of_birth'.

    returns:
        dataframe: Un dataframe contenant toutes les lignes suspectées d'être des doublons, 
                    triées par nom pour faciliter la comparaison manuelle.
        None: Si aucune duplication n'est trouvée.
    """
    # On cherche les lignes où le nom et la date de naissance sont identiques
    duplicates = df_players[df_players.duplicated(subset=['name', 'date_of_birth'], keep=False)]
    
    if not duplicates.empty:
        print(f"{len(duplicates)} doublons potentiels détectés (même nom et date de naissance).")
        return duplicates.sort_values(by='name')
    else:
        print("Aucun doublon de joueur détecté sur le combo Nom/Date de naissance.")
        return None



def get_latest_player_valuations(df_valuations, plot_freshness=True):
    """
    Extrait la valeur marchande la plus récente pour chaque joueur et analyse la fraîcheur
    du dataset.

    Cette fonction traite l'historique complet des valeurs marchandes pour isoler la
    dernière situation connue de chaque joueur. Elle inclut une étape de diagnostic temporel
    pour vérifier si les données sont à jour ou si le dataset contient des évaluations
    obsolètes.

    arguments:
        df_valuations : Le dataset historique des valeurs contenant 'player_id' et 'date'.
        plot_freshness (bool): Si True, affiche un histogramme de la répartition des dates 
                            de mise à jour. Par défaut : True.

    Returns:
        dataframe: Un dataframe contenant une seule ligne par joueur (la plus récente).
    """
    
    # Copie et conversion immédiate
    df = df_valuations.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Tri et dédoublonnage
    # On trie par date pour que 'last' soit bien la plus récente
    df_latest = df.sort_values('date').drop_duplicates('player_id', keep='last')
    
    # Affichage des statistiques
    print(f"Nombre total d'évaluations dans l'historique : {len(df)}")
    print(f"Nombre de joueurs uniques évalués : {df_latest['player_id'].nunique()}")
    print("-" * 30)
    
    # On utilise .min() et .max() sur la colonne déjà convertie en datetime
    print(f"Date de l'évaluation la plus ancienne (historique) : {df['date'].min().date()}")
    print(f"Date de l'évaluation la plus récente (marché actuel) : {df['date'].max().date()}")

    # Visualisation
    if plot_freshness:
        plt.figure(figsize=(10, 4))
        sns.histplot(df_latest['date'], bins=30, color='darkcyan')
        plt.title("Répartition des dates de dernières évaluations")
        plt.xlabel("Année de mise à jour")
        plt.ylabel("Nombre de joueurs")
        plt.show()

    return df_latest



def check_referential_integrity(df_players, df_valuations):
    """
    Vérifie si tous les IDs de joueurs présents dans les évaluations (prices) 
    existent bien dans le référentiel des profils joueurs.
    
    arguments:
        df_players : Le dataset des joueurs
        df_valuations : Le dataset des prix
        
    returns:
        set: La liste des IDs orphelins
    """
    
    # Extraction des sets d'IDs pour comparaison rapide
    ids_in_players = set(df_players['player_id'])
    ids_in_valuations = set(df_valuations['player_id'])

    # Identification des orphelins (présents dans prix mais pas dans profils)
    orphans = ids_in_valuations - ids_in_players


    print(f"Analyse de l'intégrité référentielle :")
    print(f"- Joueurs dans le fichier Profil : {len(ids_in_players)}")
    print(f"- Joueurs dans le fichier Valuations : {len(ids_in_valuations)}")
    print("-" * 40)

    if orphans:
        print(f"Alerte : {len(orphans)} joueurs ont des prix mais n'ont pas de profil.")
        print(f"Exemples d'IDs orphelins : {list(orphans)[:5]}")
    else:
        print("Intégrité parfaite : Tous les prix sont reliés à un profil joueur.")
    
    return orphans
