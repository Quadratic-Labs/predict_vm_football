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






def get_latest_player_valuations(df_valuations, plot_freshness=True):
    """
    Nettoie l'historique des valeurs marchandes pour ne garder que la plus récente
    par joueur et analyse la fraîcheur des données.
    """
    
    # 1. Copie et conversion immédiate
    df = df_valuations.copy()
    df['date'] = pd.to_datetime(df['date']) # Conversion cruciale ici
    
    # 2. Tri et dédoublonnage
    # On trie par date pour que 'last' soit bien la plus récente
    df_latest = df.sort_values('date').drop_duplicates('player_id', keep='last')
    
    # 3. Affichage des statistiques (Utilisation de 'df' converti)
    print(f"Nombre total d'évaluations dans l'historique : {len(df)}")
    print(f"Nombre de joueurs uniques évalués : {df_latest['player_id'].nunique()}")
    print("-" * 30)
    
    # On utilise .min() et .max() sur la colonne déjà convertie en datetime
    print(f"Date de l'évaluation la plus ancienne (historique) : {df['date'].min().date()}")
    print(f"Date de l'évaluation la plus récente (marché actuel) : {df['date'].max().date()}")

    # 4. Visualisation
    if plot_freshness:
        plt.figure(figsize=(10, 4))
        sns.histplot(df_latest['date'], bins=30, color='darkcyan')
        plt.title("Répartition des dates de dernières évaluations")
        plt.xlabel("Année de mise à jour")
        plt.ylabel("Nombre de joueurs")
        plt.show()

    return df_latest





def filter_valuation_freshness(df_latest_values, threshold_date='2025-10-01', plot_confirmation=True):
    """
    Filtre le dataset des valeurs marchandes pour ne conserver que les données récentes.
    
    Args:
        df_latest_values (pd.DataFrame): DataFrame des dernières valeurs par joueur.
        threshold_date (str): Date charnière au format 'YYYY-MM-DD'.
        plot_confirmation (bool): Si True, affiche l'histogramme après filtrage.
        
    Returns:
        pd.DataFrame: DataFrame filtré avec uniquement les données fraîches.
    """
    
    # 1. Conversion de la date seuil
    date_seuil = pd.to_datetime(threshold_date)
    
    # 2. Application du filtre
    # On s'assure que la colonne 'date' est bien au format datetime
    df_latest_values['date'] = pd.to_datetime(df_latest_values['date'])
    df_filtered = df_latest_values[df_latest_values['date'] >= date_seuil].copy()
    
    # 3. Bilan du filtrage
    print(f"Filtre de fraîcheur appliqué (Seuil : {threshold_date}) :")
    print(f"- Joueurs conservés : {len(df_filtered)}")
    print(f"- Joueurs écartés (données trop anciennes) : {len(df_latest_values) - len(df_filtered)}")
    print(f"- Plage finale : du {df_filtered['date'].min().date()} au {df_filtered['date'].max().date()}")

    print(f"Date la plus ancienne conservée : {df_latest_values['date'].min().date()}")
    print(f"Date la plus récente : {df_latest_values['date'].max().date()}")


    # 4. Visualisation de confirmation
    if plot_confirmation:
        plt.figure(figsize=(10, 3))
        df_filtered['date'].hist(bins=20, color='teal', grid=False, rwidth=0.9)
        plt.title(f"Distribution des dates après filtrage (Min: {threshold_date})")
        plt.xlabel("Date de mise à jour")
        plt.ylabel("Nombre de joueurs")
        plt.show()
        
    return df_filtered




def detect_outliers_zscore(df, column):
    """
    Détecte les valeurs aberrantes statistiquement (Z-score > 3).
    """
    data = df[column].dropna()
    mean = np.mean(data)
    std = np.std(data)
    
    threshold = 3
    outliers = df[np.abs((df[column] - mean) / std) > threshold]
    
    print(f"Analyse de {column} :")
    print(f"- Moyenne : {mean:.2f} | Écart-type : {std:.2f}")
    print(f"- Nombre d'outliers détectés (> 3 std) : {len(outliers)}")
    
    return outliers[[column]].sort_values(by=column, ascending=False)







def plot_outliers_jitter(df, column, outliers_df):
    data = df[column].dropna()
    mean = np.mean(data)
    std = np.std(data)

    is_outlier = df[column].isin(outliers_df[column])

    np.random.seed(42)
    jitter = np.random.uniform(-0.4, 0.4, size=len(df))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0d0f14')
    ax.set_facecolor('#161921')

    # Points normaux
    mask_normal = ~is_outlier & df[column].notna()
    ax.scatter(
        df.loc[mask_normal, column],
        jitter[mask_normal],
        color='#4a9eff', alpha=0.5, s=20, linewidths=0.5,
        edgecolors='#4a9eff', label='Valeur normale'
    )

    # Points outliers
    mask_out = is_outlier & df[column].notna()
    ax.scatter(
        df.loc[mask_out, column],
        jitter[mask_out],
        color='#ff5c5c', alpha=0.85, s=35, linewidths=1,
        edgecolors='#ff5c5c', label='Outlier (|z| > 3)'
    )

    # Seuils ±3σ
    for seuil, label in [(mean - 3*std, '−3σ'), (mean + 3*std, '+3σ')]:
        if seuil > 0:
            ax.axvline(seuil, color='#f0c040', linewidth=1.2,
                       linestyle='--', alpha=0.6)
            if seuil >= 1e6:
                seuil_label = f'{seuil/1e6:.1f}M€'
            elif seuil >= 1e3:
                seuil_label = f'{seuil/1e3:.0f}K€'
            else:
                seuil_label = f'{seuil:.0f}€'
            ax.text(seuil, 0.45, seuil_label, color='#f0c040',
                    fontsize=8, ha='center', va='bottom')

    # Mise en forme des axes
    ax.set_xscale('log')
    ax.set_xlabel('Valeur marchande (€) — échelle log',
                  color='#5a6072', fontsize=9)
    ax.set_ylabel('Jitter (dispersion visuelle)',
                  color='#5a6072', fontsize=9)
    ax.set_title('Distribution des valeurs marchandes — Outliers Z-score > 3',
                 color='#e8eaf0', fontsize=13, fontweight='bold', pad=14)

    ax.tick_params(colors='#5a6072')
    for spine in ax.spines.values():
        spine.set_edgecolor('#252a35')

    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])

    # Formatage des ticks X
    from matplotlib.ticker import FuncFormatter
    def fmt(x, _):
        if x >= 1e6: return f'{x/1e6:.0f}M€'
        if x >= 1e3: return f'{x/1e3:.0f}K€'
        return f'{x:.0f}€'
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))

    # Légende
    legend = ax.legend(
        handles=[
            mpatches.Patch(color='#4a9eff', label=f'Normal ({(~mask_out).sum()})'),
            mpatches.Patch(color='#ff5c5c', label=f'Outlier ({mask_out.sum()})')
        ],
        facecolor='#161921', edgecolor='#252a35',
        labelcolor='#e8eaf0', fontsize=9
    )

    plt.tight_layout()
    plt.show()




def check_player_duplicates(df_players):
    """
    Identifie les doublons potentiels basés sur le nom et la date de naissance.
    """
    # On cherche les lignes où le nom et la date de naissance sont identiques
    duplicates = df_players[df_players.duplicated(subset=['name', 'date_of_birth'], keep=False)]
    
    if not duplicates.empty:
        print(f"{len(duplicates)} doublons potentiels détectés (même nom et date de naissance).")
        return duplicates.sort_values(by='name')
    else:
        print("Aucun doublon de joueur détecté sur le combo Nom/Date de naissance.")
        return None



def check_referential_integrity(df_players, df_valuations):
    """
    Vérifie si tous les IDs de joueurs présents dans les évaluations (prices) 
    existent bien dans le référentiel des profils joueurs.
    
    Args:
        df_players (pd.DataFrame): Le dataset 'players.csv' (référentiel)
        df_valuations (pd.DataFrame): Le dataset des prix (consolidé ou brut)
        
    Returns:
        set: La liste des IDs orphelins (si besoin de traitement ultérieur)
    """
    
    # 1. Extraction des sets d'IDs pour comparaison rapide
    ids_in_players = set(df_players['player_id'])
    ids_in_valuations = set(df_valuations['player_id'])

    # 2. Identification des orphelins (présents dans prix mais pas dans profils)
    orphans = ids_in_valuations - ids_in_players

    # 3. Affichage du bilan d'intégrité
    print(f"Analyse de l'intégrité référentielle :")
    print(f"- Joueurs dans le fichier Profil : {len(ids_in_players)}")
    print(f"- Joueurs dans le fichier Valuations : {len(ids_in_valuations)}")
    print("-" * 40)

    if orphans:
        print(f"ALERTE : {len(orphans)} joueurs ont des prix mais n'ont pas de profil.")
        print(f"Ces prix seront perdus lors de la jointure (Merge).")
        print(f"Exemples d'IDs orphelins : {list(orphans)[:5]}")
    else:
        print("INTÉGRITÉ PARFAITE : Tous les prix sont reliés à un profil joueur.")
    
    return orphans




def process_market_data(df_players, df_valuations, threshold_date='2025-10-01', min_value=1_000_000):
    """
    Réalise le pipeline complet de nettoyage des données de marché :
    Conversion, Unicité, Filtres de fraîcheur/valeur, Fusion et Nettoyage de colonnes.
    """
    
    # --- 1. PRÉPARATION DES VALUATIONS ---
    df_v = df_valuations.copy()
    df_v['date'] = pd.to_datetime(df_v['date'])

    # Harmonisation du nom de la colonne de prix
    if 'market_value' in df_v.columns and 'market_value_in_eur' not in df_v.columns:
        df_v = df_v.rename(columns={'market_value': 'market_value_in_eur'})
    
    col_prix = 'market_value_in_eur'

    # --- 2. UNICITÉ & FILTRES ---
    # Garder la valeur la plus récente
    df_v = df_v.sort_values('date').drop_duplicates('player_id', keep='last')
    
    # Filtre de fraîcheur
    date_seuil = pd.to_datetime(threshold_date)
    df_v = df_v[df_v['date'] >= date_seuil]
    
    # Filtre de valeur (>= 1M€)
    df_v = df_v[df_v[col_prix] >= min_value]

    # --- 3. FUSION AVEC PLAYERS ---
    # Éviter les collisions de colonnes lors du merge
    if col_prix in df_players.columns:
        df_players_tmp = df_players.drop(columns=[col_prix])
    else:
        df_players_tmp = df_players

    # Fusion Inner pour ne garder que ceux qui ont un profil ET un prix valide
    df_final = df_v.merge(df_players_tmp, on='player_id', how='inner')

    # --- 4. NETTOYAGE FINAL DES COLONNES ---
    cols_to_drop = ['current_national_team_id', 'agent_name', 'city_of_birth', 'competition_id']
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])

    # --- 5. BILAN ---
    print("PIPELINE DE VALORISATION TERMINÉ")
    print(f"- Joueurs conservés : {len(df_final)}")
    print(f"- Valeur min : {df_final[col_prix].min():,.0f} €")
    print(f"- Plage temporelle : du {df_final['date'].min().date()} au {df_final['date'].max().date()}")
    
    return df_final






def process_fbref_performance(df_fbref, min_minutes_played=450):
    """
    Analyse la fiabilité statistique et filtre le dataset FBref par temps de jeu et poste.
    
    Args:
        df_fbref (pd.DataFrame): Dataset brut FBref.
        min_minutes_played (int): Seuil de minutes pour la représentativité.
        
    Returns:
        tuple: (df_field_players, df_keepers)
    """
    # --- 1. VALIDATION STATISTIQUE (Courbe de Volatilité) ---
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

    # --- 2. FILTRAGE ET SÉPARATION ---
    # Filtrage de représentativité
    df_rep = df_fbref[df_fbref['Min'] >= min_minutes_played].copy()
    
    # Séparation Champ / Gardiens
    df_field = df_rep[~df_rep['Pos'].str.contains('GK', na=False)].copy()
    df_keepers = df_rep[df_rep['Pos'].str.contains('GK', na=False)].copy()
    
    # --- 3. BILAN ---
    print(f"FILTRAGE TERMINÉ")
    print(f"- Joueurs initiaux : {len(df_fbref)}")
    print(f"- Joueurs conservés : {len(df_rep)} (Soit {len(df_field)} joueurs de champ)")
    
    return df_field, df_keepers





def prune_fbref_columns(df_input, threshold=0.80, verbose=True):
    """
    Supprime les colonnes vides (NA) et redondantes (techniques) du dataset FBref.
    Affiche également les graphiques justificatifs.
    
    Returns:
        pd.DataFrame: Le dataset nettoyé.
    """
    df_work = df_input.copy()
    initial_cols = df_work.shape[1]

    # --- 1. Gestion des Valeurs Manquantes (NA) ---
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
    cols_to_drop_na = df_work.columns[df_work.isnull().sum() > limit].tolist()
    df_clean = df_work.drop(columns=cols_to_drop_na)

    # --- 2. Gestion des Redondances ---
    redundant_patterns = ['stats_', 'Rk', 'Born', 'Nation_']
    cols_redundant = [col for col in df_clean.columns 
                      if any(pat in col for pat in redundant_patterns) 
                      and col not in ['Player', 'Min', '90s']]

    if verbose and cols_redundant:
        redundancy_summary = {pat: len([c for c in cols_redundant if pat in c]) for pat in redundant_patterns}
        plt.figure(figsize=(8, 4))
        sns.barplot(x=list(redundancy_summary.keys()), y=list(redundancy_summary.values()), palette='viridis', hue=list(redundancy_summary.keys()), legend=False)
        plt.title("Répartition des colonnes redondantes identifiées")
        plt.ylabel("Nombre de colonnes")
        plt.show()

    df_final = df_clean.drop(columns=cols_redundant)

    if verbose:
        print(f"ÉLAGAGE DES VARIABLES TERMINÉ")
        print(f"---------------------------------------------------")
        print(f"- Colonnes initiales : {initial_cols}")
        print(f"- Supprimées (> {int(threshold*100)}% NA) : {len(cols_to_drop_na)}")
        print(f"- Supprimées (Redondances) : {len(cols_redundant)}")
        print(f"- Colonnes finales conservées : {df_final.shape[1]}")

    return df_final





def resolve_player_duplicates(df_input, player_col='Player', metric_col='Min'):
    """
    Gère les doublons de joueurs (dus aux transferts en cours de saison).
    Garde la ligne où le joueur a le plus de temps de jeu.
    """
    df = df_input.copy()
    
    # 1. Identification pour le log
    duplicate_counts = df[player_col].value_counts()
    players_with_dup = duplicate_counts[duplicate_counts > 1].index.tolist()
    nb_duplicates_initial = len(df) - df[player_col].nunique()

    if players_with_dup:
        print(f"Joueurs avec plusieurs lignes (transferts/doublons) : {len(players_with_dup)}")
    
    # 2. Résolution : Tri par joueur puis par la métrique (Minutes) décroissante
    # On garde la ligne "first" qui sera celle avec le max de minutes
    df_unique = df.sort_values(by=[player_col, metric_col], ascending=[True, False])
    df_unique = df_unique.drop_duplicates(subset=[player_col], keep='first')
    
    # 3. Bilan
    print(f"Bilan de l'unicité :")
    print(f"- Lignes traitées : {len(df)}")
    print(f"- Lignes après dédoublonnage : {len(df_unique)}")
    print(f"- 'Doublons de transfert' supprimés : {nb_duplicates_initial}")
    print("-" * 40)
    
    return df_unique





def normalize_fbref_formats(df_input):
    """
    Normalise les formats de données FBref :
    - Convertit l'âge '27-125' en 27 (numérique).
    - Impute les valeurs manquantes par 0 pour les colonnes numériques.
    """
    df = df_input.copy()
    
    # 1. Conversion de l'Âge (ex: '27-125' -> 27.0)
    if 'Age' in df.columns:
        if df['Age'].dtype == 'object':
            # On split sur le tiret et on prend le premier élément
            df['Age'] = df['Age'].str.split('-').str[0]
            # Conversion en float (pour supporter les éventuels NaNs avant imputation)
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # 2. Imputation des colonnes numériques
    # On remplace les NaN par 0 (ex: un joueur sans tir a NaN en % de tirs cadrés)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    print(f"Normalisation des formats terminée.")
    print(f"- Colonnes numériques imputées : {len(num_cols)}")
    print(f"- Valeurs manquantes totales restantes : {df.isnull().sum().sum()}")
    
    return df







def audit_fbref_outliers(df, metrics=['Gls', 'Ast', 'Sh/90', 'Age'], top_n=5):
    """
    Génère des boxplots pour détecter les outliers sur des métriques clés
    et affiche les joueurs les plus extrêmes pour la première métrique.
    """
    # 1. Visualisation graphique
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

    # 2. Identification des cas extrêmes (sur la première métrique, souvent les buts)
    primary_metric = metrics[0]
    if primary_metric in df.columns:
        print(f"Top {top_n} 'Outliers' Performance ({primary_metric}) :")
        top_players = df[['Player', 'Squad', primary_metric, 'Min']].sort_values(by=primary_metric, ascending=False).head(top_n)
        print(top_players)
        print("-" * 40)
    
    return






def check_football_data_updates(season="2526"):
    """
    Vérifie la date de dernière modification des fichiers CSV 
    sur football-data.co.uk pour les 5 grands championnats.
    """
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

    # Retourne un DataFrame pour un affichage propre
    df_updates = pd.DataFrame(updates)
    return df_updates






def process_football_data(df_raw):
    """
    Filtre et nettoie le dataset Football-Data pour ne garder que le signal utile.
    Affiche un audit structurel des données.
    """
    # 1. Sélection stratégique des colonnes
    cols_match = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    cols_stats = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    cols_odds  = ['B365H', 'B365D', 'B365A', 'AvgH', 'AvgD', 'AvgA']
    
    # 2. Élagage pour éviter la fragmentation (PerformanceWarning)
    df = df_raw[cols_match + cols_stats + cols_odds].copy()
    
    # 3. Nettoyage
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.dropna(subset=['FTHG', 'FTAG'])
    
    # 4. Affichage de l'Audit
    print(f"Audit Football-Data :")
    print(f"- Nombre de colonnes initiales : {df_raw.shape[1]}")
    print(f"- Nombre de colonnes après élagage : {df.shape[1]}")
    print(f"- Matchs validés : {len(df)}")
    print("-" * 30)
    
    return df






def audit_data_quality(df, subset_duplicates=['Date', 'HomeTeam', 'AwayTeam']):
    """
    Analyse ciblée : identifie uniquement les colonnes avec des NA 
    et vérifie les doublons.
    """
    # 1. Analyse des doublons
    dup_count = df.duplicated(subset=subset_duplicates).sum()
    
    # 2. Analyse des NA par colonne
    na_summary = df.isnull().sum()
    na_only = na_summary[na_summary > 0].reset_index()
    na_only.columns = ['Variable', 'Nombre de NA']
    
    # 3. Affichage
    print(f"ANALYSE QUALITÉ :")
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
    Effectue un audit de cohérence logique sur le dataset Football-Data :
    - Validité des scores (non négatifs)
    - Cohérence entre scores et résultat final (FTR)
    - Anomalies sur les cotes (cotes < 1)
    - Détection de doublons (Date/Equipes)
    """
    # 1. Vérification des scores négatifs
    invalid_scores = df[(df['FTHG'] < 0) | (df['FTAG'] < 0)]
    
    # 2. Vérification de la cohérence FTR (Full Time Result)
    # On s'assure que le résultat 'H', 'D', 'A' correspond mathématiquement aux buts
    home_win_error = df[(df['FTR'] == 'H') & (df['FTHG'] <= df['FTAG'])]
    away_win_error = df[(df['FTR'] == 'A') & (df['FTAG'] <= df['FTHG'])]
    draw_error = df[(df['FTR'] == 'D') & (df['FTHG'] != df['FTAG'])]
    
    total_res_errors = len(home_win_error) + len(away_win_error) + len(draw_error)

    # 3. Audit des Cotes (Recherche de valeurs aberrantes < 1.0)
    # On vérifie uniquement sur les colonnes de cotes moyennes présentes
    odd_cols = ['AvgH', 'AvgD', 'AvgA']
    present_odd_cols = [c for c in odd_cols if c in df.columns]
    
    odd_error_count = 0
    if present_odd_cols:
        odd_error = df[(df[present_odd_cols] < 1).any(axis=1)]
        odd_error_count = len(odd_error)

    # 4. Vérification des Doublons
    # On utilise les colonnes d'origine ou les clés si elles existent
    dup_cols = ['Date', 'HomeTeam', 'AwayTeam']
    match_duplicates = df.duplicated(subset=dup_cols).sum()

    # --- AFFICHAGE DU BILAN ---
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
    Affiche la date de début et de fin pour une colonne spécifique d'un DataFrame.
    Utile pour vérifier la synchronisation des sources (FBref vs Football-Data).
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
    Renomme les codes ligues en noms clairs et affiche la répartition des matchs.
    """
    # 1. Mapping officiel des 5 grands championnats
    mapping = {
        "E0": "Premier League",
        "SP1": "Liga",
        "I1": "Serie A",
        "F1": "Ligue 1",
        "D1": "Bundesliga"
    }
    
    # 2. Copie pour éviter de modifier le DF original par référence (si besoin)
    df_mapped = df.copy()
    
    # 3. Remplacement des noms
    df_mapped[league_col] = df_mapped[league_col].replace(mapping)
    
    # 4. Calcul et affichage
    counts = df_mapped[league_col].value_counts()
    
    print("Répartition par Compétition :")
    print("-" * 30)
    print(counts.to_string())
    print("-" * 30)
    print(f"Total Matchs : {len(df_mapped)}")
    
    return df_mapped






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



def get_statsbomb_last_update(df_matches):
    """
    Trouve la date de la dernière mise à jour des données 
    et le match le plus récent dans le dataset.
    """
    # 1. Date du match le plus récent (réalité sportive)
    df_matches['match_date'] = pd.to_datetime(df_matches['match_date'])
    most_recent_match = df_matches['match_date'].max()

    # 2. Date de dernière modification des données (fraîcheur du fichier)
    # StatsBomb utilise souvent 'last_updated' pour l'ingestion des données
    update_col = 'last_updated' if 'last_updated' in df_matches.columns else None
    
    last_data_update = None
    if update_col:
        df_matches[update_col] = pd.to_datetime(df_matches[update_col])
        last_data_update = df_matches[update_col].max()

    print(f"FRAÎCHEUR DES DONNÉES STATSBOMB")
    print(f"-------------------------------------------")
    print(f"• Match le plus récent au calendrier : {most_recent_match.strftime('%d/%m/%Y')}")
    
    if last_data_update:
        print(f"• Dernière mise à jour du flux API  : {last_data_update.strftime('%d/%m/%Y à %H:%M')}")
    else:
        print("• Dernière mise à jour du flux API  : Non disponible (colonne absente)")
    print(f"-------------------------------------------")
    
    return most_recent_match, last_data_update




import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_lineup_analysis(df_lineups):
    """
    Analyse la structure des effectifs.
    """
    plt.figure(figsize=(10, 5))
    
    # Nombre de joueurs par équipe par match (pour vérifier la complétude)
    players_per_match = df_lineups.groupby(['match_id', 'team_name']).size()
    
    sns.histplot(players_per_match, bins=20, kde=True, color='skyblue')
    plt.title('Nombre de joueurs répertoriés par feuille de match')
    plt.xlabel('Nombre de joueurs')
    plt.ylabel('Fréquence')
    plt.show()