import pandas as pd
import numpy as np
import unicodedata
import re
from rapidfuzz import process, fuzz


def prepare_transfermarkt_data(df_players, df_valuations):
    """
    Fusionne et nettoie les datasets Transfermarkt joueurs + valuations.

    Objectifs :
    - conserver la valeur marchande issue de player_valuations
      (une ligne par joueur et par date)
    - supprimer les colonnes dupliquées créées par le merge pandas
    - harmoniser les noms de colonnes

    arguments:
        df_players (dataframe): dataset players.csv
        df_valuations (dataframe): dataset player_valuations.csv

    returns:
        dataframe: dataframe Transfermarkt nettoyé
    """

    df_tm = pd.merge(
        df_players,
        df_valuations,
        on="player_id",
        how="inner"
    )

    # valeur marchande dynamique (historique)
    df_tm = df_tm.rename(columns={
        "market_value_in_eur_y": "market_value_in_eur"
    })

    # suppression des doublons issus du merge
    cols_to_drop = [
        "market_value_in_eur_x",
        "current_club_id_y",
        "current_club_name_y"
    ]

    df_tm = df_tm.drop(
        columns=cols_to_drop,
        errors="ignore"
    )

    # harmonisation des colonnes conservées
    df_tm = df_tm.rename(columns={
        "current_club_id_x": "current_club_id",
        "current_club_name_x": "current_club_name"
    })

    return df_tm



def aggregate_injuries_by_season(df_blessures):
    """Regroupe l'historique des blessures par joueur et par saison et génère 
    des indicateurs médicaux avancés.

    Cette fonction prend un historique brut de blessures (une ligne par blessure)
    et l'agrège pour obtenir une ligne unique par couple (joueur, saison). 
    Elle catégorise les types de blessures à l'aide de mots-clés textuels et calcule
    automatiquement pour chaque famille :
    1. Le nombre exact d'occurrences (`_count`)
    2. Le nombre total de jours d'absence cumulés (`_nb_d`)
    3. Le nombre total de matchs manqués cumulés (`_nb_m`)
    4. Un indicateur binaire (0 ou 1) de présence de la blessure (`injury_`)

    Toute blessure ne correspondant à aucun mot-clé des familles spécifiques 
    (musculaire, genou, etc.) est automatiquement capturée et comptabilisée 
    dans la catégorie exclusive 'minor_unknown'.

    arguments :
        df_blessures : dataframe

    returns :
        dataframe regroupé par joueur et par saison avec les indicateurs médicaux avancés
    """
    df = df_blessures.copy()

    # Étape 1 : Nettoyage et typage numérique
    df["Jours_num"] = (
        df["Jours"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0)
    )
    df["Matchs_Manques_num"] = pd.to_numeric(
        df["Matchs_Manques"], errors="coerce"
    ).fillna(0)
    df["Blessure_lower"] = df["Blessure"].astype(str).str.lower()

    # Étape 2 : Définition des familles spécifiques et de leurs mots-clés associés
    familles_specifiques = {
        "musculaire": [
            "muscular",
            "hamstring",
            "ischio",
            "thigh",
            "cuisse",
            "tear",
            "strain",
            "adductor",
            "groin",
            "fibre",
        ],
        "genou": [
            "knee",
            "genou",
            "cruciate",
            "ligament",
            "croisé",
            "meniscus",
            "ménisque",
            "patella",
        ],
        "cheville_pied": [
            "ankle",
            "cheville",
            "foot",
            "pied",
            "sprain",
            "entorse",
            "malleolus",
            "achilles",
        ],
        "mollet_tibia": ["calf", "mollet", "shin", "tibia", "fibula"],
        "dos_bassin": [
            "back",
            "dos",
            "lumbar",
            "lombaire",
            "vertebra",
            "pubalgie",
            "pelvis",
            "spine",
        ],
        "trauma_severe": [
            "fracture",
            "broken",
            "surgery",
            "opération",
            "concussion",
            "trauma",
        ],
        "medical_repos": [
            "corona",
            "illness",
            "maladie",
            "cold",
            "grippe",
            "influenza",
            "infection",
            "appendicitis",
            "rest",
            "repos",
        ],
    }

    # Liste pour suivre quelles lignes ont été classées
    # Au début, aucune ligne n'est classée (Série remplie de False)
    est_classe = pd.Series(False, index=df.index)

    # Étape 3 : Création des colonnes pour les familles spécifiques
    for nom_famille, mots_cles in familles_specifiques.items():
        pattern = "|".join(mots_cles)
        mask = df["Blessure_lower"].str.contains(pattern, na=False)

        # On met à jour l'indicateur global des lignes classées
        est_classe = est_classe | mask

        df[f"is_{nom_famille}"] = mask.astype(int)
        df[f"days_{nom_famille}"] = np.where(mask, df["Jours_num"], 0)
        df[f"matches_{nom_famille}"] = np.where(
            mask, df["Matchs_Manques_num"], 0
        )

    # Le masque correspond à l'inverse exact de tout ce qui a été classé au-dessus
    mask_fourre_tout = ~est_classe

    df["is_minor_unknown"] = mask_fourre_tout.astype(int)
    df["days_minor_unknown"] = np.where(mask_fourre_tout, df["Jours_num"], 0)
    df["matches_minor_unknown"] = np.where(
        mask_fourre_tout, df["Matchs_Manques_num"], 0
    )

    # Liste complète de toutes les familles pour la suite de l'algorithme
    toutes_familles = list(familles_specifiques.keys()) + ["minor_unknown"]

    # ÉTAPE 4 : Configuration de l'agrégation finale
    agg_dict = {
        "Blessure": "count",
        "Jours_num": "sum",
        "Matchs_Manques_num": "max",
    }

    colonnes_de_base_a_garder = [
        col
        for col in df_blessures.columns
        if col
        not in [
            "player_id",
            "Saison",
            "Debut",
            "Fin",
            "Blessure",
            "Jours",
            "Matchs_Manques",
            "Jours_num",
            "Matchs_Manques_num",
            "Blessure_lower",
        ]
    ]

    for col in colonnes_de_base_a_garder:
        agg_dict[col] = "last"

    # On applique dynamiquement les agrégations pour absolument toutes les familles
    for nom_famille in toutes_familles:
        agg_dict[f"is_{nom_famille}"] = "sum"
        agg_dict[f"days_{nom_famille}"] = "sum"
        agg_dict[f"matches_{nom_famille}"] = "sum"

    df_grouped = (
        df.groupby(["player_id", "Saison"]).agg(agg_dict).reset_index()
    )

    # Étape 5 : Renommage propre des colonnes
    rename_dict = {
        "Blessure": "injury_nb_total",
        "Jours_num": "injury_days_total",
        "Matchs_Manques_num": "injury_matches_max_single",
    }

    for nom_famille in toutes_familles:
        rename_dict[f"is_{nom_famille}"] = f"injury_{nom_famille}_count"
        rename_dict[f"days_{nom_famille}"] = f"injury_{nom_famille}_nb_d"
        rename_dict[f"matches_{nom_famille}"] = f"injury_{nom_famille}_nb_m"

    df_grouped = df_grouped.rename(columns=rename_dict)

    # Génération de la variable binaire (0 ou 1)
    for nom_famille in toutes_familles:
        df_grouped[f"injury_{nom_famille}"] = np.where(
            df_grouped[f"injury_{nom_famille}_count"] > 0, 1, 0
        )

    # Étape 6 : Réorganisation esthétique des colonnes
    colonnes_finales = (
        ["player_id", "Saison"]
        + colonnes_de_base_a_garder
        + [
            col
            for col in df_grouped.columns
            if col not in ["player_id", "Saison"] + colonnes_de_base_a_garder
        ]
    )

    return df_grouped[colonnes_finales]


def match_player_data(df_mapping, df_soccerdata, df_tm, df_blessures):
    """
    Normalise les noms et les dates de naissance pour harmoniser trois sources de données
    footballistiques.

    Cette fonction prépare les dataframes pour une jointure ultérieure en créant des clés 
    de jointure standardisées ('join_key') et en uniformisant les formats de date de naissance. 
    Elle traite spécifiquement les encodages de caractères, la suppression des accents/majuscules 
    et l'extraction de l'année de naissance.

    arguments:
        df_mapping (dataframe): Dataframe de correspondance (issu de worldfootballR) 
            contenant au moins la colonne 'PlayerFBref'.
        df_soccerdata (dataframe): Dataframe contenant les statistiques de jeu 
            (Soccerdata) avec les colonnes 'player' et 'born'.
        df_tm (dataframe): Dataframe issu de Transfermarkt contenant les colonnes 
            'first_name', 'last_name', 'name', 'date_of_birth' et 'date'.

    returns:
        tuple[dataframe, dataframe, dataframe]: Un triplet contenant :
            - df_mapping_clean : Mapping avec 'join_key' normalisée.
            - df_sd_clean : Statistiques SoccerData avec 'join_key' et 'dob_key' (année).
            - df_tm_clean : Données Transfermarkt avec 'join_key', 'join_key_full',
              'dob_key' (année) et la valorisation la plus récente par 'valuation_season_year'.
    """
    
    # Nettoyage de df_mapping (issu de worldfootballR)
    df_mapping_clean = df_mapping.copy()
    df_mapping_clean['PlayerFBref'] = df_mapping_clean['PlayerFBref'].apply(fix_encoding)
    df_mapping_clean['join_key'] = df_mapping_clean['PlayerFBref'].apply(normalize_name)
    df_mapping_clean = df_mapping_clean.drop_duplicates(subset=['join_key', 'tm_id'])

    # Nettoyage de df_soccerdata
    df_sd_clean = df_soccerdata.copy()
    df_sd_clean['join_key'] = df_sd_clean['player'].apply(normalize_name)
    # On s'assure que dob_key est bien une chaîne de l'année (ex: "1998")
    df_sd_clean['dob_key'] = df_sd_clean['born'].astype(str).str.strip()

    def extract_season_start(season):
        season = str(season)

        if len(season) == 4:
            return 2000 + int(season[:2])

        return None

    df_sd_clean['season_year'] = (
        df_sd_clean['season']
        .apply(extract_season_start)
    )

    # Nettoyage de df_tm
    df_tm_clean = df_tm.copy()
    
    # Conversion de la date de valorisation en datetime (à faire plus haut pour le filtrage)
    df_tm_clean['date'] = pd.to_datetime(df_tm_clean['date'], errors='coerce')

    # Suppression du mois d'août
    df_tm_clean = df_tm_clean[df_tm_clean['date'].dt.month != 8]
    
    # Création de la clé par concaténation Prénom + Nom
    df_tm_clean['join_key'] = (
        df_tm_clean['first_name'].apply(normalize_name) + ' ' + 
        df_tm_clean['last_name'].apply(normalize_name)
    ).str.strip()
    
    # Clé alternative sur le nom complet
    df_tm_clean['join_key_full'] = df_tm_clean['name'].apply(normalize_name)
    
    # Extraction de l'année de naissance
    df_tm_clean['dob_key'] = pd.to_datetime(
        df_tm_clean['date_of_birth'], errors='coerce'
    ).dt.strftime('%Y')

    df_tm_clean['valuation_season_year'] = np.where(
        df_tm_clean['date'].dt.month < 7,
        df_tm_clean['date'].dt.year - 1,
        df_tm_clean['date'].dt.year
    )

    # On trie par date pour s'assurer que la plus récente soit à la fin
    df_tm_clean = df_tm_clean.sort_values('date')

    # On garde la valorisation la plus récente de la saison
    df_tm_clean = (
        df_tm_clean
        .groupby(['player_id', 'valuation_season_year'], as_index=False)
        .last()
    )

    # Le dataframe des blessures Transfermarkt

    # Fonction pour convertir le format "23/24" de Transfermarkt en année de début
    def convert_injury_season(saison_str):
        saison_str = str(saison_str).strip()
        parts = saison_str.split("/")
        if len(parts) == 2:
            try:
                annee_courte = int(parts[0])
                # Gère le passage à l'an 2000 (ex: "99/00" vs "23/24")
                if annee_courte > 50:
                    return 1900 + annee_courte
                else:
                    return 2000 + annee_courte
            except ValueError:
                return None
        return None

    # On crée une colonne temporelle standardisée 'injury_season_year' (ex: 2023)
    df_blessures["injury_season_year"] = df_blessures[
        "Saison"
    ].apply(convert_injury_season)

    return df_mapping_clean, df_sd_clean, df_tm_clean, df_blessures

# Fonctions nécessaires pour la synchronisation des noms FBref et ceux du mapping

def fix_encoding(name):
    """
    Corrige les erreurs d'encodage courantes dans les noms de joueurs.
    
    Tente de réparer les chaînes de caractères où des caractères UTF-8 ont été 
    interprétés comme du Latin-1 (ex: "Mesut Ã–zil" -> "Mesut Özil").

    arguments:
        name (str): La chaîne de caractères potentiellement mal encodée.

    returns:
        str: La chaîne corrigée ou la chaîne originale en cas d'échec.
    """
    if not isinstance(name, str): return ""
    try:
        return name.encode('raw_unicode_escape').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name

def normalize_name(name):
    """
    Standardise un nom pour faciliter la comparaison entre différentes sources.
    
    Le processus inclut :
    1. La correction d'encodage via `fix_encoding`.
    2. La décomposition NFD pour séparer les accents.
    3. La suppression des caractères non-ASCII.
    4. La mise en minuscule et le remplacement des caractères spéciaux par des espaces.
    5. Le nettoyage des espaces multiples.

    arguments:
        name (str): Le nom brut du joueur.

    returns:
        str: Le nom normalisé (ex: "José-María Callejón" -> "jose maria callejon").
    """
    if not isinstance(name, str): return ""
    normalized = unicodedata.normalize('NFD', fix_encoding(name))
    ascii_name = normalized.encode('ascii', 'ignore').decode("utf-8").lower().strip()
    ascii_name = re.sub(r'[^a-z0-9 ]', ' ', ascii_name)
    return re.sub(r'\s+', ' ', ascii_name).strip()


def remove_matched(df, keys_matched):
    """
    Filtre un dataframe pour ne conserver que les joueurs non identifiés.

    Cette fonction est utilisée entre chaque étape de matching pour réduire 
    le dataset 'remaining' et éviter de traiter plusieurs fois les mêmes joueurs.

    arguments:
        df (dataframe): Le dataframe contenant les joueurs restants.
        keys_matched (set): Un ensemble de 'join_key' ayant déjà trouvé 
            une correspondance dans l'étape précédente.

    returns:
        dataframe: Une copie du Dataframe original excluant les clés fournies.
    """
    return df[~df['join_key'].isin(keys_matched)].copy()


def run_player_matching(df_soccerdata, df_mapping, df_tm, df_blessures, score_min=90):
    """
    Exécute un pipeline de réconciliation itératif pour lier les données SoccerData à Transfermarkt.

    Le processus suit une logique de filtrage successif :
    1. Match exact : Jointure directe sur la clé 'join_key' via le mapping.
    2. Fuzzy Match : Pour les joueurs restants, recherche par similarité textuelle
            (fuzz.token_sort_ratio) dans le mapping.
    3. Consolidation : Fusion des IDs trouvés avec le dataset Transfermarkt complet pour récupérer 
       les métadonnées (valeurs, postes, etc.).

    arguments:
        df_soccerdata (dataframe): Données sources contenant les statistiques joueurs.
        df_mapping (dataframe): Table de référence faisant le lien entre les noms FBref et
            les ids Transfermarkt.
        df_tm (dataframe): Dataframe complet de Transfermarkt.
        score_min (int, optional): Seuil de similarité pour le fuzzy matching (0-100). 
            Par défaut à 90.

    returns:
        tuple[dataframe, dataframe]: Un tuple contenant :
            - df_final : Dataframe des joueurs matchés avec toutes leurs informations cumulées.
            - remaining : Dataframe des joueurs n'ayant trouvé aucune correspondance.
    """
    results = []
    remaining = df_soccerdata.copy()
    
    # Match exact via le mapping
    # On ne merge que les colonnes nécessaires pour identifier le match
    merge_1 = pd.merge(
        remaining, 
        df_mapping[['join_key', 'tm_id']], 
        on='join_key', 
        how='inner' # Inner pour ne garder que ceux qui ont matché
    )
    
    if not merge_1.empty:
        merge_1['match_method'] = 'exact_name_mapping'
        results.append(merge_1)
        # Mise à jour de remaining : on retire ceux dont la join_key est dans merge_1
        remaining = remaining[~remaining['join_key'].isin(merge_1['join_key'])]
    
    print(f"[1] Nom exact (mapping)     : {len(merge_1):>5} | restants : {len(remaining)}")

    # Fuzzy matching sur le mapping
    if not remaining.empty:
        mapping_keys = df_mapping['join_key'].tolist()
        fuzzy_rows = []

        for _, row in remaining.iterrows():
            res = process.extractOne(row['join_key'], mapping_keys, scorer=fuzz.token_sort_ratio)
            if res and res[1] >= score_min:
                tm_id = df_mapping[df_mapping['join_key'] == res[0]].iloc[0]['tm_id']
                # On combine les données de la ligne avec l'ID trouvé
                match_data = row.to_dict()
                match_data['tm_id'] = tm_id
                match_data['match_method'] = f'fuzzy_name({res[1]})'
                fuzzy_rows.append(match_data)

        merge_2 = pd.DataFrame(fuzzy_rows)
        if not merge_2.empty:
            results.append(merge_2)
            remaining = remaining[~remaining['join_key'].isin(merge_2['join_key'])]
        
        print(f"[2] Fuzzy nom (mapping)     : {len(merge_2):>5} | restants : {len(remaining)}")

    # Assemblage et fusion finale
    if not results:
        print("Aucun match trouvé.")
        return pd.DataFrame(), remaining

    # Assemblage de tous les niveaux de résultats
    df_with_id = pd.concat(results, ignore_index=True)

    # Préparation de df_tm pour la fusion
    df_tm_final = df_tm.copy()

    # garder uniquement les colonnes utiles
    cols_to_keep = [
        'player_id',
        'valuation_season_year',
        'market_value_in_eur',
        'date',
        'date_of_birth',
        'name',
        'join_key',
        'join_key_full',
        'dob_key'
    ]

    df_tm_final = df_tm_final[cols_to_keep]

    df_tm_final = df_tm_final.rename(columns={
        'join_key': 'tm_join_key',
        'join_key_full': 'tm_join_key_full',
        'dob_key': 'tm_dob_key'
    })

    # Fusion finale pour récupérer les infos de Transfermarkt
    df_final = pd.merge(
        df_with_id,
        df_tm_final,
        left_on=['tm_id', 'season_year'],
        right_on=['player_id', 'valuation_season_year'],
        how='left'
    )

    df_blessures_final = df_blessures.copy()
    # Supprime les colonnes portant exactement le même nom dans le df_blessures
    df_blessures_final = df_blessures_final.loc[:, ~df_blessures_final.columns.duplicated()]

    # Éviter les conflits de colonnes si d'anciennes versions de colonnes de base s'y trouvent
    # On ne garde que la clé temporelle calculée et les variables de blessures calculées
    cols_inj_to_keep = ["player_id", "injury_season_year"] + [
        col
        for col in df_blessures_final.columns
        if col.startswith("injury_") and col != "injury_season_year"
    ]
    df_blessures_final = df_blessures_final[cols_inj_to_keep]

    # Left join pour préserver toutes les lignes de performance construites au-dessus
    df_final = pd.merge(
        df_final,
        df_blessures_final,
        left_on=["player_id", "season_year"],
        right_on=["player_id", "injury_season_year"],
        how="left",
    )

    # Les joueurs sans correspondance médicale n'ont pas eu de blessures cette saison-là
    cols_medicales = [
        col
        for col in df_blessures_final.columns
        if col.startswith("injury_")
    ]
    df_final[cols_medicales] = df_final[cols_medicales].fillna(0)

    # Suppression de la clé de jointure dupliquée devenue inutile
    df_final = df_final.drop(columns=["injury_season_year"], errors="ignore")

    return df_final, remaining