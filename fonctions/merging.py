import pandas as pd
import numpy as np
import unicodedata
import re
from rapidfuzz import process, fuzz
from fuzzywuzzy import fuzz, process
from unidecode import unidecode


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
        how="left"
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

    # Étape 4 : Configuration de l'agrégation finale
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
    colonnes_sans_count = [
        col for col in df_grouped.columns if not col.endswith("_count")
    ]

    colonnes_finales = (
        ["player_id", "Saison"]
        + colonnes_de_base_a_garder
        + [
            col
            for col in colonnes_sans_count
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
    df_sd_clean['dob_key'] = (
        pd.to_numeric(df_sd_clean['born'], errors='coerce')
        .astype('Int64')
        .astype(str)
        .str.strip()
    )

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
        df_tm_clean['date'].dt.month < 9,
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


def clean_text(text):
    if pd.isna(text): return ""
    return unidecode(str(text)).lower().strip()

def run_player_matching(df_soccerdata, df_mapping, df_tm, df_blessures, score_min=90):
    df_soccerdata = df_soccerdata.loc[:, ~df_soccerdata.columns.duplicated()].copy()
    df_mapping = df_mapping.loc[:, ~df_mapping.columns.duplicated()].copy()
    df_tm = df_tm.loc[:, ~df_tm.columns.duplicated()].copy()
    df_blessures = df_blessures.loc[:, ~df_blessures.columns.duplicated()].copy()

    results = []
    remaining = df_soccerdata.copy()
    remaining['join_key_clean'] = remaining['join_key'].apply(clean_text)
    remaining['team_clean'] = remaining['team'].apply(clean_text)
    
    # Match exact via le mapping

    df_mapping_temp = df_mapping.copy()
    df_mapping_temp['join_key_clean'] = df_mapping_temp['join_key'].apply(clean_text)
    
    merge_1 = pd.merge(
        remaining, 
        df_mapping_temp[['join_key_clean', 'tm_id']].drop_duplicates(subset=['join_key_clean']), 
        on='join_key_clean', 
        how='inner'
    )
    
    if not merge_1.empty:
        merge_1['match_method'] = 'exact_name_mapping'
        results.append(merge_1)
        remaining = remaining[~remaining['join_key_clean'].isin(merge_1['join_key_clean'])]
    
    print(f"[1] Nom exact (mapping)     : {len(merge_1):>5} | restants : {len(remaining)}")

    # Fuzzy matching sur le mapping + validation Équipe

    if not remaining.empty and not df_mapping_temp.empty:
        mapping_dict = df_mapping_temp.set_index('join_key_clean')['tm_id'].to_dict()
        mapping_keys = [k for k in mapping_dict.keys() if k] # On filtre les clés vides
        fuzzy_rows = []

        if mapping_keys:
            for _, row in remaining.iterrows():
                current_key = row['join_key_clean']
                if not current_key:
                    continue
                
                res_set = process.extractOne(current_key, mapping_keys, scorer=fuzz.token_set_ratio)
                res_sort = process.extractOne(current_key, mapping_keys, scorer=fuzz.token_sort_ratio)

                best_res = None

                if res_set and res_sort and isinstance(res_set, tuple) and isinstance(res_sort, tuple):
                    if res_set[0] == res_sort[0]:
                        best_score = max(res_set[1], res_sort[1])
                        if best_score >= (score_min - 5):
                            best_res = (res_set[0], best_score)
                    else:
                        candidate, score_sort = res_sort[0], res_sort[1]
                        if score_sort >= 75: 
                            best_res = (candidate, score_sort)

                if best_res:
                    match_data = row.to_dict()
                    match_data['tm_id'] = mapping_dict[best_res[0]]
                    match_data['match_method'] = f'fuzzy_mapping({best_res[1]:.1f})'
                    
                    if best_res[1] < 85: 
                        if fuzz.ratio(current_key, best_res[0]) < 60:
                            continue 
                            
                    fuzzy_rows.append(match_data)

        if fuzzy_rows:
            merge_2 = pd.DataFrame(fuzzy_rows)
            results.append(merge_2)
            remaining = remaining[~remaining['join_key_clean'].isin(merge_2['join_key_clean'])]
        
        print(f"[2] Fuzzy nom (mapping)     : {len(fuzzy_rows):>5} | restants : {len(remaining)}")

    # Match sur la base Transfermarkt

    if not remaining.empty and not df_tm.empty:
        cols_to_extract = ['name', 'player_id']
        if 'current_club_name' in df_tm.columns:
            cols_to_extract.append('current_club_name')
            
        df_tm_unique = df_tm[cols_to_extract].drop_duplicates(subset=['name']).copy()
        df_tm_unique['name_clean'] = df_tm_unique['name'].apply(clean_text)
        if 'current_club_name' in df_tm_unique.columns:
            df_tm_unique['club_clean'] = df_tm_unique['current_club_name'].apply(clean_text)
        
        df_tm_unique = df_tm_unique.drop_duplicates(subset=['name_clean'])
        
        # Match exact direct
        merge_3_exact = pd.merge(
            remaining,
            df_tm_unique,
            left_on='join_key_clean',
            right_on='name_clean',
            how='inner'
        )
        
        if not merge_3_exact.empty:
            merge_3_exact['tm_id'] = merge_3_exact['player_id']
            merge_3_exact['match_method'] = 'exact_direct_tm'
            results.append(merge_3_exact)
            remaining = remaining[~remaining['join_key_clean'].isin(merge_3_exact['join_key_clean'])]
            print(f"[3.1] Match direct TM (Exact) : {len(merge_3_exact):>5} | restants : {len(remaining)}")

        # Match fuzzy direct
        if not remaining.empty:
            tm_dict = df_tm_unique.set_index('name_clean').to_dict(orient='index')
            tm_keys = [k for k in tm_dict.keys() if k] # On filtre les clés vides
            tm_fuzzy_rows = []

            if tm_keys:
                for _, row in remaining.iterrows():
                    current_key = row['join_key_clean']
                    current_team = row['team_clean']
                    if not current_key:
                        continue
                    
                    res = process.extractOne(current_key, tm_keys, scorer=fuzz.token_sort_ratio)
                    
                    # PROTECTION SÉCURISÉE DU NONETYPE ICI AUSSI
                    if res and isinstance(res, tuple) and len(res) >= 2:
                        meta_tm = tm_dict[res[0]]
                        
                        if 'club_clean' in meta_tm:
                            score_club = fuzz.token_set_ratio(current_team, meta_tm['club_clean'])
                            if score_club < 50 and res[1] < 95:
                                continue
                        
                        match_data = row.to_dict()
                        match_data['tm_id'] = meta_tm['player_id']
                        match_data['match_method'] = f'fuzzy_direct_tm({res[1]:.1f})'
                        tm_fuzzy_rows.append(match_data)

            if tm_fuzzy_rows:
                merge_3_fuzzy = pd.DataFrame(tm_fuzzy_rows)
                results.append(merge_3_fuzzy)
                remaining = remaining[~remaining['join_key_clean'].isin(merge_3_fuzzy['join_key_clean'])]
            print(f"[3.2] Match direct TM (Fuzzy) : {len(tm_fuzzy_rows):>5} | restants : {len(remaining)}")

    # Nettoyage
    remaining = remaining.drop(columns=['join_key_clean', 'team_clean'], errors='ignore')
    if not results: return pd.DataFrame(), remaining

    # Assemblage final
    df_with_id = pd.concat(results, ignore_index=True)
    
    cols_to_clean = ['join_key_clean', 'team_clean', 'name_clean', 'club_clean', 'current_club_name', 'player_id']
    df_with_id = df_with_id.drop(columns=[c for c in cols_to_clean if c in df_with_id.columns], errors='ignore')
    
    cols_identity = ['player_id', 'name', 'date_of_birth', 'sub_position', 'position', 'foot', 'height_in_cm']
    cols_id_existing = [c for c in cols_identity if c in df_tm.columns]
    df_tm_identity = df_tm[cols_id_existing].drop_duplicates(subset=['player_id'])

    cols_intersection = [c for c in df_tm_identity.columns if c in df_with_id.columns and c != 'player_id']
    df_with_id = df_with_id.drop(columns=cols_intersection, errors='ignore')

    df_final = pd.merge(df_with_id, df_tm_identity, left_on='tm_id', right_on='player_id', how='left')
    
    if 'player_id' not in df_final.columns:
        df_final['player_id'] = df_final['tm_id']
    else:
        df_final['player_id'] = df_final['player_id'].fillna(df_final['tm_id'])

    df_final['name'] = df_final['name'].fillna(df_final['player'])

    if 'valuation_season_year' in df_tm.columns and 'market_value_in_eur' in df_tm.columns:
        cols_values = ['player_id', 'valuation_season_year', 'market_value_in_eur']
        df_tm_values = df_tm[cols_values].dropna(subset=['valuation_season_year']).copy()
        df_tm_values['valuation_season_year'] = df_tm_values['valuation_season_year'].astype(int)

        df_final = pd.merge(df_final, df_tm_values, left_on=['player_id', 'season_year'], right_on=['player_id', 'valuation_season_year'], how='left')

    cols_inj_metrics = [col for col in df_blessures.columns if col.startswith("injury_") and col != "injury_season_year"]
    if cols_inj_metrics and 'injury_season_year' in df_blessures.columns:
        cols_inj_to_keep = ["player_id", "injury_season_year"] + cols_inj_metrics
        df_blessures_final = df_blessures[cols_inj_to_keep]

        df_final = pd.merge(df_final, df_blessures_final, left_on=["player_id", "season_year"], right_on=["player_id", "injury_season_year"], how="left")
        df_final[cols_inj_metrics] = df_final[cols_inj_metrics].fillna(0)

    df_final = df_final.drop(columns=["injury_season_year", "valuation_season_year", "player_id_x", "player_id_y"], errors="ignore")

    return df_final, remaining