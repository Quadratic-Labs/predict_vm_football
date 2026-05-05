import pandas as pd
import unicodedata
import re
from rapidfuzz import process, fuzz
import os
import sys


def match_player_data(df_mapping, df_soccerdata, df_tm):
    """
    Normalise les noms et les dates de naissance, puis effectue le mapping 
    entre FBref (via soccerdata) et Transfermarkt.
    """
    
    # 1. Nettoyage de df_mapping (FBref Reference)
    df_mapping_clean = df_mapping.copy()
    df_mapping_clean['PlayerFBref'] = df_mapping_clean['PlayerFBref'].apply(fix_encoding)
    df_mapping_clean['join_key'] = df_mapping_clean['PlayerFBref'].apply(normalize_name)

    # 2. Nettoyage de df_soccerdata (Stats)
    df_sd_clean = df_soccerdata.copy()
    df_sd_clean['join_key'] = df_sd_clean['player'].apply(normalize_name)
    # On s'assure que dob_key est bien une chaîne de l'année (ex: "1998")
    df_sd_clean['dob_key'] = df_sd_clean['born'].astype(str).str.strip()

    # 3. Nettoyage de df_tm (Market Value / Bio)
    df_tm_clean = df_tm.copy()
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

    return df_mapping_clean, df_sd_clean, df_tm_clean

# Fonctions nécessaires pour la synchronisation des noms FBref et ceux du mapping

def fix_encoding(name):
    if not isinstance(name, str): return ""
    try:
        return name.encode('raw_unicode_escape').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name

def normalize_name(name):
    if not isinstance(name, str): return ""
    normalized = unicodedata.normalize('NFD', fix_encoding(name))
    ascii_name = normalized.encode('ascii', 'ignore').decode("utf-8").lower().strip()
    ascii_name = re.sub(r'[^a-z0-9 ]', ' ', ascii_name)
    return re.sub(r'\s+', ' ', ascii_name).strip()


def remove_matched(df, keys_matched):
    return df[~df['join_key'].isin(keys_matched)].copy()


def run_player_matching(df_soccerdata, df_mapping, df_tm, score_min=90):
    """
    Réalise le matching itératif entre SoccerData et Transfermarkt.
    """
    results = []
    remaining = df_soccerdata.copy()
    
    # --- NIVEAU 1 : Match exact via le mapping ---
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

    # --- NIVEAU 2 : Fuzzy matching sur le mapping ---
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

    # --- ASSEMBLAGE ET FUSION FINALE ---
    if not results:
        print("Aucun match trouvé.")
        return pd.DataFrame(), remaining

    # Assemblage de tous les niveaux de résultats
    df_with_id = pd.concat(results, ignore_index=True)

    # Préparation de df_tm pour la fusion
    df_tm_final = df_tm.rename(columns={
        'join_key':      'tm_join_key',
        'join_key_full': 'tm_join_key_full',
        'dob_key':       'tm_dob_key'
    })

    # Fusion finale pour récupérer les infos de Transfermarkt (valeur, etc.)
    df_final = pd.merge(
        df_with_id, 
        df_tm_final, 
        left_on='tm_id', 
        right_on='player_id', 
        how='left'
    )

    return df_final, remaining