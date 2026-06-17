import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import pickle


def nettoyer_age_et_dates(df, colonnes_dates=None):
    """Nettoie et normalise les colonnes de dates et calcule l'âge exact du joueur

    au moment de la saison de manière robuste.
    """
    # Copie de sécurité pour éviter le SettingWithCopyWarning
    df_clean = df.copy()

    # 1. NETTOYAGE DES COLONNES DE DATES
    # Si l'utilisateur n'a pas fourni de liste, on détecte automatiquement les colonnes de date
    if colonnes_dates is None:
        colonnes_dates = [
            col
            for col in df_clean.columns
            if "date" in col.lower() or "dob" in col.lower()
        ]

    if colonnes_dates:
        print(f"Traitement des colonnes de dates : {colonnes_dates}")
        for col in colonnes_dates:
            if col in df_clean.columns:
                # Conversion au format datetime Pandas officiel
                df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
                # Écrase l'heure en la figeant strictly à minuit
                df_clean[col] = df_clean[col].dt.normalize()
        print(" -> Toutes les heures ont été remises à minuit.")
    else:
        print("Aucune colonne de date détectée ou fournie.")

    # 2. EXTRACTION DE L'ANNÉE DE NAISSANCE ('born')
    if "date_of_birth" in df_clean.columns:
        df_clean["born"] = df_clean["date_of_birth"].dt.year
        print(" -> Colonne 'born' (année de naissance) extraite.")
    else:
        df_clean["born"] = np.nan
        print(" -> Attention : 'date_of_birth' manquante, 'born' initialisée à NaN.")

    # 3. CALCUL ET NETTOYAGE ROBUSTE DE L'ÂGE
    print("Calcul et nettoyage de la colonne 'age'...")

    # Étape A : On calcule l'âge théorique basé sur la saison et l'année de naissance
    if "season_year" in df_clean.columns and "date_of_birth" in df_clean.columns:
        df_clean["age_calcule"] = df_clean["season_year"] - df_clean["born"]
    else:
        df_clean["age_calcule"] = np.nan

    # Étape B : On nettoie l'âge brut textuel (ex: '23-328' -> 23) au cas où on en aurait besoin
    if "age" in df_clean.columns:
        df_clean["age_brut"] = (
            df_clean["age"].astype(str).str.extract(r"^(\d+)")
        )
        df_clean["age_brut"] = pd.to_numeric(
            df_clean["age_brut"], errors="coerce"
        )
    else:
        df_clean["age_brut"] = np.nan

    # Étape C : Stratégie de fusion (On prend l'âge calculé, sinon l'âge brut de FBref)
    df_clean["age"] = df_clean["age_calcule"].fillna(df_clean["age_brut"])

    # Étape D : Sécurité absolue pour les NaN restants avant la conversion en Int
    # Remplacement par la médiane des âges du dataset pour ne pas fausser le Random Forest
    if df_clean["age"].isna().sum() > 0:
        valeur_remplacement = df_clean["age"].median()
        # Si tout est vide (cas extrême), on met une valeur par défaut
        if pd.isna(valeur_remplacement):
            valeur_remplacement = 25
        df_clean["age"] = df_clean["age"].fillna(valeur_remplacement)
        print(
            f" -> Attention : {df_clean['age'].isna().sum()} âges manquants remplacés par la médiane ({int(valeur_remplacement)} ans)."
        )

    # Conversion finale STRICTE en entier
    df_clean["age"] = df_clean["age"].astype(int)

    # Nettoyage des colonnes temporaires pour garder le DataFrame propre
    colonnes_temporaires = ["age_calcule", "age_brut"]
    df_clean = df_clean.drop(
        columns=[col for col in colonnes_temporaires if col in df_clean.columns]
    )

    print(" -> Colonne 'age' convertie strictement en entiers (int).")
    print("Nettoyage de l'âge et des dates terminé.\n")

    return df_clean

def verifier_doublons_metier_et_techniques(dataframe):
    """Analyse et distingue les doublons de mercato des doublons techniques

    dans une base de données de statistiques de football.
    """
    print("Recherche de doublons")

    # Vérification de la présence des colonnes indispensables
    col_joueur = "player" if "player" in dataframe.columns else None
    col_saison = "season" if "season" in dataframe.columns else None
    col_club = "team" if "team" in dataframe.columns else None

    if not (col_joueur and col_saison and col_club):
        print(
            "Impossible de lancer le diagnostic : les colonnes 'player', 'season' ou 'team' sont introuvables."
        )
        return None

    # Calcul des doublons globaux (Joueur + Saison) -> Mercato + Technique
    nb_doublons_saison = dataframe.duplicated(
        subset=[col_joueur, col_saison]
    ).sum()

    # Calcul des doublons stricts (Joueur + Saison + Club) -> Technique uniquement
    nb_doublons_techniques = dataframe.duplicated(
        subset=[col_joueur, col_saison, col_club]
    ).sum()

    # Déduction logique des doublons liés purement au Mercato (changement de club)
    nb_doublons_mercato = nb_doublons_saison - nb_doublons_techniques

    # Rapport des résultats

    # Présence de doublons techniques
    if nb_doublons_techniques > 0:
        print(
            f" Attention : {nb_doublons_techniques} lignes sont des doublons techniques stricts."
        )
        print(
            "(Même joueur, même saison, même club -> Erreur d'extraction/jointure)"
        )
        print("\nExemple de lignes techniques concernées :")
        lignes_doublons_tech = dataframe[
            dataframe.duplicated(
                subset=[col_joueur, col_saison, col_club], keep=False
            )
        ]
        print(
            lignes_doublons_tech[[col_joueur, col_saison, col_club]].head(6)
        )
    else:
        print(
            " Parfait ! Aucune répétition technique (Même joueur, même saison, même club)."
        )

    print()

    # Présence de doublons de Mercato
    if nb_doublons_mercato > 0:
        print(
            f"{nb_doublons_mercato} lignes correspondent à des doublons de mercato"
        )
        print(
            "(Même joueur, même saison, mais clubs différents -> Transferts de mi-saison)"
        )
        print("\nExemple de joueurs transférés concernés :")

        # Pour n'isoler que le mercato, on filtre les doublons joueur+saison qui n'ont pas le même club
        lignes_tous_doublons = dataframe[
            dataframe.duplicated(subset=[col_joueur, col_saison], keep=False)
        ]
        lignes_mercato = lignes_tous_doublons.drop_duplicates(
            subset=[col_joueur, col_saison, col_club]
        )
        print(lignes_mercato[[col_joueur, col_saison, col_club]].head(6))
    else:
        print(
            "Aucun doublon de mercato détecté (Chaque joueur n'a qu'un seul club par saison)."
        )

    # Retourne les résultats sous forme de dictionnaire (optionnel)
    return {
        "doublons_techniques": nb_doublons_techniques,
        "doublons_mercato": nb_doublons_mercato,
    }


def fusionner_doublons_techniques(df, chemin_sauvegarde=None):
    """Fusionne les doublons techniques (Joueur + Saison + Club) en combinant

    les informations de toutes les lignes pour ne perdre aucune donnée.
    """
    print(f"Format initial de la base : {df.shape}")

    # Copie de sécurité
    df_travail = df.copy()

    # On trie le dataset par nos clés métier
    df_tri = df_travail.sort_values(by=["player", "season", "team"])

    # Utilisation de groupby + first()
    # .first() dans un groupby Pandas prend automatiquement la première valeur NON-NAN
    # trouvée pour chaque colonne de manière indépendante.
    df_fusionne = df_tri.groupby(
        ["player", "season", "team"], as_index=False
    ).first()

    print(
        f"Format après fusion intelligente des doublons : {df_fusionne.shape}"
    )

    if chemin_sauvegarde:
        df_fusionne.to_csv(chemin_sauvegarde, index=False)
        print(f"Base fusionnée sauvegardée dans : {chemin_sauvegarde}")

    return df_fusionne



def fusionner_et_recalculer_mercato(df):
    """Fusionne les doublons liés au mercato (Joueur + Saison), puis recalcule

    exactement toutes les variables de ratio et statistiques par 90 minutes.
    """
    print(f"Format avant fusion mercato : {df.shape}")

    aggregation_rules = {}

    # Mots-clés des variables numériques à conserver telles quelles (sans somme)
    # Plus souple et sécurisé contre les variantes de casse ou d'espaces
    mots_cles_exceptions = [
        "born",
        "market_value",
        "value_in_eur",
        "season_year",
        "dob_key",
        "tm_id",
        "player_id",
        "valuation_season_year",
        "tm_dob_key",
    ]

    # Liste automatique des colonnes numériques
    cols_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()


    for col in df.columns:
        # On ignore les clés du groupby
        if col in ["player", "season"]:
            continue

        col_lower = col.lower().strip()

        # Les exceptions techniques à ne pas modifier (dernier en date)
        if any(keyword in col_lower for keyword in mots_cles_exceptions):
            aggregation_rules[col] = "last"

        # Les colonnes de pourcentage (%) ou ratios intermédiaires (moyenne)
        elif (
            "pct" in col_lower
            or "%" in col
            or "rate" in col_lower
            or "accuracy" in col_lower
            or "/90" in col_lower
            or "per 90" in col_lower
            or "playing time_mn/mp" in col_lower
            or "starts_mn/starts" in col_lower
            or "/starts" in col_lower
            or "subs_mn/subs" in col_lower
        ):
            aggregation_rules[col] = "mean"

        # Les autres colonnes numériques (Buts, Minutes, Passes...) (somme)
        elif col in cols_numeriques:
            aggregation_rules[col] = "sum"

        # Les colonnes textuelles (team, league...) (dernier club en date)
        else:
            aggregation_rules[col] = "last"

    # Application de la fusion par GroupBy après un tri chronologique/alphabétique
    df_tri = df.sort_values(by=["player", "season"])
    df_fusion_mercato = df_tri.groupby(
        ["player", "season"], as_index=False
    ).agg(aggregation_rules)

    print(f"Format après fusion mercato : {df_fusion_mercato.shape}")


    print("Recalcul des ratios et statistiques par 90 minutes...")

    # Gestion des divisions par zéro
    # On remplace temporairement les 0 par des NaN dans les dénominateurs
    t90s = df_fusion_mercato["Playing Time_90s"].replace(0, np.nan)
    sh = df_fusion_mercato["Standard_Sh"].replace(0, np.nan)
    sot = df_fusion_mercato["Standard_SoT"].replace(0, np.nan)
    mp = df_fusion_mercato["Playing Time_MP"].replace(0, np.nan)


    # Ratios de buts et passes décisives par 90 minutes
    df_fusion_mercato["Per 90 Minutes_Gls"] = (
        df_fusion_mercato["Performance_Gls"] / t90s
    )
    df_fusion_mercato["Per 90 Minutes_Ast"] = (
        df_fusion_mercato["Performance_Ast"] / t90s
    )
    df_fusion_mercato["Per 90 Minutes_G+A"] = (
        df_fusion_mercato["Per 90 Minutes_Gls"]
        + df_fusion_mercato["Per 90 Minutes_Ast"]
    )

    # Efficacité face au but (Buts par tir et par tir cadré)
    df_fusion_mercato["Standard_G/Sh"] = (
        df_fusion_mercato["Standard_Gls"] / sh
    )
    df_fusion_mercato["Standard_G/SoT"] = (
        df_fusion_mercato["Standard_Gls"] / sot
    )

    # Précision des tirs (% de tirs cadrés)
    df_fusion_mercato["Standard_SoT%"] = (
        df_fusion_mercato["Standard_SoT"] / sh
    ) * 100

    # Volume de tirs par 90 minutes
    df_fusion_mercato["Standard_Sh/90"] = (
        df_fusion_mercato["Standard_Sh"] / t90s
    )
    df_fusion_mercato["Standard_SoT/90"] = (
        df_fusion_mercato["Standard_SoT"] / t90s
    )

    # Gestion du temps de jeu (Minutes par match et % de minutes jouées)
    df_fusion_mercato["Playing Time_Mn/MP"] = (
        df_fusion_mercato["Playing Time_Min"] / mp
    )
    df_fusion_mercato["Playing Time_Min%"] = (
        df_fusion_mercato["Playing Time_Min"] / (mp * 90)
    ) * 100

    # Nettoyage et finitions
    list_recalcul = [
        "Per 90 Minutes_Gls",
        "Per 90 Minutes_Ast",
        "Per 90 Minutes_G+A",
        "Standard_G/Sh",
        "Standard_G/SoT",
        "Standard_SoT%",
        "Standard_Sh/90",
        "Standard_SoT/90",
        "Playing Time_Mn/MP",
        "Playing Time_Min%",
    ]

    # Remplacement des infinis (générés par x / 0) et des NaN par des 0 propres
    for col in list_recalcul:
        if col in df_fusion_mercato.columns:
            df_fusion_mercato[col] = df_fusion_mercato[col].replace(
                [np.inf, -np.inf], np.nan
            )

    # Limitation stricte à 2 chiffres après la virgule
    df_fusion_mercato[list_recalcul] = df_fusion_mercato[list_recalcul].round(2)

    print(
        "Base de données fusionnée et variables recalculées avec exactitude.\n"
    )

    return df_fusion_mercato


import pandas as pd


def supprimer_colonnes_du_dataset(df, colonnes_a_supprimer):
    """Supprime une liste de colonnes spécifiée d'un DataFrame de manière sécurisée.

    param df: Le DataFrame d'origine.
    param colonnes_a_supprimer: Liste de chaînes de caractères (noms des
    colonnes).
    return: Un nouveau DataFrame nettoyé.
    """
    # On filtre la liste pour ne garder que les colonnes qui existent vraiment dans le DF
    # (Évite que le code ne plante si tu lances la cellule deux fois d'affilée)
    colonnes_existantes = [c for c in colonnes_a_supprimer if c in df.columns]

    # Suppression
    if colonnes_existantes:
        df_nettoye = df.drop(columns=colonnes_existantes)
        print(
            f"{len(colonnes_existantes)} colonne(s) supprimée(s) : {colonnes_existantes}"
        )
    else:
        df_nettoye = df.copy()
        print("Aucune des colonnes spécifiées n'a été trouvée dans le dataset.")

    return df_nettoye


def diagnostiquer_valeurs_manquantes(df, seuil=0.30):
    """Analyse, affiche et retourne les colonnes dont le taux de valeurs manquantes

    dépasse le seuil spécifié.
    """
    print(
        f"Diagnostic des valeurs manquantes (Seuil > {seuil*100:.0f}%)"
    )

    # Calcul du taux de valeurs manquantes pour chaque colonne
    taux_manquants = df.isnull().mean()

    # Filtrage des colonnes qui dépassent le seuil
    colonnes_vides_serie = taux_manquants[taux_manquants > seuil].sort_values(
        ascending=False
    )

    if not colonnes_vides_serie.empty:
        # Affichage détaillé de chaque colonne trouvée
        for col, tx in colonnes_vides_serie.items():
            print(f"   • {col} : {tx*100:.1f}% de valeurs manquantes")

        # Extraction de la liste des noms de colonnes pour pouvoir la retourner
        liste_colonnes_vides = colonnes_vides_serie.index.tolist()
        print(
            f"\nTotal : {len(liste_colonnes_vides)} colonnes dépassent le seuil de {seuil*100:.0f}%."
        )
    else:
        print(f"   • Aucune colonne ne dépasse {seuil*100:.0f}% de lignes vides.")
        liste_colonnes_vides = []


def nettoyer_valeurs_manquantes_ciblees(df):
    """Effectue le traitement ciblé des NaN :

    1. Remplace par 0 les variables de performance spécifiques.
    2. Propage (ffill/bfill) les données fixes des joueurs d'une saison à l'autre.
    3. Remplace les NaN résiduels par des valeurs neutres ('Inconnu' ou 0).
    """
    print("Début du traitement ciblé des valeurs manquantes...")
    
    # Copie locale pour éviter les warnings "SettingWithCopy"
    df_clean = df.copy()

    # Remplacement des NA par des 0 dans les colonnes de performance spécifiques
    cols_NA = [
        "Penalty Kicks_Save%", "Performance_CS%", "Performance_Save%",
        "Standard_G/SoT", "Standard_G/Sh", "Standard_SoT%", "Per 90 Minutes_Gls",
        "Per 90 Minutes_Ast", "Per 90 Minutes_G+A", "Standard_SoT/90", "Standard_Sh/90"
    ]
    # On ne filtre que les colonnes réellement présentes pour éviter les plantages
    cols_NA_presentes = [c for c in cols_NA if c in df_clean.columns]
    
    if cols_NA_presentes:
        df_clean[cols_NA_presentes] = df_clean[cols_NA_presentes].fillna(0)
    print(f" -> {len(cols_NA_presentes)} colonnes de performance nettoyées (NaN -> 0).")

    # Propagation inter-saisons des données fixes des joueurs
    cols_a_propager = [
        "foot", "tm_dob_key", "date_of_birth", "sub_position", 
        "player_id", "tm_join_key_full", "tm_join_key", "name", "position"
    ]
    cols_propager_presentes = [c for c in cols_a_propager if c in df_clean.columns]

    if cols_propager_presentes and "player" in df_clean.columns:
        # Optimisation : on groupe et on applique ffill puis bfill d'un coup sur le bloc de colonnes
        df_clean[cols_propager_presentes] = (
            df_clean.groupby("player")[cols_propager_presentes]
            .ffill()
            .bfill()
        )
        print(f" -> Propagation inter-saisons terminée pour {len(cols_propager_presentes)} colonnes fixes.")

    # Traitement des NA résiduels
    for col in cols_propager_presentes:
        if col == "player_id":
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = df_clean[col].fillna("Inconnu")
            
    print("Finitions terminées (Derniers NaN résiduels convertis en valeurs neutres).")
    
    return df_clean


def lister_variables_categorielles(df):
    """Identifie, affiche le nombre de modalités uniques et retourne

    la liste des variables catégorielles présentes dans le dataframe.
    """
    print("Variables catégorielles du dataset :")

    # Sélection automatique des colonnes catégorielles (object et category)
    cols_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if cols_cat:
        print("Liste des variables catégorielles détectées :")
        # Boucle pour calculer et afficher les modalités de chaque colonne
        for col in cols_cat:
            nb_uniques = df[col].nunique()
            print(f"   • {col} ({nb_uniques} modalités uniques)")

        print(f"\nTotal : {len(cols_cat)} variables catégorielles trouvées.")
    else:
        print("   • Aucune variable catégorielle détectée dans ce dataset.")


    # Retourne la liste des noms de colonnes
    return cols_cat



def encoder_dataset_football(
    df, colonnes_categoriques, df_fifa_historique=None
):
    """Fonction modulaire mise à jour. Remplace la colonne 'nation' par 10

    colonnes binaires (classement_FIFA_1 à 10) basées sur le rang réel de la
    nation du joueur pour la saison donnée.

    Paramètres:
    -----------
    df : dataframe
        Le dataset principal (1 ligne = 1 joueur pour 1 saison).
        Doit contenir les colonnes 'nation' et 'season_year'.
    colonnes_categoriques : list of str
        La liste des colonnes à encoder ('pos', 'league', 'foot', etc.).
    df_fifa_historique : dataframe
        Le dataframe du Top 10 FIFA filtré contenant les colonnes:
        ['country_abrv', 'rank', 'season_year'].
    """
    df_encoded = df.copy()
    print(f"Format initial avant encodage : {df_encoded.shape}")

    cols_a_traiter = list(colonnes_categoriques)

    # Encodage des nationalités selon le classement FIFA
    if "nation" in cols_a_traiter:
        cols_a_traiter.remove("nation")

        if df_fifa_historique is not None and "nation" in df_encoded.columns:
            print(
                "Encodage de 'nation' en 10 colonnes binaires Top FIFA (par saison)..."
            )

            # Harmonisation des chaînes de caractères pour garantir le match lors du merge
            df_encoded["nation_join"] = (
                df_encoded["nation"].astype(str).str.lower().str.strip()
            )

            df_fifa_temp = df_fifa_historique.copy()
            df_fifa_temp["nation_join"] = (
                df_fifa_temp["country_abrv"].astype(str).str.lower().str.strip()
            )

            # Jointure pour récupérer le rang ('rank') du pays pour la bonne saison
            df_encoded = pd.merge(
                df_encoded,
                df_fifa_temp[["nation_join", "season_year", "rank"]],
                on=["nation_join", "season_year"],
                how="left",
            )

            # Remplissage des NaN : si un pays n'est pas dans le top 10 historique,
            # on lui attribue une valeur neutre hors du top 10 (ex: 999)
            df_encoded["rank"] = df_encoded["rank"].fillna(999).astype(int)

            # Création dynamique des 10 colonnes binaires (0 ou 1)
            for i in range(1, 11):
                df_encoded[f"classement_FIFA_{i}"] = (
                    df_encoded["rank"] == i
                ).astype(int)

            # Suppression des colonnes devenues inutiles (ancienne nation, rang brut et clé de jointure)
            df_encoded = df_encoded.drop(columns=["rank", "nation_join"])
            print("   • Les 10 colonnes classement_FIFA_X ont été injectées.")
        else:
            print(
                "Impossible d'appliquer le filtre FIFA (df_fifa_historique manquant ou colonne 'nation' absente)."
            )

    # Traitement de la position
    if "pos" in cols_a_traiter:
        cols_a_traiter.remove("pos")

        if "pos" in df_encoded.columns:
            nb_modalites_pos = df_encoded["pos"].nunique()

            if nb_modalites_pos <= 1:
                df_encoded = df_encoded.drop(columns=["pos"])
                print("Profil Gardien détecté : Suppression de la colonne 'pos'.")
            else:
                print(
                    f"Profil Joueurs de champ détecté : Encodage Multi-Label de 'pos'."
                )
                listes_postes = (
                    df_encoded["pos"]
                    .astype(str)
                    .str.split(",")
                    .apply(lambda x: [i.strip() for i in x])
                )

                mlb = MultiLabelBinarizer()
                matrice_postes = mlb.fit_transform(listes_postes)

                df_postes_encodes = pd.DataFrame(
                    matrice_postes,
                    columns=[f"pos_{classe}" for classe in mlb.classes_],
                    index=df_encoded.index,
                )

                df_encoded = pd.concat([df_encoded, df_postes_encodes], axis=1)
                df_encoded = df_encoded.drop(columns=["pos"])

    # Encodage des autres variables
    variables_finales_a_encoder = [
        col for col in cols_a_traiter if col in df_encoded.columns
    ]

    if variables_finales_a_encoder:
        print(f"Encodage One-Hot des colonnes : {variables_finales_a_encoder}")
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=variables_finales_a_encoder,
            drop_first=False,
            dtype=int,
        )

    print(f"Format final après encodage : {df_encoded.shape}\n")
    return df_encoded



def detecter_tous_outliers_iqr_trie_filtre(df):
    """Détecte automatiquement les outliers sur toutes les variables numériques

    continues (hors binaires), trie les résultats et n'affiche QUE les
    variables ayant strictement plus de 10% d'outliers.
    """
    # Sélection automatique des colonnes numériques continues
    toutes_cols_numeriques = df.select_dtypes(include=[np.number]).columns
    colonnes_numeriques = [
        col
        for col in toutes_cols_numeriques
        if df[col].nunique() > 2 and not col.startswith("injury")
    ]

    rapport_outliers = {}
    index_tous_outliers = set()
    liste_pour_tri = []

    print(
        f"Analyse en cours sur {len(colonnes_numeriques)} variables numériques continues..."
    )
    print(
        f"({len(toutes_cols_numeriques) - len(colonnes_numeriques)} variables binaires exclues)\n"
    )

    # Calcul des outliers
    for col in colonnes_numeriques:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        borne_inf = Q1 - 1.5 * IQR
        borne_sup = Q3 + 1.5 * IQR

        outliers_indices = df[
            (df[col] < borne_inf) | (df[col] > borne_sup)
        ].index.tolist()

        nb_outliers = len(outliers_indices)

        if nb_outliers > 0:
            pourcentage = round((nb_outliers / len(df)) * 100, 2)

            rapport_outliers[col] = {
                "nombre_outliers": nb_outliers,
                "pourcentage": pourcentage,
                "indices": outliers_indices,
            }

            index_tous_outliers.update(outliers_indices)

            liste_pour_tri.append(
                {"colonne": col, "pourcentage": pourcentage, "nombre": nb_outliers}
            )
        else:
            liste_pour_tri.append(
                {"colonne": col, "pourcentage": 0.0, "nombre": 0}
            )

    # Tri par pourcentage décroissant
    liste_pour_tri = sorted(
        liste_pour_tri, key=lambda x: x["pourcentage"], reverse=True
    )

    # Affichage filtré (Uniquement > 5%)
    print("Variables contenant strictement plus de 5% d'outliers :")

    nb_colonnes_masquees = 0

    for item in liste_pour_tri:
        if item["pourcentage"] > 5.0:
            print(
                f"Colonne '{item['colonne']}' : {item['nombre']} outliers détectés ({item['pourcentage']}% du dataset)"
            )
        else:
            nb_colonnes_masquees += 1


    print(
        f"{nb_colonnes_masquees} autres colonnes numériques ont un taux d'outliers inférieur ou égal à 5% (masquées)."
    )
    print(
        f"\nBilan global (hors binaires) : {len(index_tous_outliers)} lignes uniques possèdent au moins un vrai outlier."
    )

    return rapport_outliers, list(index_tous_outliers)



def calculer_jours_contrat_restants(df):
    """Calcule le nombre de jours de contrat restants pour chaque joueur

    par rapport à une date de référence fixe de fin de saison.
    """
    print("Début du calcul de la durée restante des contrats...")

    # Copie locale pour éviter les warnings
    df_clean = df.copy()

    # Vérification des colonnes nécessaires
    if "season" not in df_clean.columns or "contract_expiration_date" not in df_clean.columns:
        print("Colonne 'season' ou 'contract_expiration_date' manquante. Calcul annulé.")
        return df_clean

    # Dictionnaire de correspondance Saisons -> Dates de fin de saison
    saison_to_date = {
        2021: "2021-06-30",
        2122: "2022-06-30",
        2223: "2023-06-30",
        2324: "2024-06-30",
        2425: "2025-06-30",
        2526: "2026-06-30",
    }

    # Création et conversion des dates de référence
    df_clean["date_ref_saison"] = df_clean["season"].map(saison_to_date)
    df_clean["date_ref_saison"] = pd.to_datetime(df_clean["date_ref_saison"])
    df_clean["contract_expiration_date"] = pd.to_datetime(
        df_clean["contract_expiration_date"], errors="coerce"
    )

    # Calcul du delta en jours
    df_clean["contrat_jours_restants"] = (
        df_clean["contract_expiration_date"] - df_clean["date_ref_saison"]
    ).dt.days

    # Résultats
    print(f"Contrats déjà expirés (valeur négative) : {(df_clean['contrat_jours_restants'] < 0).sum()}")
    print(f"Contrats manquants (NaN)                  : {df_clean['contrat_jours_restants'].isna().sum()}")
    
    print()
    
    print("Statistiques descriptives de la variable calculée :")
    print(df_clean["contrat_jours_restants"].describe().to_string())

    # Nettoyage de la colonne temporaire
    df_clean.drop(columns=["date_ref_saison"], inplace=True)

    print("Calcul terminé avec succès.")

    return df_clean


def executer_pipeline_preprocessing(
    df,
    dossier_sortie,
    cols_a_normaliser,
    col_poste="position",
    methode_split="temporel",
    seed_aleatoire=42,
    height_min=155,
    height_max=210,
):
    """Exécute l'intégralité du pipeline de préprocessing : split temporel,

    traitement des outliers (par la médiane du poste sur le train),
    normalisation MinMax et sauvegardes.
    """

    if "season_year" in df.columns:
        df_filtré = df[df["season_year"] != 2025].copy()
        nb_exclus = len(df) - len(df_filtré)
        if nb_exclus > 0:
            print(
                f"{nb_exclus} lignes de la saison 2025 ont été écartées du pipeline."
            )
    else:
        df_filtré = df.copy()

    if methode_split == "temporel":
        # Split Temporel classique (Train: 2020-2022, Val: 2023, Test: 2024)
        df_train = df_filtré[
            df_filtré["season_year"].isin([2020, 2021, 2022])
        ].copy()
        df_val = df_filtré[df_filtré["season_year"].isin([2023])].copy()
        df_test = df_filtré[df_filtré["season_year"].isin([2024])].copy()

    elif methode_split == "aleatoire":
        # Split Aléatoire (ex: 70% Train, 15% Val, 15% Test)
        # Étape 1 : On isole le Train (70%) et un bloc temporaire de Validation + Test (30%)
        df_train, df_temp = train_test_split(
            df_filtré, test_size=0.30, random_state=seed_aleatoire
        )
        # Étape 2 : On coupe le bloc temporaire en deux parts égales (15% Val, 15% Test)
        df_val, df_test = train_test_split(
            df_temp, test_size=0.50, random_state=seed_aleatoire
        )

        # Re-conversion explicite en copie pour éviter les warnings Pandas
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_test = df_test.copy()

    else:
        raise ValueError(
            f"Méthode de split '{methode_split}' inconnue. Choisissez 'temporel' ou 'aleatoire'."
        )
    
    print(
        f"Split effectué -> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}"
    )

    # Vérification de la présence de la colonne de poste
    if col_poste not in df_train.columns:
        raise ValueError(
            f"La colonne de poste '{col_poste}' est absente du dataset."
        )

    # Traitement des outliers par poste (Taille en cm)
    # On calcule les médianes par poste uniquement sur le train pour éviter le data leakage
    medianes_par_poste = (
        df_train.groupby(col_poste)["height_in_cm"].median().to_dict()
    )

    # Médiane globale de secours au cas où un poste bizarre n'aurait pas de médiane
    mediane_globale_train = df_train["height_in_cm"].median()

    outlier_stats = {
        "height_in_cm": {
            "lower": height_min,
            "upper": height_max,
            "medianes_par_poste": medianes_par_poste,
            "mediane_globale": mediane_globale_train,
        }
    }

    # Application de la correction sur les 3 splits
    for df_split in [df_train, df_val, df_test]:
        # Détection des lignes qui ont un problème de taille
        mask_outlier = (df_split["height_in_cm"] < height_min) | (
            df_split["height_in_cm"] > height_max
        )

        if mask_outlier.any():
            # Pour les lignes en anomalie, on récupère la médiane correspondant à leur poste
            valeurs_remplacement = df_split[col_poste].map(medianes_par_poste)

            # Sécurité : si un poste n'était pas dans le train, on met la médiane globale
            valeurs_remplacement = valeurs_remplacement.fillna(
                mediane_globale_train
            )

            # On applique le remplacement uniquement là où le masque est Vrai
            df_split.loc[mask_outlier, "height_in_cm"] = valeurs_remplacement[
                mask_outlier
            ]

    # Sauvegarde du dictionnaire complet des stats d'outliers
    os.makedirs(dossier_sortie, exist_ok=True)
    with open(os.path.join(dossier_sortie, "outlier_stats.pkl"), "wb") as f:
        pickle.dump(outlier_stats, f)
    print(
        f"Outliers traités (Bornes: [{height_min}, {height_max}] | Remplacement par la médiane du poste du Train)."
    )

    # Normalisation MinMax [0-1]
    cols_manquantes = [c for c in cols_a_normaliser if c not in df_train.columns]
    if cols_manquantes:
        raise ValueError(f"Colonnes absentes du dataset : {cols_manquantes}")

    scaler = MinMaxScaler()
    scaler.fit(df_train[cols_a_normaliser])

    for df_split in [df_train, df_val, df_test]:
        normalized = scaler.transform(df_split[cols_a_normaliser])
        for i, col in enumerate(cols_a_normaliser):
            df_split[f"{col}_nor"] = normalized[:, i]

    # Sauvegarde du scaler
    with open(os.path.join(dossier_sortie, "normalisation.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Normalisation MinMax appliquée avec succès.")

    # Sauvegarde des fichiers finaux
    df_train.to_csv(
        os.path.join(dossier_sortie, "train.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    df_val.to_csv(
        os.path.join(dossier_sortie, "val.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    df_test.to_csv(
        os.path.join(dossier_sortie, "test.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Pipeline terminé ! Fichiers sauvegardés dans : {dossier_sortie}\n")
    return df_train, df_val, df_test