import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def nettoyer_age_et_dates(df, colonnes_dates=None):
    """Nettoie la colonne 'age' (extraction et conversion en int) et applique le

    format datetime normalisé (sans les heures visibles) aux colonnes de dates.

    Paramètres:
    -----------
    df : dataframe
        Le dataset de football à nettoyer.
    colonnes_dates : list of str, optional
        La liste explicite des colonnes de dates à traiter (ex: ['date_of_birth',
        'date']).
        Si None, la fonction cherchera automatiquement les colonnes contenant
        'date' ou 'dob' dans leur nom.
    """
    # Copie de sécurité pour éviter le SettingWithCopyWarning
    df_clean = df.copy()

    # Nettoyage de l'âge
    if "age" in df_clean.columns:
        print("Nettoyage de la colonne 'age'")
        # Force en texte, extrait les chiffres avant le tiret (ex: 23-328 -> 23)
        df_clean["age"] = df_clean["age"].astype(str).str.extract(r"^(\d+)")

        # Sécurité pour les NaN : remplacement temporaire par une valeur neutre avant conversion
        df_clean["age"] = df_clean["age"].fillna(0)

        # Conversion finale en entier (int)
        df_clean["age"] = df_clean["age"].astype(int)
        print("Colonne 'age' convertie en entiers avec succès.")
    else:
        print("Colonne 'age' introuvable dans le dataframe.")

    # Nettoyage des colonnes de dates
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

                # Ecrase l'heure en la figeant strictement à minuit (00:00:00)
                # Tout en conservant le format datetime64[ns] idéal pour les calculs.
                df_clean[col] = df_clean[col].dt.normalize()
        print("Toutes les heures ont été retirées (remises à minuit).")
    else:
        print("Aucune colonne de date détectée ou fournie.")

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


def imputer_donnees_physiques_et_ages(df):
    """Nettoie la taille des joueurs en imputant les valeurs aberrantes

    par la médiane de leur poste, puis calcule l'année de naissance et l'âge.
    """
    print("Début du traitement de la taille et du calcul des âges...")

    # Copie locale pour éviter les warnings "SettingWithCopy"
    df_clean = df.copy()

    # Nettoyage et imputation de la taille
    if "height_in_cm" in df_clean.columns:
        # Remplacement des valeurs aberrantes par NaN
        df_clean.loc[
            (df_clean["height_in_cm"] < 150)
            | (df_clean["height_in_cm"] > 220),
            "height_in_cm",
        ] = np.nan

        # Imputation par la médiane du poste (version optimisée sans lambda)
        if "position" in df_clean.columns:
            df_clean["height_in_cm"] = df_clean["height_in_cm"].fillna(
                df_clean.groupby("position")["height_in_cm"].transform("median")
            )
            print(" -> Valeurs aberrantes de taille imputées par la médiane du poste.")
        else:
            # Sécurité si la colonne 'position' n'existe pas, on prend la médiane globale
            df_clean["height_in_cm"] = df_clean["height_in_cm"].fillna(
                df_clean["height_in_cm"].median()
            )
            print(" -> Valeurs aberrantes de taille imputées par la médiane globale (colonne 'position' manquante).")

    # Calcul correct de l'âge
    if "date_of_birth" in df_clean.columns and "season_year" in df_clean.columns:
        # Extraction de l'année de naissance
        df_clean["born"] = pd.to_datetime(
            df_clean["date_of_birth"], errors="coerce"
        ).dt.year

        # Calcul de l'âge au moment de la saison
        df_clean["age"] = df_clean["season_year"] - df_clean["born"]
        print(" -> Colonnes 'born' (année) et 'age' calculées avec succès.")
    else:
        print("Colonne 'date_of_birth' ou 'season_year' manquante. Calcul de l'âge annulé.")

    print("Traitement terminé.")

    return df_clean


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


def supprimer_colonnes_inutiles_et_leakage(df):
    """Supprime les identifiants techniques, les doublons d'information

    et les variables de fuite de données (Data Leakage) avant la modélisation.
    """
    print("Début du nettoyage des colonnes avant modélisation...")

    # Copie locale pour éviter les warnings
    df_clean = df.copy()

    # Dictionnaire explicatif des suppressions
    categories_suppression = {
        "Identifiants techniques": [
            "join_key", "tm_join_key", "tm_join_key_full", "tm_id",
            "player_id", "dob_key", "tm_dob_key", "match_method"
        ],
        "Doublons d'information": [
            "name", "date_of_birth", "dob_year", "date", "season"
        ],
        "Fuites de données (Data Leakage)": [
            "valuation_season_year", "contract_expiration_date"
        ],
    }

    total_supprimees = 0

    # Analyse et suppression par catégorie pour un affichage propre
    for categorie, colonnes in categories_suppression.items():
        # On ne garde que les colonnes qui existent réellement dans le DataFrame
        cols_presentes = [c for c in colonnes if c in df_clean.columns]
        
        if cols_presentes:
            df_clean.drop(columns=cols_presentes, inplace=True)
            print(f"{categorie:35s} : {len(cols_presentes):2d} colonnes supprimées ({', '.join(cols_presentes)})")
            total_supprimees += len(cols_presentes)
        else:
            print(f"{categorie:35s} : 0 colonne supprimée")

    print()
    print(f"Nettoyage terminé. Total de colonnes supprimées : {total_supprimees}")
    print(f"Dimensions actuelles du DataFrame : {df_clean.shape}")

    return df_clean


def normaliser_variables_continues(df):
    """Identifie les variables continues (non binaires, non encodées) et crée

    leurs versions normalisées (MinMax 0-1) avec le suffixe '_nor'.
    """
    print("Début du processus de normalisation (MinMax)...")

    # Copie locale pour éviter les warnings
    df_clean = df.copy()

    # Identification des colonnes à exclure de la normalisation
    cols_exclure_normalisation = [
        *[c for c in df_clean.columns if c.startswith("pos_")],
        *[c for c in df_clean.columns if c.startswith("sub_position_")],
        *[c for c in df_clean.columns if c.startswith("league_")],
        *[c for c in df_clean.columns if c.startswith("foot_")],
        *[c for c in df_clean.columns if c.startswith("classement_FIFA_")],
        *[
            c
            for c in df_clean.columns
            if "injury_musculaire" in c
            and df_clean[c].dropna().isin([0, 1]).all()
        ],
        *[
            c
            for c in df_clean.columns
            if c.startswith("injury_") and df_clean[c].dropna().isin([0, 1]).all()
        ],
        "season_year",
    ]

    # Séparation automatique des colonnes numériques (continues vs binaires)
    numeric_cols = df_clean.select_dtypes(include="number").columns.tolist()
    binary_cols = [
        c for c in numeric_cols if df_clean[c].dropna().isin([0, 1]).all()
    ]

    # Isolation des colonnes qui doivent être normalisées
    cols_a_normaliser = [
        c
        for c in numeric_cols
        if c not in cols_exclure_normalisation and c not in binary_cols
    ]

    print(
        f"Colonnes binaires (déjà en 0/1, à ne pas normaliser) : {len(binary_cols)}"
    )
    print(
        f"Colonnes continues à dupliquer en version '_nor'     : {len(cols_a_normaliser)}"
    )

    print()

    # Application du MinMaxScaler (0-1)
    if cols_a_normaliser:
        scaler = MinMaxScaler()
        cols_normalisees_noms = [f"{c}_nor" for c in cols_a_normaliser]

        # Imputation temporaire par la médiane pour le fit_transform (sécurité Sklearn)
        df_clean[cols_normalisees_noms] = scaler.fit_transform(
            df_clean[cols_a_normaliser].fillna(df_clean[cols_a_normaliser].median())
        )
        print(
            f"\nCréation des colonnes '_nor' terminée. Total de colonnes actuel : {df_clean.shape[1]}"
        )
    else:
        print("\nAucune colonne continue à normaliser n'a été trouvée.")

    return df_clean