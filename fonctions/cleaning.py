import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


# Mapping des codes FBref vers les Confédérations FIFA (logique sportive, pas géographique)
CODE_TO_CONFEDERATION = {
    # UEFA (Europe + Turquie, Chypre, Russie, Caucase, Kosovo)
    "ALB": "UEFA", "ARM": "UEFA", "AUT": "UEFA", "BEL": "UEFA", "BIH": "UEFA",
    "BUL": "UEFA", "CRO": "UEFA", "CYP": "UEFA", "CZE": "UEFA", "DEN": "UEFA",
    "ENG": "UEFA", "ESP": "UEFA", "EST": "UEFA", "FIN": "UEFA", "FRA": "UEFA",
    "FRO": "UEFA", "GEO": "UEFA", "GER": "UEFA", "GRE": "UEFA", "HUN": "UEFA",
    "IRL": "UEFA", "ISL": "UEFA", "ISR": "UEFA", "ITA": "UEFA", "KVX": "UEFA",
    "LTU": "UEFA", "LUX": "UEFA", "LVA": "UEFA", "MDA": "UEFA", "MKD": "UEFA",
    "MLT": "UEFA", "MNE": "UEFA", "NED": "UEFA", "NIR": "UEFA", "NOR": "UEFA",
    "POL": "UEFA", "POR": "UEFA", "ROU": "UEFA", "RUS": "UEFA", "SCO": "UEFA",
    "SRB": "UEFA", "SUI": "UEFA", "SVK": "UEFA", "SVN": "UEFA", "SWE": "UEFA",
    "TUR": "UEFA", "UKR": "UEFA", "WAL": "UEFA",

    # CONMEBOL (Amérique du Sud)
    "ARG": "CONMEBOL", "BOL": "CONMEBOL", "BRA": "CONMEBOL", "CHI": "CONMEBOL",
    "COL": "CONMEBOL", "ECU": "CONMEBOL", "PAR": "CONMEBOL", "PER": "CONMEBOL",
    "URU": "CONMEBOL", "VEN": "CONMEBOL", "GUF": "CONMEBOL", "SUR": "CONMEBOL",

    # CONCACAF (Amérique du Nord/Centrale/Caraïbes)
    "CAN": "CONCACAF", "CRC": "CONCACAF", "CUW": "CONCACAF", "DOM": "CONCACAF",
    "GLP": "CONCACAF", "GRN": "CONCACAF", "HAI": "CONCACAF", "HON": "CONCACAF",
    "JAM": "CONCACAF", "MEX": "CONCACAF", "MSR": "CONCACAF", "MTQ": "CONCACAF",
    "PAN": "CONCACAF", "PUR": "CONCACAF", "SKN": "CONCACAF", "USA": "CONCACAF",

    # CAF (Afrique)
    "ALG": "CAF", "ANG": "CAF", "BDI": "CAF", "BEN": "CAF", "BFA": "CAF",
    "CGO": "CAF", "CHA": "CAF", "CIV": "CAF", "CMR": "CAF", "COD": "CAF",
    "COM": "CAF", "CPV": "CAF", "CTA": "CAF", "EGY": "CAF", "EQG": "CAF",
    "GAB": "CAF", "GAM": "CAF", "GHA": "CAF", "GNB": "CAF", "GUI": "CAF",
    "KEN": "CAF", "LBY": "CAF", "MAD": "CAF", "MAR": "CAF", "MLI": "CAF",
    "MOZ": "CAF", "MTN": "CAF", "NGA": "CAF", "RSA": "CAF", "SEN": "CAF",
    "SLE": "CAF", "TAN": "CAF", "TOG": "CAF", "TUN": "CAF", "UGA": "CAF",
    "ZAM": "CAF", "ZIM": "CAF",

    # AFC (Asie, Australie incluse depuis 2006)
    "AUS": "AFC", "BAN": "AFC", "CHN": "AFC", "IDN": "AFC", "IRN": "AFC",
    "IRQ": "AFC", "JOR": "AFC", "JPN": "AFC", "KOR": "AFC", "KSA": "AFC",
    "MAS": "AFC", "PHI": "AFC", "SYR": "AFC", "UAE": "AFC", "UZB": "AFC",

    # OFC (Océanie, hors Australie)
    "NCL": "OFC", "NZL": "OFC",
}

def get_confederation(code_fbref):
    if pd.isna(code_fbref):
        return None
    return CODE_TO_CONFEDERATION.get(code_fbref, "Inconnu")

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
    nb_doublons_saison = dataframe.duplicated(subset=[col_joueur, col_saison]).sum()

    # Calcul des doublons stricts (Joueur + Saison + Club) -> Technique uniquement
    nb_doublons_techniques = dataframe.duplicated(subset=[col_joueur, col_saison, col_club]).sum()

    # Déduction des doublons liés purement au Mercato (changement de club)
    nb_doublons_mercato = nb_doublons_saison - nb_doublons_techniques

    # Rapport des résultats

    # Présence de doublons techniques
    if nb_doublons_techniques > 0:
        print(
            f"{nb_doublons_techniques} lignes sont des doublons techniques stricts."
        )
        print(
            "(Même joueur, même saison, même club : Erreur d'extraction/jointure)"
        )
    else:
        print(
            " Aucune répétition technique (Même joueur, même saison, même club)."
        )

    print()

    # Présence de doublons de Mercato
    if nb_doublons_mercato > 0:
        print(
            f"{nb_doublons_mercato} lignes correspondent à des doublons de mercato"
        )
        print(
            "(Même joueur, même saison, mais clubs différents : transferts de mi-saison)"
        )
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

    df_fusionne = df_tri.groupby(
        ["player", "season", "team"], as_index=False
    ).first()

    print(f"Format après fusion intelligente des doublons : {df_fusionne.shape}")

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
        # lambda x: x.sum(min_count=1) permet de garder NaN si toutes les valeurs sont NaN
        elif col in cols_numeriques:
            aggregation_rules[col] = lambda x: x.sum(min_count=1)

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
    list_recalcul = ["Per 90 Minutes_Gls", "Per 90 Minutes_Ast", "Per 90 Minutes_G+A", "Standard_G/Sh",
                     "Standard_G/SoT", "Standard_SoT%", "Standard_Sh/90", "Standard_SoT/90",
                     "Playing Time_Mn/MP", "Playing Time_Min%"]

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


def nettoyer_age_et_dates(df, colonnes_dates=None):
    """Nettoie et normalise les colonnes de dates et calcule l'âge exact du joueur

    au moment de la saison de manière robuste.
    """
    # Copie de sécurité pour éviter le SettingWithCopyWarning
    df_clean = df.copy()

    # Nettoyage des colonnes de dates
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
        print("Toutes les heures ont été remises à minuit.")
    else:
        print("Aucune colonne de date détectée ou fournie.")

    # Extraction de l'année de naissance à partir de la date de naissance
    if "date_of_birth" in df_clean.columns:
        df_clean["born"] = df_clean["date_of_birth"].dt.year
    else:
        df_clean["born"] = np.nan
        print("Attention : 'date_of_birth' manquante, 'born' initialisée à NaN.")

    # Calcul de l'âge exact du joueur pour la saison en cours
    print("Calcul et nettoyage de la colonne 'age'...")

    # On calcule l'âge théorique basé sur la saison et l'année de naissance
    if "season_year" in df_clean.columns and "date_of_birth" in df_clean.columns:
        df_clean["age_calcule"] = df_clean["season_year"] - df_clean["born"]
    else:
        df_clean["age_calcule"] = np.nan

    # On nettoie l'âge brut textuel (ex: '23-328' -> 23) au cas où on en aurait besoin
    if "age" in df_clean.columns:
        df_clean["age_brut"] = (
            df_clean["age"].astype(str).str.extract(r"^(\d+)")
        )
        df_clean["age_brut"] = pd.to_numeric(
            df_clean["age_brut"], errors="coerce"
        )
    else:
        df_clean["age_brut"] = np.nan

    # On prend l'âge calculé, sinon l'âge brut de FBref
    df_clean["age"] = df_clean["age_calcule"].fillna(df_clean["age_brut"])


    # Remplacement par la médiane des âges du dataset
    if df_clean["age"].isna().sum() > 0:
        valeur_remplacement = df_clean["age"].median()
        # Si tout est vide (cas extrême), on met une valeur par défaut
        if pd.isna(valeur_remplacement):
            valeur_remplacement = 25
        df_clean["age"] = df_clean["age"].fillna(valeur_remplacement)
        print(
            f"{df_clean['age'].isna().sum()} âges manquants remplacés par la médiane ({int(valeur_remplacement)} ans)."
        )

    # Conversion finale stricte en entier
    df_clean["age"] = df_clean["age"].astype(int)

    # Nettoyage des colonnes temporaires pour garder le DataFrame propre
    colonnes_temporaires = ["age_calcule", "age_brut"]
    df_clean = df_clean.drop(
        columns=[col for col in colonnes_temporaires if col in df_clean.columns]
    )

    print(" -> Colonne 'age' convertie strictement en entiers (int).")
    print("Nettoyage de l'âge et des dates terminé.\n")

    return df_clean


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
            print(f"   {col} : {tx*100:.1f}% de valeurs manquantes")

        # Extraction de la liste des noms de colonnes pour pouvoir la retourner
        liste_colonnes_vides = colonnes_vides_serie.index.tolist()
        print(
            f"\nTotal : {len(liste_colonnes_vides)} colonnes dépassent le seuil de {seuil*100:.0f}%."
        )
    else:
        print(f"   Aucune colonne ne dépasse {seuil*100:.0f}% de lignes vides.")
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

    if "market_value_in_eur" in df_clean.columns:
        nb_avant = len(df_clean)
        # dropna supprime les lignes où la colonne spécifiée contient un NaN
        df_clean = df_clean.dropna(subset=["market_value_in_eur"])
        nb_apres = len(df_clean)
        print(f" {nb_avant - nb_apres} lignes supprimées car 'market_value_in_eur' était manquant.")
    else:
        print(" Attention : la colonne 'market_value_in_eur' est introuvable.")

    # Remplacement des NA par des 0 dans les colonnes de performance spécifiques
    cols_NA = [
        "Penalty Kicks_Save%", "Performance_CS%", "Performance_Save%",
        "Standard_G/SoT", "Standard_G/Sh", "Standard_SoT%", "Per 90 Minutes_Gls",
        "Per 90 Minutes_Ast", "Per 90 Minutes_G+A", "Standard_SoT/90", "Standard_Sh/90",
        "Performance_PKwon", "Performance_PKcon", "Performance_Saves", "Performance_GA90",
        "Performance_GA", "Performance_SoTA", "Penalty Kicks_PKm", "Penalty Kicks_PKsv",
        "Penalty Kicks_PKatt", "Performance_CS", "Penalty Kicks_PKA", "Performance_W",
        "Performance_D", "Performance_L"
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
        

    # Traitement des NA résiduels
    for col in cols_propager_presentes:
        if col == "player_id":
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = df_clean[col].fillna("Inconnu")
    
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
            print(f"   {col} ({nb_uniques} modalités uniques)")

        print(f"\nTotal : {len(cols_cat)} variables catégorielles trouvées.")
    else:
        print("   Aucune variable catégorielle détectée dans ce dataset.")

    # Retourne la liste des noms de colonnes
    return cols_cat


def encoder_dataset(
    df, colonnes_categoriques, df_fifa_historique=None
):
    """Fonction modulaire mise à jour. Remplace la colonne 'nation' par 10

    colonnes binaires (classement_FIFA_1 à 10) basées sur le rang réel de la
    nation du joueur pour la saison donnée.

    arguments:
        df : dataframe
            Le dataset principal (1 ligne = 1 joueur pour 1 saison).
            Doit contenir les colonnes 'nation' et 'season_year'.
        colonnes_categoriques : list of str
            La liste des colonnes à encoder.
        df_fifa_historique : dataframe
            Le dataframe du Top 10 FIFA filtré.
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

            df_encoded["est_anglais"] = (df_encoded["nation_join"] == "eng").astype(int)
            print("   • Variable 'est_anglais' créée.")

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


            # Ajout de la variable confédération FIFA
            df_encoded["confederation"] = df_encoded["nation"].apply(get_confederation)

            df_encoded = pd.get_dummies(
                df_encoded, columns=["confederation"], prefix="confederation"
            )
            for col in [c for c in df_encoded.columns if c.startswith("confederation_")]:
                df_encoded[col] = df_encoded[col].astype(int)
            print("   Variable 'confederation' encodée en colonnes binaires.")

            # Suppression des colonnes devenues inutiles (ancienne nation, rang brut et clé de jointure)
            df_encoded = df_encoded.drop(columns=["rank", "nation_join"])
            print("   Les 10 colonnes classement_FIFA_X ont été injectées.")
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


def supprimer_colonnes_du_dataset(df, colonnes_a_supprimer):
    """Supprime une liste de colonnes spécifiée d'un DataFrame de manière sécurisée.

    param df: Le DataFrame d'origine.
    param colonnes_a_supprimer: Liste de chaînes de caractères (noms des
    colonnes).
    return: Un nouveau DataFrame nettoyé.
    """
    # On filtre la liste pour ne garder que les colonnes qui existent vraiment dans le dataframe
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


def executer_pipeline_preprocessing(df, dossier_sortie, col_poste="position", col_joueur="player",
                                    col_valeur_marchande="market_value_in_eur", methode_split="temporel",
                                    seed_aleatoire=42, height_min=155, height_max=210,):
    
    """Exécute l'intégralité du pipeline de préprocessing : split temporel/aléatoire,
    traitement des outliers, imputation des NA, calcul de log_prev_value 
    (valeur marchande log de la saison précédente) et sauvegardes.
    """

    # Split des données
    if methode_split == "temporel":
        df_train = df[df["season_year"].isin([2020, 2021, 2022])].copy()
        df_val = df[df["season_year"].isin([2023])].copy()
        df_test = df[df["season_year"].isin([2024])].copy()
        df_en_cours = df[df["season_year"].isin([2025])].copy()
    elif methode_split == "aleatoire":
        df_train, df_temp = train_test_split(
            df, test_size=0.30, random_state=seed_aleatoire
        )
        df_val, df_test = train_test_split(
            df_temp, test_size=0.50, random_state=seed_aleatoire
        )
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_test = df_test.copy()
        # NB: en split "aleatoire" il n'y a pas de notion de saison "en cours" ;
        # on garde un DataFrame vide de même structure pour ne pas casser la suite du pipeline.
        df_en_cours = df.iloc[0:0].copy()
    else:
        raise ValueError(
            f"Méthode de split '{methode_split}' inconnue. Choisissez 'temporel' ou 'aleatoire'."
        )

    print(f"Split effectué, Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # Vérification de la présence de la colonne de poste
    if col_poste not in df_train.columns:
        raise ValueError(f"La colonne de poste '{col_poste}' est absente du dataset.")

    # Traitement des outliers par poste (Taille en cm)
    medianes_par_poste = df_train.groupby(col_poste)["height_in_cm"].median().to_dict()
    mediane_globale_train = df_train["height_in_cm"].median()

    for df_split in [df_train, df_val, df_test, df_en_cours]:
        mask_outlier = (df_split["height_in_cm"] < height_min) | (df_split["height_in_cm"] > height_max)
        if mask_outlier.any():
            valeurs_remplacement = df_split[col_poste].map(medianes_par_poste)
            valeurs_remplacement = valeurs_remplacement.fillna(mediane_globale_train)
            df_split.loc[mask_outlier, "height_in_cm"] = valeurs_remplacement[mask_outlier]

    print(f"Outliers traités (Bornes: [{height_min}, {height_max}]. Remplacement par la médiane du poste du Train).")

    # Imputation de xg et Performance_Gls (Poste × Ligue)
    COLS_CIBLES = [c for c in ["xg", "Performance_Gls"] if c in df_train.columns]

    if COLS_CIBLES:
        # Préparation des colonnes temporaires de regroupement pour les 4 splits
        for df_split in [df_train, df_val, df_test, df_en_cours]:
            # Identification de la ligue active (ex: league_Ligue1, etc.)
            league_cols = [c for c in df_split.columns if c.startswith("league_")]
            if league_cols:
                df_split["_ligue"] = df_split[league_cols].idxmax(axis=1).where(
                    df_split[league_cols].max(axis=1) == 1, other="Autre"
                )
            else:
                df_split["_ligue"] = "Autre"

            # Identification du poste
            df_split["_position"] = df_split[col_poste].fillna("Inconnu")

        # Application de l'imputation (Poste × Ligue)
        for col in COLS_CIBLES:
            # Calcul des médianes sur le TRAIN uniquement
            med_1_train = df_train.groupby(["_position", "_ligue"])[col].median().rename("_med_1")

            # Application sur Train, Val, Test et En Cours
            for df_name, df_split in [("Train", df_train), ("Val", df_val), ("Test", df_test), ("En Cours", df_en_cours)]:
                n_nan_initial = df_split[col].isna().sum()
                if n_nan_initial == 0:
                    continue

                # Jointure sur Poste × Ligue (permet de matcher même si la saison change !)
                df_split = df_split.join(med_1_train, on=["_position", "_ligue"])
                df_split[col] = df_split[col].fillna(df_split["_med_1"])
                df_split.drop(columns=["_med_1"], inplace=True)

                # Réassignation des DataFrames
                if df_name == "Train":
                    df_train = df_split
                elif df_name == "Val":
                    df_val = df_split
                elif df_name == "Test":
                    df_test = df_split
                elif df_name == "En Cours":
                    df_en_cours = df_split

                n_nan_final = df_split[col].isna().sum()

        # Nettoyage des colonnes temporaires
        for df_split in [df_train, df_val, df_test, df_en_cours]:
            df_split.drop(columns=["_ligue", "_position"], inplace=True, errors='ignore')

        print("Imputation des NA terminée.")
    else:
        print("Aucune colonne cible trouvée pour l'imputation.")


    # Valeur marchande de la saison précédente (log_prev_value)

    if col_valeur_marchande in df_train.columns and col_joueur in df_train.columns:

        # Base d'historique propre et triée par temps (tous splits confondus)
        df_history = pd.concat([df_train, df_val, df_test, df_en_cours], axis=0)
        df_history = df_history.sort_values([col_joueur, "season_year"])

        # Shift sur cet historique de référence
        df_history["log_prev_value"] = np.log1p(
            df_history.groupby(col_joueur)[col_valeur_marchande].shift(1)
        )

        # Redécoupage du dataset en train/val/test/en_cours
        df_train = df_history[df_history["season_year"].isin([2020, 2021, 2022])].copy()
        df_val = df_history[df_history["season_year"] == 2023].copy()
        df_test = df_history[df_history["season_year"] == 2024].copy()
        df_en_cours = df_history[df_history["season_year"] == 2025].copy()

        # Colonnes de ligue disponibles
        league_cols_prev = [c for c in df_train.columns if c.startswith("league_")]

        def extraire_nom_ligue(df_in):
            df_temp = df_in.copy()
            if league_cols_prev:
                df_temp["league_name"] = df_temp[league_cols_prev].idxmax(axis=1).where(
                    df_temp[league_cols_prev].max(axis=1) == 1, "league_Other"
                )
            else:
                df_temp["league_name"] = "league_Other"
            return df_temp

        # Médianes apprises uniquement sur le Train, par Ligue x Poste x Saison
        df_train_temp = extraire_nom_ligue(df_train)

        prev_pos_league_year = (
            df_train_temp
            .groupby(["league_name", col_poste, "season_year"])["log_prev_value"]
            .median()
        )

        # Secours 1 : Ligue × Poste (sans saison)
        prev_pos_league = (
            df_train_temp
            .groupby(["league_name", col_poste])["log_prev_value"]
            .median()
        )

        # Secours 2 : Poste seul
        prev_pos_backup = (
            df_train_temp
            .groupby(col_poste)["log_prev_value"]
            .median()
        )

        # Secours 3 : médiane globale
        prev_global = df_train_temp["log_prev_value"].median()

        def impute_log_prev_value(df_in):
            df_with_league = extraire_nom_ligue(df_in)
            original_index = df_with_league.index

            # Niveau 1 : Ligue × Poste × Saison (groupe le plus fin)
            mapped_1 = (
                df_with_league
                .set_index(["league_name", col_poste, "season_year"])
                .index.map(prev_pos_league_year)
                .to_series(index=original_index)
            )
            df_with_league["log_prev_value"] = df_with_league["log_prev_value"].fillna(mapped_1)

            # Niveau 2 : Ligue × Poste (si saison absente des médianes train)
            mapped_2 = (
                df_with_league
                .set_index(["league_name", col_poste])
                .index.map(prev_pos_league)
                .to_series(index=original_index)
            )
            df_with_league["log_prev_value"] = df_with_league["log_prev_value"].fillna(mapped_2)

            # Niveau 3 : Poste seul
            df_with_league["log_prev_value"] = df_with_league["log_prev_value"].fillna(
                df_with_league[col_poste].map(prev_pos_backup)
            )

            # Niveau 4 : médiane globale
            df_with_league["log_prev_value"] = df_with_league["log_prev_value"].fillna(prev_global)

            return df_with_league.drop(columns=["league_name"])

        df_train = impute_log_prev_value(df_train)
        df_val = impute_log_prev_value(df_val)
        df_test = impute_log_prev_value(df_test)
        df_en_cours = impute_log_prev_value(df_en_cours)

        print("Colonne 'log_prev_value' calculée et imputée.")
    else:
        print(f"Colonnes '{col_valeur_marchande}' et/ou '{col_joueur}' absentes : 'log_prev_value' non calculée.")

    # Sauvegarde des fichiers finaux
    os.makedirs(dossier_sortie, exist_ok=True)
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
    df_en_cours.to_csv(
        os.path.join(dossier_sortie, "en_cours.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Pipeline terminé. Fichiers sauvegardés dans : {dossier_sortie}\n")
    return df_train, df_val, df_test, df_en_cours


def executer_pipeline_nettoyage(df, df_fifa):
    """Exécute l'intégralité du pipeline de nettoyage enchaîné."""
    
    df_nettoye = df.copy()
    
    # Doublons
    verifier_doublons_metier_et_techniques(df_nettoye)
    df_nettoye = fusionner_doublons_techniques(df_nettoye)
    df_nettoye = fusionner_et_recalculer_mercato(df_nettoye)
    
    # Âge et Dates
    colonnes_dates = ["date_of_birth", "contract_expiration_date"]
    df_nettoye = nettoyer_age_et_dates(df_nettoye, colonnes_dates=colonnes_dates)
    
    # Valeurs manquantes
    diagnostiquer_valeurs_manquantes(df_nettoye, seuil=0.01)
    df_nettoye = nettoyer_valeurs_manquantes_ciblees(df_nettoye)
    diagnostiquer_valeurs_manquantes(df_nettoye, seuil=0.01)
    
    # Encodage
    mes_variables = ["pos", "sub_position", "nation", "league", "foot"]
    df_nettoye = encoder_dataset(
        df=df_nettoye,
        colonnes_categoriques=mes_variables,
        df_fifa_historique=df_fifa,
    )
    
    # Suppression colonnes
    colonnes_redondantes = [
        "Starts_Starts", "Standard_PK", "Standard_PKatt", "Standard_Gls", "90s", "Playing Time_Min%",
        "Performance_SoTA", "Performance_G+A", "Team Success_+/-", "Team Success_+/-90", "Playing Time_Min", 
        "Penalty Kicks_PKatt", "born", "np_xg", "xg_chain", "Per 90 Minutes_G+A-PK", "Per 90 Minutes_G-PK",
        "join_key", "tm_join_key", "tm_join_key_full", "tm_id", "player_id", "dob_key", "tm_dob_key",
        "match_method", "name", "date_of_birth", "dob_year", "date", "season", "valuation_season_year",
        "contract_expiration_date", "Starts_Mn/Start"
    ]
    cols_a_supprimer = [c for c in colonnes_redondantes if c in df_nettoye.columns]
    df_nettoye = supprimer_colonnes_du_dataset(df_nettoye, cols_a_supprimer)
    
    return df_nettoye