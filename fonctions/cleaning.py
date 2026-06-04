import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer



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
            f" ATTENTION : {nb_doublons_techniques} lignes sont des doublons techniques stricts."
        )
        print(
            "   (Même joueur, même saison, même club -> Erreur d'extraction/jointure)"
        )
        print("\n   Exemple de lignes techniques concernées :")
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

    # Présence de doublons de Mercato
    if nb_doublons_mercato > 0:
        print(
            f"={nb_doublons_mercato} lignes correspondent à des doublons de mercato"
        )
        print(
            "   (Même joueur, même saison, mais CLUBS DIFFÉRENTS -> Transferts de mi-saison)"
        )
        print("\n   Exemple de joueurs transférés concernés :")

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
    # trouvée pour CHAQUE colonne de manière indépendante.
    df_fusionne = df_tri.groupby(
        ["player", "season", "team"], as_index=False
    ).first()

    print(
        f"Format après fusion intelligente des doublons : {df_fusionne.shape}"
    )

    # Sauvegarde optionnelle
    if chemin_sauvegarde:
        df_fusionne.to_csv(chemin_sauvegarde, index=False)
        print(f"Base fusionnée sauvegardée dans : {chemin_sauvegarde}")

    return df_fusionne




def verifier_doublons_mercato(df):
    """Diagnostique et affiche les doublons de type Joueur + Saison

    généralement causés par les transferts ou prêts de mi-saison (mercato).
    """
    print("Diagnostic des doublons liés au mercato (transferts de mi-saison)")

    # Identification automatique des colonnes clés
    col_joueur = "player" if "player" in df.columns else None
    col_saison = "season" if "season" in df.columns else None
    col_club = "team" if "team" in df.columns else None

    if col_joueur and col_saison:
        # Calcul du nombre de lignes en doublon sur le couple Joueur + Saison
        nb_doublons = df.duplicated(subset=[col_joueur, col_saison]).sum()

        if nb_doublons > 0:
            print(
                f"Attention : {nb_doublons} lignes sont des doublons pour le même joueur lors de la même saison !"
            )
            print("   Cela indique la présence de transferts ou de prêts à la mi-saison.\n")
            print("   Exemple de lignes concernées :")

            # On prépare les colonnes à afficher pour l'exemple (on ajoute le club si dispo pour plus de clarté)
            cols_affichage = [col_joueur, col_saison]
            if col_club:
                cols_affichage.append(col_club)

            # Extraction et affichage des premiers exemples trouvés
            exemples = df[df.duplicated(subset=[col_joueur, col_saison], keep=False)]
            print(exemples[cols_affichage].head(6))

        else:
            print(
                "Parfait ! Aucun doublon détecté pour le couple [Joueur + Saison]."
            )
            print("   Chaque joueur ne possède qu'une seule et unique ligne par saison.")

    else:
        print(
            "Impossible de vérifier : les colonnes 'player' ou 'season' sont manquantes dans le dataset."
        )




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

        # Priorité 1 : Les exceptions techniques à ne pas modifier (Dernier en date)
        if any(keyword in col_lower for keyword in mots_cles_exceptions):
            aggregation_rules[col] = "last"

        # Priorité 2 : Les colonnes de pourcentage (%) ou ratios intermédiaires (Moyenne)
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

        # Priorité 3 : Les autres colonnes numériques (Buts, Minutes, Passes...) (Somme)
        elif col in cols_numeriques:
            aggregation_rules[col] = "sum"

        # Priorité 4 : Les colonnes textuelles (team, league...) (Dernier club en date)
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


    # Retourne la liste des colonnes (utile pour l'étape de suppression)
    return liste_colonnes_vides





def lister_variables_categorielles(df):
    """Identifie, affiche le nombre de modalités uniques et retourne

    la liste des variables catégorielles présentes dans le DataFrame.
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




def encoder_dataset_football(df, top_n_nations=10):
    """Fonction unique pour encoder les variables catégorielles (league, nation, pos)

    s'adaptant automatiquement aux Gardiens ou aux Joueurs de champ.
    """
    # Copie de sécurité pour éviter les avertissements de type SettingWithCopyWarning
    df_encoded = df.copy()

    print(f"Format initial avant encodage : {df_encoded.shape}")

    # Traitement de la colonne position
    if "pos" in df_encoded.columns:
        nb_modalites_pos = df_encoded["pos"].nunique()

        # Cas 1 : Base de Gardiens (Une seule modalité, ex: 'GK')
        if nb_modalites_pos <= 1:
            df_encoded = df_encoded.drop(columns=["pos"])
            print("Profil Gardien détecté : Suppression de la colonne 'pos'.")

        # Cas 2 : Base de Joueurs de champ (Multi-labels / Plusieurs postes)
        else:
            print(
                f"Profil Joueurs de champ détecté : Encodage Multi-Label des {nb_modalites_pos} modalités de postes."
            )

            # On transforme la chaîne "DF,FW" en une vraie liste Python ['DF', 'FW']
            listes_postes = (
                df_encoded["pos"]
                .astype(str)
                .str.split(",")
                .apply(lambda x: [i.strip() for i in x])
            )

            # On crée la matrice binaire de 0 et de 1
            mlb = MultiLabelBinarizer()
            matrice_postes = mlb.fit_transform(listes_postes)

            # Conversion en DataFrame propre aligné sur les index d'origine
            df_postes_encodes = pd.DataFrame(
                matrice_postes,
                columns=[f"pos_{classe}" for classe in mlb.classes_],
                index=df_encoded.index,
            )

            # Fusion et suppression de l'ancienne colonne texte
            df_encoded = pd.concat([df_encoded, df_postes_encodes], axis=1)
            df_encoded = df_encoded.drop(columns=["pos"])

            print(
                f"   • Encodage des postes terminé. Classes détectées : {list(mlb.classes_)}"
            )
    else:
        print(
            "Colonne 'pos' absente (Traitement déjà effectué ou base de gardiens pré-nettoyée)."
        )

    # Regroupement des nationalités
    if "nation" in df_encoded.columns:
        # On calcule les N nations les plus représentées spécifiques à CE dataset
        top_nations = df_encoded["nation"].value_counts().head(top_n_nations).index

        # Application du filtre
        df_encoded["nation_group"] = df_encoded["nation"].apply(
            lambda x: x if x in top_nations else "Autre"
        )

        # Suppression de l'ancienne colonne de texte brute
        df_encoded = df_encoded.drop(columns=["nation"])
        print(
            f"Colonne 'nation' regroupée : Top {top_n_nations} + 'Autre' ({df_encoded['nation_group'].nunique()} modalités au total)."
        )

    # Encodage des ligues et des nationalités
    variables_a_encoder = []
    if "league" in df_encoded.columns:
        variables_a_encoder.append("league")
    if "nation_group" in df_encoded.columns:
        variables_a_encoder.append("nation_group")

    if variables_a_encoder:
        # Encodage en gardant toutes les colonnes (drop_first=False)
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=variables_a_encoder,
            drop_first=False,
            dtype=int,
        )

    print(f"Format final après encodage : {df_encoded.shape}\n")

    return df_encoded