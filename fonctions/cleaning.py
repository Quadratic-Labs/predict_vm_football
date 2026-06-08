import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer



def nettoyer_age_et_dates(df, colonnes_dates=None):
    """Nettoie la colonne 'age' (extraction et conversion en int) et applique le

    format datetime normalisé (sans les heures visibles) aux colonnes de dates.

    Paramètres:
    -----------
    df : DataFrame
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
        print("Nettoyage de la colonne 'age'...")
        # Force en texte, extrait les chiffres avant le tiret (ex: 23-328 -> 23)
        df_clean["age"] = df_clean["age"].astype(str).str.extract(r"^(\d+)")

        # Sécurité pour les NaN : remplacement temporaire par une valeur neutre avant conversion
        df_clean["age"] = df_clean["age"].fillna(0)

        # Conversion finale en entier (int)
        df_clean["age"] = df_clean["age"].astype(int)
        print("Colonne 'age' convertie en entiers avec succès.")
    else:
        print("Colonne 'age' introuvable dans le DataFrame.")

    # Nettoyage des colonnes de dates
    # Si l'utilisateur n'a pas fourni de liste, on détecte automatiquement les colonnes de date
    if colonnes_dates is None:
        colonnes_dates = [
            col
            for col in df_clean.columns
            if "date" in col.lower() or "dob" in col.lower()
        ]

    if colonnes_dates:
        print(f"Traitement des colonnes de dates : {colonnes_dates}...")
        for col in colonnes_dates:
            if col in df_clean.columns:
                # Conversion au format datetime Pandas officiel
                df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")

                # Ecrase l'heure en la figeant strictement à minuit (00:00:00)
                # Tout en conservant le format datetime64[ns] idéal pour les calculs.
                df_clean[col] = df_clean[col].dt.normalize()
        print("   • Toutes les heures ont été retirées (remises à minuit).")
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




import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def encoder_dataset_football(
    df, colonnes_categoriques, df_fifa_historique=None
):
    """Fonction modulaire mise à jour. Remplace la colonne 'nation' par 10

    colonnes binaires (classement_FIFA_1 à 10) basées sur le rang réel de la
    nation du joueur pour la saison donnée.

    Paramètres:
    -----------
    df : DataFrame
        Le dataset principal (1 ligne = 1 joueur pour 1 saison).
        Doit contenir les colonnes 'nation' et 'season_year'.
    colonnes_categoriques : list of str
        La liste des colonnes à encoder ('pos', 'league', 'foot', etc.).
    df_fifa_historique : DataFrame
        Le DataFrame du Top 10 FIFA filtré contenant les colonnes:
        ['country_abrv', 'rank', 'season_year'].
    """
    df_encoded = df.copy()
    print(f"Format initial avant encodage : {df_encoded.shape}")

    cols_a_traiter = list(colonnes_categoriques)

    # ==============================================================================
    # 1. ENCODAGE DE LA NATION EN 10 COLONNES BINAIRES DE CLASSEMENT FIFA
    # ==============================================================================
    if "nation" in cols_a_traiter:
        cols_a_traiter.remove("nation")

        if df_fifa_historique is not None and "nation" in df_encoded.columns:
            print(
                "🏆 Encodage de 'nation' en 10 colonnes binaires Top FIFA (par saison)..."
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
                "⚠️ Impossible d'appliquer le filtre FIFA (df_fifa_historique manquant ou colonne 'nation' absente)."
            )

    # ==============================================================================
    # 2. TRAITEMENT DE LA POSITION (MULTI-LABEL)
    # ==============================================================================
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

    # ==============================================================================
    # 3. ENCODAGE ONE-HOT CLASSIQUE (League, Foot, etc.)
    # ==============================================================================
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