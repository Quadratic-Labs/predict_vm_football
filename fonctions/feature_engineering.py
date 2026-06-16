import numpy as np
import pandas as pd


def generer_feature_engineering_football(df):
    """Calcule et ajoute des variables explicatives complexes (ratios tactiques,

    tendances inter-saisons, statut de l'âge et contexte équipe) au DataFrame
    en utilisant les colonnes exactes fournies.
    """
    print("Début de la création des nouvelles variables (Feature Engineering)...")

    # Copie locale pour éviter les conflits de modification
    df_fe = df.copy()

    # Tri chronologique par joueur
    if "player" in df_fe.columns and "season_year" in df_fe.columns:
        df_fe = df_fe.sort_values(by=["player", "season_year"]).reset_index(
            drop=True
        )

    # Profils de style de jeu et indices tactiques
    print("Calcul des ratios tactiques et styles de jeu...")

    # Indice de verticalité
    if "Standard_G/SoT" in df_fe.columns and "Standard_SoT%" in df_fe.columns:
        # Indique l'efficacité face au but par rapport à la précision globale
        df_fe["indice_danger_tirs"] = (
            df_fe["Standard_G/SoT"] * df_fe["Standard_SoT%"] / 100.0
        )

    # Ratio d'agressivité défensive (Fautes commises / Interceptions + Tacles Gagnés)
    if all(
        c in df_fe.columns
        for c in ["Performance_Fls", "Performance_Int", "Performance_TklW"]
    ):
        denominateur_def = df_fe["Performance_Int"] + df_fe["Performance_TklW"]
        df_fe["ratio_agressivite"] = np.where(
            denominateur_def > 0,
            df_fe["Performance_Fls"] / denominateur_def,
            0,
        )

    # Indice de menace de gardien (Taille * Efficacité des arrêts pour GK)
    if "Performance_Save%" in df_fe.columns and "height_in_cm" in df_fe.columns:
        df_fe["taux_arretXtaille_gardien"] = (
            df_fe["Performance_Save%"] / 100.0
        ) * df_fe["height_in_cm"]

    # Ratio de Danger Converti (Efficacité devant le but : Buts réels / xG attendus)
    if "Performance_Gls" in df_fe.columns and "xg" in df_fe.columns:
        df_fe["efficacite_devant_but"] = np.where(
            df_fe["xg"] > 1,
            df_fe["Performance_Gls"] / df_fe["xg"],
            1.0,  # Valeur neutre de 1 si pas d'xg
        )

    # Indicateurs de dynamiques
    print("Calcul des dynamiques et trajectoires inter-saisons...")

    if "player" in df_fe.columns and "season_year" in df_fe.columns:
        
        # On récupère l'année de la saison de la ligne précédente pour chaque joueur
        df_fe["saison_precedente_reelle"] = df_fe.groupby("player")["season_year"].shift(1)
        
        # On crée un masque booléen : True si la ligne précédente est bien la saison N-1
        # (Si le joueur a un trou dans sa carrière, la condition sera False)
        saison_consecutive = (df_fe["season_year"] - df_fe["saison_precedente_reelle"]) == 1

        # Tendance du temps de jeu (Delta minutes)
        if "Playing Time_Min" in df_fe.columns:
            # On récupère la valeur précédente
            minutes_prec = df_fe.groupby("player")["Playing Time_Min"].shift(1)
            
            # On calcule le delta brut
            df_fe["delta_minutes_jouees"] = df_fe["Playing Time_Min"] - minutes_prec
            
            # Si ce n'est pas consécutif, on force à 0 (ou on laisse le joueur neutre)
            df_fe["delta_minutes_jouees"] = np.where(saison_consecutive, df_fe["delta_minutes_jouees"], 0)
            
            # Gestion des NaN pour la toute première saison connue du joueur
            df_fe["delta_minutes_jouees"] = df_fe["delta_minutes_jouees"].fillna(0)

        # Tendance des Expected Goals (xg)
        if "xg" in df_fe.columns:
            # On récupère la valeur précédente
            xg_prec = df_fe.groupby("player")["xg"].shift(1)
            
            # On calcule le delta brut
            df_fe["delta_xg"] = df_fe["xg"] - xg_prec
            
            # Si ce n'est pas consécutif, on force à 0
            df_fe["delta_xg"] = np.where(saison_consecutive, df_fe["delta_xg"], 0)
            
            # Gestion des NaN pour la toute première saison connue du joueur
            df_fe["delta_xg"] = df_fe["delta_xg"].fillna(0)

        # Nettoyage de la colonne technique temporaire
        df_fe.drop(columns=["saison_precedente_reelle"], inplace=True)

    # Contextualisation de la valeur du joueur dans son équipe
    print("Calcul du poids du joueur dans son équipe...")

    # Poids offensif du joueur (Buts + Passes Décisives du joueur / Total Buts de l'équipe)
    if all(
        c in df_fe.columns
        for c in ["Performance_Gls", "Performance_Ast", "team", "season_year"]
    ):
        # Somme des buts de l'équipe par saison
        buts_equipe = (
            df_fe.groupby(["team", "season_year"])["Performance_Gls"]
            .sum()
            .reset_index()
        )
        buts_equipe.columns = ["team", "season_year", "total_buts_equipe"]

        # Fusion
        df_fe = pd.merge(
            df_fe, buts_equipe, on=["team", "season_year"], how="left"
        )

        # Calcul du ratio d'implication
        df_fe["contribution_offensive_equipe"] = np.where(
            df_fe["total_buts_equipe"] > 0,
            (df_fe["Performance_Gls"] + df_fe["Performance_Ast"])
            / df_fe["total_buts_equipe"],
            0,
        )
        df_fe.drop(columns=["total_buts_equipe"], inplace=True)

    # Taux d'indisponibilité annuel
    if "injury_days_total" in df_fe.columns:
        df_fe["taux_indisponibilite"] = (df_fe["injury_days_total"] / 365.0).clip(upper=1.0)


    # Score lié à la nation
    # On initialise le score à 0
    df_fe["score_hype_nation"] = 0

    # Boucle pour calculer dynamiquement la somme pondérée : (11 - i) * classement_FIFA_i
    for i in range(1, 11):
        col_fifa = f"classement_FIFA_{i}"
        if col_fifa in df_fe.columns:
            poids = 11 - i
            df_fe["score_hype_nation"] += poids * df_fe[col_fifa]


    # Rentabilité des buts par minute
    if "Performance_Gls" in df_fe.columns and "Playing Time_Min" in df_fe.columns:
        df_fe["rentabilite_buts_minutes"] = np.where(
            df_fe["Playing Time_Min"] > 0,
            df_fe["Performance_Gls"] / df_fe["Playing Time_Min"],
            0.0,  # 0 si le joueur n'a joué aucune minute cette saison
        )


    # Urgence contractuelle (1 si contrat_jours_restants <= 365, sinon 0)
    if "contrat_jours_restants" in df_fe.columns:
        df_fe["urgence_contractuelle"] = np.where(
            df_fe["contrat_jours_restants"] <= 365, 1, 0
        )
    
    
    print()
    print(
        f"Feature Engineering terminé ! Nombre total de colonnes : {df_fe.shape[1]}"
    )

    return df_fe