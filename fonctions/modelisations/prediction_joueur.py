import numpy as np
import pandas as pd
import difflib
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin


class StackingModel(BaseEstimator, RegressorMixin):
    """Combine les 3 modèles de base déjà entraînés via le méta-modèle du stacking."""

    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model

    def fit(self, X, y):
        # Les modèles sont déjà entraînés.
        return self

    def predict(self, X):
        # Chaque modèle de base prédit en log(valeur), on repasse en euros avec expm1.
        X_meta = np.column_stack([
            np.expm1(model.predict(X)) for model in self.models
        ])
        # Le méta-modèle combine les 3 prédictions et prédit directement en euros.
        return self.meta_model.predict(X_meta)


def formater_euros(valeur):
    """Affiche un montant en euros avec des espaces comme séparateurs de milliers."""
    return f"{valeur:,.0f} €".replace(",", " ")



def predire_et_afficher(modele_stack, historique_predictions, X_ligne, nom_affiche, valeur_reelle=None, mode="manuel", saison=None):
    """Calcule la prédiction, l'affiche de façon lisible, et l'ajoute à l'historique."""
    prediction = modele_stack.predict(X_ligne)[0]

    print(f" {nom_affiche}")
    if saison is not None:
        print(f"   Saison               : {saison}")
    print(f"    Valeur marchande prédite : {formater_euros(prediction)}")

    ecart_pct = None
    if valeur_reelle is not None:
        ecart_pct = (prediction - valeur_reelle) / valeur_reelle * 100
        print(f"    Valeur réelle connue    : {formater_euros(valeur_reelle)}")
        print(f"    Écart                   : {ecart_pct:+.1f} %")

    historique_predictions.append({
        "date_prediction": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "joueur": nom_affiche,
        "mode": mode,
        "saison": saison,
        "valeur_predite_euros": round(float(prediction), 2),
        "valeur_reelle_euros": float(valeur_reelle) if valeur_reelle is not None else None,
        "ecart_pct": round(ecart_pct, 2) if ecart_pct is not None else None,
    })

    return prediction


def chercher_joueur(nom, datasets, colonne_joueur= "player", colonne_saison="season_year"):
    """Retourne toutes les correspondances (nom exact, saison, split, index) pour un nom recherché."""
    correspondances = []
    for split, (df_split, X_split) in datasets.items():
        mask = df_split[colonne_joueur].str.contains(nom, case=False, na=False)
        for idx in df_split[mask].index:
            correspondances.append(
                (df_split.loc[idx, colonne_joueur], df_split.loc[idx, colonne_saison], split, idx)
            )
    return correspondances


def predire_joueur_existant(nom, datasets, modele_stack, historique_predictions, saison=None, colonne_joueur= "player", colonne_saison="season_year", colonne_cible="market_value_in_eur"):
    """Recherche un joueur par son nom et affiche la prédiction du modèle pour la saison demandée (la plus récente par défaut), avec comparaison à la valeur réelle connue."""
    correspondances = chercher_joueur(nom, datasets, colonne_joueur, colonne_saison)

    if not correspondances:
        tous_les_noms = pd.concat([df[colonne_joueur] for df, _ in datasets.values()]).unique()
        suggestions = difflib.get_close_matches(nom, tous_les_noms, n=5, cutoff=0.5)
        print(f"Aucun joueur trouvé pour « {nom} ».")
        if len(suggestions):
            print("   Suggestions proches :", ", ".join(suggestions))
        return None

    noms_trouves = sorted(set(c[0] for c in correspondances))
    if len(noms_trouves) > 1:
        print(f"Plusieurs joueurs correspondent à « {nom} » : {noms_trouves}")
        print("   Merci de préciser un nom plus exact.")
        return None

    nom_exact = noms_trouves[0]
    correspondances_joueur = [c for c in correspondances if c[0] == nom_exact]

    if saison is not None:
        choix = [c for c in correspondances_joueur if c[1] == saison]
        if not choix:
            saisons_dispo = sorted(c[1] for c in correspondances_joueur)
            print(f"Saison {saison} non trouvée pour {nom_exact}. Saisons disponibles : {saisons_dispo}")
            return None
        _, saison_choisie, split, idx = choix[0]
    else:
        _, saison_choisie, split, idx = max(correspondances_joueur, key=lambda c: c[1])
        if len(correspondances_joueur) > 1:
            print(
                f"{len(correspondances_joueur)} saisons trouvées pour {nom_exact}, "
                f"on utilise la plus récente ({saison_choisie}). "
                f"Précisez saison=... pour en choisir une autre."
            )

    df_split, X_split = datasets[split]
    X_ligne = X_split.loc[[idx]]
    valeur_reelle = df_split.loc[idx, colonne_cible]

    return predire_et_afficher(modele_stack, historique_predictions, X_ligne, nom_exact, valeur_reelle=valeur_reelle, mode="recherche", saison=saison_choisie)


def ligne_par_defaut(X_train):
    """Construit une ligne de caractéristiques 'joueur moyen' : médiane pour les variables numériques,
    valeur la plus fréquente pour les variables binaires (0/1), calculées sur le jeu d'entraînement."""
    ligne = {}
    for col in X_train.columns:
        valeurs = X_train[col].dropna()
        if valeurs.empty:
            ligne[col] = 0
            continue
        est_binaire = set(valeurs.unique()) <= {0, 1}
        ligne[col] = int(valeurs.mode().iloc[0]) if est_binaire else valeurs.median()
    return pd.DataFrame([ligne], columns=X_train.columns)


# Correspondance entre les choix "métier" et les colonnes réelles attendues par le modèle
POSTES = {"Gardien": "pos_GK", "Défenseur": "pos_DF", "Milieu": "pos_MF", "Attaquant": "pos_FW"}
PIEDS = {"Droit": "foot_right", "Gauche": "foot_left", "Ambidextre": "foot_both"}
CHAMPIONNATS = {
    "Premier League": "league_ENG-Premier League",
    "La Liga": "league_ESP-La Liga",
    "Ligue 1": "league_FRA-Ligue 1",
    "Bundesliga": "league_GER-Bundesliga",
    "Serie A": "league_ITA-Serie A",
    "Autre / non renseigné": None,
}


def appliquer_saisie_metier(ligne, saisie):
    """Applique un dictionnaire de saisie 'métier' sur une ligne de caractéristiques existante. Utilisée à la fois pour la saisie manuelle et pour la table de sensibilité
    des curseurs PowerBI. Les champs non reconnus sont ignorés plutôt que de faire planter le calcul."""
    champs_ignores = []

    def poser(col, valeur):
        if col in ligne.columns:
            ligne.at[0, col] = valeur
        else:
            champs_ignores.append(col)

    if "age" in saisie:
        poser("age", saisie["age"])
        poser("age_sq", saisie["age"] ** 2)

    if "taille_cm" in saisie:
        poser("height_in_cm", saisie["taille_cm"])

    if "pied" in saisie:
        for label, col in PIEDS.items():
            poser(col, 1 if label == saisie["pied"] else 0)

    if "poste" in saisie:
        for label, col in POSTES.items():
            poser(col, 1 if label == saisie["poste"] else 0)

    if "championnat" in saisie:
        for label, col in CHAMPIONNATS.items():
            if col is not None:
                poser(col, 1 if label == saisie["championnat"] else 0)

    if "valeur_marchande_precedente_euros" in saisie:
        poser("log_prev_value", np.log1p(saisie["valeur_marchande_precedente_euros"]))

    if "classement_equipe" in saisie:
        poser("classement", saisie["classement_equipe"])

    if "buts" in saisie:
        poser("Performance_Gls", saisie["buts"])

    if "passes_decisives" in saisie:
        poser("Performance_Ast", saisie["passes_decisives"])

    if "matchs_joues" in saisie:
        poser("Playing Time_MP", saisie["matchs_joues"])

    if "minutes_jouees_par_match" in saisie:
        poser("Playing Time_Mn/MP", saisie["minutes_jouees_par_match"])

    return champs_ignores


def construire_ligne_manuelle(saisie, X_train):
    """Transforme un dictionnaire de saisie 'métier' en une ligne de caractéristiques complète, en partant du profil moyen et en appliquant la saisie par-dessus."""
    ligne = ligne_par_defaut(X_train)
    champs_ignores = appliquer_saisie_metier(ligne, saisie)
    if champs_ignores:
        print(f" Champs ignorés (colonne absente du modèle) : {champs_ignores}")
    return ligne


def predire_joueur_manuel(modele_stack, historique_predictions, saisie, X_train, nom_affiche="Joueur simulé"):
    """Construit la ligne de caractéristiques à partir de la saisie 'métier', puis affiche la prédiction."""
    X_ligne = construire_ligne_manuelle(saisie, X_train)
    return predire_et_afficher(modele_stack, historique_predictions, X_ligne, nom_affiche, valeur_reelle=None, mode="manuel")


def construire_export_historique(datasets, modele_stack, colonne_joueur="player", colonne_team="team", colonne_saison="season_year", colonne_cible="market_value_in_eur"):
    """Calcule, pour tous les joueurs et toutes les saisons disponibles (train/val/test/en_cours), la VM réelle et la VM prédite par le modèle."""
    lignes = []
    for split, (df_split, X_split) in datasets.items():
        predictions = modele_stack.predict(X_split)
        for position, idx in enumerate(df_split.index):
            vm_reelle = df_split.loc[idx, colonne_cible]
            vm_predite = predictions[position]
            lignes.append({
                "joueur": df_split.loc[idx, colonne_joueur],
                "equipe": df_split.loc[idx, colonne_team] if colonne_team in df_split.columns else None,
                "saison": df_split.loc[idx, colonne_saison],
                "jeu": split,  # train / val / test / en_cours
                "vm_reelle_euros": float(vm_reelle),
                "vm_predite_euros": float(vm_predite),
                "ecart_pct": round((vm_predite - vm_reelle) / vm_reelle, 2) if vm_reelle else None,
            })
    return pd.DataFrame(lignes)


def construire_table_sensibilite(modele_stack, X_train, ligne_reference, nom_reference):
    """Pour une ligne de référence donnée (profil moyen ou joueur réel), fait varier une à une les 4 variables 'curseur' sur une plage de valeurs réalistes,
    toutes les autres variables restant fixes, et calcule la prédiction du modèle pour chaque valeur."""
    plages = {
        "age": np.arange(int(X_train["age"].min()), int(X_train["age"].max()) + 1, 1),
        "buts": np.arange(0, int(np.ceil(X_train["Performance_Gls"].max())) + 1, 1),
        "classement_equipe": np.arange(1, int(np.ceil(X_train["classement"].max())) + 1, 1),
        # Plage linéaire simple
        "valeur_marchande_precedente_euros": np.arange(0, 200_000_001, 2_000_000),
    }

    resultats = []
    for variable, valeurs in plages.items():
        # On construit toutes les variantes de cette variable en une seule fois
        lignes_variees = []
        for valeur in valeurs:
            ligne_test = ligne_reference.copy()
            ligne_test.index = [0]
            appliquer_saisie_metier(ligne_test, {variable: valeur})
            lignes_variees.append(ligne_test)
        batch = pd.concat(lignes_variees, ignore_index=True)
        predictions = modele_stack.predict(batch)

        for valeur, prediction in zip(valeurs, predictions):
            resultats.append({
                "profil_reference": nom_reference,
                "variable": variable,
                "valeur": float(valeur),
                "vm_predite_euros": float(prediction),
            })

    return pd.DataFrame(resultats)