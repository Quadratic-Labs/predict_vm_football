import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import time
from sklearn.linear_model import LinearRegression

def preparer_donnees(df_train, df_val, df_test, df_en_cours, colonne_cible):
    """Nettoie les valeurs manquantes cibles, sépare features/cible,

    calcule les cibles log & sample weights, et génère des versions
    sans NaN pour les modèles linéaires/SVR.
    """
    # Suppression des lignes avec valeur manquante sur la cible
    df_train = df_train.dropna(subset=[colonne_cible])
    df_val = df_val.dropna(subset=[colonne_cible])
    df_test = df_test.dropna(subset=[colonne_cible])
    df_en_cours = df_en_cours.dropna(subset=[colonne_cible])

    # Suppression des colonnes normalisées (_nor)
    colonnes_nor = [c for c in df_train.columns if c.endswith("_nor")]
    df_train = df_train.drop(columns=colonnes_nor, errors="ignore")
    df_val = df_val.drop(columns=colonnes_nor, errors="ignore")
    df_test = df_test.drop(columns=colonnes_nor, errors="ignore")
    df_en_cours = df_en_cours.drop(columns=colonnes_nor, errors="ignore")

    # Définition et vérification des colonnes à exclure des features
    cols_a_supprimer_base = [ colonne_cible, "player", "team", "nation", "position", ]
    cols_a_supprimer = [c for c in cols_a_supprimer_base if c in df_train.columns]

    # Séparation des features (X) et de la cible (y)
    X_train = df_train.drop(columns=cols_a_supprimer)
    y_train = df_train[colonne_cible]

    X_val = df_val.drop(columns=cols_a_supprimer)
    y_val = df_val[colonne_cible]

    X_test = df_test.drop(columns=cols_a_supprimer)
    y_test = df_test[colonne_cible]

    X_en_cours = df_en_cours.drop(columns=cols_a_supprimer)
    y_en_cours = df_en_cours[colonne_cible]

    # Cibles log
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)
    y_en_cours_log = np.log1p(y_en_cours)

    # Poids d'entraînement (sur-pondération des petites valeurs)
    sample_weights_train = 1 / np.log1p(y_train)
    sample_weights_train = sample_weights_train / sample_weights_train.mean()

    print(
        f"X_train : {X_train.shape} | X_val : {X_val.shape} | X_test : {X_test.shape} | X_en_cours : {X_en_cours.shape}"
    )

    # Retour sous forme de dictionnaire pour un accès plus clair et structuré
    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_log": y_train_log,
        "sample_weights_train": sample_weights_train,
        "X_val": X_val,
        "y_val": y_val,
        "y_val_log": y_val_log,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_log": y_test_log,
        "X_en_cours": X_en_cours,
        "y_en_cours": y_en_cours,
        "y_en_cours_log": y_en_cours_log,
    }


def initialiser_modeles_finaux(params, random_state=42):
    """Initialise et retourne un dictionnaire contenant les modèles XGBoost,

    LightGBM et CatBoost configurés.
    """
    xgb_final = XGBRegressor(
        **params["XGBoost"],
        tree_method="hist",
        enable_categorical=True,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    lgbm_final = LGBMRegressor( **params["LightGBM"], random_state=random_state, n_jobs=-1, verbose=-1 )

    cat_final = CatBoostRegressor(
        **params["CatBoost"],
        bootstrap_type="Bernoulli",
        random_state=random_state,
        early_stopping_rounds=50,
        verbose=0,
    )

    return {
        "XGBoost (log)": xgb_final,
        "LightGBM (log)": lgbm_final,
        "CatBoost (log)": cat_final,
    }


def evaluer_log(nom, modele, X_tr, y_tr_log, X_ev, y_ev, w=None):
    fit_kwargs = {"sample_weight": w} if w is not None else {}

    debut = time.time()
    if isinstance(modele, CatBoostRegressor):
        modele.fit(X_tr, y_tr_log, eval_set=(X_ev, np.log1p(y_ev)), verbose=False, **fit_kwargs)
    else:
        modele.fit(X_tr, y_tr_log, **fit_kwargs)
    duree_entrainement = time.time() - debut
    print(f"  Temps d'entraînement {nom:15s} : {duree_entrainement:6.2f} s")

    preds = np.expm1(modele.predict(X_ev))
    mae   = mean_absolute_error(y_ev, preds)
    rmse  = np.sqrt(mean_squared_error(y_ev, preds))
    mape  = mean_absolute_percentage_error(y_ev, preds)
    r2    = r2_score(y_ev, preds)
    n, p  = len(y_ev), X_ev.shape[1]
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"  {nom:12s} | MAPE : {mape:.2%} | MAE : {mae:>12,.0f} € | R² : {r2:.4f} | R²adj : {r2_adj:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "R2 Ajusté": r2_adj, "Temps entraînement (s)": duree_entrainement}


def reconstruire_meta_modele(meta_config, base_models, positive=True):
    """Reconstruit un méta-modèle LinearRegression à partir d'une configuration.

    Parameters:
    -----------
    meta_config : dict
        Dictionnaire contenant 'weights' (dict de poids) et 'intercept' (float).
    base_models : list
        Liste des noms des modèles de base correspondant aux clés des poids.
    positive : bool, default=True
        Force les coefficients à être positifs si ré-entraîné.

    Returns:
    --------
    meta : LinearRegression
        Le méta-modèle instancié et configuré.
    """
    meta = LinearRegression(positive=positive)

    # Injecter les poids et l'intercept
    meta.coef_ = np.array([meta_config["weights"][nom] for nom in base_models])
    meta.intercept_ = float(meta_config["intercept"])

    # Affichage des informations de reconstruction
    poids_dict = dict(zip(base_models, meta.coef_.round(3)))
    print("Méta-modèle reconstruit avec succès !")
    print("Poids :", poids_dict)
    print(f"Intercept : {meta.intercept_:,.2f} €")

    return meta


def entrainer_et_predire_stacking(base_models, modeles_finaux, meta, X_train, y_train_log, X_val,
                                  y_val_log, X_test, X_en_cours, sample_weights_train=None):
    """Entraîne les modèles de base, génère leurs prédictions et applique le méta-modèle.

    Returns:
    --------
    dict: Contient les prédictions finales (euros et log) et les métriques de temps.
    """
    debut_stacking = time.time()
    print("\nEntraînement des modèles de base...")
    temps_entrainement_stacking = {}

    # Entraînement des modèles de base
    for nom in base_models:
        modele = modeles_finaux[nom]
        debut_modele = time.time()

        if isinstance(modele, CatBoostRegressor):
            modele.fit(X_train, y_train_log, eval_set=(X_val, y_val_log), sample_weight=sample_weights_train,
                       verbose=False)
        else:
            modele.fit(X_train, y_train_log, sample_weight=sample_weights_train)

        duree_modele = time.time() - debut_modele
        temps_entrainement_stacking[nom] = duree_modele
        print(f"  {nom} : entraîné en {duree_modele:.2f} s")

    # Prédictions échelle log
    preds_val_log = {nom: modeles_finaux[nom].predict(X_val) for nom in base_models}
    preds_test_log = {nom: modeles_finaux[nom].predict(X_test) for nom in base_models}
    preds_en_cours_log = {nom: modeles_finaux[nom].predict(X_en_cours) for nom in base_models}

    # Conversion en euros
    preds_val_euros = {nom: np.expm1(preds_val_log[nom]) for nom in base_models}
    preds_test_euros = {nom: np.expm1(preds_test_log[nom]) for nom in base_models}
    preds_en_cours_euros = {nom: np.expm1(preds_en_cours_log[nom]) for nom in base_models}

    # DataFrames d'entrées pour le méta-modèle
    X_meta_val = pd.DataFrame(preds_val_euros)
    X_meta_test = pd.DataFrame(preds_test_euros)
    X_meta_en_cours = pd.DataFrame(preds_en_cours_euros)

    # Prédictions finales du méta-modèle
    pred_val_stack = meta.predict(X_meta_val)
    pred_test_stack = meta.predict(X_meta_test)
    pred_en_cours_stack = meta.predict(X_meta_en_cours)

    duree_stacking_totale = time.time() - debut_stacking

    # Affichage du bilan
    print("\nPrédictions du Stacking calculées avec succès.")
    print( f"\nTemps total Stacking OOF (entraînement + prédictions) : {duree_stacking_totale:.2f} s" )
    for nom, duree in temps_entrainement_stacking.items():
        print(f"    dont {nom:15s} : {duree:.2f} s")

    return {
        "pred_val_stack": pred_val_stack,
        "pred_test_stack": pred_test_stack,
        "pred_en_cours_stack": pred_en_cours_stack,
        "preds_val_log": preds_val_log,
        "preds_test_log": preds_test_log,
        "preds_en_cours_log": preds_en_cours_log,
        "preds_val_euros": preds_val_euros,
        "preds_test_euros": preds_test_euros,
        "preds_en_cours_euros": preds_en_cours_euros,
        "temps_entrainement": temps_entrainement_stacking,
        "duree_totale": duree_stacking_totale,
    }


def analyser_erreurs_par_tranches(y_reel, y_pred, tranches=None, labels_tranches=None, afficher_resultat=True):
    """Analyse la répartition des erreurs du modèle par tranches de valeur.

    Parameters:
    -----------
    y_reel : pd.Series ou array-like
        Valeurs réelles du jeu de test.
    y_pred : pd.Series ou array-like
        Valeurs prédites par le modèle.
    tranches : list, optional
        Bornes des tranches de valeur.
    labels_tranches : list, optional
        Noms des tranches.
    afficher_resultat : bool, default=True
        Affiche le DataFrame résumé si True.

    Returns:
    --------
    pd.DataFrame : Synthèse des métriques par tranche.
    """
    # Valeurs par défaut des tranches si non fournies
    if tranches is None:
        tranches = [0, 1_000_000, 5_000_000, 20_000_000, 50_000_000, np.inf]
    if labels_tranches is None:
        labels_tranches = ["<1M€", "1-5M€", "5-20M€", "20-50M€", ">50M€"]

    # Transformation en Series pandas si besoin
    y_true_values = (y_reel.values if isinstance(y_reel, pd.Series) else np.array(y_reel))
    y_pred_values = (y_pred.values if isinstance(y_pred, pd.Series) else np.array(y_pred))

    # Création du DataFrame d'analyse
    df_erreur = pd.DataFrame({"y_reel": y_true_values, "y_pred": y_pred_values})

    df_erreur["erreur_abs"] = (df_erreur["y_reel"] - df_erreur["y_pred"]).abs()

    df_erreur["erreur_pct"] = (df_erreur["erreur_abs"] / df_erreur["y_reel"]) * 100
    df_erreur["tranche"] = pd.cut(df_erreur["y_reel"], bins=tranches, labels=labels_tranches)

    # Aggrégations par tranche
    synthese_tranches = df_erreur.groupby("tranche", observed=True).agg(
        n_joueurs=("y_reel", "size"),
        valeur_moyenne=("y_reel", "mean"),
        MAE=("erreur_abs", "mean"),
        MAPE_pct=("erreur_pct", "mean"),
    )

    # Calculs relatifs sur l'ensemble
    total_joueurs = len(df_erreur)
    total_erreur_abs = df_erreur["erreur_abs"].sum()

    synthese_tranches["% du total joueurs"] = (synthese_tranches["n_joueurs"] / total_joueurs * 100)
    synthese_tranches["% de l'erreur absolue totale"] = (
        df_erreur.groupby("tranche", observed=True)["erreur_abs"].sum()
        / total_erreur_abs
        * 100
    )

    if afficher_resultat:
        print(synthese_tranches)

    return synthese_tranches


def afficher_graphiques_analyse_erreurs(y_reel, y_pred, synthese_tranches=None, colonne_mape="MAPE_pct",
                                        random_state=42):
    """Génère les graphiques d'analyse de l'erreur par tranche et le scatter plot

    Valeurs Réelles vs Prédites.
    """
    # 1. Si synthese_tranches n'est pas fourni, on la génère brièvement
    if synthese_tranches is None:
        synthese_tranches = analyser_erreurs_par_tranches(y_reel, y_pred, afficher_resultat=False)

    # Vérification du nom de la colonne MAPE dans le DataFrame
    if colonne_mape not in synthese_tranches.columns:
        colonne_mape = "MAPE"

    # Graphiques par tranche de valeur
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    synthese_tranches[colonne_mape].plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Erreur relative (MAPE) par tranche de valeur")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].tick_params(axis="x", rotation=0)

    synthese_tranches["% de l'erreur absolue totale"].plot(kind="bar", ax=axes[1], color="indianred")
    axes[1].set_title("Part de l'erreur absolue totale par tranche")
    axes[1].set_ylabel("% de l'erreur totale")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.show()


def analyser_et_afficher_erreurs_par_age(y_reel, y_pred, ages, tranches_age=None, labels_age=None,
                                         afficher_graphique=True):
    """Analyse les erreurs du modèle par tranche d'âge et affiche les graphiques associes.

    Parameters:
    -----------
    y_reel : pd.Series ou array-like
        Valeurs réelles du jeu de test.
    y_pred : pd.Series ou array-like
        Valeurs prédites par le modèle.
    ages : pd.Series ou array-like
        Âges des joueurs correspondants (ex: df_test['age']).
    tranches_age : list, optional
        Bornes des tranches d'âge.
    labels_age : list, optional
        Libellés des tranches.
    afficher_graphique : bool, default=True
        Affiche les barplots si True.

    Returns:
    --------
    pd.DataFrame : Synthèse des métriques par tranche d'âge.
    """
    if tranches_age is None:
        tranches_age = [0, 20, 24, 28, 32, 100]
    if labels_age is None:
        labels_age = ["<21 ans", "21-24 ans", "25-28 ans", "29-32 ans", "33+ ans"]

    # Extraction des numpy arrays
    y_true_vals = (y_reel.values if isinstance(y_reel, pd.Series) else np.array(y_reel))
    y_pred_vals = (y_pred.values if isinstance(y_pred, pd.Series) else np.array(y_pred))
    age_vals = ages.values if isinstance(ages, pd.Series) else np.array(ages)

    # DataFrame de travail
    df_erreur_age = pd.DataFrame({"y_reel": y_true_vals, "y_pred": y_pred_vals, "age": age_vals})

    df_erreur_age["erreur_abs"] = (df_erreur_age["y_reel"] - df_erreur_age["y_pred"]).abs()
    df_erreur_age["erreur_pct"] = (df_erreur_age["erreur_abs"] / df_erreur_age["y_reel"]) * 100
    df_erreur_age["tranche_age"] = pd.cut(df_erreur_age["age"], bins=tranches_age, labels=labels_age)

    # Aggrégations
    synthese_age = df_erreur_age.groupby("tranche_age", observed=True).agg(
        n_joueurs=("y_reel", "size"),
        age_moyen=("age", "mean"),
        valeur_moyenne=("y_reel", "mean"),
        MAE=("erreur_abs", "mean"),
        MAPE_pct=("erreur_pct", "mean"),
    )

    total_joueurs = len(df_erreur_age)
    total_erreur_abs = df_erreur_age["erreur_abs"].sum()

    synthese_age["% du total joueurs"] = (synthese_age["n_joueurs"] / total_joueurs * 100)
    synthese_age["% de l'erreur absolue totale"] = (
        df_erreur_age.groupby("tranche_age", observed=True)["erreur_abs"].sum()
        / total_erreur_abs
        * 100
    )

    print(synthese_age)

    # Display des graphiques
    if afficher_graphique:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        synthese_age["MAPE_pct"].plot(kind="bar", ax=axes[0], color="steelblue")
        axes[0].set_title("Erreur relative (MAPE %) par tranche d'âge")
        axes[0].set_ylabel("MAPE (%)")
        axes[0].tick_params(axis="x", rotation=0)

        synthese_age["% de l'erreur absolue totale"].plot(
            kind="bar", ax=axes[1], color="indianred"
        )
        axes[1].set_title("Part de l'erreur absolue totale par tranche d'âge")
        axes[1].set_ylabel("% de l'erreur totale")
        axes[1].tick_params(axis="x", rotation=0)

        plt.tight_layout()
        plt.show()

    return synthese_age


def analyser_et_afficher_erreurs_par_poste(y_reel, y_pred, positions, seuil_min_joueurs=0,
                                           afficher_graphique=True):
    """Analyse les erreurs du modèle selon les postes/positions des joueurs

    et affiche les barplots associés.

    Parameters:
    -----------
    y_reel : pd.Series ou array-like
        Valeurs réelles du jeu de test.
    y_pred : pd.Series ou array-like
        Valeurs prédites par le modèle.
    positions : pd.Series ou array-like
        Postes des joueurs correspondants (ex: df_test['position']).
    seuil_min_joueurs : int, default=0
        Nombre minimal de joueurs pour conserver un poste dans l'analyse.
    afficher_graphique : bool, default=True
        Affiche les graphiques si True.

    Returns:
    --------
    pd.DataFrame : Synthèse des métriques par poste.
    """

    y_true_vals = (y_reel.values if isinstance(y_reel, pd.Series) else np.array(y_reel))
    y_pred_vals = (y_pred.values if isinstance(y_pred, pd.Series) else np.array(y_pred))
    pos_vals = (
        positions.values
        if isinstance(positions, pd.Series)
        else np.array(positions)
    )

    # DataFrame de travail
    df_erreur_poste = pd.DataFrame({"y_reel": y_true_vals, "y_pred": y_pred_vals, "position": pos_vals})

    df_erreur_poste["erreur_abs"] = (df_erreur_poste["y_reel"] - df_erreur_poste["y_pred"]).abs()

    df_erreur_poste["erreur_pct"] = (df_erreur_poste["erreur_abs"] / df_erreur_poste["y_reel"]) * 100

    # Filtrage des postes selon l'effectif
    if seuil_min_joueurs > 0:
        effectifs = df_erreur_poste["position"].value_counts()
        postes_retenus = effectifs[effectifs >= seuil_min_joueurs].index
        df_erreur_poste = df_erreur_poste[
            df_erreur_poste["position"].isin(postes_retenus)
        ]

    # Aggrégations
    synthese_poste = (
        df_erreur_poste.groupby("position", observed=True)
        .agg(
            n_joueurs=("y_reel", "size"),
            valeur_moyenne=("y_reel", "mean"),
            MAE=("erreur_abs", "mean"),
            MAPE_pct=("erreur_pct", "mean"),
        )
        .sort_values("MAPE_pct")
    )

    total_joueurs = len(df_erreur_poste)
    total_erreur_abs = df_erreur_poste["erreur_abs"].sum()

    synthese_poste["% du total joueurs"] = (synthese_poste["n_joueurs"] / total_joueurs * 100)
    synthese_poste["% de l'erreur absolue totale"] = (
        df_erreur_poste.groupby("position", observed=True)["erreur_abs"].sum()
        / total_erreur_abs
        * 100
    )

    print(synthese_poste)

    # Display des graphiques
    if afficher_graphique:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        synthese_poste["MAPE_pct"].plot(kind="bar", ax=axes[0], color="steelblue")
        axes[0].set_title("Erreur relative (MAPE %) par poste")
        axes[0].set_ylabel("MAPE (%)")
        axes[0].tick_params(axis="x", rotation=30)

        synthese_poste["% de l'erreur absolue totale"].plot(kind="bar", ax=axes[1], color="indianred")
        axes[1].set_title("Part de l'erreur absolue totale par poste")
        axes[1].set_ylabel("% de l'erreur totale")
        axes[1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.show()

    return synthese_poste