import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import time
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.base import clone
from catboost import CatBoostRegressor

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from matplotlib.patches import Patch



def preparer_donnees(df_train, df_val, df_test, colonne_cible):
    """Nettoie les valeurs manquantes cibles, sépare features/cible,

    calcule les cibles log & sample weights, et génère des versions
    sans NaN pour les modèles linéaires/SVR.
    """
    # Suppression des lignes avec valeur manquante sur la cible
    df_train = df_train.dropna(subset=[colonne_cible])
    df_val = df_val.dropna(subset=[colonne_cible])
    df_test = df_test.dropna(subset=[colonne_cible])

    # Suppression des colonnes normalisées (_nor)
    colonnes_nor = [c for c in df_train.columns if c.endswith("_nor")]
    df_train = df_train.drop(columns=colonnes_nor, errors="ignore")
    df_val = df_val.drop(columns=colonnes_nor, errors="ignore")
    df_test = df_test.drop(columns=colonnes_nor, errors="ignore")

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

    # Cibles log (log1p)
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)

    # Poids d'entraînement (sur-pondération des petites valeurs)
    sample_weights_train = 1 / np.log1p(y_train)
    sample_weights_train = sample_weights_train / sample_weights_train.mean()

    # Sélection des colonnes sans aucun NaN (basée sur le train set)
    colonnes_sans_nan = X_train.columns[X_train.isna().sum() == 0].tolist()

    print( f"X_train : {X_train.shape} | X_val : {X_val.shape} | X_test : {X_test.shape}" )
    print("Filtrage pour les modèles linéaires/SVR :")
    print(f"{len(colonnes_sans_nan)} colonnes conservées sur {X_train.shape[1]}.")

    X_train_sans_nan = X_train[colonnes_sans_nan]
    X_val_sans_nan = X_val[colonnes_sans_nan]
    X_test_sans_nan = X_test[colonnes_sans_nan]

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
        "X_train_sans_nan": X_train_sans_nan,
        "X_val_sans_nan": X_val_sans_nan,
        "X_test_sans_nan": X_test_sans_nan,
    }


def evaluer_modeles(modeles, data, colonne_cible="value", mode="val"):
    """Entraîne les modèles et évalue leurs performances sur la Validation ou le Test.

    Parameters:
    -----------
    modeles : dict
        Dictionnaire des modèles {nom: instance_modele}
    data : dict
        Dictionnaire renvoyé par preparer_donnees
    colonne_cible : str
        Nom de la colonne cible (pour l'affichage)
    mode : str
        'val' pour évaluer sur la validation, 'test' pour évaluer sur le test
    """
    mode = mode.lower()
    if mode not in ["val", "test"]:
        raise ValueError("Le paramètre 'mode' doit être 'val' ou 'test'.")

    # Sélection des jeux de données d'évaluation selon le mode
    X_train = data["X_train"]
    y_train = data["y_train"]

    if mode == "val":
        X_eval, y_eval = data["X_val"], data["y_val"]
        label_eval = "Validation"
    else:
        X_eval, y_eval = data["X_test"], data["y_test"]
        label_eval = "Test"

    resultats = {}

    for nom, modele in modeles.items():
        print(f"Entraînement de {nom}...")
        modele.fit(X_train, y_train)
        preds = modele.predict(X_eval)

        mae = mean_absolute_error(y_eval, preds)
        r2 = r2_score(y_eval, preds)
        rmse = np.sqrt(mean_squared_error(y_eval, preds))
        mape = mean_absolute_percentage_error(y_eval, preds)

        n, p = len(y_eval), X_eval.shape[1]
        r2_ajuste = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        resultats[nom] = {
            f"MAE {label_eval}": mae,
            f"RMSE {label_eval}": rmse,
            f"MAPE {label_eval}": mape,
            f"R² {label_eval}": r2,
            f"R² Ajusté {label_eval}": r2_ajuste,
        }

        print( f"   -> {label_eval} | MAE : {mae:,.0f} € | RMSE : {rmse:,.0f} € | MAPE : {mape:.2%} | R² : {r2:.2%} | R² Ajusté : {r2_ajuste:.2%}" )

    print(f"\nClassement final {label_eval} (trié par MAE croissante)")
    tableau = []
    for nom, m in resultats.items():
        tableau.append(
            {
                "Modèle": nom,
                f"Erreur moyenne {label_eval} (MAE)": f"{m[f'MAE {label_eval}']:,.0f} €",
                f"Score R² {label_eval}": f"{m[f'R² {label_eval}']:.2%}",
                f"Score R² Ajusté {label_eval}": f"{m[f'R² Ajusté {label_eval}']:.2%}",
            }
        )

    df_resultats = pd.DataFrame(tableau).sort_values(
        f"Erreur moyenne {label_eval} (MAE)"
    )
    print(df_resultats.to_string(index=False))

    return df_resultats, resultats


def tune_forest(model_class, trial, use_log=False, random_state=42, X_train=None, y_train=None,
                y_train_log=None, X_val=None, y_val=None):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]
        ),
        "random_state": random_state,
        "n_jobs": -1,
    }

    model = model_class(**params)

    # Adaptations selon la cible (normal ou log)
    y_target = y_train_log if use_log else y_train
    model.fit(X_train, y_target)

    preds = model.predict(X_val)
    if use_log:
        preds = np.expm1(preds)

    return np.sqrt(mean_squared_error(y_val, preds))


def tune_xgb( model_class, trial, transfo=None, random_state=42, X_train=None, y_train=None, X_val=None, y_val=None, ):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 5000, step=250),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.2, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "tree_method": "hist",
        "enable_categorical": True,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }

    model = model_class(**params)

    # Application de la transformation
    y_tr = (
        transfo["transform"](y_train) if transfo is not None else y_train.copy()
    )
    model.fit(X_train, y_tr)

    # Prédictions et inversion
    preds = model.predict(X_val)
    if transfo is not None:
        preds = transfo["inverse"](preds)

    return np.sqrt(mean_squared_error(y_val, preds))


def tune_lgbm( model_class, trial, transfo=None, random_state=42, X_train=None, y_train=None, X_val=None, y_val=None, ):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 5000, step=250),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.2, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = model_class(**params)

    # Application de la transformation
    y_tr = (
        transfo["transform"](y_train) if transfo is not None else y_train.copy()
    )
    model.fit(X_train, y_tr)

    # Prédictions et inversion
    preds = model.predict(X_val)
    if transfo is not None:
        preds = transfo["inverse"](preds)

    return np.sqrt(mean_squared_error(y_val, preds))


def tune_catboost( model_class, trial, transfo=None, random_state=42, X_train=None, y_train=None, X_val=None, y_val=None, ):
    # Adapter la stratégie de recherche si une transformation est appliquée
    if transfo is not None:
        lr = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
        patience = max(50, int(100 * (0.05 / lr)))
        iterations_range = (1000, 10000)
        depth_range = (4, 10)
    else:
        lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        patience = 50
        iterations_range = (500, 3000)
        depth_range = (4, 8)

    params = {
        "iterations": trial.suggest_int(
            "iterations", *iterations_range, step=500
        ),
        "depth": trial.suggest_int("depth", *depth_range),
        "learning_rate": lr,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "bootstrap_type": "Bernoulli",
        "random_state": random_state,
        "early_stopping_rounds": patience,
        "verbose": 0,
    }

    model = model_class(**params)

    # Transformations de y_train et y_val
    if transfo is not None:
        y_tr = transfo["transform"](y_train)
        y_v = transfo["transform"](y_val)
    else:
        y_tr = y_train.copy()
        y_v = y_val.copy()

    model.fit(X_train, y_tr, eval_set=(X_val, y_v), verbose=False)

    # Prédictions et inversion
    preds = model.predict(X_val)
    if transfo is not None:
        preds = transfo["inverse"](preds)

    return np.sqrt(mean_squared_error(y_val, preds))


def tune_svr( trial, use_log=False,
             X_train_sans_nan=None, y_train=None, y_train_log=None, X_val_sans_nan=None, y_val=None):
    params = {
        "C": trial.suggest_float("C", 0.01, 100, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.001, 1, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }

    pipeline = Pipeline([("scaler", StandardScaler()), ("svr", SVR(**params))])

    # Sélection de la cible (brute ou log)
    y_target = y_train_log if use_log else y_train
    pipeline.fit(X_train_sans_nan, y_target)

    # Prédiction et conversion inverse si log
    preds = pipeline.predict(X_val_sans_nan)
    if use_log:
        preds = np.expm1(preds)

    return np.sqrt(mean_squared_error(y_val, preds))


def tune_enet(trial, use_log=False, random_state=42,
              X_train_sans_nan=None, y_train=None, y_train_log=None, X_val_sans_nan=None, y_val=None, ):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "random_state": random_state,
        "max_iter": 10000,
    }

    pipeline = Pipeline([("scaler", StandardScaler()), ("enet", ElasticNet(**params))])

    # Sélection de la cible (brute ou log)
    y_target = y_train_log if use_log else y_train
    pipeline.fit(X_train_sans_nan, y_target)

    # Prédiction et conversion inverse si log
    preds = pipeline.predict(X_val_sans_nan)
    if use_log:
        preds = np.expm1(preds)

    return np.sqrt(mean_squared_error(y_val, preds))


def evaluer_modeles_tuned( modeles_tuned, modeles_tuned_sans_nan, data, mode="val", use_log=False, ):
    """Entraîne et évalue les modèles optimisés sur les jeux de Validation ou de Test.

    Parameters:
    -----------
    modeles_tuned : dict
        Dictionnaire des modèles gérant les NaN.
    modeles_tuned_sans_nan : dict
        Dictionnaire des modèles/Pipelines nécessitant des données sans NaN.
    data : dict
        Dictionnaire renvoyé par la fonction preparer_donnees.
    mode : str
        'val' pour évaluer sur la validation, 'test' pour le jeu de test.
    use_log : bool
        Si True, entraîne sur y_train_log et applique np.expm1 sur les prédictions.
    """
    mode = mode.lower()
    if mode not in ["val", "test"]:
        raise ValueError("Le paramètre 'mode' doit être 'val' ou 'test'.")

    # Sélection de la cible d'entraînement selon use_log
    y_train_target = data["y_train_log"] if use_log else data["y_train"]

    # Sélection des jeux de validation ou test
    if mode == "val":
        X_eval = data["X_val"]
        X_eval_sn = data["X_val_sans_nan"]
        y_eval = data["y_val"]
        label = "Validation"
    else:
        X_eval = data["X_test"]
        X_eval_sn = data["X_test_sans_nan"]
        y_eval = data["y_test"]
        label = "Test"

    def evaluer(modele, X_tr, y_tr, X_ev, y_ev):
        modele.fit(X_tr, y_tr)
        preds = modele.predict(X_ev)

        # Repassage à l'échelle d'origine si entraînement en log
        if use_log:
            preds = np.expm1(preds)

        mae = mean_absolute_error(y_ev, preds)
        rmse = np.sqrt(mean_squared_error(y_ev, preds))
        mape = mean_absolute_percentage_error(y_ev, preds)
        r2 = r2_score(y_ev, preds)

        n, p = len(y_ev), X_ev.shape[1]
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2,
            "R2 Ajusté": r2_adj,
        }

    resultats = {}

    # Évaluation des modèles standards
    for nom, modele in modeles_tuned.items():
        print(f"Entraînement (tuné) de {nom}...")
        resultats[nom] = evaluer(
            modele, data["X_train"], y_train_target, X_eval, y_eval
        )

    # Évaluation des modèles sans NaN
    if modeles_tuned_sans_nan:
        for nom, modele in modeles_tuned_sans_nan.items():
            print(f"Entraînement (tuné) de {nom}...")
            resultats[nom] = evaluer(
                modele, data["X_train_sans_nan"], y_train_target, X_eval_sn, y_eval
            )

    # Construction du tableau récapitulatif
    df_resultats = pd.DataFrame(resultats).T.sort_values("MAE")

    print(f"\nClassement ({label}, modèles tunés)")
    print(df_resultats)

    return df_resultats


def preparer_meta_features(
    base_models_names, modeles_dict, data, cv=5, n_jobs=-1
):
    """Génère les méta-features (OOF, Validation et Test) pour le Stacking.

    Returns:
    --------
    X_meta_train, X_meta_val, X_meta_test, temps_entrainement
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    y_train_log = data["y_train_log"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    oof_preds = {}
    preds_val_base = {}
    preds_test_base = {}
    temps_entrainement = {}

    print( f"Génération des OOF predictions (cross_val_predict, cv={cv})..." )

    for nom in base_models_names:
        debut_modele = time.time()
        modele = modeles_dict[nom]

        # Génération des OOF sur train
        oof_log = cross_val_predict(
            modele, X_train, y_train_log, cv=cv, n_jobs=n_jobs
        )
        oof_preds[nom] = np.expm1(oof_log)

        # Re-fit du modèle sur tout le train + prédictions test
        modele.fit(X_train, y_train_log)
        preds_test_base[nom] = np.expm1(modele.predict(X_test))

        # Prédictions sur le jeu de Validation
        preds_val_base[nom] = np.expm1(modele.predict(X_val))

        duree = time.time() - debut_modele
        temps_entrainement[nom] = duree

        rmse_oof = np.sqrt(mean_squared_error(y_train, oof_preds[nom]))
        print(f"  {nom} : OOF RMSE = {rmse_oof:,.0f} € ({duree:.1f}s)")

    X_meta_train = pd.DataFrame(oof_preds)
    X_meta_val = pd.DataFrame(preds_val_base)
    X_meta_test = pd.DataFrame(preds_test_base)

    return X_meta_train, X_meta_val, X_meta_test, temps_entrainement


def evaluer_ensembles( X_meta_train, X_meta_val, X_meta_test, data, modeles_tuned_log,
                      nom_modele_seul="CatBoost (log)", ):
    """Calcule les prédictions par moyenne et par stacking OOF, puis construit le DataFrame de comparaison.

    Returns:
    --------
    pred_val_moyenne, pred_test_moyenne, pred_val_stack, pred_test_stack, meta_modele, comparaison_blend
    """
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    base_models_names = X_meta_train.columns.tolist()

    # Blend 1 : moyenne simple
    pred_val_moyenne = X_meta_val.mean(axis=1)
    pred_test_moyenne = X_meta_test.mean(axis=1)

    print("Blend (moyenne simple)")
    print( f"  Val  -> MAE : {mean_absolute_error(y_val, pred_val_moyenne):,.0f} € | "
        f"R² : {r2_score(y_val, pred_val_moyenne):.4f}" )
    print( f"  Test -> MAE : {mean_absolute_error(y_test, pred_test_moyenne):,.0f} € | "
        f"R² : {r2_score(y_test, pred_test_moyenne):.4f}" )

    # Blend 2 : stacking OOF (méta-modèle entraîné sur les prédictions OOF)
    debut_meta = time.time()
    meta_modele = LinearRegression(positive=True)
    meta_modele.fit(X_meta_train, y_train)

    pred_val_stack = meta_modele.predict(X_meta_val)
    pred_test_stack = meta_modele.predict(X_meta_test)

    poids = dict(zip(base_models_names, meta_modele.coef_.round(3)))

    print(f"\nPoids appris par le méta-modèle : {poids}")
    print(f"Intercept : {meta_modele.intercept_:,.0f} €")
    print("\nStacking OOF (méta-modèle)")
    print( f"  Val  -> MAE : {mean_absolute_error(y_val, pred_val_stack):,.0f} € | "
        f"R² : {r2_score(y_val, pred_val_stack):.4f}" )
    print( f"  Test -> MAE : {mean_absolute_error(y_test, pred_test_stack):,.0f} € | "
        f"R² : {r2_score(y_test, pred_test_stack):.4f}\n" )

    # Prédictions du modèle seul pour comparaison
    modele_seul = modeles_tuned_log[nom_modele_seul]
    pred_val_seul = np.expm1(modele_seul.predict(X_val))
    pred_test_seul = np.expm1(modele_seul.predict(X_test))

    # Construction du DataFrame comparatif
    comparaison_blend = pd.DataFrame({
        "MAE Val": {
            f"{nom_modele_seul} seul": mean_absolute_error(
                y_val, pred_val_seul
            ),
            "Blend (moyenne)": mean_absolute_error(y_val, pred_val_moyenne),
            "Blend (stacking OOF)": mean_absolute_error(y_val, pred_val_stack),
        },
        "R2 Val": {
            f"{nom_modele_seul} seul": r2_score(y_val, pred_val_seul),
            "Blend (moyenne)": r2_score(y_val, pred_val_moyenne),
            "Blend (stacking OOF)": r2_score(y_val, pred_val_stack),
        },
        "MAE Test": {
            f"{nom_modele_seul} seul": mean_absolute_error(
                y_test, pred_test_seul
            ),
            "Blend (moyenne)": mean_absolute_error(y_test, pred_test_moyenne),
            "Blend (stacking OOF)": mean_absolute_error(
                y_test, pred_test_stack
            ),
        },
        "R2 Test": {
            f"{nom_modele_seul} seul": r2_score(y_test, pred_test_seul),
            "Blend (moyenne)": r2_score(y_test, pred_test_moyenne),
            "Blend (stacking OOF)": r2_score(y_test, pred_test_stack),
        },
    }).sort_values("MAE Test")

    print(comparaison_blend)

    return (pred_val_moyenne, pred_test_moyenne, pred_val_stack, pred_test_stack,
            meta_modele, comparaison_blend)


def comparaison_des_blend(data, modeles_tuned_log, pred_val_moyenne, pred_test_moyenne,
                          pred_val_stack, pred_test_stack, nom_modele_seul="CatBoost (log)", ):
    """Construit et affiche le DataFrame de comparaison entre le meilleur modèle seul,

    le blend par moyenne et le blend par stacking.

    Parameters:
    -----------
    data : dict
        Dictionnaire contenant y_val, y_test, X_val, X_test.
    modeles_tuned_log : dict
        Dictionnaire contenant le modèle individuel entraîné.
    pred_val_moyenne, pred_test_moyenne : array-like
        Prédictions sur Val/Test pour le blend par moyenne.
    pred_val_stack, pred_test_stack : array-like
        Prédictions sur Val/Test pour le stacking OOF.
    nom_modele_seul : str
        Clé du modèle individuel dans modeles_tuned_log.

    Returns:
    --------
    pd.DataFrame
        Tableau comparatif trié par MAE Test.
    """
    y_val = data["y_val"]
    y_test = data["y_test"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    # Prédictions du modèle seul
    modele_seul = modeles_tuned_log[nom_modele_seul]
    pred_val_seul = np.expm1(modele_seul.predict(X_val))
    pred_test_seul = np.expm1(modele_seul.predict(X_test))

    # Construction du DataFrame
    comparaison_blend = pd.DataFrame({
        "MAE Val": {
            f"{nom_modele_seul} seul": mean_absolute_error(
                y_val, pred_val_seul
            ),
            "Blend (moyenne)": mean_absolute_error(y_val, pred_val_moyenne),
            "Blend (stacking OOF)": mean_absolute_error(y_val, pred_val_stack),
        },
        "R2 Val": {
            f"{nom_modele_seul} seul": r2_score(y_val, pred_val_seul),
            "Blend (moyenne)": r2_score(y_val, pred_val_moyenne),
            "Blend (stacking OOF)": r2_score(y_val, pred_val_stack),
        },
        "MAE Test": {
            f"{nom_modele_seul} seul": mean_absolute_error(
                y_test, pred_test_seul
            ),
            "Blend (moyenne)": mean_absolute_error(y_test, pred_test_moyenne),
            "Blend (stacking OOF)": mean_absolute_error(
                y_test, pred_test_stack
            ),
        },
        "R2 Test": {
            f"{nom_modele_seul} seul": r2_score(y_test, pred_test_seul),
            "Blend (moyenne)": r2_score(y_test, pred_test_moyenne),
            "Blend (stacking OOF)": r2_score(y_test, pred_test_stack),
        },
    }).sort_values("MAE Test")

    print(comparaison_blend)
    return comparaison_blend


def analyser_erreurs_par_tranches(y_test, meilleure_prediction_test, tranches=None,
                                  labels_tranches=None, random_state=42):
    """Analyse les erreurs de prédiction par tranches de valeur réelle et génère les graphiques associés.

    Parameters:
    -----------
    y_test : pd.Series ou array-like
        Valeurs réelles du jeu de test.
    meilleure_prediction_test : array-like
        Prédictions sur le jeu de test.
    tranches : list, optional
        Bornes des tranches de valeur.
    labels_tranches : list, optional
        Noms des tranches.
    random_state : int
        Graine pour le sous-échantillonnage du scatter plot.

    Returns:
    --------
    pd.DataFrame
        Tableau de synthèse des erreurs par tranche.
    """
    if tranches is None:
        tranches = [0, 5_000_000, 20_000_000, 50_000_000, np.inf]
    if labels_tranches is None:
        labels_tranches = ["<5M€", "5-20M€", "20-50M€", ">50M€"]

    # Alignment des séries
    y_true_vals = y_test.values if hasattr(y_test, "values") else y_test

    df_erreur = pd.DataFrame({
        "y_reel": y_true_vals,
        "y_pred": meilleure_prediction_test,
    })
    df_erreur["erreur_abs"] = (df_erreur["y_reel"] - df_erreur["y_pred"]).abs()
    df_erreur["erreur_pct"] = df_erreur["erreur_abs"] / df_erreur["y_reel"]
    df_erreur["tranche"] = pd.cut(
        df_erreur["y_reel"], bins=tranches, labels=labels_tranches
    )

    # Synthèse par tranche
    synthese_tranches = df_erreur.groupby("tranche", observed=True).agg(
        n_joueurs=("y_reel", "size"),
        valeur_moyenne=("y_reel", "mean"),
        MAE=("erreur_abs", "mean"),
        MAPE=("erreur_pct", "mean"),
    )
    synthese_tranches["% du total joueurs"] = (
        synthese_tranches["n_joueurs"] / len(df_erreur) * 100
    )
    synthese_tranches["% de l'erreur absolue totale"] = (
        df_erreur.groupby("tranche", observed=True)["erreur_abs"].sum()
        / df_erreur["erreur_abs"].sum()
        * 100
    )

    print(synthese_tranches)

    # Figure 1 : Graphiques à barres
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    synthese_tranches["MAPE"].plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Erreur relative (MAPE) par tranche de valeur")
    axes[0].set_ylabel("MAPE")
    axes[0].tick_params(axis="x", rotation=0)

    synthese_tranches["% de l'erreur absolue totale"].plot(
        kind="bar", ax=axes[1], color="indianred"
    )
    axes[1].set_title("Part de l'erreur absolue totale par tranche")
    axes[1].set_ylabel("% de l'erreur totale (MAE x n)")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.show()

    # Figure 2 : Scatter plot Réel vs Prédit
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    sample = df_erreur.sample(
        min(2000, len(df_erreur)), random_state=random_state
    )
    ax.scatter(
        sample["y_reel"] / 1e6,
        sample["y_pred"] / 1e6,
        alpha=0.3,
        s=8,
        color="teal",
    )
    lim = max(sample["y_reel"].max(), sample["y_pred"].max()) / 1e6
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Prédiction parfaite")
    ax.set_xlabel("Valeur réelle (M€)")
    ax.set_ylabel("Valeur prédite (M€)")
    ax.set_title("Réel vs Prédit")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return synthese_tranches


def afficher_feature_importances(modele, feature_names, top_n=25, nom_modele="Modèle",
                                 figsize=(8, 8), color="teal", ):
    """Calcule et affiche l'importance des variables pour un modèle donné.

    Parameters:
    -----------
    modele : estimator object
        Le modèle entraîné (CatBoost, XGBoost, LightGBM, Random Forest, etc.).
    feature_names : Index ou list
        La liste des noms de variables (ex: X_train.columns).
    top_n : int, default=25
        Le nombre de variables top à afficher.
    nom_modele : str, default="Modèle"
        Nom du modèle pour le titre du graphique.
    figsize : tuple, default=(8, 8)
        Taille de la figure Matplotlib.
    color : str, default="teal"
        Couleur des barres du graphique.

    Returns:
    --------
    pd.Series
        Série contenant toutes les importances triées par ordre décroissant.
    """
    # Extraction dynamique des importances selon le type de modèle
    if hasattr(modele, "get_feature_importance"):
        # Spécifique CatBoost
        raw_importances = modele.get_feature_importance()
    elif hasattr(modele, "feature_importances_"):
        # Tree-based Scikit-Learn, LightGBM, XGBoost
        raw_importances = modele.feature_importances_
    elif hasattr(modele, "coef_"):
        # Modèles linéaires / Ridge / Lasso
        raw_importances = (
            modele.coef_[0] if modele.coef_.ndim > 1 else modele.coef_
        )
    else:
        raise ValueError( f"Le modèle {type(modele).__name__} ne possède pas d'attribut d'importance des variables direct." )

    # Création et tri de la série
    importances = pd.Series(raw_importances, index=feature_names).sort_values(ascending=False)

    # Graphique
    plt.figure(figsize=figsize)
    importances.head(top_n).sort_values().plot(kind="barh", color=color)
    plt.title(f"Top {top_n} variables les plus importantes — {nom_modele}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    print(importances.head(top_n))

    return importances


def analyser_transformations_cible(y_train, random_state=42):
    """Compare visuellement différentes transformations sur la variable cible.

    Parameters:
    -----------
    y_train : pd.Series ou array-like
        Série contenant la variable cible brute.
    random_state : int, default=42
        Graine pour le QuantileTransformer.

    Returns:
    --------
    dict
        Dictionnaire contenant les séries transformées.
    """
    y_series = (
        y_train if isinstance(y_train, pd.Series) else pd.Series(y_train)
    )
    y_vals = y_series.values.reshape(-1, 1)

    transformations_viz = {
        "Brute (€)": y_series,
        "log1p": np.log1p(y_series),
        "log10": np.log10(y_series),
        "sqrt": np.sqrt(y_series),
        "Yeo-Johnson": pd.Series(
            PowerTransformer(method="yeo-johnson")
            .fit_transform(y_vals)
            .flatten(),
            index=y_series.index,
        ),
        "MinMax [0,1]": pd.Series(
            MinMaxScaler().fit_transform(y_vals).flatten(),
            index=y_series.index,
        ),
        "Rang centile": pd.Series(
            QuantileTransformer(
                output_distribution="uniform", random_state=random_state
            )
            .fit_transform(y_vals)
            .flatten(),
            index=y_series.index,
        ),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for ax, (label, y_t) in zip(axes.flatten(), transformations_viz.items()):
        ax.hist(y_t, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
        skew = float(pd.Series(y_t).skew())
        ax.set_title(f"{label}\nskewness = {skew:.2f}")
        ax.tick_params(labelsize=7)

    # Masquer le dernier axe inutilisé
    if len(transformations_viz) < len(axes.flatten()):
        fig.delaxes(axes.flatten()[-1])

    plt.suptitle("Distribution de la cible selon la transformation", fontsize=13)
    plt.tight_layout()
    plt.show()

    print("Statistiques de la cible brute :")
    print(y_series.describe())

    return transformations_viz


def preparer_transformations_cible(y_train, random_state=42):
    """Instancie, ajuste les transformateurs data-dependent sur y_train

    et retourne un dictionnaire de transformations (fonctions transform & inverse).

    Parameters:
    -----------
    y_train : pd.Series ou array-like
        Série de la variable cible sur l'ensemble d'entraînement.
    random_state : int, default=42
        Graine pour la reproductibilité des QuantileTransformers.

    Returns:
    --------
    dict
        Dictionnaire contenant les transformations et leurs fonctions inverses.
    """
    y_vals = (
        y_train.values.reshape(-1, 1)
        if hasattr(y_train, "values")
        else np.array(y_train).reshape(-1, 1)
    )

    # Fit des transformateurs data-dependent sur train uniquement
    yj_transformer = PowerTransformer(method="yeo-johnson")
    yj_transformer.fit(y_vals)

    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(y_vals)

    quantile_unif = QuantileTransformer(
        output_distribution="uniform",
        random_state=random_state,
        n_quantiles=1000,
    )
    quantile_unif.fit(y_vals)

    quantile_norm = QuantileTransformer(
        output_distribution="normal",
        random_state=random_state,
        n_quantiles=1000,
    )
    quantile_norm.fit(y_vals)

    # Helper pour sécuriser le formatage des entrées Series/Arrays
    def _to_2d(y):
        return (
            y.values.reshape(-1, 1)
            if hasattr(y, "values")
            else np.array(y).reshape(-1, 1)
        )

    # Dictionnaire de fonctions de transformation / inversion
    transformations = {
        "Brute": {
            "transform": lambda y: y,
            "inverse": lambda y: np.clip(y, 0, None),
            "description": "Aucune transformation. MAE en euros bruts.",
        },
        "log1p (baseline)": {
            "transform": lambda y: np.log1p(y),
            "inverse": lambda y: np.expm1(y),
            "description": "log(1+y) : transformation de référence.",
        },
        "log10": {
            "transform": lambda y: np.log10(np.clip(y, 1, None)),
            "inverse": lambda y: np.power(10, y),
            "description": "log10(y) : espace plus lisible (6 = 1M€, 7 = 10M€).",
        },
        "sqrt": {
            "transform": lambda y: np.sqrt(y),
            "inverse": lambda y: np.square(np.clip(y, 0, None)),
            "description": "Racine carrée : compression plus douce que le log.",
        },
        "Yeo-Johnson": {
            "transform": lambda y: yj_transformer.transform(_to_2d(y)).flatten(),
            "inverse": lambda y: np.clip(
                yj_transformer.inverse_transform(_to_2d(y)).flatten(), 0, None
            ),
            "description": "Box-Cox généralisée optimisée par maximum de vraisemblance.",
        },
        "MinMax [0,1]": {
            "transform": lambda y: minmax_scaler.transform(_to_2d(y)).flatten(),
            "inverse": lambda y: np.clip(
                minmax_scaler.inverse_transform(_to_2d(y)).flatten(), 0, None
            ),
            "description": "Normalisation dans [0,1], sensible aux outliers.",
        },
        "Rang centile [0,1]": {
            "transform": lambda y: quantile_unif.transform(_to_2d(y)).flatten(),
            "inverse": lambda y: np.clip(
                quantile_unif.inverse_transform(_to_2d(y)).flatten(), 0, None
            ),
            "description": "Transforme en rangs uniformes [0,1], robuste aux outliers.",
        },
    }

    print(f"{len(transformations)} transformations définies.")
    return transformations


def evaluer_complet(label, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse_log = np.sqrt(
        mean_squared_error(
            np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))
        )
    )
    return {
        "Transformation": label,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R²": r2,
        "RMSE_log": rmse_log,
    }


def evaluer_transformations_stacking(transformations, modeles_dict, data, sample_weights_train=None,
                                     cv=5, n_jobs=-1, ):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    y_test_vals = y_test.values if hasattr(y_test, "values") else y_test
    sw = (
        sample_weights_train.values
        if hasattr(sample_weights_train, "values")
        else sample_weights_train
    )

    resultats = []
    predictions_test = {}

    for label, transfo in transformations.items():
        print(f"\n{label} (Stacking OOF)")
        print(f"  {transfo['description'][:80]}")

        try:
            y_tr_t = transfo["transform"](y_train)

            oof_preds_dict = {}
            preds_test_dict = {}

            fit_kwargs = {"sample_weight": sw} if sw is not None else {}

            for nom_m, modele_original in modeles_dict.items():
                modele = clone(modele_original)

                # Rétrocompatibilité automatique entre fit_params et params
                try:
                    oof_t = cross_val_predict(modele, X_train, y_tr_t, cv=cv, n_jobs=n_jobs,
                                              fit_params=fit_kwargs, )
                except TypeError:
                    oof_t = cross_val_predict(modele, X_train, y_tr_t, cv=cv, n_jobs=n_jobs, params=fit_kwargs)

                oof_preds_dict[nom_m] = oof_t

                modele.fit(X_train, y_tr_t, **fit_kwargs)
                preds_test_dict[nom_m] = modele.predict(X_test)

            X_meta_train = pd.DataFrame(oof_preds_dict)
            X_meta_test = pd.DataFrame(preds_test_dict)

            meta_modele = LinearRegression(positive=True)
            meta_modele.fit(X_meta_train, y_tr_t)

            preds_stack_t = meta_modele.predict(X_meta_test)

            preds_eur = transfo["inverse"](
                preds_stack_t
                if isinstance(preds_stack_t, np.ndarray)
                else np.array(preds_stack_t)
            )

            res = evaluer_complet(label, y_test_vals, preds_eur)
            resultats.append(res)
            predictions_test[label] = preds_eur

            print( f"  MAPE: {res['MAPE']:.2%} | R²: {res['R²']:.4f} | "
                f"MAE: {res['MAE']:>10,.0f}€ | RMSE_log: {res['RMSE_log']:.4f}" )

        except Exception as e:
            print(f"  Erreur lors de l'évaluation de {label} : {e}")

    if not resultats:
        raise ValueError(
            "Aucune transformation n'a pu être évaluée avec succès."
        )

    df_resultats = pd.DataFrame(resultats).sort_values("MAE")
    return df_resultats, predictions_test


def afficher_synthese_transformations(resultats):
    """Affiche le classement sous forme de tableau formaté et génère 3 barplots

    comparatifs (MAPE, R², RMSE_log) selon les transformations de la cible.
    """
    if isinstance(resultats, list):
        df_res = pd.DataFrame(resultats)
    else:
        df_res = resultats.copy()

    if "Transformation" in df_res.columns:
        df_res = df_res.set_index("Transformation")

    df_res_sorted = df_res.sort_values("MAPE")

    # Affichage du tableau
    print("Classement par MAPE décroissant")
    df_display = df_res_sorted.copy()
    df_display["MAE"] = df_display["MAE"].apply(lambda x: f"{x:>12,.0f} €")
    df_display["RMSE"] = df_display["RMSE"].apply(lambda x: f"{x:>12,.0f} €")
    df_display["MAPE"] = df_display["MAPE"].apply(lambda x: f"{x:.2%}")
    df_display["R²"] = df_display["R²"].apply(lambda x: f"{x:.4f}")
    df_display["RMSE_log"] = df_display["RMSE_log"].apply(lambda x: f"{x:.4f}")
    print(df_display.to_string())

    # Graphiques
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = [
        "indianred" if "baseline" in str(label).lower() else "steelblue"
        for label in df_res_sorted.index
    ]

    # Barplot MAPE
    df_res_sorted["MAPE"].plot( kind="bar", ax=axes[0], color=colors, edgecolor="none" )
    axes[0].set_title("MAPE — Test (↓ meilleur)")
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    axes[0].tick_params(axis="x", rotation=35)

    # Barplot R²
    df_res_sorted["R²"].plot( kind="bar", ax=axes[1], color=colors, edgecolor="none" )
    axes[1].set_title("R² — Test (↑ meilleur)")
    axes[1].tick_params(axis="x", rotation=35)

    # Barplot RMSE_log
    df_res_sorted["RMSE_log"].plot( kind="bar", ax=axes[2], color=colors, edgecolor="none" )
    axes[2].set_title("RMSE log — Test (↓ meilleur)")
    axes[2].tick_params(axis="x", rotation=35)

    plt.suptitle( "Impact de la transformation de la cible sur les performances", fontsize=13, )
    plt.tight_layout()
    plt.show()

    return df_res_sorted


def analyser_mape_par_tranches(predictions_test, y_test, tranches=None, labels_tranches=None,
                               figsize_per_plot=(12, 3.5), ):
    """Calcule et affiche le MAPE par tranche de montant pour différentes prédictions,

    et génère un barplot pour chaque transformation.
    """
    if tranches is None:
        tranches = [0, 5_000_000, 20_000_000, 50_000_000, np.inf]
    if labels_tranches is None:
        labels_tranches = ["<5M€", "5–20M€", "20–50M€", ">50M€"]

    y_test_vals = y_test.values if hasattr(y_test, "values") else y_test
    n_preds = len(predictions_test)

    # Affichage
    header_tranches = " ".join([f"{lbl:>8}" for lbl in labels_tranches])
    print("MAPE par tranche selon la transformation")
    print(f"{'Transformation':30s} {header_tranches}")
    print()

    resultats_tranches = []

    # Graphiques
    fig, axes = plt.subplots( n_preds, 1, figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_preds) )
    if n_preds == 1:
        axes = [axes]

    for ax, (label, preds) in zip(axes, predictions_test.items()):
        df_err = pd.DataFrame({"y": y_test_vals, "pred": preds})
        df_err["err_pct"] = (df_err["y"] - df_err["pred"]).abs() / df_err["y"]
        df_err["tranche"] = pd.cut(
            df_err["y"], bins=tranches, labels=labels_tranches
        )

        mapes = df_err.groupby("tranche", observed=True)["err_pct"].mean()

        # Log console
        ligne_str = f"{label:30s} " + " ".join(
            [f"{mapes.get(lbl, np.nan):>8.1%}" for lbl in labels_tranches]
        )
        print(ligne_str)

        # Structure pour DataFrame de sortie
        res_dict = {"Transformation": label}
        res_dict.update(
            {lbl: mapes.get(lbl, np.nan) for lbl in labels_tranches}
        )
        resultats_tranches.append(res_dict)

        # Plot
        mapes.plot(kind="bar", ax=ax, color="steelblue", edgecolor="none")
        ax.set_title(f"MAPE par tranche — {label}")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}")
        )
        ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(resultats_tranches).set_index("Transformation")


def preparer_meilleure_transformation(df_res, transformations, y_train, y_val, y_test, metric="MAPE",
                                      ascending=True):
    """Identifie la meilleure transformation selon une métrique donnée et prépare

    les séries cibles (y_train, y_val, y_test) transformées.

    Parameters:
    -----------
    df_res : pd.DataFrame
        Tableau de résultats (indexé par le nom des transformations ou contenant
        la colonne 'Transformation').
    transformations : dict
        Dictionnaire des transformations définies.
    y_train, y_val, y_test : pd.Series
        Séries originales de la cible.
    metric : str, default='MAPE'
        Métrique servant au classement ('MAPE', 'MAE', 'R²', 'RMSE_log').
    ascending : bool, default=True
        True si une valeur plus faible est meilleure (MAPE, MAE), False sinon
        (R²).

    Returns:
    --------
    dict:
        Contient le nom de la meilleure transformation, les séries cibles
        transformées (y_tr_best, y_val_best, y_te_best) et l'objet de la
        transformation.
    """
    # Alignement du DataFrame si 'Transformation' est une colonne
    if "Transformation" in df_res.columns:
        df_res_idx = df_res.set_index("Transformation")
    else:
        df_res_idx = df_res

    # Identification de la meilleure transformation
    if ascending:
        meilleure_transfo = df_res_idx[metric].idxmin()
    else:
        meilleure_transfo = df_res_idx[metric].idxmax()

    # Affichage des métriques clés
    print(f"Sélection de la meilleure transformation ({metric})")
    print(f"Meilleure transformation : {meilleure_transfo}")
    print(f"MAPE : {df_res_idx.loc[meilleure_transfo, 'MAPE']:.2%}")
    print(f"R²   : {df_res_idx.loc[meilleure_transfo, 'R²']:.4f}")
    if "MAE" in df_res_idx.columns:
        print(f"MAE  : {df_res_idx.loc[meilleure_transfo, 'MAE']:>10,.0f} €")

    # Application de la transformation
    transfo_best = transformations[meilleure_transfo]
    y_tr_best = pd.Series( transfo_best["transform"](y_train), index=y_train.index )
    y_val_best = pd.Series( transfo_best["transform"](y_val), index=y_val.index )
    y_te_best = pd.Series(transfo_best["transform"](y_test), index=y_test.index)

    print( f"\nDistribution de la cible transformée Train ({meilleure_transfo}) :" )
    print(y_tr_best.describe())

    return {
        "nom": meilleure_transfo,
        "transfo": transfo_best,
        "y_tr_best": y_tr_best,
        "y_val_best": y_val_best,
        "y_te_best": y_te_best,
    }


def entrainer_et_evaluer_stacking_tuned(modeles_dict, transfo_dict, meilleure_transfo_label, data,
                                        sample_weights_train=None, cv=5, n_jobs=-1, evaluer_fn=None, ):
    """Génère les prédictions OOF, ré-entraîne les modèles de base tunés,

    ajuste le méta-modèle LinearRegression et évalue le pipeline complet.
    """
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    y_test_vals = y_test.values if hasattr(y_test, "values") else y_test
    sw = (
        sample_weights_train.values
        if hasattr(sample_weights_train, "values")
        else sample_weights_train
    )

    # Récupération de la transformation sélectionnée
    transfo = transfo_dict[meilleure_transfo_label]
    fn_inv = transfo.get("inverse") or transfo.get("inverse_transform")

    y_tr_best = transfo["transform"](y_train)
    y_val_best = transfo["transform"](y_val)

    # Génération des prédictions OOF
    print("Génération OOF")
    oof = {}
    fit_kwargs = {"sample_weight": sw} if sw is not None else {}

    for nom, m in modeles_dict.items():
        m_clone = clone(m)
        try:
            oof_t = cross_val_predict(m_clone, X_train, y_tr_best, cv=cv, n_jobs=n_jobs, fit_params=fit_kwargs)
        except TypeError:
            oof_t = cross_val_predict(m_clone, X_train, y_tr_best, cv=cv, n_jobs=n_jobs, params=fit_kwargs)

        oof[nom] = oof_t
        preds_eur_oof = fn_inv(oof_t)
        mape_oof = mean_absolute_percentage_error(y_train, preds_eur_oof)
        print(f"  {nom}: OOF MAPE = {mape_oof:.2%}")

    X_meta_tr = pd.DataFrame(oof)

    # Ré-entraînement complet des modèles de base
    modeles_fit = {}
    preds_val_dict = {}
    preds_test_dict = {}

    for nom, m in modeles_dict.items():
        m_fit = clone(m)

        # Gestion spécifique CatBoost (ou prévenance early_stopping)
        if "CatBoost" in m_fit.__class__.__name__:
            m_fit.fit(
                X_train,
                y_tr_best,
                eval_set=(X_val, y_val_best),
                sample_weight=sw,
                verbose=False,
            )
        else:
            m_fit.fit(X_train, y_tr_best, **fit_kwargs)

        modeles_fit[nom] = m_fit
        preds_val_dict[nom] = m_fit.predict(X_val)
        preds_test_dict[nom] = m_fit.predict(X_test)

    X_meta_val = pd.DataFrame(preds_val_dict)
    X_meta_te = pd.DataFrame(preds_test_dict)

    # Méta-modèle de régression linéaire
    meta = LinearRegression(positive=True)
    meta.fit(X_meta_tr, y_tr_best)

    poids = { nom: round(w, 3) for nom, w in zip(modeles_dict.keys(), meta.coef_) }
    print(f"Poids : {poids}")

    # Prédictions finales et évaluation
    preds_test_stack_t = meta.predict(X_meta_te)
    pred_test_tuned_eur = fn_inv(
        preds_test_stack_t
        if isinstance(preds_test_stack_t, np.ndarray)
        else np.array(preds_test_stack_t)
    )

    label_res = f"Tuné + Stack ({meilleure_transfo_label})"
    if evaluer_fn is not None:
        res_tuned = evaluer_fn(label_res, y_test_vals, pred_test_tuned_eur)
    else:
        res_tuned = {
            "Transformation": label_res,
            "MAPE": mean_absolute_percentage_error(
                y_test_vals, pred_test_tuned_eur
            ),
            "MAE": np.mean(np.abs(y_test_vals - pred_test_tuned_eur)),
        }

    print(f"\nRésultat final ({label_res})")
    print(f"  MAPE     : {res_tuned['MAPE']:.2%}")
    if "R²" in res_tuned:
        print(f"  R²       : {res_tuned['R²']:.4f}")
    print(f"  MAE      : {res_tuned['MAE']:>12,.0f} €")
    if "RMSE_log" in res_tuned:
        print(f"  RMSE_log : {res_tuned['RMSE_log']:.4f}")

    return {
        "resultat": res_tuned,
        "predictions_test_eur": pred_test_tuned_eur,
        "meta_model": meta,
        "modeles_base_fit": modeles_fit,
        "poids_meta": poids,
    }



def afficher_classement_final(resultats, reference=None, sort_by="MAPE", ascending=True):
    """Génère et affiche le classement final des modèles/transformations.

    Parameters:
    -----------
    resultats : list ou pd.DataFrame
        Liste de dictionnaires ou DataFrame contenant les métriques d'évaluation.
    reference : dict, optional
        Dictionnaire représentant une ligne de référence à ajouter (ex: v4_ref).
    sort_by : str, default='MAPE'
        Métrique sur laquelle trier le classement.
    ascending : bool, default=True
        Sens du tri (True pour MAPE/MAE/RMSE, False pour R²).

    Returns:
    --------
    pd.DataFrame:
        DataFrame bruts filtrés et triés (non formatés en chaîne) pour réutilisation.
    """
    # Normalisation des données en DataFrame
    if isinstance(resultats, list):
        data_list = list(resultats)
    elif isinstance(resultats, pd.DataFrame):
        data_list = resultats.to_dict(orient="records")
    else:
        raise TypeError("resultats doit être une liste ou un pd.DataFrame")

    # Ajout de la référence si fournie
    if reference is not None:
        data_list.append(reference)

    df_final = pd.DataFrame(data_list)

    # Passage en index de la colonne Transformation si présente
    if "Transformation" in df_final.columns:
        df_final = df_final.set_index("Transformation")

    # Tri du DataFrame
    df_sorted = df_final.sort_values(by=sort_by, ascending=ascending)

    # Formatage à l'affichage (sur une copie pour préserver les valeurs numériques brutes)
    df_display = df_sorted.copy()

    for col in df_display.columns:
        if col == "MAPE":
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.2%}" if pd.notnull(x) else "—"
            )
        elif col in ["MAE", "RMSE"]:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:>12,.0f} €" if pd.notnull(x) else "—"
            )
        elif col in ["R²", "RMSE_log"]:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.4f}" if pd.notnull(x) else "—"
            )

    cols_to_show = [
        c for c in ["MAPE", "R²", "MAE", "RMSE_log"] if c in df_display.columns
    ]

    print("Classement final")
    print(df_display[cols_to_show].to_string())

    return df_sorted


def entrainer_stack(xgb, lgbm, cat, X_tr, y_tr_log, X_val_, y_val_log_, X_te, y_val, w=None):
    """Entraîne XGB+LGBM+CAT + stacking OOF. Renvoie preds_val, preds_test."""

    modeles = {"XGBoost": xgb, "LightGBM": lgbm, "CatBoost": cat}
    kw = {"sample_weight": w} if w is not None else {}

    # OOF
    oof = {}
    for nom, m in modeles.items():
        oof[nom] = cross_val_predict(m, X_tr, y_tr_log, cv=5, n_jobs=-1)

    # Ré-entraînement
    for nom, m in modeles.items():
        if isinstance(m, CatBoostRegressor):
            m.fit(X_tr, y_tr_log, eval_set=(X_val_, y_val_log_), verbose=False, **kw)
        else:
            m.fit(X_tr, y_tr_log, **kw)

    X_meta_tr  = pd.DataFrame(oof)
    X_meta_val = pd.DataFrame({n: m.predict(X_val_) for n,m in modeles.items()})
    X_meta_te  = pd.DataFrame({n: m.predict(X_te)  for n,m in modeles.items()})

    meta = LinearRegression(positive=True)
    meta.fit(X_meta_tr, y_tr_log)

    return (np.expm1(meta.predict(X_meta_val)),
            np.expm1(meta.predict(X_meta_te)),
            modeles)


def preprocess_and_evaluate_pca( X_train, X_val, X_test, n_top_bars=30, random_state=42 ):
    """Impute les NaN à la médiane, standardise les données, applique une ACP

    et affiche les graphiques de variance expliquée.
    """
    # Imputation
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_imp = pd.DataFrame( imputer.transform(X_val), columns=X_val.columns, index=X_val.index )
    X_test_imp = pd.DataFrame( imputer.transform(X_test), columns=X_test.columns, index=X_test.index )

    print(
        f"NaN restants — train: {X_train_imp.isnull().sum().sum()} | "
        f"val: {X_val_imp.isnull().sum().sum()} | "
        f"test: {X_test_imp.isnull().sum().sum()}"
    )

    # Standardisation
    scaler_acp = StandardScaler()
    X_train_scaled = scaler_acp.fit_transform(X_train_imp)
    X_val_scaled = scaler_acp.transform(X_val_imp)
    X_test_scaled = scaler_acp.transform(X_test_imp)

    # ACP
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_train_scaled)

    variance_cumul = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = np.argmax(variance_cumul >= 0.95) + 1
    n_99 = np.argmax(variance_cumul >= 0.99) + 1
    n_999 = np.argmax(variance_cumul >= 0.999) + 1

    print(f"Dimensions originales : {X_train.shape[1]} features")
    print(f"Composantes pour 95%  de variance : {n_95}")
    print(f"Composantes pour 99%  de variance : {n_99}")
    print(f"Composantes pour 99.9% de variance : {n_999}")

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Variance cumulée
    axes[0].plot(variance_cumul, lw=2, color="steelblue")
    thresholds = [
        (0.95, n_95, "red"),
        (0.99, n_99, "orange"),
        (0.999, n_999, "green"),
    ]
    for thresh, n, color in thresholds:
        axes[0].axhline(
            thresh,
            linestyle="--",
            color=color,
            alpha=0.7,
            label=f"{thresh:.1%} → {n} composantes",
        )
        axes[0].axvline(n - 1, linestyle="--", color=color, alpha=0.5)

    axes[0].set_xlabel("Nombre de composantes")
    axes[0].set_ylabel("Variance expliquée cumulée")
    axes[0].set_title("Variance expliquée — ACP")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Variance par composante
    top_n = min(n_top_bars, len(pca_full.explained_variance_ratio_))
    axes[1].bar(
        range(top_n),
        pca_full.explained_variance_ratio_[:top_n] * 100,
        color="steelblue",
        edgecolor="none",
    )
    axes[1].set_xlabel("Composante")
    axes[1].set_ylabel("% variance expliquée")
    axes[1].set_title(f"Variance par composante (top {top_n})")

    plt.tight_layout()
    plt.show()

    # Dictionnaire de retour des objets transformés
    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "imputer": imputer,
        "scaler": scaler_acp,
        "pca": pca_full,
        "n_components": {"95%": n_95, "99%": n_99, "99.9%": n_999},
    }


def evaluer_acp_seuils(X_train_scaled, X_val_scaled, X_test_scaled, y_train_log, y_val_log,
                       y_val, y_test, sample_weights_train, modeles_base, res_pca,
                       seuils=[0.95, 0.99, 0.999], random_state=42):
    """Teste plusieurs seuils de variance expliquée pour l'ACP, entraîne la fonction stack

    et retourne les résultats d'évaluation.
    """
    xgb, lgbm, cat = modeles_base
    n_components_dict = res_pca["n_components"]

    # Association dynamique des seuils avec leur nombre de composantes
    map_seuils = {0.95: "95%", 0.99: "99%", 0.999: "99.9%"}

    resultats = []

    # Conversion préalable des cibles et poids en arrays pour éviter les problèmes d'index
    y_train_log_arr = ( y_train_log.values if hasattr(y_train_log, "values") else y_train_log )
    y_val_log_series = (
        pd.Series(y_val_log.values)
        if hasattr(y_val_log, "values")
        else pd.Series(y_val_log)
    )
    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test
    weights_arr = (
        sample_weights_train.values
        if hasattr(sample_weights_train, "values")
        else sample_weights_train
    )

    for seuil in seuils:
        key = map_seuils.get(seuil, f"{seuil:.1%}")
        n_comp = n_components_dict[key]
        label_acp = f"ACP {seuil:.1%}"

        print(f"\n{label_acp} ({n_comp} composantes)")

        # Application de l'ACP
        pca = PCA(n_components=n_comp, random_state=random_state)
        Xtr_pca = pca.fit_transform(X_train_scaled)
        Xval_pca = pca.transform(X_val_scaled)
        Xte_pca = pca.transform(X_test_scaled)

        # Conversion en DataFrames
        cols = [f"PC{i+1}" for i in range(n_comp)]
        Xtr_pca = pd.DataFrame(Xtr_pca, columns=cols)
        Xval_pca = pd.DataFrame(Xval_pca, columns=cols)
        Xte_pca = pd.DataFrame(Xte_pca, columns=cols)

        # Entraînement et prédictions
        pred_val_acp, pred_te_acp, _ = entrainer_stack(xgb, lgbm, cat, Xtr_pca, y_train_log_arr,
                                                       Xval_pca, y_val_log_series, Xte_pca, y_val,
                                                       w=weights_arr)
        # Évaluation
        res = evaluer_complet(label_acp, y_test_arr, pred_te_acp)
        print(
            f"  MAPE : {res['MAPE']:.2%} | R² : {res['R²']:.4f} | RMSE_log : {res['RMSE_log']:.4f}"
        )

        resultats.append(res)

    return resultats



def calculer_importances_xgboost(selector_model, X_train, y_train_log, sample_weights_train=None,
                                 top_n_plot=30, random_state=42, ):
    """Entraîne un modèle XGBoost rapide pour extraire et visualiser l'importance des features (gain).

    Returns:
        pd.Series: Les importances triées par ordre décroissant.
    """
    # Entraînement du modèle XGBoost rapide
    
    selector_model.fit( X_train, y_train_log, sample_weight=sample_weights_train )

    # Extraction et tri des importances
    importances = pd.Series( selector_model.feature_importances_, index=X_train.columns )
    importances = importances.sort_values(ascending=False)

    # Affichage des statistiques
    print(f"Features total : {len(importances)}")
    print(f"Features avec importance > 0     : {(importances > 0).sum()}")
    print( f"Features avec importance > 0.01 (1%) : {(importances > 0.01).sum()}" )
    print( f"Features avec importance > 0.005 (0.5%) : {(importances > 0.005).sum()}" )
    print("\nTop 20 features :")
    print(importances.head(20).to_string())

    # Graphique du top N
    fig, ax = plt.subplots(figsize=(8, 8))
    top_n = min(top_n_plot, len(importances))
    importances.head(top_n).sort_values().plot( kind="barh", ax=ax, color="teal", edgecolor="none" )
    ax.set_title(f"Top {top_n} features — importance XGBoost (gain)")
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    plt.show()

    return importances


def calculer_permutation_importance(model, X_val, y_val_log, feature_names, importances_gain=None, n_repeats=10,
                                    scoring="neg_mean_squared_error", top_n_plot=30, random_state=42):
    """Calcule la permutation importance sur le jeu de validation

    et affiche une comparaison avec la gain importance (si fournie).

    Returns:
        tuple: (perm_imp, perm_std) contenant les Series Pandas des importances
        et écarts-types.
    """
    print("Calcul permutation importance (peut prendre 2-3 minutes)...")

    perm_result = permutation_importance(model, X_val, y_val_log, n_repeats=n_repeats,
                                         random_state=random_state, scoring=scoring, n_jobs=-1, )

    # Structuration des résultats
    perm_imp = pd.Series(perm_result.importances_mean, index=feature_names)
    perm_std = pd.Series(perm_result.importances_std, index=feature_names)
    perm_imp = perm_imp.sort_values(ascending=False)

    print(f"\nFeatures avec permutation importance > 0   : {(perm_imp > 0).sum()}")
    print( f"Features avec permutation importance < 0   : {(perm_imp < 0).sum()}" )
    print(f"\nTop {min(top_n_plot, len(perm_imp))} permutation importance :")
    print(perm_imp.head(top_n_plot).to_string())

    top_n = min(top_n_plot, len(perm_imp))

    # Graphique unique si l'importance par gain n'est pas transmise
    if importances_gain is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        perm_imp.head(top_n).sort_values().plot( kind="barh", ax=ax, color="steelblue", edgecolor="none" )
        ax.set_title( f"Top {top_n} — Permutation Importance (Validation)", fontsize=12 )
        ax.set_xlabel("Importance moyenne (dégradation score)")
    else:
        # Comparaison côte-à-côte
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        perm_imp.head(top_n).sort_values().plot( kind="barh", ax=axes[0], color="steelblue", edgecolor="none" )
        axes[0].set_title(f"Top {top_n} — Permutation Importance (non biaisée)")
        axes[0].set_xlabel("Importance moyenne (dégradation MSE log)")

        importances_gain.head(top_n).sort_values().plot( kind="barh", ax=axes[1], color="teal", edgecolor="none" )
        axes[1].set_title( f"Top {top_n} — Importance par gain (XGBoost, biaisée)" )
        axes[1].set_xlabel("Importance (gain)")

        plt.suptitle( "Comparaison : Permutation vs Gain Importance", fontsize=13, y=1.01 )

    plt.tight_layout()
    plt.show()

    return perm_imp, perm_std


def comparer_permutation_vs_gain_importance(selector_model, X_val, y_val_log, importances_gain,
                                            top_n=30, n_repeats=10, random_state=None,
                                            scoring="neg_mean_squared_error", n_jobs=-1, figsize=(16, 8), ):
    print("Calcul permutation importance (peut prendre 2-3 minutes)...")

    perm_result = permutation_importance(selector_model, X_val, y_val_log, n_repeats=n_repeats,
                                         random_state=random_state, scoring=scoring, n_jobs=n_jobs, )

    perm_imp = pd.Series(perm_result.importances_mean, index=X_val.columns)
    perm_std = pd.Series(perm_result.importances_std, index=X_val.columns)
    perm_imp = perm_imp.sort_values(ascending=False)

    print(f"\nFeatures avec permutation importance > 0   : {(perm_imp > 0).sum()}")
    print(f"Features avec permutation importance < 0   : {(perm_imp < 0).sum()} (mélanger améliore → bruit)")
    print(f"\nTop {top_n} permutation importance :")
    print(perm_imp.head(top_n).to_string())

    # Comparaison gain vs permutation importance
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    perm_imp.head(top_n).sort_values().plot( kind="barh", ax=axes[0], color="steelblue", edgecolor="none" )
    axes[0].set_title(f"Top {top_n} — Permutation Importance (non biaisée)")
    axes[0].set_xlabel("Importance moyenne (dégradation MSE log)")

    importances_gain.head(top_n).sort_values().plot( kind="barh", ax=axes[1], color="teal", edgecolor="none" )
    axes[1].set_title(f"Top {top_n} — Importance par gain (biaisée)")
    axes[1].set_xlabel("Importance (gain)")

    plt.suptitle("Comparaison : Permutation vs Gain Importance", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

    return perm_imp, perm_std


def evaluer_selection_permutation(perm_imp, X_train, y_train_log, X_val, y_val_log, X_test, y_val,
                                  y_test, sample_weights_train, modeles_base,
                                  seuils_perm=[(0, "Perm > 0  (toutes utiles)"),
                                               (0.001, "Perm > 0.001"),
                                               (0.005, "Perm > 0.005"), ],
                                  min_features=5, ):
    """Filtre les features selon différents seuils de permutation importance,

    réentraîne la stack sur ces sous-ensembles et évalue les performances.

    Returns:
        tuple: (resultats, dictionnaire_features_selectionnees)
    """
    xgb, lgbm, cat = modeles_base
    resultats = []
    features_selectionnees = {}

    # Conversion sécurisée en numpy arrays / pandas objects si besoin
    weights_arr = (
        sample_weights_train.values
        if hasattr(sample_weights_train, "values")
        else sample_weights_train
    )

    for seuil_perm, label_perm in seuils_perm:
        # Filtrage des variables dont l'importance dépasse le seuil
        features_perm = perm_imp[perm_imp > seuil_perm].index.tolist()

        # Vérification du nombre minimal de variables retenues
        if len(features_perm) < min_features:
            print( f"Trop peu de features ({len(features_perm)}) pour le seuil {seuil_perm}, étape ignorée." )
            continue

        print(f"\n{label_perm} ({len(features_perm)} features)")
        features_selectionnees[label_perm] = features_perm

        # Entraînement de la stack sur les variables filtrées
        pred_val_perm, pred_te_perm, _ = entrainer_stack(xgb, lgbm, cat, X_train[features_perm],
                                                         y_train_log, X_val[features_perm],
                                                         y_val_log, X_test[features_perm], y_val,
                                                         w=weights_arr)

        # Évaluation
        res = evaluer_complet(label_perm, y_test, pred_te_perm)
        print( f"  MAPE : {res['MAPE']:.2%} | R² : {res['R²']:.4f} | RMSE_log : {res['RMSE_log']:.4f}" )

        resultats.append(res)

    return resultats, features_selectionnees


def afficher_comparaison_finale(resultats_comparaison, label_baseline="Sans"):
    """Formate et affiche le tableau récapitulatif des performances

    ainsi que les graphiques comparatifs (MAPE, R², RMSE log).

    Gère automatiquement la clé 'Transformation' ou 'Label'.
    """
    # Création du DataFrame brut
    df_raw = pd.DataFrame(resultats_comparaison)

    # Solution 2 : Renommage automatique de 'Transformation' vers 'Label'
    if "Transformation" in df_raw.columns and "Label" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"Transformation": "Label"})

    if "Label" not in df_raw.columns:
        raise KeyError(
            "Aucune colonne 'Label' ou 'Transformation' trouvée dans resultats_comparaison."
        )

    # Mise en forme du tableau d'affichage
    df_display = df_raw.copy().set_index("Label")

    for col, fmt in [
        ("MAE", "{:>=12,.0f} €"),
        ("RMSE", "{:>=12,.0f} €"),
        ("MAPE", "{:.2%}"),
        ("R²", "{:.4f}"),
        ("RMSE_log", "{:.4f}"),
    ]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: fmt.format(x) if isinstance(x, (int, float)) 
                                                    else str(x) )

    print("Comparaison finale sur le test")
    print(df_display.to_string())

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Recherche de la ligne baseline
    df_base_rows = df_raw[ df_raw["Label"].str.contains(label_baseline, case=False, na=False) ]
    has_baseline = not df_base_rows.empty

    colors = [
        "indianred" if label_baseline.lower() in str(l).lower() else "steelblue"
        for l in df_raw["Label"]
    ]

    # Graphique MAPE
    df_raw.set_index("Label")["MAPE"].plot( kind="bar", ax=axes[0], color=colors, edgecolor="none" )
    axes[0].set_title("MAPE (↓ meilleur)")
    axes[0].yaxis.set_major_formatter( plt.FuncFormatter(lambda x, _: f"{x:.0%}") )
    axes[0].tick_params(axis="x", rotation=30)
    if has_baseline:
        axes[0].axhline(
            df_base_rows["MAPE"].values[0],
            color="red",
            lw=1.5,
            linestyle="--",
            label="baseline",
        )

    # Graphique R²
    df_raw.set_index("Label")["R²"].plot( kind="bar", ax=axes[1], color=colors, edgecolor="none" )
    axes[1].set_title("R² (↑ meilleur)")
    axes[1].tick_params(axis="x", rotation=30)
    if has_baseline:
        axes[1].axhline( df_base_rows["R²"].values[0], color="red", lw=1.5, linestyle="--" )

    # Graphique RMSE_log
    df_raw.set_index("Label")["RMSE_log"].plot(
        kind="bar", ax=axes[2], color=colors, edgecolor="none"
    )
    axes[2].set_title("RMSE log (↓ meilleur)")
    axes[2].tick_params(axis="x", rotation=30)
    if has_baseline:
        axes[2].axhline( df_base_rows["RMSE_log"].values[0], color="red", lw=1.5, linestyle="--", )

    plt.suptitle( "Réduction de dimensionnalité — Impact sur les performances", fontsize=13, )
    plt.tight_layout()
    plt.show()

    return df_display


def separer_gardiens_et_champ(X_train, y_train, X_val, y_val, X_test, y_test, col_gk="pos_GK",
                              cols_a_supprimer=["pos_DF", "pos_FW", "pos_GK", "pos_MF"], ):
    """Sépare les jeux de données (train/val/test) entre gardiens (GK) et joueurs de champ,

    puis retire les colonnes spécifiées (ex: indicateurs de poste).

    Returns:
        dict: Contient tous les sous-ensembles X et y séparés.
    """
    # Masques boléens
    mask_train_gk = X_train[col_gk] == 1
    mask_val_gk = X_val[col_gk] == 1
    mask_test_gk = X_test[col_gk] == 1

    # Séparation Gardiens
    X_train_gk, y_train_gk = X_train[mask_train_gk], y_train[mask_train_gk]
    X_val_gk, y_val_gk = X_val[mask_val_gk], y_val[mask_val_gk]
    X_test_gk, y_test_gk = X_test[mask_test_gk], y_test[mask_test_gk]

    # Séparation Joueurs de champ
    X_train_champ, y_train_champ = ( X_train[~mask_train_gk], y_train[~mask_train_gk], )
    X_val_champ, y_val_champ = X_val[~mask_val_gk], y_val[~mask_val_gk]
    X_test_champ, y_test_champ = X_test[~mask_test_gk], y_test[~mask_test_gk]

    # Affichage des effectifs
    print( f"Gardiens         : {len(X_train_gk):>5} train | {len(X_val_gk):>4} val | {len(X_test_gk):>4} test" )
    print( f"Joueurs de champ : {len(X_train_champ):>5} train | {len(X_val_champ):>4} val | {len(X_test_champ):>4} test" )

    # Suppression des colonnes de position
    cols_existant_train = [
        c for c in cols_a_supprimer if c in X_train.columns
    ]

    X_train_gk_feat = X_train_gk.drop(columns=cols_existant_train)
    X_val_gk_feat = X_val_gk.drop(columns=cols_existant_train)
    X_test_gk_feat = X_test_gk.drop(columns=cols_existant_train)

    X_train_champ_feat = X_train_champ.drop(columns=cols_existant_train)
    X_val_champ_feat = X_val_champ.drop(columns=cols_existant_train)
    X_test_champ_feat = X_test_champ.drop(columns=cols_existant_train)

    # Dictionnaire de retour
    return {
        "gk": { "X_train": X_train_gk_feat, "y_train": y_train_gk, "X_val": X_val_gk_feat,
               "y_val": y_val_gk, "X_test": X_test_gk_feat, "y_test": y_test_gk, },
        "champ": { "X_train": X_train_champ_feat, "y_train": y_train_champ, "X_val": X_val_champ_feat,
                  "y_val": y_val_champ, "X_test": X_test_champ_feat, "y_test": y_test_champ, },
    }


def espace_forest(trial):
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth":         trial.suggest_int("max_depth", 5, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]),
    }

def espace_xgb(trial):
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 3000, step=200),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
    }

def espace_lgbm(trial):
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 3000, step=200),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 255),
        "max_depth":        trial.suggest_int("max_depth", 3, 15),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
    }

def tuner_modele(fixed_params, model_class, espace_fn, X_tr, y_tr, X_ev, y_ev, n_trials, random_state):
    fixed = fixed_params[model_class]
    def objective(trial):
        params = {**espace_fn(trial), **fixed}
        model = model_class(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_ev)
        return np.sqrt(mean_squared_error(y_ev, preds))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def construire_et_entrainer_modeles_groupes( groupes, tuning_par_groupe, modeles_a_tuner, fixed_params ):
    """Instancie les modèles avec les meilleurs hyperparamètres issus d'Optuna

    et les entraîne sur leurs groupes respectifs.

    Returns:
        tuple: (modeles_finaux_par_groupe, modeles_groupe_entraines)
    """
    modeles_finaux_par_groupe = {}
    modeles_groupe_entraines = {}

    for nom_groupe, (X_tr, y_tr, *_) in groupes.items():
        modeles_finaux_par_groupe[nom_groupe] = {}
        modeles_groupe_entraines[nom_groupe] = {}

        print(f"\nEntraînement groupe : {nom_groupe}")

        for nom_modele, (model_class, _) in modeles_a_tuner.items():
            # Récupération de l'étude Optuna et instanciation
            study = tuning_par_groupe[nom_groupe][nom_modele]
            params_fixes = fixed_params.get(model_class, {})

            modele = model_class(**study.best_params, **params_fixes)
            modeles_finaux_par_groupe[nom_groupe][nom_modele] = modele

            # Entraînement du modèle
            modele.fit(X_tr, y_tr)
            modeles_groupe_entraines[nom_groupe][nom_modele] = modele

            print(f"  {nom_groupe} / {nom_modele} : entraîné")

    return modeles_finaux_par_groupe, modeles_groupe_entraines


def comparer_modeles_global_vs_specifique(X_global, modeles_globaux, modeles_specifiques, modeles_a_tuner,
                                          dict_groupes, split_name="Validation", ):
    comparaison_groupes = []

    for nom_groupe, (X_group_feat, y_ev) in dict_groupes.items():
        # Extraction dynamique de X_global pour ce sous-groupe (conserve toutes les colonnes d'origine)
        X_ev_global = X_global.loc[X_group_feat.index]

        for nom_modele in modeles_a_tuner:
            # Prédictions du modèle global
            modele_global = modeles_globaux[nom_modele]
            preds_global = modele_global.predict(X_ev_global)
            mae_global = mean_absolute_error(y_ev, preds_global)
            r2_global = r2_score(y_ev, preds_global)

            # Prédictions du modèle spécifique au groupe
            modele_groupe = modeles_specifiques[nom_groupe][nom_modele]
            preds_groupe = modele_groupe.predict(X_group_feat)
            mae_groupe = mean_absolute_error(y_ev, preds_groupe)
            r2_groupe = r2_score(y_ev, preds_groupe)

            # Stockage des résultats
            comparaison_groupes.append(
                { "Groupe": nom_groupe, "Modèle": nom_modele, "MAE global": mae_global,
                 "R² global": r2_global, "MAE spécifique": mae_groupe, "R² spécifique": r2_groupe,
                 "Gain MAE": mae_global - mae_groupe, }
            )

    df_comparaison = pd.DataFrame(comparaison_groupes)

    print(
        f"Classement ({split_name}) : modèle global vs modèle spécifique"
    )
    print(df_comparaison.to_string(index=False))

    return df_comparaison


def get_feature_names(modele):
    """Récupère la liste des features attendues par un modèle, quel que soit son type."""
    if hasattr(modele, "get_booster"):
        try:
            fn = modele.get_booster().feature_names
            if fn: return fn
        except Exception:
            pass
    if hasattr(modele, "booster_"):
        try:
            return list(modele.booster_.feature_name())
        except Exception:
            pass
    if hasattr(modele, "feature_names_"):
        return list(modele.feature_names_)
    if hasattr(modele, "feature_names_in_"):
        return list(modele.feature_names_in_)
    return None


def aligner_X(X, feature_names, nom_modele=""):
    """Réaligne les colonnes de X sur celles attendues par le modèle :

    colonnes manquantes -> ajoutées à 0, colonnes en trop -> supprimées.
    """
    if feature_names is None:
        return X
    manquantes = [c for c in feature_names if c not in X.columns]
    en_trop    = [c for c in X.columns if c not in feature_names]
    if manquantes:
        print(f"    {nom_modele} : {len(manquantes)} colonne(s) manquante(s) "
              f"réinjectée(s) à 0 : {manquantes[:5]}{'...' if len(manquantes) > 5 else ''}")
        X = X.copy()
        for c in manquantes:
            X[c] = 0
    if en_trop:
        X = X.drop(columns=en_trop)
    return X[feature_names]


def predire_aligne(modele, X, nom_modele=""):
    """Prédit avec `modele` après avoir réaligné X sur les features qu'il attend."""
    fn = get_feature_names(modele)
    return modele.predict(aligner_X(X, fn, nom_modele))


def construire_tableau_comparatif_final(X_test_final, y_test_final, X_test_sans_nan_final, modeles_tuned,
                                        modeles_tuned_sans_nan, modeles_tuned_log, modeles_tuned_log_sans_nan,
                                        base_models_log, meta_modele, meilleure_transfo_label,
                                        pred_test_tuned_eur, modeles_finaux, base_models_b5, meta_modele_b5):
    taille_ref = len(y_test_final)
    print(f"Référence : {taille_ref} lignes (test set Partie A, figé)")

    results = []

    def verif(name, arr):
        arr = np.array(arr, dtype=float)
        if len(arr) != taille_ref:
            print(f"    {name} ignoré : taille {len(arr)} ≠ {taille_ref}")
            return None
        return np.clip(arr, 0, None)

    def calcul_metriques(name, y_pred):
        y_pred = verif(name, y_pred)
        if y_pred is None:
            return
        mae      = mean_absolute_error(y_test_final, y_pred)
        rmse     = np.sqrt(mean_squared_error(y_test_final, y_pred))
        mape     = mean_absolute_percentage_error(y_test_final, y_pred)
        r2       = r2_score(y_test_final, y_pred)
        rmse_log = np.sqrt(mean_squared_error(np.log1p(y_test_final), np.log1p(y_pred)))
        n, p     = X_test_final.shape
        r2_adj   = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        results.append({"Modèle": name, "MAPE": mape, "RMSE_log": rmse_log,
                         "R²": r2, "R²_adj": r2_adj, "MAE": mae, "RMSE": rmse, "preds": y_pred})

    # Partie B3
    preds_test_base_final = {
        nom: np.expm1(predire_aligne(modeles_tuned_log[nom], X_test_final, nom))
        for nom in base_models_log
    }
    X_meta_test_b3 = pd.DataFrame(preds_test_base_final)

    pred_test_stack_final   = meta_modele.predict(X_meta_test_b3)
    pred_test_moyenne_final = X_meta_test_b3.mean(axis=1)

    calcul_metriques("Blend Moyenne (B3 — Stacking initial)", pred_test_moyenne_final)
    calcul_metriques("Stacking OOF (B3 — Stacking initial)",  pred_test_stack_final)

    # Partie B4
    calcul_metriques(f"Stacking Tuné ({meilleure_transfo_label}) (B4 — Transfo)", pred_test_tuned_eur)

    # Parties B1 et B2
    modeles_bruts = {
        "Random Forest (B1)": (modeles_tuned["Random Forest"],       X_test_final,          False),
        "Extra Trees (B1)":   (modeles_tuned["Extra Trees"],         X_test_final,          False),
        "XGBoost (B1)":       (modeles_tuned["XGBoost"],             X_test_final,          False),
        "LightGBM (B1)":      (modeles_tuned["LightGBM"],            X_test_final,          False),
        "CatBoost (B1)":      (modeles_tuned["CatBoost"],            X_test_final,          False),
        "SVR (B1)":           (modeles_tuned_sans_nan["SVR"],        X_test_sans_nan_final, False),
        "ElasticNet (B1)":    (modeles_tuned_sans_nan["ElasticNet"], X_test_sans_nan_final, False),
    }
    modeles_log = {
        "Random Forest log (B2)": (modeles_tuned_log["Random Forest (log)"],       X_test_final,          True),
        "Extra Trees log (B2)":   (modeles_tuned_log["Extra Trees (log)"],         X_test_final,          True),
        "XGBoost log (B2)":       (modeles_tuned_log["XGBoost (log)"],             X_test_final,          True),
        "LightGBM log (B2)":      (modeles_tuned_log["LightGBM (log)"],            X_test_final,          True),
        "CatBoost log (B2)":      (modeles_tuned_log["CatBoost (log)"],            X_test_final,          True),
        "SVR log (B2)":           (modeles_tuned_log_sans_nan["SVR (log)"],        X_test_sans_nan_final, True),
        "ElasticNet log (B2)":    (modeles_tuned_log_sans_nan["ElasticNet (log)"], X_test_sans_nan_final, True),
    }
    for name, (modele, X_ev, is_log) in {**modeles_bruts, **modeles_log}.items():
        y_pred_raw = np.expm1(predire_aligne(modele, X_ev, name)) if is_log else predire_aligne(modele, X_ev, name)
        calcul_metriques(name, y_pred_raw)

    # Partie B5
    preds_b5_final = {
        nom: np.expm1(predire_aligne(modeles_finaux[nom], X_test_final, nom))
        for nom in base_models_b5
    }
    X_meta_test_final  = pd.DataFrame(preds_b5_final)
    pred_test_b5_final = meta_modele_b5.predict(X_meta_test_final)  # déjà en €, pas de expm1 ici
    calcul_metriques("Stacking OOF, train sans outliers (B5)", pred_test_b5_final)

    # Partie B6
    print("B6 (ACP / SFM / Permutation) exclu du tableau : transformeurs non sauvegardés, "
          "voir note ci-dessus pour les inclure proprement.")

    # Tableau comparatif final
    df_all         = pd.DataFrame(results).set_index("Modèle").sort_values("MAE")
    df_recap_final = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)

    df_disp = df_all.drop(columns=["preds"]).copy()
    df_disp["MAPE"]     = df_disp["MAPE"].apply(lambda x: f"{x:.2%}")
    df_disp["RMSE_log"] = df_disp["RMSE_log"].apply(lambda x: f"{x:.4f}")
    df_disp["R²"]       = df_disp["R²"].apply(lambda x: f"{x:.4f}")
    df_disp["R²_adj"]   = df_disp["R²_adj"].apply(lambda x: f"{x:.4f}")
    df_disp["MAE"]      = df_disp["MAE"].apply(lambda x: f"{x:>12,.0f} €")
    df_disp["RMSE"]     = df_disp["RMSE"].apply(lambda x: f"{x:>12,.0f} €")

    print("CLASSEMENT FINAL — Toutes approches (Test figé Partie A, trié par MAE croissant)")
    print(df_disp.to_string())

    return df_all, df_recap_final


def visualiser_comparaison_finale(df_all):
    def get_color(lbl):
        if "B1" in lbl: return "#2196F3"
        if "B2" in lbl: return "#9C27B0"
        if "B3" in lbl: return "#FF9800"
        if "B4" in lbl: return "#4CAF50"
        if "B5" in lbl: return "#F44336"
        if "B6" in lbl: return "#00BCD4"
        return "#9E9E9E"

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(df_all) * 0.4)))
    for ax, col, title, ascending in [
        (axes[0], "MAPE",     "MAPE ↓ (meilleur)",      True),
        (axes[1], "R²",       "R² ↑ (meilleur)",         False),
        (axes[2], "RMSE_log", "RMSE log ↓ (meilleur)",   True),
    ]:
        # Sélection uniquement des 5 premiers modèles pour la métrique courante
        df_top5 = df_all[col].sort_values(ascending=ascending).head(5)
        colors_top5 = [get_color(l) for l in df_top5.index]

        df_top5.plot(kind="barh", ax=ax, color=colors_top5, edgecolor="none")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.invert_yaxis()  # Le #1 se retrouve en haut du graphique
        ax.tick_params(axis="y", labelsize=9)

        if col == "MAPE":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    legend = [
        Patch(facecolor="#2196F3", label="B1 — Cible brute"),
        Patch(facecolor="#9C27B0", label="B2 — Cible log"),
        Patch(facecolor="#FF9800", label="B3 — Stacking OOF"),
        Patch(facecolor="#4CAF50", label="B4 — Transfo optimale"),
        Patch(facecolor="#F44336", label="B5 — Sans outliers"),
        Patch(facecolor="#00BCD4", label="B6 — Réd. dimensionnalité"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("Comparaison finale — Toutes approches (Test)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def selectionner_modele_champion(df_recap_final):
    champion_row   = df_recap_final.iloc[0]
    champion_lbl   = champion_row["Modèle"]
    preds_champion = champion_row["preds"]

    print(f"Modèle retenu : {champion_lbl}")
    print(f"  MAPE     : {champion_row['MAPE']:.2%}")
    print(f"  R²       : {champion_row['R²']:.4f}")
    print(f"  MAE      : {champion_row['MAE']:>12,.0f} €")
    print(f"  RMSE_log : {champion_row['RMSE_log']:.4f}")

    return champion_lbl, preds_champion, champion_row
