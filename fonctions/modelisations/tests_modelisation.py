import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.base import clone



def preparer_donnees(df_train, df_val, df_test, colonne_cible):
    """Nettoie les valeurs manquantes cibles, sépare features/cible,

    calcule les cibles log & sample weights, et génère des versions
    sans NaN pour les modèles linéaires/SVR.
    """
    # 1. Suppression des lignes avec valeur manquante sur la cible
    df_train = df_train.dropna(subset=[colonne_cible])
    df_val = df_val.dropna(subset=[colonne_cible])
    df_test = df_test.dropna(subset=[colonne_cible])

    # 2. Suppression des colonnes normalisées (_nor)
    colonnes_nor = [c for c in df_train.columns if c.endswith("_nor")]
    df_train = df_train.drop(columns=colonnes_nor, errors="ignore")
    df_val = df_val.drop(columns=colonnes_nor, errors="ignore")
    df_test = df_test.drop(columns=colonnes_nor, errors="ignore")

    # 3. Définition et vérification des colonnes à exclure des features
    cols_a_supprimer_base = [
        colonne_cible,
        "player",
        "team",
        "nation",
        "position",
    ]
    cols_a_supprimer = [c for c in cols_a_supprimer_base if c in df_train.columns]

    # 4. Séparation des features (X) et de la cible (y)
    X_train = df_train.drop(columns=cols_a_supprimer)
    y_train = df_train[colonne_cible]

    X_val = df_val.drop(columns=cols_a_supprimer)
    y_val = df_val[colonne_cible]

    X_test = df_test.drop(columns=cols_a_supprimer)
    y_test = df_test[colonne_cible]

    # 5. Cibles log (log1p)
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)

    # 6. Poids d'entraînement (sur-pondération des petites valeurs)
    sample_weights_train = 1 / np.log1p(y_train)
    sample_weights_train = sample_weights_train / sample_weights_train.mean()

    # 7. Sélection des colonnes sans aucun NaN (basée sur le train set)
    colonnes_sans_nan = X_train.columns[X_train.isna().sum() == 0].tolist()

    print(
        f"X_train : {X_train.shape} | X_val : {X_val.shape} | X_test : {X_test.shape}"
    )
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

        print(
            f"   -> {label_eval} | MAE : {mae:,.0f} € | RMSE : {rmse:,.0f} € | MAPE : {mape:.2%} | R² : {r2:.2%} | R² Ajusté : {r2_ajuste:.2%}"
        )

    print(f"\nCLASSEMENT FINAL {label_eval} (trié par MAE croissante)")
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


def tune_forest(model_class, trial, use_log=False, random_state=42, X_train=None, y_train=None, y_train_log=None, X_val=None, y_val=None):
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


import numpy as np
from sklearn.metrics import mean_squared_error


def tune_xgb(
    model_class,
    trial,
    transfo=None,
    random_state=42,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
):
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


def tune_lgbm(
    model_class,
    trial,
    transfo=None,
    random_state=42,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
):
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


def tune_catboost(
    model_class,
    trial,
    transfo=None,
    random_state=42,
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
):
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

    # 1. Évaluation des modèles standards
    for nom, modele in modeles_tuned.items():
        print(f"Entraînement (tuné) de {nom}...")
        resultats[nom] = evaluer(
            modele, data["X_train"], y_train_target, X_eval, y_eval
        )

    # 2. Évaluation des modèles sans NaN (Pipelines)
    for nom, modele in modeles_tuned_sans_nan.items():
        print(f"Entraînement (tuné) de {nom}...")
        resultats[nom] = evaluer(
            modele, data["X_train_sans_nan"], y_train_target, X_eval_sn, y_eval
        )

    # Construction du tableau récapitulatif
    df_resultats = pd.DataFrame(resultats).T.sort_values("MAE")

    print(f"\nCLASSEMENT ({label}, modèles tunés)")
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

    print(
        f"Génération des OOF predictions (cross_val_predict, cv={cv})..."
    )

    for nom in base_models_names:
        debut_modele = time.time()
        modele = modeles_dict[nom]

        # 1. Génération des OOF sur Train
        oof_log = cross_val_predict(
            modele, X_train, y_train_log, cv=cv, n_jobs=n_jobs
        )
        oof_preds[nom] = np.expm1(oof_log)

        # 2. Re-fit du modèle sur TOUT le train + prédictions Test
        modele.fit(X_train, y_train_log)
        preds_test_base[nom] = np.expm1(modele.predict(X_test))

        # 3. Prédictions sur le jeu de Validation
        preds_val_base[nom] = np.expm1(modele.predict(X_val))

        duree = time.time() - debut_modele
        temps_entrainement[nom] = duree

        rmse_oof = np.sqrt(mean_squared_error(y_train, oof_preds[nom]))
        print(f"  {nom} : OOF RMSE = {rmse_oof:,.0f} € ({duree:.1f}s)")

    X_meta_train = pd.DataFrame(oof_preds)
    X_meta_val = pd.DataFrame(preds_val_base)
    X_meta_test = pd.DataFrame(preds_test_base)

    return X_meta_train, X_meta_val, X_meta_test, temps_entrainement


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def evaluer_ensembles(
    X_meta_train,
    X_meta_val,
    X_meta_test,
    data,
    modeles_tuned_log,
    nom_modele_seul="CatBoost (log)",
):
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
    print(
        f"  Val  -> MAE : {mean_absolute_error(y_val, pred_val_moyenne):,.0f} € | "
        f"R² : {r2_score(y_val, pred_val_moyenne):.4f}"
    )
    print(
        f"  Test -> MAE : {mean_absolute_error(y_test, pred_test_moyenne):,.0f} € | "
        f"R² : {r2_score(y_test, pred_test_moyenne):.4f}"
    )

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
    print(
        f"  Val  -> MAE : {mean_absolute_error(y_val, pred_val_stack):,.0f} € | "
        f"R² : {r2_score(y_val, pred_val_stack):.4f}"
    )
    print(
        f"  Test -> MAE : {mean_absolute_error(y_test, pred_test_stack):,.0f} € | "
        f"R² : {r2_score(y_test, pred_test_stack):.4f}\n"
    )

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

    return (
        pred_val_moyenne,
        pred_test_moyenne,
        pred_val_stack,
        pred_test_stack,
        meta_modele,
        comparaison_blend,
    )


def comparaison_des_blend(
    data,
    modeles_tuned_log,
    pred_val_moyenne,
    pred_test_moyenne,
    pred_val_stack,
    pred_test_stack,
    nom_modele_seul="CatBoost (log)",
):
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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyser_erreurs_par_tranches(
    y_test,
    meilleure_prediction_test,
    tranches=None,
    labels_tranches=None,
    random_state=42,
):
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

    # Figure 1 : Graphiques à barres (MAPE & % de l'erreur totale)
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


import matplotlib.pyplot as plt
import pandas as pd


def afficher_feature_importances(
    modele,
    feature_names,
    top_n=25,
    nom_modele="Modèle",
    figsize=(8, 8),
    color="teal",
):
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
        raise ValueError(
            f"Le modèle {type(modele).__name__} ne possède pas d'attribut d'importance des variables direct."
        )

    # Création et tri de la série
    importances = pd.Series(raw_importances, index=feature_names).sort_values(
        ascending=False
    )

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

    # Masquer le dernier axe inutilisé (7 transformations pour 8 emplacements)
    if len(transformations_viz) < len(axes.flatten()):
        fig.delaxes(axes.flatten()[-1])

    plt.suptitle("Distribution de la cible selon la transformation", fontsize=13)
    plt.tight_layout()
    plt.show()

    print("Statistiques de la cible brute :")
    print(y_series.describe())

    return transformations_viz


import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
)


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

    # 1. Fit des transformateurs data-dependent sur TRAIN uniquement
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

    # 2. Dictionnaire de fonctions de transformation / inversion
    transformations = {
        "Brute": {
            "transform": lambda y: y,
            "inverse": lambda y: np.clip(y, 0, None),
            "description": "Aucune transformation. MAE en euros bruts.",
        },
        "log1p (baseline)": {
            "transform": lambda y: np.log1p(y),
            "inverse": lambda y: np.expm1(y),
            "description": "log(1+y) — transformation de référence. Compresse l'échelle.",
        },
        "log10": {
            "transform": lambda y: np.log10(np.clip(y, 1, None)),
            "inverse": lambda y: np.power(10, y),
            "description": "log10(y) — espace plus lisible (6 = 1M€, 7 = 10M€).",
        },
        "sqrt": {
            "transform": lambda y: np.sqrt(y),
            "inverse": lambda y: np.square(np.clip(y, 0, None)),
            "description": "Racine carrée — compression plus douce que le log.",
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
            "description": "Normalisation dans [0,1]. Sensible aux outliers.",
        },
        "Rang centile [0,1]": {
            "transform": lambda y: quantile_unif.transform(_to_2d(y)).flatten(),
            "inverse": lambda y: np.clip(
                quantile_unif.inverse_transform(_to_2d(y)).flatten(), 0, None
            ),
            "description": "Transforme en rangs uniformes [0,1]. Robuste aux outliers.",
        },
    }

    print(f"{len(transformations)} transformations définies.")
    return transformations


import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_predict


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


def evaluer_transformations_stacking(
    transformations,
    modeles_dict,
    data,
    sample_weights_train=None,
    cv=5,
    n_jobs=-1,
):
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
        print(f"\n--- {label} (Stacking OOF) ---")
        print(f"  {transfo['description'][:80]}...")

        try:
            y_tr_t = transfo["transform"](y_train)

            oof_preds_dict = {}
            preds_test_dict = {}

            fit_kwargs = {"sample_weight": sw} if sw is not None else {}

            for nom_m, modele_original in modeles_dict.items():
                modele = clone(modele_original)

                # Rétrocompatibilité automatique entre fit_params et params
                try:
                    oof_t = cross_val_predict(
                        modele,
                        X_train,
                        y_tr_t,
                        cv=cv,
                        n_jobs=n_jobs,
                        fit_params=fit_kwargs,
                    )
                except TypeError:
                    oof_t = cross_val_predict(
                        modele,
                        X_train,
                        y_tr_t,
                        cv=cv,
                        n_jobs=n_jobs,
                        params=fit_kwargs,
                    )

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

            print(
                f"  → MAPE: {res['MAPE']:.2%} | R²: {res['R²']:.4f} | "
                f"MAE: {res['MAE']:>10,.0f}€ | RMSE_log: {res['RMSE_log']:.4f}"
            )

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

    # --- 1. AFFICHAGE DU TABLEAU FORMATÉ ---
    print("=== CLASSEMENT PAR MAPE (↓ meilleur) ===")
    df_display = df_res_sorted.copy()
    df_display["MAE"] = df_display["MAE"].apply(lambda x: f"{x:>12,.0f} €")
    df_display["RMSE"] = df_display["RMSE"].apply(lambda x: f"{x:>12,.0f} €")
    df_display["MAPE"] = df_display["MAPE"].apply(lambda x: f"{x:.2%}")
    df_display["R²"] = df_display["R²"].apply(lambda x: f"{x:.4f}")
    df_display["RMSE_log"] = df_display["RMSE_log"].apply(lambda x: f"{x:.4f}")
    print(df_display.to_string())

    # --- 2. GRAPHIQUES BARPLOT ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = [
        "indianred" if "baseline" in str(label).lower() else "steelblue"
        for label in df_res_sorted.index
    ]

    # Barplot MAPE
    df_res_sorted["MAPE"].plot(
        kind="bar", ax=axes[0], color=colors, edgecolor="none"
    )
    axes[0].set_title("MAPE — Test (↓ meilleur)")
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    axes[0].tick_params(axis="x", rotation=35)

    # Barplot R²
    df_res_sorted["R²"].plot(
        kind="bar", ax=axes[1], color=colors, edgecolor="none"
    )
    axes[1].set_title("R² — Test (↑ meilleur)")
    axes[1].tick_params(axis="x", rotation=35)

    # Barplot RMSE_log
    df_res_sorted["RMSE_log"].plot(
        kind="bar", ax=axes[2], color=colors, edgecolor="none"
    )
    axes[2].set_title("RMSE log — Test (↓ meilleur)")
    axes[2].tick_params(axis="x", rotation=35)

    plt.suptitle(
        "Impact de la transformation de la cible sur les performances",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()

    return df_res_sorted


def analyser_mape_par_tranches(
    predictions_test,
    y_test,
    tranches=None,
    labels_tranches=None,
    figsize_per_plot=(12, 3.5),
):
    """Calcule et affiche le MAPE par tranche de montant pour différentes prédictions,

    et génère un barplot pour chaque transformation.
    """
    if tranches is None:
        tranches = [0, 5_000_000, 20_000_000, 50_000_000, np.inf]
    if labels_tranches is None:
        labels_tranches = ["<5M€", "5–20M€", "20–50M€", ">50M€"]

    y_test_vals = y_test.values if hasattr(y_test, "values") else y_test
    n_preds = len(predictions_test)

    # --- 1. AFFICHAGE DU CONSOLE LOG ---
    header_tranches = " ".join([f"{lbl:>8}" for lbl in labels_tranches])
    print("=== MAPE PAR TRANCHE SELON LA TRANSFORMATION ===")
    print(f"{'Transformation':30s} {header_tranches}")
    print("-" * (31 + 9 * len(labels_tranches)))

    resultats_tranches = []

    # Graphiques
    fig, axes = plt.subplots(
        n_preds, 1, figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_preds)
    )
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


def preparer_meilleure_transformation(
    df_res,
    transformations,
    y_train,
    y_val,
    y_test,
    metric="MAPE",
    ascending=True,
):
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
    print(f"=== SÉLECTION DE LA MEILLEURE TRANSFORMATION ({metric}) ===")
    print(f"Meilleure transformation : {meilleure_transfo}")
    print(f"MAPE : {df_res_idx.loc[meilleure_transfo, 'MAPE']:.2%}")
    print(f"R²   : {df_res_idx.loc[meilleure_transfo, 'R²']:.4f}")
    if "MAE" in df_res_idx.columns:
        print(f"MAE  : {df_res_idx.loc[meilleure_transfo, 'MAE']:>10,.0f} €")

    # Application de la transformation
    transfo_best = transformations[meilleure_transfo]
    y_tr_best = pd.Series(
        transfo_best["transform"](y_train), index=y_train.index
    )
    y_val_best = pd.Series(
        transfo_best["transform"](y_val), index=y_val.index
    )
    y_te_best = pd.Series(transfo_best["transform"](y_test), index=y_test.index)

    print(
        f"\nDistribution de la cible transformée Train ({meilleure_transfo}) :"
    )
    print(y_tr_best.describe())
    print(f"Skewness : {y_tr_best.skew():.3f}\n")

    return {
        "nom": meilleure_transfo,
        "transfo": transfo_best,
        "y_tr_best": y_tr_best,
        "y_val_best": y_val_best,
        "y_te_best": y_te_best,
    }


import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_predict


def entrainer_et_evaluer_stacking_tuned(
    modeles_dict,
    transfo_dict,
    meilleure_transfo_label,
    data,
    sample_weights_train=None,
    cv=5,
    n_jobs=-1,
    evaluer_fn=None,
):
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

    # 1. Génération des prédictions OOF
    print("Génération OOF...")
    oof = {}
    fit_kwargs = {"sample_weight": sw} if sw is not None else {}

    for nom, m in modeles_dict.items():
        m_clone = clone(m)
        try:
            oof_t = cross_val_predict(
                m_clone,
                X_train,
                y_tr_best,
                cv=cv,
                n_jobs=n_jobs,
                fit_params=fit_kwargs,
            )
        except TypeError:
            oof_t = cross_val_predict(
                m_clone,
                X_train,
                y_tr_best,
                cv=cv,
                n_jobs=n_jobs,
                params=fit_kwargs,
            )

        oof[nom] = oof_t
        preds_eur_oof = fn_inv(oof_t)
        mape_oof = mean_absolute_percentage_error(y_train, preds_eur_oof)
        print(f"  {nom}: OOF MAPE = {mape_oof:.2%}")

    X_meta_tr = pd.DataFrame(oof)

    # 2. Ré-entraînement complet des modèles de base
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

    # 3. Méta-modèle LinearRegression (poids positifs)
    meta = LinearRegression(positive=True)
    meta.fit(X_meta_tr, y_tr_best)

    poids = {
        nom: round(w, 3) for nom, w in zip(modeles_dict.keys(), meta.coef_)
    }
    print(f"Poids méta : {poids}")

    # 4. Prédictions finales et évaluation
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

    print(f"\n=== RÉSULTAT FINAL ({label_res}) ===")
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


import pandas as pd


def afficher_classement_final(
    resultats, reference=None, sort_by="MAPE", ascending=True
):
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
    # 1. Normalisation des données en DataFrame
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

    # 2. Tri du DataFrame
    df_sorted = df_final.sort_values(by=sort_by, ascending=ascending)

    # 3. Formatage à l'affichage (sur une copie pour préserver les valeurs numériques brutes)
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

    print("=== CLASSEMENT FINAL ===")
    print(df_display[cols_to_show].to_string())

    return df_sorted