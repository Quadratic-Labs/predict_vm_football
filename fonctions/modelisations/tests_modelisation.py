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


def tune_xgb(model_class, trial, use_log=False, random_state=42,
             X_train=None, y_train=None, y_train_log=None, X_val=None, y_val=None):
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

    # Sélection de la cible (brute ou log)
    y_target = y_train_log if use_log else y_train
    model.fit(X_train, y_target)

    # Prédiction et passage à l'échelle d'origine si log
    preds = model.predict(X_val)
    if use_log:
        preds = np.expm1(preds)

    return np.sqrt(mean_squared_error(y_val, preds))


def tune_lgb(model_class, trial, use_log=False, random_state=42,
             X_train=None, y_train=None, y_train_log=None, X_val=None, y_val=None, ):
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

    # Sélection de la cible (brute ou log)
    y_target = y_train_log if use_log else y_train
    model.fit(X_train, y_target)

    # Prédiction et passage à l'échelle d'origine si log
    preds = model.predict(X_val)
    if use_log:
        preds = np.expm1(preds)

    return np.sqrt(mean_squared_error(y_val, preds))


def tune_catboost(model_class, trial, use_log=False, random_state=42,
                  X_train=None, y_train=None, y_train_log=None, X_val=None, y_val=None, y_val_log=None, ):
    # Plages de recherche adaptées selon l'utilisation du log
    if use_log:
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

    # Sélection de la cible et du jeu d'évaluation
    if use_log:
        y_target = y_train_log
        eval_set = (X_val, y_val_log)
    else:
        y_target = y_train
        eval_set = (X_val, y_val)

    model.fit(X_train, y_target, eval_set=eval_set, verbose=False)

    # Prédiction et conversion inverse si log
    preds = model.predict(X_val)
    if use_log:
        preds = np.expm1(preds)

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


def comparaison_blend(
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