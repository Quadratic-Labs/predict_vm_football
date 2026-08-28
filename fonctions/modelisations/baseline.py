import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nettoyer_donnees_manquantes(df_train, df_val, variables_x, variable_y):
    """
    Filtre les DataFrames d'entraînement et de validation pour ne garder
    que les lignes sans valeurs manquantes sur X et y.
    """
    cols_a_verifier = variables_x + [variable_y]

    df_train_clean = df_train.dropna(subset=cols_a_verifier)
    df_val_clean = df_val.dropna(subset=cols_a_verifier)

    print(f"Lignes après suppression des NA -> Train: {len(df_train_clean)} (vs {len(df_train)}) | Val: {len(df_val_clean)} (vs {len(df_val)})\n")

    return df_train_clean, df_val_clean


def calculer_metriques(y_reel, y_pred):
    mae = mean_absolute_error(y_reel, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
    r2 = r2_score(y_reel, y_pred)
    return mae, rmse, r2


def affichage_performances(mae_train, rmse_train, r2_train, mae_val, rmse_val, r2_val):
    print("Performances du modèle naïf (baseline)")
    print()
    print(f"Jeu d'entraînement (train)")
    print(f"- MAE  (Erreur Moyenne Absolue) : {mae_train:,.2f} €")
    print(f"- RMSE (Écart-type des erreurs) : {rmse_train:,.2f} €")
    print(f"- R²   (Pouvoir explicatif)     : {r2_train:.4f} ({r2_train*100:.1f}%)")
    print()
    print(f"Jeu de validation (val)")
    print(f"- MAE  (Erreur Moyenne Absolue) : {mae_val:,.2f} €")
    print(f"- RMSE (Écart-type des erreurs) : {rmse_val:,.2f} €")
    print(f"- R²   (Pouvoir explicatif)     : {r2_val:.4f} ({r2_val*100:.1f}%)")


def afficher_top_erreurs_predictions(df_val_clean, y_val, y_pred_val, top_n=5, nom_modele="naïf"):
    """
    Génère un graphique en barres comparant la valeur réelle et la valeur prédite
    pour les N plus grosses erreurs de prédiction.
    """
    # Construction du dataframe de comparaison
    df_comparaison = pd.DataFrame({
        "Joueur": df_val_clean["player"],
        "Saison": df_val_clean["season_year"],
        "Valeur Réelle": y_val,
        "Prédiction": y_pred_val,
        "Erreur (Ecart)": np.abs(y_val - y_pred_val)
    })

    # Extraction du Top N des plus grosses erreurs
    top_df = df_comparaison.sort_values(by="Erreur (Ecart)", ascending=False).head(top_n)

    # Création du graphique
    x = np.arange(len(top_df))
    largeur = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - largeur/2, top_df["Valeur Réelle"] / 1e6, largeur, label="Valeur réelle", color="#2a78d6")
    ax.bar(x + largeur/2, top_df["Prédiction"] / 1e6, largeur, label=f"Prédiction ({nom_modele})", color="#eb6834")

    ax.set_ylabel("Valeur (M €)")
    ax.set_title(f"Modèle {nom_modele} : sous-estimation des plus grosses valeurs marchandes")
    ax.set_xticks(x)
    ax.set_xticklabels(top_df["Joueur"], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

    return top_df


def afficher_scatter_reelle_vs_predite(y_val, y_pred_val, nom_modele="naïf", figsize=(7, 7)):
    """
    Affiche un nuage de points comparant la valeur réelle et la valeur prédite
    sur le jeu de validation avec la ligne de référence y = x.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Nuage de points
    ax.scatter(y_val / 1e6, y_pred_val / 1e6, alpha=0.4, s=20, color="#2a78d6", edgecolor="none")

    # Ligne de référence y = x
    max_val = max(y_val.max(), y_pred_val.max()) / 1e6
    ax.plot([0, max_val], [0, max_val], color="#e34948", linestyle="--", linewidth=1.5, label="Prédiction parfaite (y = x)")

    ax.set_xlabel("Valeur marchande réelle (M €)")
    ax.set_ylabel("Valeur marchande prédite (M €)")
    ax.set_title(f"Modèle {nom_modele} — Valeur réelle vs valeur prédite (validation)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()