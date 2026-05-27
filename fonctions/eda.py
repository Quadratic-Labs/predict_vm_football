import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import warnings
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE = "viridis"

COLORS = {
    "blue":   "#61AFEF",
    "green":  "#98C379",
    "red":    "#E06C75",
    "purple": "#C678DD",
    "yellow": "#E5C07B",
    "cyan":   "#56B6C2",
}



# On définit le dossier de destination
dossier_sortie = Path("../output")

# On force la création du dossier s'il n'existe pas encore
dossier_sortie.mkdir(exist_ok=True)


def analyser_types_et_stats(df):
    """Affiche le nombre de variables par type (numériques/catégorielles)

    ainsi que les statistiques descriptives des variables numériques.
    """
    print("Analyse des types de variables")

    # Extraction des colonnes par type
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Affichage des comptes
    print(f"Variables numériques   : {len(num_cols)}")
    print(f"Variables catégorielles: {len(cat_cols)}")

    # Affichage des statistiques descriptives s'il y a des colonnes numériques
    print("\nStatistiques descriptives (variables numériques)")
    if len(num_cols) > 0:
        print(df[num_cols].describe().T.to_string())
    else:
        print("Aucune variable numérique dans ce DataFrame.")






def analyser_valeurs_manquantes(df, dossier_sauvegarde=None):
    """Calcule les valeurs manquantes d'un DataFrame, affiche le récapitulatif

    et génère un graphique en barres du Top 30 des colonnes les plus vides.
    """
    print("Analyse des valeurs manquantes")

    # Calcul du nombre et du pourcentage de valeurs manquantes
    missing = (
        df.isnull()
        .sum()
        .to_frame("nb_missing")
        .assign(pct=lambda x: (x["nb_missing"] / len(df) * 100).round(2))
        .sort_values("pct", ascending=False)
    )

    # Affichage des colonnes qui ont au moins une valeur manquante
    df_missing_only = missing[missing["nb_missing"] > 0]

    if df_missing_only.empty:
        print("Parfait ! Aucune valeur manquante détectée.")
        return  # On arrête la fonction ici puisqu'il n'y a rien à tracer

    print(df_missing_only.to_string())

    # Génération du graphique (Top 30 des colonnes)
    top_missing = df_missing_only.head(30).index

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(top_missing, missing.loc[top_missing, "pct"], color="#E06C75")
    ax.set_title("Taux de valeurs manquantes")
    ax.set_ylabel("% manquant")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Gestion de la sauvegarde si un dossier est fourni
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(
            exist_ok=True
        )  # Crée le dossier s'il n'existe pas
        chemin_fichier = chemin_dossier / "valeurs_manquantes.png"

        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    # Affichage à l'écran
    plt.show()




def analyser_doublons(df, id_cols=None):
    """Vérifie la présence de lignes dupliquées globales dans le DataFrame

    et sur un sous-ensemble de colonnes (clés d'identification).
    """
    if id_cols is None:
        id_cols = ["player_id", "season"]

    print("Analyse des doublons")

    # Doublons globaux (lignes strictement identiques)
    nb_dup = df.duplicated().sum()
    print(f"Lignes dupliquées (toutes colonnes) : {nb_dup}")

    # Doublons sur les clés spécifiques
    # On vérifie si toutes les colonnes demandées existent dans le DataFrame
    colonnes_presentes = [c for c in id_cols if c in df.columns]

    if len(colonnes_presentes) == len(id_cols):
        nb_dup_id = df.duplicated(subset=id_cols).sum()
        print(f"Doublons sur ({', '.join(id_cols)}) : {nb_dup_id}")
    else:
        # Si certaines colonnes manquent, on prévient l'utilisateur sans faire planter le code
        manquantes = set(id_cols) - set(colonnes_presentes)
        print(
            f"Impossible de vérifier les doublons spécifiques. Colonne(s) manquante(s) : {', '.join(manquantes)}"
        )






def analyser_variable_cible(df, target_col="valeur_marchande", dossier_sauvegarde=None):
    """Affiche les statistiques descriptives (skewness, kurtosis) de la variable cible

    et génère trois graphiques : distribution brute, log-transformée et boxplot.
    """
    print(f"Analyse de la variable cible : {target_col} ===")

    # Extraction et nettoyage des données manquantes
    if target_col not in df.columns:
        print(f"Erreur : La colonne '{target_col}' n'existe pas dans le DataFrame.")
        return

    vm = df[target_col].dropna()

    if vm.empty:
        print(f"Erreur : La colonne '{target_col}' ne contient que des valeurs manquantes ou est vide.")
        return

    # Affichage des statistiques descriptives
    print(vm.describe().apply(lambda x: f"{x:,.0f}"))
    print(f"\nSkewness : {vm.skew():.2f}")
    print(f"Kurtosis : {vm.kurt():.2f}")

    # Création des graphiques
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Distribution de la variable : {target_col}", fontsize=14, fontweight="bold")

    # Graphique 1 : Distribution brute (conversion en millions si les valeurs sont grandes)
    # Note : On divise par 1e6 les valeurs financières brutes (ex: 5 000 000 €)
    axes[0].hist(vm / 1e6, bins=80, color="#61AFEF", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel(f"{target_col} (M€)")
    axes[0].set_title("Distribution brute")

    # Graphique 2 : Distribution log-transformée
    vm_log = np.log1p(vm)
    axes[1].hist(vm_log, bins=60, color="#98C379", edgecolor="white", linewidth=0.3)
    axes[1].set_xlabel(f"log({target_col} + 1)")
    axes[1].set_title("Distribution log-transformée")

    # Graphique 3 : Boxplot
    axes[2].boxplot(vm / 1e6, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#C678DD", alpha=0.6))
    axes[2].set_ylabel(f"{target_col} (M€)")
    axes[2].set_title("Boxplot")

    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / f"distribution_{target_col}.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    # Affichage
    plt.show()






def analyser_distributions_numeriques(df, cols_a_analyser=None, dossier_sauvegarde=None):
    """Génère une grille d'histogrammes pour les variables numériques clés,

    avec une ligne verticale indiquant la médiane de chaque distribution.
    """
    print("Distributions des variables numériques clés")

    # Liste des colonnes par défaut si aucune n'est fournie
    if cols_a_analyser is None:
        cols_a_analyser = [
            "age",
            "Playing Time_Min",
            "Performance_Gls",
            "Performance_Ast",
            "xg",
            "xa",
            "injury_days_total",
            "injury_nb_total",
        ]

    # Filtrage pour ne garder que les colonnes réellement présentes dans le DataFrame
    key_num = [c for c in cols_a_analyser if c in df.columns]

    if not key_num:
        print("Aucune des colonnes spécifiées n'est présente dans le DataFrame.")
        return

    # Configuration dynamique de la grille de graphiques
    n_cols = 4
    n_rows = int(np.ceil(len(key_num) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    
    # Si la grille n'a qu'une seule ligne ou colonne, on s'assure qu'axes soit un tableau plat
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    # Boucle de traçage des histogrammes
    i = 0
    for i, col in enumerate(key_num):
        # Conversion numérique forcée et suppression des NaN
        data = pd.to_numeric(df[col], errors="coerce").dropna()
        
        if data.empty:
            axes[i].text(0.5, 0.5, "Données vides\naprès conversion", 
                         ha="center", va="center", color="gray")
            axes[i].set_title(col)
            continue

        # Tracé de l'histogramme
        axes[i].hist(data, bins=50, color="#56B6C2", edgecolor="white", linewidth=0.3)
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        
        # Ajout de la ligne de médiane
        mediane = data.median()
        axes[i].axvline(mediane, color="#E06C75", linestyle="--", 
                        label=f"Médiane={mediane:.1f}")
        axes[i].legend(fontsize=8)

    # Masquer les sous-graphiques vides de la grille
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distributions de variables numériques", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "distributions_variables_numeriques_cles.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    # Affichage
    plt.show()







def analyser_variables_categorielles(df, target_col, dossier_sauvegarde=None):
    """Analyse les variables catégorielles du DataFrame : affiche les comptages textuels,

    génère les graphiques de répartition et croise les catégories avec la variable cible (boxplots).
    """
    print("Analyse des variables catégorielles")

    # Affichage textuel des value_counts pour les variables clés
    key_cat = ["pos", "league", "season"]
    key_cat = [c for c in key_cat if c in df.columns]

    for col in key_cat:
        vc = df[col].value_counts()
        print(f"\n{col} ({vc.shape[0]} modalités) :")
        print(vc.head(20).to_string())

    # Préparation de la sauvegarde
    sauvegarder = dossier_sauvegarde is not None
    if sauvegarder:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)

    # Graphique : Top 20 Nationalités
    if "nation" in df.columns:
        top_nations = df["nation"].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(12, 5))
        top_nations.plot(kind="bar", ax=ax, color="#E5C07B")
        ax.set_title("Top 20 nationalités")
        ax.set_ylabel("Nombre de joueurs")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if sauvegarder:
            plt.savefig(chemin_dossier / "players_top_nations.png")
            print("\nplayers_top_nations.png sauvegardé")
        plt.show()

    # Graphiques : Répartition Poste & Championnat
    for col in ["pos", "league"]:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        vc.plot(kind="bar", ax=ax, color="#61AFEF")
        ax.set_title(f"Répartition : {col}")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        
        if sauvegarder:
            plt.savefig(chemin_dossier / f"{col}.png")
            print(f"{col}.png sauvegardé")
        plt.show()

    # Vérification de la présence de la cible pour les analyses croisées
    if target_col not in df.columns:
        print(f"\n[Poste / League vs Cible] Annulé : La colonne cible '{target_col}' est absente.")
        return

    # Boxplot : Variable Cible par Poste
    if "pos" in df.columns:
        df_pos = df[[target_col, "pos"]].dropna()
        if not df_pos.empty:
            order = df_pos.groupby("pos")[target_col].median().sort_values(ascending=False).index
            fig, ax = plt.subplots(figsize=(14, 5))
            groups = [df_pos.loc[df_pos["pos"] == p, target_col].values / 1e6 for p in order]
            ax.boxplot(groups, labels=order, patch_artist=True,
                       boxprops=dict(facecolor="#98C379", alpha=0.5))
            ax.set_title(f"Valeur Marchande par poste (Trié par médiane décroissante)")
            ax.set_ylabel("VM (M€)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            
            if sauvegarder:
                plt.savefig(chemin_dossier / "vm_par_pos.png")
                print("vm_par_pos.png sauvegardé")
            plt.show()

    # Boxplot : Variable Cible par Championnat
    if "league" in df.columns:
        df_ligue = df[[target_col, "league"]].dropna()
        if not df_ligue.empty:
            order_l = df_ligue.groupby("league")[target_col].median().sort_values(ascending=False).index
            fig, ax = plt.subplots(figsize=(14, 5))
            groups_l = [df_ligue.loc[df_ligue["league"] == l, target_col].values / 1e6 for l in order_l]
            ax.boxplot(groups_l, labels=order_l, patch_artist=True,
                       boxprops=dict(facecolor="#C678DD", alpha=0.5))
            ax.set_title(f"Valeur Marchande par championnat (Trié par médiane décroissante)")
            ax.set_ylabel("VM (M€)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            
            if sauvegarder:
                plt.savefig(chemin_dossier / "vm_par_league.png")
                print("vm_par_league.png sauvegardé")
            plt.show()






def analyser_outliers(df, cols_a_analyser=None):
    """Détecte le nombre d'outliers pour les variables numériques

    en utilisant deux méthodes : l'Écart Interquartile (IQR) et le Z-score.
    """
    print("Détection des outliers")

    # Liste des colonnes par défaut si aucune n'est fournie
    if cols_a_analyser is None:
        cols_a_analyser = [
            "age",
            "Playing Time_Min",
            "Performance_Gls",
            "Performance_Ast",
            "xg",
            "xa",
            "injury_days_total",
            "injury_nb_total",
        ]

    # Filtrage pour ne garder que les colonnes réellement présentes
    key_num = [c for c in cols_a_analyser if c in df.columns]

    if not key_num:
        print("Aucune des colonnes spécifiées n'est présente dans le DataFrame.")
        return

    # Calcul des outliers par colonne
    outlier_report = []
    for col in key_num:
        # Conversion numérique forcée et suppression des NaN
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        
        if s.empty:
            continue

        # Méthode 1 : IQR (Interquartile Range)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_iqr = ((s < lower_bound) | (s > upper_bound)).sum()

        # Méthode 2 : Z-score (Seuil strict à |Z| > 3)
        # Note : On s'assure qu'il y a de la variance pour éviter une division par zéro
        if s.std() > 0:
            z = np.abs(stats.zscore(s))
            n_z = (z > 3).sum()
        else:
            n_z = 0

        outlier_report.append({
            "variable": col, 
            "outliers_IQR": n_iqr, 
            "outliers_Zscore": n_z, 
            "total": len(s)
        })

    # Structuration et affichage du tableau de rapport
    if not outlier_report:
        print("Impossible de calculer les valeurs aberrantes (données vides ou textuelles).")
        return

    df_out = pd.DataFrame(outlier_report)
    df_out["pct_IQR"] = (df_out["outliers_IQR"] / df_out["total"] * 100).round(1)
    
    # Réorganisation esthétique des colonnes
    colonnes_ordre = ["variable", "outliers_IQR", "pct_IQR", "outliers_Zscore", "total"]
    print(df_out[colonnes_ordre].to_string(index=False))






def analyser_relations_cible(df, target_col, variables_regresseurs=None, dossier_sauvegarde=None):
    """Génère une grille de graphiques en nuages de points (scatter plots) avec droites de régression

    pour analyser la relation entre des variables explicatives et la variable cible.
    """
    print(f"Analyse des relations avec la variable cible : {target_col}")

    # Vérification de la présence de la cible
    if target_col not in df.columns:
        print(f"Erreur : La colonne cible '{target_col}' n'existe pas dans le DataFrame.")
        return

    # Liste des variables et labels par défaut
    if variables_regresseurs is None:
        variables_regresseurs = [
            ("age", "Âge"),
            ("xg", "xG"),
            ("xa", "xA"),
            ("Performance_Gls", "Buts"),
            ("Performance_Ast", "Passes décisives"),
            ("Playing Time_Min", "Minutes jouées"),
            ("injury_days_total", "Jours blessés"),
            ("injury_nb_total", "Nb blessures"),
        ]

    # Filtrage pour ne garder que les colonnes réellement présentes
    scatter_vars = [(c, l) for c, l in variables_regresseurs if c in df.columns]

    if not scatter_vars:
        print("Aucune des variables spécifiées n'est présente dans le DataFrame.")
        return

    # Configuration dynamique de la grille de graphiques
    n_cols_s = 4
    n_rows_s = int(np.ceil(len(scatter_vars) / n_cols_s))
    
    fig, axes = plt.subplots(n_rows_s, n_cols_s, figsize=(22, 5 * n_rows_s))
    
    # Sécurité pour s'assurer qu'axes est toujours un tableau plat
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])

    # Boucle de traçage des scatter plots
    i = 0
    for i, (col, label) in enumerate(scatter_vars):
        # Nettoyage des données par paire (variable, cible)
        sub = df[[col, target_col]].dropna()
        
        # S'il n'y a pas assez de données (minimum 2 points requis pour une régression)
        if len(sub) < 2:
            axes[i].text(0.5, 0.5, "Pas assez de données\npour corréler", 
                         ha="center", va="center", color="gray")
            axes[i].set_title(label)
            continue

        try:
            # Calcul du coefficient de corrélation de Pearson (avec log1p sur la cible comme ton code d'origine)
            r, p = stats.pearsonr(sub[col], np.log1p(sub[target_col]))
            p_label = '<0.001' if p < 0.001 else f'{p:.3f}'
            titre_graphique = f"{label}\n(r={r:.2f}, p={p_label})"
            
            # Droite de régression (sur la valeur brute en millions)
            m, b = np.polyfit(sub[col], sub[target_col] / 1e6, 1)
            x_line = np.linspace(sub[col].min(), sub[col].max(), 100)
            axes[i].plot(x_line, m * x_line + b, color="#E06C75", linewidth=1.5)
            
        except Exception:
            # En cas de variance nulle ou d'erreur mathématique
            titre_graphique = f"{label}\n(Calcul corrélation impossible)"

        # Tracé du nuage de points (valeurs de la cible divisées par 1e6 pour l'échelle M€)
        axes[i].scatter(sub[col], sub[target_col] / 1e6,
                        alpha=0.15, s=10, color="#56B6C2", rasterized=True)
        
        axes[i].set_xlabel(label)
        axes[i].set_ylabel(f"{target_col} (M€)")
        axes[i].set_title(titre_graphique)

    # Masquer les sous-graphiques vides de la grille
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Relation entre chaque variable et la variable : {target_col}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / f"relations_{target_col}.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    # Affichage
    plt.show()






def analyser_correlations(df, target_col, cols_cles=None, dossier_sauvegarde=None):
    """Calcule et affiche les corrélations des variables avec la cible (target_col).

    Génère une heatmap Seaborn pour les variables clés et liste le Top 20 des corrélations globales.
    """
    print(f"Analyse des corrélations (cible : {target_col})")

    # Vérification de la présence de la cible
    if target_col not in df.columns:
        print(f"Erreur : La colonne cible '{target_col}' n'existe pas dans le DataFrame.")
        return

    # Définition des variables clés par défaut si non fournies
    if cols_cles is None:
        cols_cles = [
            "age",
            "xg",
            "xa",
            "Performance_Gls",
            "Performance_Ast",
            "Playing Time_Min",
            "injury_days_total",
            "injury_nb_total",
        ]

    # Construction de la liste des colonnes présentes pour la Heatmap
    corr_vars = [c for c in cols_cles if c in df.columns] + [target_col]
    corr_vars = list(set(corr_vars))  # Évite les doublons si target_col était déjà dans cols_cles

    # Calcul et affichage de la matrice de corrélation restreinte
    if len(corr_vars) > 1:
        print("\nCorrélation linéaire avec la cible (variables clés)")
        corr_matrix = df[corr_vars].corr()
        
        # Affichage textuel du tri par rapport à la cible
        print(corr_matrix[[target_col]].sort_values(target_col, ascending=False).to_string())

        # Génération de la Heatmap (Masque triangulaire supérieur pour éviter la redondance)
        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_title(f"Matrice de corrélation", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Gestion de la sauvegarde de la Heatmap
        if dossier_sauvegarde is not None:
            chemin_dossier = Path(dossier_sauvegarde)
            chemin_dossier.mkdir(exist_ok=True)
            chemin_fichier = chemin_dossier / "matrice_correlation.png"
            plt.savefig(chemin_fichier)
            print(f"\nmatrice_correlation.png sauvegardé sous : {chemin_fichier}")
            
        plt.show()
    else:
        print("Pas assez de variables clés valides pour générer une Heatmap.")

    # Corrélation complète (toutes variables numériques du dataframe) avec la cible
    print(f"\nTop 20 des corrélations absolues avec {target_col} (Toutes variables numériques)")
    
    # Identification automatique de toutes les colonnes numériques du dataframe
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_unique = [c for c in num_cols if c != target_col]

    if num_cols_unique:
        # Calcul des corrélations en valeur absolue pour capter les fortes relations négatives et positives
        full_corr = (
            df[num_cols_unique + [target_col]]
            .apply(pd.to_numeric, errors="coerce")
            .corr()[target_col]
            .drop(target_col, errors="ignore")
            .abs()
            .sort_values(ascending=False)
        )
        print(full_corr.head(20).to_string())
    else:
        print("Aucune autre variable numérique trouvée dans le dataframe pour l'analyse globale.")






def analyser_profil_par_poste(df, target_col, dossier_sauvegarde=None, couleurs_dict=None):
    """Calcule le profil médian normalisé des joueurs par poste (GK, DF, MF, FW),

    génère une heatmap comparative et un graphique en barres de la valeur marchande médiane.
    """
    print("Analyse du profil médian par poste")

    # Vérifications initiales et filtres
    if "pos_simple" not in df.columns:
        print("Annulé : La colonne 'pos_simple' est absente du DataFrame.")
        return

    pos_order = ["GK", "DF", "MF", "FW"]
    df_pos = df[df["pos_simple"].isin(pos_order)].copy()

    if df_pos.empty:
        print("Annulé : Aucun joueur ne correspond aux postes (GK, DF, MF, FW) dans la colonne 'pos_simple'.")
        return

    # Gestion de la couleur par défaut si COLORS n'est pas passé
    if couleurs_dict is None:
        couleurs_dict = {"blue": "#61AFEF"}

    # Définition et filtrage des colonnes du profil
    profile_cols = {
        "xG": "xg",
        "xA": "xa",
        "Buts": "Performance_Gls",
        "Assists": "Performance_Ast",
        "Minutes": "Playing Time_Min",
        "J. blessés": "injury_days_total",
        "Nb blessures": "injury_nb_total",
        "VM (M€)": target_col,
    }
    # On ne garde que les colonnes qui existent réellement dans le DataFrame
    profile_cols = {k: v for k, v in profile_cols.items() if v in df.columns}

    if not profile_cols:
        print("Aucune des colonnes de profil spécifiées n'est présente dans le DataFrame.")
        return

    # Calcul du profil médian et normalisation Min-Max
    profile = (
        df_pos.groupby("pos_simple")[list(profile_cols.values())]
        .median()
        .reindex(pos_order)
    )
    profile.columns = list(profile_cols.keys())

    # Normalisation min-max par colonne (ajout de 1e-9 pour éviter la division par zéro)
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    # Construction des graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle("Profil médian par poste", fontsize=13, fontweight="bold")

    # Graphique 1 : Heatmap du profil normalisé avec annotations réelles
    sns.heatmap(
        profile_norm.T, 
        annot=profile.T.round(1), 
        fmt="g",
        cmap="YlOrRd", 
        ax=axes[0], 
        linewidths=0.5, 
        cbar_kws={"label": "Normalisé 0-1"}
    )
    axes[0].set_title("Valeurs normalisées (annotation = médiane réelle)")
    axes[0].set_xlabel("Poste")

    # Graphique 2 : Valeurs brutes VM (uniquement si la cible est présente)
    if target_col in df.columns:
        vm_by_pos = df_pos.groupby("pos_simple")[target_col].median().reindex(pos_order) / 1e6
        vm_by_pos.plot(kind="bar", ax=axes[1], color=couleurs_dict.get("blue", "#61AFEF"), edgecolor="white")
        axes[1].set_title("VM médiane par poste")
        axes[1].set_ylabel("VM (M€)")
        axes[1].set_xlabel("Poste")
        axes[1].set_xticklabels(pos_order, rotation=0)
    else:
        axes[1].text(0.5, 0.5, f"Colonne cible '{target_col}'\nintrouvable pour la VM", 
                     ha="center", va="center", color="gray")
        axes[1].set_title("VM médiane par poste (Indisponible)")

    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "segmentation_par_poste_profil_poste.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    # Affichage
    plt.show()







def analyser_correlations_specifiques_poste(df, target_col, dossier_sauvegarde=None, couleurs_dict=None):
    """Génère des graphiques de corrélation ciblés (xG et xA vs Valeur Marchande)

    spécifiquement pour les Attaquants (FW) et les Milieux (MF).
    """
    print("Analyse des corrélations spécifiques par poste")

    # Vérifications initiales
    if "pos_simple" not in df.columns:
        print("Annulé : La colonne 'pos_simple' est absente du DataFrame.")
        return
        
    if target_col not in df.columns:
        print(f"Annulé : La colonne cible '{target_col}' est absente.")
        return

    # Gestion des couleurs par défaut
    if couleurs_dict is None:
        couleurs_dict = {"cyan": "#56B6C2", "red": "#E06C75"}

    # Définition des paires à analyser (Poste, Feature, Titre visuel)
    pairs = [
        ("FW", "xg", "xG → VM  (Attaquants)"),
        ("MF", "xg", "xG → VM  (Milieux)"),
        ("MF", "xa", "xA → VM  (Milieux)"),
    ]

    # Filtrer les paires pour ne garder que les features existantes
    pairs = [(p, f, t) for p, f, t in pairs if f in df.columns]

    if not pairs:
        print("Aucune des fonctionnalités (xg, xa) n'est présente dans le DataFrame.")
        return

    # Configuration de la grille de graphiques
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Corrélations spécifiques par poste", fontsize=13, fontweight="bold")

    # Si axes n'est pas un tableau (au cas où il n'y aurait qu'un seul graphique généré)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Boucle de traçage avec zip
    for ax, (pos, feat, title) in zip(axes, pairs):
        # Extraction et nettoyage des données pour le couple (Poste, Feature)
        sub = df[(df["pos_simple"] == pos) & df[[feat, target_col]].notna().all(axis=1)].copy()
        
        # Sécurité : Vérification du nombre de lignes disponibles
        if len(sub) < 2:
            ax.text(0.5, 0.5, f"Pas assez de données\npour {pos} ({feat})", 
                    ha="center", va="center", color="gray")
            ax.set_title(title)
            continue

        try:
            # Calcul du coefficient de corrélation de Pearson (avec log1p sur la cible)
            r, p = pearsonr(sub[feat], np.log1p(sub[target_col]))
            p_label = '<0.001' if p < 0.001 else f'{p:.3f}'
            titre_graphique = f"{title}\nr={r:.2f}  p={p_label}"
            
            # Calcul de la droite de régression (sur la valeur brute en millions)
            m, b = np.polyfit(sub[feat], sub[target_col] / 1e6, 1)
            x_l = np.linspace(sub[feat].min(), sub[feat].max(), 100)
            ax.plot(x_l, m * x_l + b, color=couleurs_dict.get("red", "#E06C75"), lw=1.8)
            
        except Exception:
            # Sécurité en cas de variance nulle (ex: tous les xG sont à 0)
            titre_graphique = f"{title}\n(Calcul impossible)"

        # Tracé du nuage de points
        ax.scatter(
            sub[feat], 
            sub[target_col] / 1e6, 
            alpha=0.2, 
            s=12,
            color=couleurs_dict.get("cyan", "#56B6C2"), 
            rasterized=True
        )
        
        ax.set_title(titre_graphique)
        ax.set_xlabel(feat)
        ax.set_ylabel("VM (M€)")

    # Masquer les axes restants si jamais il y a moins de 3 graphiques tracés
    for j in range(len(pairs), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "segmentation_par_poste_corr_poste.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    # Affichage
    plt.show()







def analyser_ages_par_poste(df, dossier_sauvegarde=None, liste_couleurs=None):
    """Génère un histogramme superposé de la distribution des âges pour chaque poste (GK, DF, MF, FW)

    et affiche les âges médians sous forme textuelle.
    """
    print("Analyse de la distribution des âges par poste")

    # Vérifications initiales
    if "pos_simple" not in df.columns:
        print("Annulé : La colonne 'pos_simple' est absente du DataFrame.")
        return
    if "age" not in df.columns:
        print("Annulé : La colonne 'age' est absente du DataFrame.")
        return

    pos_order = ["GK", "DF", "MF", "FW"]
    df_pos = df[df["pos_simple"].isin(pos_order)].copy()

    if df_pos.empty:
        print("Annulé : Aucun joueur trouvé pour les postes (GK, DF, MF, FW).")
        return

    # Gestion de la palette de couleurs par défaut (One Dark Pro style comme tes codes précédents)
    if liste_couleurs is None:
        liste_couleurs = ["#61AFEF", "#98C379", "#E5C07B", "#E06C75"]  # bleu, vert, jaune, rouge

    # Construction de l'histogramme superposé
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Distribution des âges par poste", fontsize=13, fontweight="bold")

    for pos, color in zip(pos_order, liste_couleurs):
        ages = df_pos[df_pos["pos_simple"] == pos]["age"].dropna()
        
        # On ne trace que s'il y a des données pour le poste en question
        if not ages.empty:
            ax.hist(ages, bins=30, alpha=0.55, label=pos, color=color, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Âge")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "segmentation_par_poste_age_par_poste.png"
        
        plt.savefig(chemin_fichier)
        print(f"Graphique sauvegardé sous : {chemin_fichier}")

    # Affichage du graphique
    plt.show()

    # Calcul et affichage des statistiques textuelles
    print("\nÂge médian par poste")
    stats_mediane = df_pos.groupby("pos_simple")["age"].median().reindex(pos_order).round(1)
    print(stats_mediane.to_string())







def analyser_feature_engineering(df, target_col="valeur_marchande", dossier_sauvegarde=None, couleurs_dict=None):
    """Calcule de nouvelles variables (Feature Engineering), affiche leur corrélation avec la cible

    et génère une grille de graphiques de régression.
    """
    print("Feature engineering exploratoire")

    # Vérifications initiales
    if target_col not in df.columns:
        print(f"Annulé : La colonne cible '{target_col}' est absente du DataFrame.")
        return

    # Gestion des couleurs par défaut
    if couleurs_dict is None:
        couleurs_dict = {"cyan": "#56B6C2", "red": "#E06C75"}

    # Calcul des nouvelles features sur une copie de travail
    dfe = df.copy()
    
    # Dictionnaire pour mapper les calculs de manière sécurisée (vérification des colonnes sources)
    colonnes_requises = ["Performance_Gls", "xg", "Playing Time_Min", "Playing Time_MP", "injury_days_total", "age"]
    manquantes = [c for c in colonnes_requises if c not in dfe.columns]
    
    if manquantes:
        print(f"Attention : Certaines colonnes sources sont manquantes ({', '.join(manquantes)}).")
        print("Le calcul de certaines nouvelles variables peut échouer.")

    # Calculs avec gestion des divisions par zéro
    dfe["ratio_buts_xg"] = (dfe.get("Performance_Gls", 0) / (dfe.get("xg", 0) + 1e-6)).clip(upper=5)
    dfe["ratio_min_mp"]  = (dfe.get("Playing Time_Min", 0) / (dfe.get("Playing Time_MP", 0) + 1e-6))
    dfe["taux_indispo"]  = (dfe.get("injury_days_total", 0) / (dfe.get("Playing Time_Min", 0) + 1)).clip(upper=1)
    dfe["age_x_xg"]      = dfe.get("age", 0) * dfe.get("xg", 0)

    # Définition des labels pour les graphiques
    new_features = {
        "ratio_buts_xg": "Buts / xG\n(sur/sous-performance)",
        "ratio_min_mp":  "Min / Match\n(titulaire régulier ?)",
        "taux_indispo":  "Jours blessés / Minutes\n(taux d'indispo)",
        "age_x_xg":      "Âge × xG\n(interaction)",
    }

    # Configuration de la grille de graphiques (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Features engineered vs {target_col}", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    # Boucle de traçage des graphiques
    for ax, (feat, label) in zip(axes, new_features.items()):
        # Nettoyage des valeurs manquantes et infinies
        sub = dfe[[feat, target_col]].dropna()
        sub = sub[np.isfinite(sub[feat]) & np.isfinite(sub[target_col])]
        
        if len(sub) < 2:
            ax.text(0.5, 0.5, f"Données insuffisantes\npour {feat}", ha="center", va="center", color="gray")
            ax.set_title(label)
            continue

        # Tracé du nuage de points
        ax.scatter(
            sub[feat], 
            sub[target_col] / 1e6, 
            alpha=0.15, 
            s=10,
            color=couleurs_dict.get("cyan", "#56B6C2"), 
            rasterized=True
        )

        try:
            # Calcul du coefficient de corrélation (Pearson sur log de la cible)
            r, p = pearsonr(sub[feat], np.log1p(sub[target_col]))
            p_label = '<0.001' if p < 0.001 else f'{p:.3f}'
            titre_graphique = f"{label}\nr={r:.2f}  p={p_label}"

            # Calcul et tracé de la droite de régression (sur la plage de quantiles 1% à 99%)
            m, b = np.polyfit(sub[feat], sub[target_col] / 1e6, 1)
            x_l = np.linspace(sub[feat].quantile(0.01), sub[feat].quantile(0.99), 100)
            ax.plot(x_l, m * x_l + b, color=couleurs_dict.get("red", "#E06C75"), lw=2)
            
        except Exception:
            titre_graphique = f"{label}\n(Calcul régression impossible)"

        ax.set_title(titre_graphique)
        ax.set_xlabel(feat)
        ax.set_ylabel(f"{target_col} (M€)")

    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "feature_engineering.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    plt.show()

    # Résumé textuel des corrélations
    print(f"\nCorrélations (Pearson sur log {target_col}) — features engineered :")
    for feat, label in new_features.items():
        sub = dfe[[feat, target_col]].dropna()
        sub = sub[np.isfinite(sub[feat]) & np.isfinite(sub[target_col])]
        
        if len(sub) >= 2:
            try:
                r, p = pearsonr(sub[feat], np.log1p(sub[target_col]))
                p_label = '<0.001' if p < 0.001 else f'{p:.3f}'
                print(f"  {feat:<20} r={r:+.3f}  p={p_label}")
            except Exception:
                print(f"  {feat:<20} Calcul de corrélation impossible.")
        else:
            print(f"  {feat:<20} Pas assez de données valides.")







def analyser_heatmap_league_poste(df, target_col, dossier_sauvegarde=None):
    """Génère une heatmap croisant les championnats (league) et les postes (pos_simple)

    pour afficher la valeur marchande médiane (en M€) de chaque segment.
    """
    print("Analyse croisée : VM médiane par championnat et par poste")

    # Vérifications initiales des colonnes
    if "pos_simple" not in df.columns:
        print("Annulé : La colonne 'pos_simple' est absente du DataFrame.")
        return
    if "league" not in df.columns:
        print("Annulé : La colonne 'league' est absente du DataFrame.")
        return
    if target_col not in df.columns:
        print(f"Annulé : La colonne cible '{target_col}' est absente.")
        return

    pos_order = ["GK", "DF", "MF", "FW"]
    
    # Construction du tableau croisé (Pivot Table)
    # Filtrage des postes valides
    df_filtrer = df[df["pos_simple"].isin(pos_order)]
    
    if df_filtrer.empty:
        print("Annulé : Aucun joueur trouvé pour les postes (GK, DF, MF, FW).")
        return

    # Calcul de la médiane par groupe, pivotement et conversion en Millions
    pivot = (
        df_filtrer.groupby(["league", "pos_simple"])[target_col]
        .median()
        .unstack("pos_simple")
        .reindex(columns=pos_order)
        / 1e6
    )

    # Sécurité : Si un championnat n'a aucun joueur à un poste donné, on remplace le NaN par 0
    pivot = pivot.fillna(0)

    # Génération de la Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        pivot.round(1), 
        annot=True, 
        fmt=".1f", 
        cmap="YlOrRd",
        linewidths=0.5, 
        ax=ax, 
        cbar_kws={"label": f"VM médiane (M€)"}
    )
    
    ax.set_title(f"{target_col} médiane (M€) par championnat × poste", fontsize=13, fontweight="bold")
    ax.set_xlabel("Poste")
    ax.set_ylabel("Championnat")
    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "feature_engineered_heatmap_league_pos.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nHeatmap sauvegardée sous : {chemin_fichier}")

    # Affichage
    plt.show()







def analyser_distribution_violin_league(df, target_col, dossier_sauvegarde=None, couleurs_dict=None):
    """Génère un graphique en violon (Violin plot) pour analyser la distribution et la densité

    de la variable cible (en M€) à travers les différents championnats (league).
    """
    print("Analyse de la distribution par championnat")

    # Vérifications initiales des colonnes
    if "league" not in df.columns:
        print("Annulé : La colonne 'league' est absente du DataFrame.")
        return
    if target_col not in df.columns:
        print(f"Annulé : La colonne cible '{target_col}' est absente.")
        return

    # Gestion des couleurs par défaut
    if couleurs_dict is None:
        couleurs_dict = {"purple": "#C678DD", "red": "#E06C75"}

    # Préparation et tri des données par médiane décroissante
    leagues = df["league"].dropna().unique().tolist()
    df_ligue = df[df["league"].isin(leagues)].copy()
    
    if df_ligue.empty:
        print("Annulé : Le DataFrame ne contient aucune donnée valide pour la colonne 'league'.")
        return

    # Calcul de l'ordre d'affichage basé sur la médiane
    order_vm = df_ligue.groupby("league")[target_col].median().sort_values(ascending=False).index

    # Extraction des données sous forme de listes de tableaux numpy (conversion en Millions d'Euros)
    data_violin = []
    labels_valides = []
    
    for l in order_vm:
        values = df_ligue[df_ligue["league"] == l][target_col].dropna().values / 1e6
        # Sécurité : On s'assure qu'il y a assez de points pour construire un violon (minimum 2 requis)
        if len(values) >= 2:
            data_violin.append(values)
            labels_valides.append(l)

    if not data_violin:
        print("Annulé : Pas assez de données numériques valides pour générer les violons.")
        return

    # Génération du Violin plot Matplotlib
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Distribution de {target_col} par championnat", fontsize=13, fontweight="bold")
    
    parts = ax.violinplot(data_violin, showmedians=True, showextrema=False)
    
    # Customisation des couleurs du corps des violons
    for pc in parts["bodies"]:
        pc.set_facecolor(couleurs_dict.get("purple", "#C678DD"))
        pc.set_alpha(0.6)
        
    # Customisation de la ligne de la médiane
    if "cmedians" in parts:
        parts["cmedians"].set_color(couleurs_dict.get("red", "#E06C75"))
        parts["cmedians"].set_linewidth(2)

    # Configuration des axes (Matplotlib indexe les violons à partir de 1)
    ax.set_xticks(range(1, len(labels_valides) + 1))
    ax.set_xticklabels(labels_valides, rotation=30, ha="right")
    ax.set_ylabel(f"{target_col} (M€)")
    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "feature_engineered_violin_league.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nViolin plot sauvegardé sous : {chemin_fichier}")

    # Affichage
    plt.show()







def analyser_profil_outliers(df, target_col, dossier_sauvegarde=None, couleurs_dict=None):
    """Filtre les joueurs d'élite (>= 100M€), affiche leur récapitulatif textuel

    et compare leur profil (Âge, xG, Blessures) avec le reste des joueurs via des boxplots.
    """
    print(f"Analyse des outliers et profils Spécifiques (cible : {target_col})")

    # Vérification de la présence de la cible
    if target_col not in df.columns:
        print(f"Annulé : La colonne cible '{target_col}' est absente du DataFrame.")
        return

    # Gestion des couleurs par défaut
    if couleurs_dict is None:
        couleurs_dict = {"blue": "#61AFEF", "red": "#E06C75"}

    seuil_elite = 100e6
    df_top = df[df[target_col] >= seuil_elite].copy()
    print(f"\nJoueurs avec {target_col} ≥ 100M€ : {len(df_top)} observations")

    # Si aucun joueur n'atteint ce seuil, on arrête proprement pour éviter des graphiques vides
    if df_top.empty:
        print("Aucun joueur ne dépasse le seuil des 100M€ dans ce jeu de données.")
        return

    # Extraction et affichage du tableau récapitulatif du top 20
    id_cols_show = ["player", "pos_simple", "league", "season_label", "age",
                    target_col, "xg", "Performance_Gls", "injury_days_total"]
    id_cols_show = [c for c in id_cols_show if c in df_top.columns]
    
    top_display = df_top[id_cols_show].sort_values(target_col, ascending=False).head(20)
    top_display[target_col] = (top_display[target_col] / 1e6).round(1)
    
    print("\nTop 20 des joueurs les plus chers")
    print(top_display.to_string(index=False))

    # Création des boxplots comparatifs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Profil des joueurs à {target_col} ≥ 100M€ vs reste", fontsize=13, fontweight="bold")

    # Utilisation d'une copie locale pour éviter d'ajouter de façon permanente la colonne 'group' au df global
    df_local = df.copy()
    df_local["group"] = np.where(df_local[target_col] >= seuil_elite, "≥100M€", "<100M€")

    # Liste des features à comparer
    features_analyse = [
        ("age", "Âge"),
        ("xg", "xG"),
        ("injury_days_total", "Jours blessés"),
    ]

    for ax, (feat, label) in zip(axes, features_analyse):
        if feat not in df_local.columns:
            ax.text(0.5, 0.5, f"Variable '{feat}'\nmanquante", ha="center", va="center", color="gray")
            ax.set_title(label)
            continue

        # Extraction des groupes
        groups = [df_local[df_local["group"] == g][feat].dropna().values for g in ["<100M€", "≥100M€"]]
        
        # Sécurité : vérifier que les deux groupes contiennent des données
        if len(groups[0]) > 0 and len(groups[1]) > 0:
            bp = ax.boxplot(groups, labels=["<100M€", "≥100M€"], patch_artist=True, boxprops=dict(alpha=0.6))
            
            # Application des couleurs du dictionnaire
            bp["boxes"][0].set_facecolor(couleurs_dict.get("blue", "#61AFEF"))
            bp["boxes"][1].set_facecolor(couleurs_dict.get("red", "#E06C75"))
        else:
            ax.text(0.5, 0.5, "Données insuffisantes\ndans l'un des groupes", ha="center", va="center", color="gray")

        ax.set_title(label)
        ax.set_ylabel(label)

    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "outliers_top100M_profil.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    # Affichage
    plt.show()







def analyser_residus_regression(df, target_col, dossier_sauvegarde=None, couleurs_dict=None):
    """Effectue une régression linéaire xG -> log(VM), calcule les résidus,

    détecte les joueurs sur/sous-évalués (> 2.5 std) et affiche les graphiques associés.
    """
    print(f"Analyse des résidus de régression (xG → log({target_col})) ===")

    # Vérifications initiales
    if "xg" not in df.columns:
        print("Annulé : La colonne 'xg' est absente du DataFrame.")
        return
    if target_col not in df.columns:
        print(f"Annulé : La colonne cible '{target_col}' est absente.")
        return

    # Gestion des couleurs par défaut
    if couleurs_dict is None:
        couleurs_dict = {"cyan": "#56B6C2", "red": "#E06C75", "green": "#98C379", "blue": "#61AFEF"}

    # Préparation et nettoyage des données
    colonnes_filtre = ["player", "pos_simple", "league", "age", "xg", target_col]
    colonnes_presentes = [c for c in colonnes_filtre if c in df.columns]
    
    df_res = df[colonnes_presentes].dropna().copy()
    df_res = df_res[np.isfinite(df_res["xg"]) & (df_res[target_col] > 0)]

    if len(df_res) < 5:
        print("Annulé : Pas assez d'observations valides pour entraîner la régression.")
        return

    # Entraînement de la régression linéaire
    X = df_res["xg"].values.reshape(-1, 1)
    y = np.log1p(df_res[target_col].values)

    reg = LinearRegression().fit(X, y)
    df_res["vm_pred_log"] = reg.predict(X)
    df_res["residual"] = y - df_res["vm_pred_log"]

    # 4. Identification des Outliers (> 2.5 écarts-types)
    std_r = df_res["residual"].std()
    df_res["outlier_type"] = "Normal"
    
    if std_r > 0:
        df_res.loc[df_res["residual"] >  2.5 * std_r, "outlier_type"] = "Sur-évalué"
        df_res.loc[df_res["residual"] < -2.5 * std_r, "outlier_type"] = "Sous-évalué"

    n_sur = (df_res["outlier_type"] == "Sur-évalué").sum()
    n_sous = (df_res["outlier_type"] == "Sous-évalué").sum()
    
    print(f"\nRégression xG → log({target_col})  |  R²={reg.score(X, y):.3f}")
    print(f"  Sur-évalués  (résidu > +2.5σ) : {n_sur}")
    print(f"  Sous-évalués (résidu < -2.5σ) : {n_sous}")

    # Mapping des couleurs
    color_map = {
        "Normal": couleurs_dict.get("cyan", "#56B6C2"), 
        "Sur-évalué": couleurs_dict.get("red", "#E06C75"), 
        "Sous-évalué": couleurs_dict.get("green", "#98C379")
    }

    # Construction de la figure (1 ligne, 2 colonnes)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Résidus de la régression xG → log({target_col})", fontsize=13, fontweight="bold")

    # Graphique 1 : Nuage de points xG vs Cible brute (M€)
    for group, color in color_map.items():
        sub = df_res[df_res["outlier_type"] == group]
        alpha = 0.12 if group == "Normal" else 0.7
        size = 8 if group == "Normal" else 20
        axes[0].scatter(
            sub["xg"], 
            sub[target_col].values / 1e6,
            alpha=alpha, 
            s=size, 
            color=color, 
            label=group, 
            rasterized=True
        )

    # Tracé de la courbe de tendance exponentielle convertie en Millions
    x_line = np.linspace(df_res["xg"].min(), df_res["xg"].max(), 100).reshape(-1, 1)
    axes[0].plot(
        x_line, 
        np.expm1(reg.predict(x_line)) / 1e6,
        color="black", 
        lw=1.5, 
        linestyle="--", 
        label="Régression"
    )
    axes[0].set_xlabel("xG")
    axes[0].set_ylabel(f"{target_col} (M€)")
    axes[0].set_title("xG vs VM")
    axes[0].legend()

    # Graphique 2 : Distribution des résidus
    axes[1].hist(
        df_res["residual"], 
        bins=60, 
        color=couleurs_dict.get("blue", "#61AFEF"), 
        edgecolor="white", 
        linewidth=0.3
    )
    axes[1].axvline(2.5 * std_r, color=couleurs_dict.get("red", "#E06C75"), linestyle="--", label="+2.5σ")
    axes[1].axvline(-2.5 * std_r, color=couleurs_dict.get("green", "#98C379"), linestyle="--", label="-2.5σ")
    axes[1].set_title("Distribution des résidus")
    axes[1].set_xlabel("Résidu (log scale)")
    axes[1].legend()

    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "outliers_residus.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    # Affichage
    plt.show()
    
    # On renvoie le DataFrame de l'analyse si l'utilisateur souhaite inspecter la liste des joueurs concernés
    return df_res






def afficher_top_anomalies_valeur(df_res, target_col):
    """Filtre et affiche les tableaux des 10 joueurs les plus sur-évalués

    et les 10 joueurs les plus sous-évalués en se basant sur les résidus de la régression.
    """
    print(f"Top joueurs atypiques (Régression vs {target_col}) ===")

    # Vérification de la présence de la colonne de classification des anomalies
    if "outlier_type" not in df_res.columns or "residual" not in df_res.columns:
        print(
            "Erreur : Le DataFrame fourni ne contient pas les colonnes 'outlier_type' ou 'residual'."
        )
        print(
            "Veuillez exécuter la fonction 'analyser_residus_regression' au préalable."
        )
        return

    # Configuration dynamique des colonnes à afficher
    cols_show = [
        "player",
        "pos_simple",
        "league",
        "age",
        "xg",
        target_col,
        "residual",
    ]
    cols_show = [c for c in cols_show if c in df_res.columns]

    # Top 10 sur-évalués
    print(f"\nTop 10 joueurs les plus SUR-évalués vs xG :")
    df_sur = df_res[df_res["outlier_type"] == "Sur-évalué"].copy()

    if not df_sur.empty:
        top_sur = df_sur.sort_values("residual", ascending=False).head(10)
        # Conversion d'affichage en M€ de manière sécurisée
        top_sur[target_col] = (top_sur[target_col] / 1e6).round(1)
        print(top_sur[cols_show].to_string(index=False))
    else:
        print(" Aucun joueur détecté comme sur-évalué au seuil défini.")

    # Top 10 sous-évalués
    print(f"\nTop 10 joueurs les plus SOUS-évalués vs xG :")
    df_sous = df_res[df_res["outlier_type"] == "Sous-évalué"].copy()

    if not df_sous.empty:
        top_sous = df_sous.sort_values("residual", ascending=True).head(10)
        # Conversion d'affichage en M€ de manière sécurisée
        top_sous[target_col] = (top_sous[target_col] / 1e6).round(1)
        print(top_sous[cols_show].to_string(index=False))
    else:
        print(" Aucun joueur détecté comme sous-évalué au seuil défini.")






def verifier_coherence_age_naissance(df):
    """Calcule l'âge théorique d'un joueur à partir de sa saison et de son année de naissance,

    puis valide la cohérence avec la colonne 'age' pour détecter d'éventuelles anomalies.
    """
    print("Contrôle de qualité et cohérence des données")

    # Vérification des colonnes requises
    colonnes_requises = ["season", "season_label", "born", "age"]
    manquantes = [c for c in colonnes_requises if c not in df.columns]
    
    if manquantes:
        print(f"Annulé : Colonnes sources manquantes dans le DataFrame : {', '.join(manquantes)}")
        return

    # Copie de travail locale pour ne pas polluer le DataFrame principal
    df_check = df[colonnes_requises].copy()

    # Nettoyage et conversion des types de base
    df_check["born"] = pd.to_numeric(df_check["born"], errors="coerce")
    df_check["age"] = pd.to_numeric(df_check["age"], errors="coerce")

    # Calcul de l'année de début de saison (Méthode 1 : depuis 'season')
    def extraire_annee_saison(x):
        x_str = str(x).strip()
        if len(x_str) == 4 and not x_str.startswith("20"):
            try:
                return int(x_str[:2]) + 2000
            except ValueError:
                return None
        else:
            try:
                return int(x_str[:4])
            except ValueError:
                return None

    df_check["season_start"] = df_check["season"].apply(extraire_annee_saison)

    # Calcul de l'année de début de saison (Méthode 2 : depuis 'season_label')
    # ex: "2022-2023" -> 2022.0
    df_check["season_start2"] = pd.to_numeric(df_check["season_label"].str[:4], errors="coerce")

    # Calcul de l'écart de cohérence
    df_check["age_calc"] = df_check["season_start2"] - df_check["born"]
    df_check["age_diff"] = (df_check["age_calc"] - df_check["age"]).abs()

    # Suppression des lignes où le calcul n'a pas pu aboutir (NaN)
    valid_diffs = df_check["age_diff"].dropna()
    total_valides = len(valid_diffs)
    total_df = len(df)

    # Affichage du rapport textuel
    print(f"\nCohérence born/age/season ({total_valides}/{total_df} lignes analysables) :")
    
    if total_valides > 0:
        print(f"  Différence nulle (exacte)      : {(valid_diffs == 0).sum():,}")
        print(f"  Différence ≤ 1 an              : {(valid_diffs <= 1).sum():,}")
        print(f"  Différence > 1 an (anomalie)   : {(valid_diffs > 1).sum():,}")
        print(f"  Différence > 5 ans             : {(valid_diffs > 5).sum():,}")
        
        # Alerte bonus si des anomalies graves sont détectées
        nb_anomalies_graves = (valid_diffs > 5).sum()
        if nb_anomalies_graves > 0:
            print(f"\n[ATTENTION] {nb_anomalies_graves} lignes présentent un écart critique (> 5 ans).")
            print("Vérifiez s'il n'y a pas un décalage de ligne ou un problème d'encodage sur ces profils.")
    else:
        print("  Aucune donnée exploitable (les conversions numériques ont toutes échoué).")






def verifier_coherence_jours_blessures(df):
    """Vérifie la cohérence entre la colonne globale 'injury_days_total'

    et la somme des différentes catégories de blessures par zone corporelle.
    """
    print("Contrôle de cohérence : jours de blessures")

    # Liste complète des composantes de blessures à vérifier
    injury_d_cols = [
        "injury_musculaire_nb_d",
        "injury_genou_nb_d",
        "injury_cheville_pied_nb_d",
        "injury_mollet_tibia_nb_d",
        "injury_dos_bassin_nb_d",
        "injury_trauma_severe_nb_d",
        "injury_medical_repos_nb_d",
        "injury_minor_unknown_nb_d",
    ]

    # Filtrage pour ne garder que les colonnes réellement présentes
    injury_d_cols = [c for c in injury_d_cols if c in df.columns]

    if not injury_d_cols:
        print("Annulé : Aucune des colonnes de sous-catégories de blessures n'est présente.")
        return

    if "injury_days_total" not in df.columns:
        print("Annulé : La colonne globale 'injury_days_total' est absente.")
        return

    # Copie de travail locale pour isoler les calculs
    df_check = df[injury_d_cols + ["injury_days_total"]].copy()

    # On force la conversion en numérique pour s'assurer qu'aucun type 'object' ou chaîne ne bloque le calcul
    for col in injury_d_cols + ["injury_days_total"]:
        df_check[col] = pd.to_numeric(df_check[col], errors="coerce")

    # Calcul de la somme recalculée et de la différence absolue
    # min_count=1 permet de renvoyer NaN si toutes les colonnes d'une ligne sont à NaN, plutôt que de renvoyer 0
    df_check["injury_sum_recomputed"] = df_check[injury_d_cols].sum(axis=1, min_count=1)
    df_check["injury_diff"] = (df_check["injury_sum_recomputed"] - df_check["injury_days_total"]).abs()

    # Élimination des lignes non comparables (NaN)
    valid_diffs = df_check["injury_diff"].dropna()
    total_valides = len(valid_diffs)
    total_df = len(df)

    # Affichage du rapport textuel de qualité
    print(f"\nCohérence injury_days_total vs somme des composantes ({total_valides}/{total_df} lignes analysées) :")
    
    if total_valides > 0:
        print(f"  Cohérents (diff = 0)           : {(valid_diffs == 0).sum():,}")
        print(f"  Diff ≤ 1 jour                  : {(valid_diffs <= 1).sum():,}")
        print(f"  Diff > 1 jour (anomalie)       : {(valid_diffs > 1).sum():,}")
        
        diff_max = valid_diffs.max()
        print(f"  Diff max                       : {diff_max:.0f} jours")
        
        # Alerte si des écarts majeurs sont constatés
        if (valid_diffs > 1).sum() > 0:
            print(f"\nDes incohérences ont été détectées. Cela peut provenir de blessures")
            print(" qui se chevauchent dans le temps ou de catégories non répertoriées ici.")
    else:
        print("  Aucune donnée comparable disponible (valeurs manquantes ou non numériques).")





def analyser_evolution_temporelle_saison(df, target_col="valeur_marchande", dossier_sauvegarde=None, couleurs_dict=None):
    """Calcule l'évolution de la valeur marchande (médiane et moyenne) ainsi que le volume de joueurs

    par saison, affiche le bilan textuel et génère les graphiques d'évolution.
    """
    print(f"Analyse de l'évolution temporelle par Saison (cible : {target_col}) ===")

    # Vérifications initiales des colonnes
    if "season_label" not in df.columns:
        print("Annulé : La colonne 'season_label' est absente du DataFrame.")
        return
    if target_col not in df.columns:
        print(f"Annulé : La colonne cible '{target_col}' est absente.")
        return

    # Gestion des couleurs par défaut
    if couleurs_dict is None:
        couleurs_dict = {"blue": "#61AFEF", "red": "#E06C75", "green": "#98C379"}

    # Agrégation et calcul des statistiques par saison
    # On filtre les valeurs manquantes sur la cible pour avoir un décompte (count) précis des joueurs évalués
    df_clean = df.dropna(subset=[target_col])
    
    if df_clean.empty:
        print("Annulé : Aucune donnée valide après suppression des valeurs manquantes sur la cible.")
        return

    vm_season = df_clean.groupby("season_label")[target_col].agg(
        mediane="median", 
        moyenne="mean", 
        count="count"
    ).copy()

    # Conversion des valeurs monétaires en Millions d'Euros (uniquement sur la médiane et la moyenne)
    vm_season["mediane"] = vm_season["mediane"] / 1e6
    vm_season["moyenne"] = vm_season["moyenne"] / 1e6

    # Renommage explicite des colonnes pour l'affichage et les graphiques
    vm_season.columns = ["Médiane (M€)", "Moyenne (M€)", "Nb joueurs"]
    
    # Tri par l'index (saisons chronologiques) pour s'assurer du bon sens des lignes de tendance
    vm_season = vm_season.sort_index()

    # Affichage du bilan textuel
    print(f"\nÉvolution VM par saison :")
    print(vm_season.to_string())

    # Configuration de la figure (1 ligne, 2 colonnes)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Évolution temporelle", fontsize=13, fontweight="bold")

    seasons = vm_season.index.tolist()

    # Graphique 1 : Courbes d'évolution (Médiane vs Moyenne)
    axes[0].plot(
        seasons, 
        vm_season["Médiane (M€)"], 
        marker="o", 
        color=couleurs_dict.get("blue", "#61AFEF"),  
        label="Médiane", 
        lw=2
    )
    axes[0].plot(
        seasons, 
        vm_season["Moyenne (M€)"], 
        marker="s", 
        color=couleurs_dict.get("red", "#E06C75"),   
        label="Moyenne", 
        lw=2, 
        linestyle="--"
    )
    axes[0].set_title(f"Moyenne et Médiane de {target_col} par saison")
    axes[0].set_ylabel("VM (M€)")
    axes[0].legend()
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Graphique 2 : Histogramme du volume de joueurs par saison
    axes[1].bar(
        seasons, 
        vm_season["Nb joueurs"], 
        color=couleurs_dict.get("green", "#98C379"), 
        edgecolor="white"
    )
    axes[1].set_title("Nombre de joueurs par saison")
    axes[1].set_ylabel("Count")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()

    # Gestion de la sauvegarde
    if dossier_sauvegarde is not None:
        chemin_dossier = Path(dossier_sauvegarde)
        chemin_dossier.mkdir(exist_ok=True)
        chemin_fichier = chemin_dossier / "evolution_temporelle.png"
        
        plt.savefig(chemin_fichier)
        print(f"\nGraphique sauvegardé sous : {chemin_fichier}")

    # Affichage
    plt.show()
    
    # Optionnel : Retourne le tableau croisé récapitulatif
    return vm_season