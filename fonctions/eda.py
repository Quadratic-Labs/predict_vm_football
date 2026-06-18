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
dossier_sortie = Path("../outputs/eda")

# On force la création du dossier s'il n'existe pas encore
dossier_sortie.mkdir(exist_ok=True)


def analyser_types(df):
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


def analyser_variable_cible(df, target_col="valeur_marchande", dossier_sauvegarde=None):
    """Affiche les statistiques descriptives (skewness, kurtosis) de la variable cible

    et génère trois graphiques : distribution brute, log-transformée et boxplot.
    """
    print(f"Analyse de la variable cible : {target_col}")

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
    Gère la colonne 'league' encodée en One-Hot (league_*).
    """
    print("Analyse des variables catégorielles")

    # Identification dynamique des colonnes de ligues encodées
    cols_league = [c for c in df.columns if c.startswith("league_")]

    # Affichage textuel des value_counts pour les variables clés restantes
    key_cat = ["pos", "season"]
    key_cat = [c for c in key_cat if c in df.columns]

    for col in key_cat:
        vc = df[col].value_counts()
        print(f"\n{col} ({vc.shape[0]} modalités) :")
        print(vc.head(20).to_string())

    # Affichage textuel pour les ligues encodées (somme des 1)
    if cols_league:
        print(f"\nleague ({len(cols_league)} modalités encodées) :")
        # On fait la somme des 1 pour chaque colonne et on nettoie le nom pour l'affichage
        vc_league = pd.Series({c.replace("league_", ""): df[c].sum() for c in cols_league})
        vc_league = vc_league.sort_values(ascending=False)
        print(vc_league.to_string())

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

    # Graphique : Répartition Poste
    if "pos" in df.columns:
        vc = df["pos"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        vc.plot(kind="bar", ax=ax, color="#61AFEF")
        ax.set_title("Répartition : pos")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        
        if sauvegarder:
            plt.savefig(chemin_dossier / "pos.png")
            print("pos.png sauvegardé")
        plt.show()

    # Graphique : Répartition Championnat (version One-Hot)
    if cols_league:
        # On récupère le total de joueurs par ligue
        league_counts = pd.Series({c.replace("league_", ""): df[c].sum() for c in cols_league}).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        league_counts.plot(kind="bar", ax=ax, color="#61AFEF")
        ax.set_title("Répartition : league (One-Hot)")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        
        if sauvegarder:
            plt.savefig(chemin_dossier / "league.png")
            print("league.png sauvegardé")
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
            ax.set_title("Valeur Marchande par poste (Trié par médiane décroissante)")
            ax.set_ylabel("VM (M€)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            
            if sauvegarder:
                plt.savefig(chemin_dossier / "vm_par_pos.png")
                print("vm_par_pos.png sauvegardé")
            plt.show()

    # Boxplot : Variable Cible par Championnat (reconstitué depuis le One-Hot)
    if cols_league:
        groups_l = []
        labels_l = []
        medians_l = []

        # Pour chaque colonne de ligue, on extrait les valeurs de la cible là où la ligue vaut 1
        for col in cols_league:
            values = df.loc[df[col] == 1, target_col].dropna().values / 1e6
            if len(values) > 0:
                groups_l.append(values)
                labels_l.append(col.replace("league_", ""))
                medians_l.append(pd.Series(values).median())
        
        if groups_l:
            # Tri des ligues par leur médiane décroissante
            sorted_indices = pd.Series(medians_l).sort_values(ascending=False).index
            groups_l = [groups_l[i] for i in sorted_indices]
            labels_l = [labels_l[i] for i in sorted_indices]

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.boxplot(groups_l, labels=labels_l, patch_artist=True,
                       boxprops=dict(facecolor="#C678DD", alpha=0.5))
            ax.set_title("Valeur Marchande par championnat (Trié par médiane décroissante)")
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
    if "position" not in df.columns:
        print("Annulé : La colonne 'position' est absente du DataFrame.")
        return

    pos_order = ["Goalkeeper", "Defender", "Midfield", "Attack"]
    df_pos = df[df["position"].isin(pos_order)].copy()

    if df_pos.empty:
        print("Annulé : Aucun joueur ne correspond aux postes (Goalkeeper, Defender, Midfield, Attack) dans la colonne 'position'.")
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
        df_pos.groupby("position")[list(profile_cols.values())]
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
        vm_by_pos = df_pos.groupby("position")[target_col].median().reindex(pos_order) / 1e6
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


def analyser_ages_par_poste(df, dossier_sauvegarde=None, liste_couleurs=None):
    """Génère un histogramme superposé de la distribution des âges pour chaque poste (GK, DF, MF, FW)

    et affiche les âges médians sous forme textuelle.
    """
    print("Analyse de la distribution des âges par poste")

    # Vérifications initiales
    if "position" not in df.columns:
        print("Annulé : La colonne 'position' est absente du DataFrame.")
        return
    if "age" not in df.columns:
        print("Annulé : La colonne 'age' est absente du DataFrame.")
        return

    pos_order = ["Goalkeeper", "Defender", "Midfield", "Attack"]
    df_pos = df[df["position"].isin(pos_order)].copy()

    if df_pos.empty:
        print("Annulé : Aucun joueur trouvé pour les postes (Goalkeeper, Defender, Midfield, Attack).")
        return

    # Gestion de la palette de couleurs par défaut (One Dark Pro style comme tes codes précédents)
    if liste_couleurs is None:
        liste_couleurs = ["#61AFEF", "#98C379", "#E5C07B", "#E06C75"]  # bleu, vert, jaune, rouge

    # Construction de l'histogramme superposé
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Distribution des âges par poste", fontsize=13, fontweight="bold")

    for pos, color in zip(pos_order, liste_couleurs):
        ages = df_pos[df_pos["position"] == pos]["age"].dropna()
        
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
    stats_mediane = df_pos.groupby("position")["age"].median().reindex(pos_order).round(1)
    print(stats_mediane.to_string())


def analyser_heatmap_league_poste(df, target_col, dossier_sauvegarde=None):
    """Génère une heatmap croisant les championnats (colonnes league_*) et les postes (position)
    pour afficher la valeur marchande médiane (en M€) de chaque segment.
    """
    print("Analyse croisée : VM médiane par championnat et par poste")

    # Vérifications initiales des colonnes
    if "position" not in df.columns:
        print("Annulé : La colonne 'position' est absente du DataFrame.")
        return
        
    if target_col not in df.columns:
        print(f"Annulé : La colonne cible '{target_col}' est absente.")
        return

    # Identification dynamique des colonnes de ligues encodées
    cols_league = [c for c in df.columns if c.startswith("league_")]
    if not cols_league:
        print("Annulé : Aucune colonne commençant par 'league_' n'a été trouvée.")
        return

    pos_order = ["Goalkeeper", "Defender", "Midfield", "Attack"]
    
    # Filtrage des postes valides à l'avance
    df_filtrer = df[df["position"].isin(pos_order)]
    if df_filtrer.empty:
        print("Annulé : Aucun joueur trouvé pour les postes (Goalkeeper, Defender, Midfield, Attack).")
        return

    rows_pivot = {}

    for col in cols_league:
        # On extrait le nom propre de la ligue (ex: "league_ESP-La Liga" -> "ESP-La Liga")
        nom_ligue = col.replace("league_", "")
        
        # On filtre le DataFrame pour n'avoir que les joueurs de cette ligue spécifique
        df_ligue = df_filtrer[df_filtrer[col] == 1]
        
        # On calcule la médiane de la cible par poste pour cette ligue
        # .reindex(pos_order) garantit que tous les postes existent (quitte à mettre du NaN)
        medians_par_poste = df_ligue.groupby("position")[target_col].median().reindex(pos_order)
        
        # On stocke le résultat (converti en Millions d'euros)
        rows_pivot[nom_ligue] = medians_par_poste / 1e6

    # Création du DataFrame final pour la heatmap
    pivot = pd.DataFrame.from_dict(rows_pivot, orient='index')

    # Sécurité : Si un championnat n'a aucun joueur à un poste donné, on remplace le NaN par 0
    pivot = pivot.fillna(0)

    # Si toutes les valeurs du pivot sont à 0, on arrête pour éviter un graphique vide
    if pivot.sum().sum() == 0:
        print("Annulé : Le tableau croisé ne contient que des zéros ou des données vides.")
        return

    # Optionnel : Trier les lignes (championnats) par la valeur médiane globale pour un plus joli visuel
    pivot = pivot.loc[pivot.median(axis=1).sort_values(ascending=False).index]

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
    de la variable cible (en M€) à travers les différents championnats (colonnes league_*).
    """
    print("Analyse de la distribution par championnat")

    # Vérifications initiales des colonnes
    if target_col not in df.columns:
        print(f"Annulé : La colonne cible '{target_col}' est absente.")
        return

    # Identification dynamique des colonnes de ligues encodées
    cols_league = [c for c in df.columns if c.startswith("league_")]
    if not cols_league:
        print("Annulé : Aucune colonne commençant par 'league_' n'a été trouvée.")
        return

    # Gestion des couleurs par défaut
    if couleurs_dict is None:
        couleurs_dict = {"purple": "#C678DD", "red": "#E06C75"}

    leagues_data = {}
    medians_l = {}

    for col in cols_league:
        nom_ligue = col.replace("league_", "")
        
        # Sélection des valeurs de la cible pour les joueurs de cette ligue (valeur == 1)
        values = df.loc[df[col] == 1, target_col].dropna().values / 1e6
        
        # Sécurité : On s'assure qu'il y a assez de points pour construire un violon (minimum 2 requis)
        if len(values) >= 2:
            leagues_data[nom_ligue] = values
            medians_l[nom_ligue] = pd.Series(values).median()

    if not leagues_data:
        print("Annulé : Pas assez de données numériques valides pour générer les violons.")
        return

    # Calcul de l'ordre d'affichage basé sur la médiane décroissante
    order_vm = pd.Series(medians_l).sort_values(ascending=False).index

    # Extraction ordonnée des données et des labels pour le graphique
    data_violin = [leagues_data[l] for l in order_vm]
    labels_valides = list(order_vm)

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
    id_cols_show = ["player", "position","age",
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
    print(f"Analyse de l'évolution temporelle par Saison (cible : {target_col})")

    # Vérifications initiales des colonnes
    if "season_year" not in df.columns:
        print("Annulé : La colonne 'season_year' est absente du DataFrame.")
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

    vm_season = df_clean.groupby("season_year")[target_col].agg(
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


def analyser_cycle_vie_financier(df):
    """Nettoie les postes des joueurs, filtre les catégories majeures,

    et génère le graphique bi-axe du cycle de vie financier selon l'âge.
    """
    print("Démarrage de l'analyse du cycle de vie financier...")

    # Copie locale pour éviter de modifier le DataFrame original en place
    df_clean = df.copy()

    # Vérification des colonnes requises
    cols_requises = ["age", "market_value_in_eur", "position"]
    manquantes = [c for c in cols_requises if c not in df_clean.columns]
    if manquantes:
        print(f"Annulé : Les colonnes suivantes sont absentes : {manquantes}")
        return df_clean

    # Harmonisation des postes (anglais / français / abréviations)
    mapping_postes = {
        "FW": "Attack",
        "MF": "Midfield",
        "DF": "Defender",
        "GK": "Goalkeeper",
        "Attaquant": "Attack",
        "Milieu": "Midfield",
        "Défenseur": "Defender",
        "Gardien": "Goalkeeper",
    }
    df_clean["position"] = (
        df_clean["position"].map(mapping_postes).fillna(df_clean["position"])
    )

    # Filtrage strict sur les 4 catégories majeures
    postes_valides = ["Attack", "Midfield", "Defender", "Goalkeeper"]
    df_clean = df_clean[df_clean["position"].isin(postes_valides)]

    print(
        f"{len(df_clean)} observations valides trouvées pour l'analyse graphique."
    )

    # Agrégation des données par âge
    age_profile = (
        df_clean.groupby("age")["market_value_in_eur"]
        .agg(["median", "count"])
        .reset_index()
    )

    # Sécurité : On retire les âges extrêmes non représentatifs (moins de 10 joueurs)
    age_profile = age_profile[age_profile["count"] >= 10]

    if age_profile.empty:
        print(
            "Annulé : Pas assez de volume de données après filtrage des âges pour générer le graphique."
        )
        return df_clean

    # Génération du graphique bi-axe (Seaborn / Matplotlib)
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(11, 5))

    # Axe principal : Courbe de la Valeur Marchande Médiane (convertie en M€)
    sns.lineplot(
        data=age_profile,
        x="age",
        y=age_profile["median"] / 1e6,
        marker="o",
        linewidth=3,
        color="#1f77b4",
        ax=ax1,
        label="Valeur Marchande Médiane",
    )
    ax1.set_ylabel("Valeur Médiane (M€)", color="#1f77b4", fontsize=12)
    ax1.set_xlabel("Âge au moment de la saison", fontsize=12)

    # Axe secondaire : Volume de données (Histogramme lissé en arrière-plan)
    ax2 = ax1.twinx()
    ax2.fill_between(
        age_profile["age"],
        age_profile["count"],
        alpha=0.1,
        color="gray",
        step="mid",
        label="Nombre d'observations",
    )
    ax2.set_ylabel("Volume de données", color="gray", fontsize=12)
    ax2.grid(False)

    # Détection dynamique et traçage de la ligne du pic financier
    age_or = age_profile.loc[age_profile["median"].idxmax(), "age"]
    ax1.axvline(
        x=age_or,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Pic de valeur moyen ({int(age_or)} ans)",
    )

    # Titre général
    ax1.set_title(
        "Cycle de vie financier d'un joueur", fontsize=14, fontweight="bold"
    )

    # Fusion propre des légendes des deux axes distincts
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()

    print()
    print("Graphique généré avec succès.")

    # On retourne le DataFrame nettoyé au cas où tu en as besoin pour la suite
    return df_clean


def analyser_distribution_prix_ligue_poste(df):
    """Reconstitue la colonne ligue à partir du One-Hot, trie les championnats

    par valeur marchande médiane décroissante et affiche le Boxplot global.
    """
    print("Démarrage de l'analyse de distribution des prix...")

    # Identification dynamique des colonnes One-Hot de ligues
    cols_league = [c for c in df.columns if c.startswith("league_")]

    if not cols_league:
        print("Annulé : Aucune colonne de type 'league_' trouvée dans le DataFrame.")
        return

    if "market_value_in_eur" not in df.columns or "position" not in df.columns:
        print("Annulé : La colonne 'market_value_in_eur' ou 'position' est absente.")
        return

    # Création d'un DataFrame temporaire de travail pour le plot
    df_plot = df.copy()

    # Reconstitution de la colonne 'league' textuelle à partir du One-Hot
    df_plot["league"] = (
        df_plot[cols_league].idxmax(axis=1).str.replace("league_", "")
    )

    # Tri des championnats par valeur marchande médiane décroissante
    league_order = (
        df_plot.groupby("league")["market_value_in_eur"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    print(f"{len(league_order)} championnats détectés et ordonnés pour l'affichage.")

    # Configuration graphique et génération du Boxplot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))

    sns.boxplot(
        data=df_plot,
        x="league",
        y="market_value_in_eur",
        hue="position",
        hue_order=["Attack", "Midfield", "Defender", "Goalkeeper"],
        order=league_order,
        palette="Set2",
    )

    # Échelle logarithmique indispensable pour écraser la dispersion des prix
    plt.yscale("log")

    plt.title(
        "Distribution des prix par Championnat et Poste (4 Catégories Majeures)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Championnat")
    plt.ylabel("Valeur Marchande (EUR - Échelle Log)")
    plt.xticks(rotation=30, ha="right")

    # Placement propre de la légende à l'extérieur pour éviter l'enchevêtrement
    plt.legend(
        title="Poste Nettoyé", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    
    plt.tight_layout()
    plt.show()

    print()
    print("Boxplot généré avec succès.")


def detecter_top_colinearites(df, top_n=15):
    """Calcule la matrice de corrélation et extrait les paires de variables

    présentant les plus fortes colinéarités (en valeur absolue), en excluant
    toutes les variables normalisées finissant par '_nor'.
    """
    print(
        f"Extraction du Top {top_n} des colinéarités (hors variables normalisées)..."
    )

    # Sélection uniquement des variables numériques (continues et binaires)
    df_num = df.select_dtypes(include=[np.number]).copy()

    # Exclusion des variables normalisées
    cols_sans_nor = [c for c in df_num.columns if not c.endswith("_nor")]
    df_num = df_num[cols_sans_nor]

    # Sécurité : On supprime les colonnes qui n'ont aucune variabilité (écart-type nul)
    df_num = df_num.loc[:, df_num.std() > 0]

    # Calcul de la matrice de corrélation
    corr_matrix = df_num.corr(method="pearson")

    # On ne garde que le triangle supérieur de la matrice pour éviter les doublons
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # On "aplatit" (unstack) la matrice pour passer à une liste de paires
    corr_pairs = upper_tri.unstack().dropna()

    # Création d'un DataFrame propre pour le tri
    df_colin = pd.DataFrame(corr_pairs).reset_index()
    df_colin.columns = ["Variable_A", "Variable_B", "Correlation"]

    # Ajout de la valeur absolue pour capturer les fortes corrélations négatives
    df_colin["Abs_Correlation"] = df_colin["Correlation"].abs()

    # Tri par corrélation absolue décroissante et sélection du Top N
    top_colin = df_colin.sort_values(by="Abs_Correlation", ascending=False).head(
        top_n
    )

    # Nettoyage de l'affichage
    top_colin = top_colin.drop(columns=["Abs_Correlation"]).reset_index(
        drop=True
    )

    # Formatage pour un affichage élégant des coefficients
    top_colin["Correlation"] = top_colin["Correlation"].round(3)

    return top_colin


def analyser_variables_les_plus_explicatives(
    df, target_col="market_value_in_eur", top_n=15, afficher_nom_technique=False, uniquement_gardiens=False
):
    """Calcule et affiche les variables les plus corrélées à la cible en brut

    (sans Log), en les regroupant par famille avec une palette de couleurs professionnelle.
    """
    print(f"Analyse des variables les plus explicatives pour : {target_col}")

    # Filtrage spécifique pour les gardiens si demandé
    if uniquement_gardiens:
        if "pos_GK" in df.columns:
            df = df[df["pos_GK"] == 1].copy()
            print("Filtrage appliqué : Analyse exclusive des Gardiens de but.")
        else:
            print("Attention : Impossible de filtrer les gardiens (colonne 'pos_GK' absente).")

    if target_col not in df.columns:
        print(f"Erreur : La colonne cible '{target_col}' est absente.")
        return

    # Dictionnaire métier
    dict_variables = {
        # Famille : Identité, âge et physique
        "player": ("Nom du joueur", "Identité et physique"),
        "team": ("Club du joueur", "Identité et physique"),
        "nation": ("Nationalité", "Identité et physique"),
        "age": ("Âge du joueur", "Identité et physique"),
        "height_in_cm": ("Taille (en cm)", "Identité et physique"),
        "position": ("Poste général", "Identité et physique"),
        "foot_both": ("Ambidextre (0/1)", "Identité et physique"),
        "foot_left": ("Gaucher (0/1)", "Identité et physique"),
        "foot_right": ("Droitier (0/1)", "Identité et physique"),
        # Famille : Postes et sub-positions
        "pos_DF": ("Défenseur (0/1)", "Postes"),
        "pos_FW": ("Attaquant (0/1)", "Postes"),
        "pos_GK": ("Gardien de but (0/1)", "Postes"),
        "pos_MF": ("Milieu de terrain (0/1)", "Postes"),
        "sub_position_Attacking Midfield": (
            "Milieu offensif (0/1)",
            "Postes",
        ),
        "sub_position_Central Midfield": (
            "Milieu central (0/1)",
            "Postes",
        ),
        "sub_position_Centre-Back": (
            "Défenseur central (0/1)",
            "Postes",
        ),
        "sub_position_Centre-Forward": (
            "Avant-centre (0/1)",
            "Postes",
        ),
        "sub_position_Defensive Midfield": (
            "Milieu défensif (0/1)",
            "Postes",
        ),
        "sub_position_Goalkeeper": (
            "Gardien (Sous-poste 0/1)",
            "Postes",
        ),
        "sub_position_Left Midfield": (
            "Milieu gauche (0/1)",
            "Postes",
        ),
        "sub_position_Left Winger": (
            "Ailier gauche (0/1)",
            "Postes",
        ),
        "sub_position_Left-Back": (
            "Latéral gauche (0/1)",
            "Postes",
        ),
        "sub_position_Right Midfield": (
            "Milieu droit (0/1)",
            "Postes",
        ),
        "sub_position_Right Winger": (
            "Ailier droit (0/1)",
            "Postes",
        ),
        "sub_position_Right-Back": (
            "Latéral droit (0/1)",
            "Postes",
        ),
        "sub_position_Second Striker": (
            "Neuf et demi / Second attaquant (0/1)",
            "Postes",
        ),
        # Famille : Championnats et contexte international
        "league_ENG-Premier League": (
            "Évolue en Premier League (0/1)",
            "Ligues et contexte international",
        ),
        "league_ESP-La Liga": (
            "Évolue en La Liga (0/1)",
            "Ligues et contexte international",
        ),
        "league_FRA-Ligue 1": (
            "Évolue en Ligue 1 (0/1)",
            "Ligues et contexte international",
        ),
        "league_GER-Bundesliga": (
            "Évolue en Bundesliga (0/1)",
            "Ligues et contexte international",
        ),
        "league_ITA-Serie A": (
            "Évolue en Serie A (0/1)",
            "Ligues et contexte international",
        ),
        "classement_FIFA_1": (
            "Sélection nationale Rang FIFA : 1",
            "Ligues et contexte international",
        ),
        "classement_FIFA_2": (
            "Sélection nationale Rang FIFA : 2",
            "Ligues et contexte international",
        ),
        "classement_FIFA_3": (
            "Sélection nationale Rang FIFA : 3",
            "Ligues et contexte international",
        ),
        "classement_FIFA_4": (
            "Sélection nationale Rang FIFA : 4",
            "Ligues et contexte international",
        ),
        "classement_FIFA_5": (
            "Sélection nationale Rang FIFA : 5",
            "Ligues et contexte international",
        ),
        "classement_FIFA_6": (
            "Sélection nationale Rang FIFA : 6",
            "Ligues et contexte international",
        ),
        "classement_FIFA_7": (
            "Sélection nationale Rang FIFA : 7",
            "Ligues et contexte international",
        ),
        "classement_FIFA_8": (
            "Sélection nationale Rang FIFA : 8",
            "Ligues et contexte international",
        ),
        "classement_FIFA_9": (
            "Sélection nationale Rang FIFA : 9",
            "Ligues et contexte international",
        ),
        "classement_FIFA_10": (
            "Sélection nationale Rang FIFA : 10",
            "Ligues et contexte international",
        ),
        # Famille : Temps de jeu, contrat et chronologie
        "season_year": ("Année de la saison", "Temps de jeu & Contrat"),
        "contrat_jours_restants": (
            "Jours de contrat restants",
            "Temps de jeu et contrat",
        ),
        "Playing Time_MP": ("Matchs disputés", "Temps de jeu et contrat"),
        "Playing Time_Starts": (
            "Titularisations",
            "Temps de jeu et contrat",
        ),
        "Playing Time_90s": (
            "Nombre de 90 minutes complétées",
            "Temps de jeu et contrat",
        ),
        "Playing Time_Mn/MP": (
            "Minutes jouées par match disputé",
            "Temps de jeu et contrat",
        ),
        "Starts_Mn/Start": (
            "Minutes par titularisation",
            "Temps de jeu et contrat",
        ),
        "Starts_Compl": (
            "Matchs commencés et terminés en entier",
            "Temps de jeu et contrat",
        ),
        "Subs_Subs": (
            "Entrées en cours de match (Remplaçant)",
            "Temps de jeu et contrat",
        ),
        "Subs_Mn/Sub": ("Minutes par entrée en jeu", "Temps de jeu et contrat"),
        "Subs_unSub": (
            "Matchs passés sur le banc sans entrer",
            "Temps de jeu et contrat",
        ),
        # Famille : Performance Offensive (Volume et efficacité)
        "Performance_Gls": ("Buts marqués", "Performance offensive"),
        "Performance_Ast": ("Passes décisives", "Performance offensive"),
        "Performance_G-PK": ("Buts hors pénaltys", "Performance offensive"),
        "Performance_PK": ("Pénaltys marqués", "Performance offensive"),
        "Performance_PKatt": ("Pénaltys tentés", "Performance offensive"),
        "Standard_Sh": ("Tirs totaux effectués", "Performance offensive"),
        "Standard_SoT": ("Tirs cadrés", "Performance offensive"),
        "Standard_SoT%": (
            "Pourcentage de tirs cadrés",
            "Performance offensive",
        ),
        "Standard_Sh/90": (
            "Tirs effectués par 90 min",
            "Performance offensive",
        ),
        "Standard_SoT/90": ("Tirs cadrés par 90 min", "Performance offensive"),
        "Standard_G/Sh": ("Buts par tir tenté", "Performance offensive"),
        "Standard_G/SoT": ("Buts par tir cadré", "Performance offensive"),
        "Performance_Off": ("Hors-jeux signalés", "Performance offensive"),
        "Performance_Crs": ("Centres vers la surface", "Performance offensive"),
        "Performance_PKwon": ("Pénaltys obtenus", "Performance offensive"),
        "Per 90 Minutes_Gls": ("Buts par 90 min", "Performance offensive"),
        "Per 90 Minutes_Ast": (
            "Passes décisives par 90 min",
            "Performance offensive",
        ),
        "Per 90 Minutes_G+A": (
            "Buts + Assists par 90 min",
            "Performance offensive",
        ),
        "Performance_Fld": ("Fautes subies", "Performance offensive"),
        # Famille : Statistiques avancées (xG, xA, Création)
        "xg": ("Expected Goals (xG)", "Statistiques avancées"),
        "xa": ("Expected Assists (xa)", "Statistiques avancées"),
        "xg_chain": (
            "Chaîne Expected Goals (xG Chain)",
            "Statistiques avancées",
        ),
        "xg_buildup": (
            "Construction Expected Goals",
            "Statistiques avancées",
        ),
        # Famille : Discipline et performance défensive
        "Performance_CrdY": ("Cartons jaunes reçus", "Discipline et défense"),
        "Performance_CrdR": ("Cartons rouges reçus", "Discipline et défense"),
        "Performance_2CrdY": (
            "Expulsions suite à 2 jaunes",
            "Discipline et défense",
        ),
        "Performance_Fls": ("Fautes commises", "Discipline et défense"),
        "Performance_Int": ("Interceptions de passes", "Discipline et défense"),
        "Performance_TklW": ("Tacles réussis", "Discipline et défense"),
        "Performance_PKcon": ("Pénaltys concédés", "Discipline et défense"),
        "Performance_OG": (
            "Buts contre son camp (OG)",
            "Discipline et défense",
        ),
        # Famille : Collectif et succès équipe
        "Team Success_PPM": (
            "Points par match glanés par l'équipe",
            "Succès équipe et collectif",
        ),
        "Team Success_onG": (
            "Buts marqués par l'équipe (si présent)",
            "Succès équipe et collectif",
        ),
        "Team Success_onGA": (
            "Buts encaissés par l'équipe (si présent)",
            "Succès équipe et collectif",
        ),
        "Team Success_On-Off": (
            "Impact On-Off de la présence du joueur",
            "Succès équipe et collectif",
        ),
        # Famille : Spécifique Gardien de but
        "Performance_GA": ("Buts encaissés", "Spécifique gardien"),
        "Performance_GA90": (
            "Buts encaissés par 90 min",
            "Spécifique gardien",
        ),
        "Performance_Saves": ("Arrêts effectués", "Spécifique gardien"),
        "Performance_Save%": ("Pourcentage d'arrêts", "Spécifique gardien"),
        "Performance_W": (
            "Victoires de l'équipe (si présent)",
            "Spécifique gardien",
        ),
        "Performance_D": (
            "Matchs nuls de l'équipe (si présent)",
            "Spécifique gardien",
        ),
        "Performance_L": (
            "Défaites de l'équipe (si présent)",
            "Spécifique gardien",
        ),
        "Performance_CS": (
            "Clean Sheets (Matchs sans but)",
            "Spécifique gardien",
        ),
        "Performance_CS%": (
            "Pourcentage de Clean Sheets",
            "Spécifique gardien",
        ),
        "Penalty Kicks_PKA": ("Pénaltys encaissés", "Spécifique gardien"),
        "Penalty Kicks_PKsv": ("Pénaltys arrêtés", "Spécifique gardien"),
        "Penalty Kicks_PKm": (
            "Pénaltys ratés par l'adversaire",
            "Spécifique gardien",
        ),
        "Penalty Kicks_Save%": (
            "Pourcentage de pénaltys arrêtés",
            "Spécifique gardien",
        ),
        # Famille : Historique médical (Blessures)
        "injury_nb_total": ("Nombre total de blessures", "Historique Médical"),
        "injury_days_total": (
            "Total des jours d'absence",
            "Historique médical",
        ),
        "injury_matches_max_single": (
            "Max de matchs manqués sur une blessure",
            "Historique médical",
        ),
        "injury_musculaire": (
            "Blessure musculaire (Présence 0/1)",
            "Historique médical",
        ),
        "injury_musculaire_nb_d": (
            "Jours d'absence - Muscle",
            "Historique médical",
        ),
        "injury_musculaire_nb_m": (
            "Matchs manqués - Muscle",
            "Historique médical",
        ),
        "injury_genou": (
            "Blessure au genou (Présence 0/1)",
            "Historique médical",
        ),
        "injury_genou_nb_d": ("Jours d'absence - Genou", "Historique médical"),
        "injury_genou_nb_m": ("Matchs manqués - Genou", "Historique médical"),
        "injury_cheville_pied": (
            "Blessure cheville/pied (Présence 0/1)",
            "Historique médical",
        ),
        "injury_cheville_pied_nb_d": (
            "Jours d'absence - Cheville/Pied",
            "Historique médical",
        ),
        "injury_cheville_pied_nb_m": (
            "Matchs manqués - Cheville/Pied",
            "Historique médical",
        ),
        "injury_mollet_tibia": (
            "Blessure mollet/tibia (Présence 0/1)",
            "Historique médical",
        ),
        "injury_mollet_tibia_nb_d": (
            "Jours d'absence - Mollet/Tibia",
            "Historique médical",
        ),
        "injury_mollet_tibia_nb_m": (
            "Matchs manqués - Mollet/Tibia",
            "Historique médical",
        ),
        "injury_dos_bassin": (
            "Blessure dos/bassin (Présence 0/1)",
            "Historique médical",
        ),
        "injury_dos_bassin_nb_d": (
            "Jours d'absence - Dos/Bassin",
            "Historique médical",
        ),
        "injury_dos_bassin_nb_m": (
            "Matchs manqués - Dos/Bassin",
            "Historique médical",
        ),
        "injury_trauma_severe": (
            "Traumatisme sévère/Opération (Présence 0/1)",
            "Historique médical",
        ),
        "injury_trauma_severe_nb_d": (
            "Jours d'absence - Traumatisme Sévère",
            "Historique médical",
        ),
        "injury_trauma_severe_nb_m": (
            "Matchs manqués - Traumatisme Sévère",
            "Historique médical",
        ),
        "injury_medical_repos": (
            "Maladie / Repos obligatoire (Présence 0/1)",
            "Historique médical",
        ),
        "injury_medical_repos_nb_d": (
            "Jours d'absence - Maladie/Repos",
            "Historique médical",
        ),
        "injury_medical_repos_nb_m": (
            "Matchs manqués - Maladie/Repos",
            "Historique médical",
        ),
        "injury_minor_unknown": (
            "Blessure mineure/inconnue (Présence 0/1)",
            "Historique médical",
        ),
        "injury_minor_unknown_nb_d": (
            "Jours d'absence - Blessure mineure",
            "Historique médical",
        ),
        "injury_minor_unknown_nb_m": (
            "Matchs manqués - Blessure mineure",
            "Historique médical",
        ),
    }

    # Sélection et traitement numérique
    df_num = df.select_dtypes(include=[np.number]).copy()
    df_num = df_num.loc[:, df_num.std() > 0]

    # Calcul des corrélations de Pearson uniquement sur la cible brute (sans log)
    corr_brute = df_num.corr(method="pearson")[target_col]

    df_importance = pd.DataFrame(
        {"Variable": corr_brute.index, "Corr_Cible_Brute": corr_brute.values}
    )

    # Filtrage de la cible et des variantes normalisées (_nor)
    exclusions = [target_col, f"{target_col}_nor"]
    df_importance = df_importance[
        ~df_importance["Variable"].isin(exclusions)
    ].copy()
    df_importance = df_importance[
        ~df_importance["Variable"].str.endswith("_nor")
    ].copy()

    # Tri par la valeur absolue de la corrélation brute
    df_importance["Abs_Corr"] = df_importance["Corr_Cible_Brute"].abs()
    top_explicatives = (
        df_importance.sort_values(by="Abs_Corr", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Traduction du texte
    top_explicatives["Description"] = top_explicatives["Variable"].apply(
        lambda x: dict_variables.get(x, (x, "Non Classifié"))[0]
    )

    if afficher_nom_technique:
        top_explicatives["Description"] = top_explicatives.apply(
            lambda r: f"{r['Description']} ({r['Variable']})", axis=1
        )

    top_explicatives["Famille"] = top_explicatives["Variable"].apply(
        lambda x: dict_variables.get(x, (x, "Non Classifié"))[1]
    )

    top_explicatives["Corr_Cible_Brute"] = top_explicatives[
        "Corr_Cible_Brute"
    ].round(3)

    # Colorométrie
    # Configuration du style graphique de fond
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#fbfbfb"})
    plt.figure(figsize=(12, 8))

    # Définition d'une palette corporate moderne aux tons mats (style Nord/Muted)
    palette_pro = {
        "Identité et physique": "#4c566a",  # Gris ardoise chic
        "Postes": "#5e81ac",  # Bleu acier
        "Ligues et contexte international": "#81a1c1",  # Bleu givré
        "Temps de jeu et contrat": "#8fbcbb",  # Vert d'eau/Sauge mat
        "Performance offensive": "#d08770",  # Terracotta doux
        "Statistiques avancées": "#b48ead",  # Vieux mauve discret
        "Discipline et défense": "#bf616a",  # Rouge brique atténué
        "Succès équipe et collectif": "#a3be8c",  # Vert olive doux
        "Spécifique gardien": "#ebcb8b",  # Ocre doux
        "Historique médical": "#e5e9f0",  # Gris clair
    }

    # Tracé des barres horizontales
    sns.barplot(
        data=top_explicatives,
        y="Description",
        x="Corr_Cible_Brute",
        hue="Famille",
        palette=palette_pro,
        dodge=False,
        edgecolor="#2e3440",
        linewidth=0.6,
    )

    plt.axvline(0, color="#2e3440", linestyle="-", linewidth=1.2)
    type_population = "Gardiens de but" if uniquement_gardiens else "Population Globale"
    plt.title(
        f"Top {top_n} des variables corrélées à la Valeur Marchande - {type_population}",
        fontsize=14,
        fontweight="bold",
        color="#2e3440",
        pad=18,
    )
    plt.xlabel(
        "Coefficient de corrélation de Pearson (r) avec market_value_in_eur",
        fontsize=11,
        fontweight="semibold",
        color="#4c566a",
        labelpad=10,
    )
    plt.ylabel("Indicateurs", fontsize=11, fontweight="semibold", color="#4c566a")

    # Placement précis de la légende pour éviter les chevauchements
    plt.legend(
        title="Familles de variables",
        title_fontproperties={"weight": "bold"},
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        facecolor="#ffffff",
    )

    # Nettoyage des bordures superflues pour un look épuré
    sns.despine(left=True, bottom=True)
    plt.grid(axis="x", linestyle="--", alpha=0.6, color="#e5e9f0")
    plt.tight_layout()
    plt.show()

    return top_explicatives[["Variable", "Description", "Famille", "Corr_Cible_Brute"]]