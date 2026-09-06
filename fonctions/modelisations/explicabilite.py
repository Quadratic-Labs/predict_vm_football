from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
import seaborn as sns
from sklearn.decomposition import PCA

DICT_VARIABLES = {
        # Famille : Identité, âge et physique
        "player": ("Nom du joueur", "Identité et physique"),
        "team": ("Club du joueur", "Identité et physique"),
        "nation": ("Nationalité", "Identité et physique"),
        "Age (age + age_sq)": ("Âge du joueur", "Identité et physique"),
        "pic_distance": ("Distance de l'âge du pic de valeur", "Identité et physique"),
        "pic_age": ("Âge du pic de valeur (0/1)", "Identité et physique"),
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
        "sub_position_Attacking Midfield": ( "Milieu offensif (0/1)", "Postes", ),
        "sub_position_Central Midfield": ( "Milieu central (0/1)", "Postes", ),
        "sub_position_Centre-Back": ( "Défenseur central (0/1)", "Postes", ),
        "sub_position_Centre-Forward": ( "Avant-centre (0/1)", "Postes", ),
        "sub_position_Defensive Midfield": ( "Milieu défensif (0/1)", "Postes", ),
        "sub_position_Goalkeeper": ( "Gardien (Sous-poste 0/1)", "Postes", ),
        "sub_position_Left Midfield": ( "Milieu gauche (0/1)", "Postes", ),
        "sub_position_Left Winger": ( "Ailier gauche (0/1)", "Postes", ),
        "sub_position_Left-Back": ( "Latéral gauche (0/1)", "Postes", ),
        "sub_position_Right Midfield": ( "Milieu droit (0/1)", "Postes", ),
        "sub_position_Right Winger": ( "Ailier droit (0/1)", "Postes", ),
        "sub_position_Right-Back": ( "Latéral droit (0/1)", "Postes", ),
        "sub_position_Second Striker": ( "Neuf et demi / Second attaquant (0/1)", "Postes", ),
        "est_polyvalent": ( "Joueur polyvalent (0/1)", "Postes", ),
        # Famille : Championnats et contexte international
        "league_ENG-Premier League": ( "Évolue en Premier League (0/1)", "Ligues et contexte international", ),
        "league_ESP-La Liga": ( "Évolue en La Liga (0/1)", "Ligues et contexte international", ),
        "league_FRA-Ligue 1": ( "Évolue en Ligue 1 (0/1)", "Ligues et contexte international", ),
        "league_GER-Bundesliga": ( "Évolue en Bundesliga (0/1)", "Ligues et contexte international", ),
        "league_ITA-Serie A": ( "Évolue en Serie A (0/1)", "Ligues et contexte international", ),
        "classement_FIFA_1": ( "Sélection nationale Rang FIFA : 1", "Ligues et contexte international", ),
        "classement_FIFA_2": ( "Sélection nationale Rang FIFA : 2", "Ligues et contexte international", ),
        "classement_FIFA_3": ( "Sélection nationale Rang FIFA : 3", "Ligues et contexte international", ),
        "classement_FIFA_4": ( "Sélection nationale Rang FIFA : 4", "Ligues et contexte international", ),
        "classement_FIFA_5": ( "Sélection nationale Rang FIFA : 5", "Ligues et contexte international", ),
        "classement_FIFA_6": ( "Sélection nationale Rang FIFA : 6", "Ligues et contexte international", ),
        "classement_FIFA_7": ( "Sélection nationale Rang FIFA : 7", "Ligues et contexte international", ),
        "classement_FIFA_8": ( "Sélection nationale Rang FIFA : 8", "Ligues et contexte international", ),
        "classement_FIFA_9": ( "Sélection nationale Rang FIFA : 9", "Ligues et contexte international", ),
        "classement_FIFA_10": ( "Sélection nationale Rang FIFA : 10", "Ligues et contexte international", ),
        "score_hype_nation": ( "Score de hype de la sélection nationale", "Ligues et contexte international", ),
        # Famille : Temps de jeu, contrat et chronologie
        "season_year": ("Année de la saison", "Temps de jeu & Contrat"),
        "delta_minutes_jouees": ( "Différence de minutes jouées par rapport à la saison précédente",
                                 "Temps de jeu et contrat", ),
        "contrat_jours_restants": ( "Jours de contrat restants", "Temps de jeu et contrat", ),
        "Playing Time_MP": ("Matchs disputés", "Temps de jeu et contrat"),
        "Playing Time_Starts": ( "Titularisations", "Temps de jeu et contrat", ),
        "Playing Time_90s": ( "Nombre de 90 minutes complétées", "Temps de jeu et contrat", ),
        "Playing Time_Mn/MP": ( "Minutes jouées par match disputé", "Temps de jeu et contrat", ),
        "Starts_Mn/Start": ( "Minutes par titularisation", "Temps de jeu et contrat", ),
        "Starts_Compl": ( "Matchs commencés et terminés en entier", "Temps de jeu et contrat", ),
        "Subs_Subs": ( "Entrées en cours de match (Remplaçant)", "Temps de jeu et contrat", ),
        "Subs_Mn/Sub": ("Minutes par entrée en jeu", "Temps de jeu et contrat"),
        "Subs_unSub": ( "Matchs passés sur le banc sans entrer", "Temps de jeu et contrat", ),
        "taux_matchs_termines": ( "Taux de matchs terminés", "Temps de jeu et contrat", ),
        # Famille : Performance Offensive (Volume et efficacité)
        "Performance_Gls": ("Buts marqués", "Performance offensive"),
        "Performance_Ast": ("Passes décisives", "Performance offensive"),
        "Performance_G-PK": ("Buts hors pénaltys", "Performance offensive"),
        "Performance_PK": ("Pénaltys marqués", "Performance offensive"),
        "Performance_PKatt": ("Pénaltys tentés", "Performance offensive"),
        "Standard_Sh": ("Tirs totaux effectués", "Performance offensive"),
        "Standard_SoT": ("Tirs cadrés", "Performance offensive"),
        "Standard_SoT%": ( "Pourcentage de tirs cadrés", "Performance offensive", ),
        "Standard_Sh/90": ( "Tirs effectués par 90 min", "Performance offensive", ),
        "Standard_SoT/90": ("Tirs cadrés par 90 min", "Performance offensive"),
        "Standard_G/Sh": ("Buts par tir tenté", "Performance offensive"),
        "Standard_G/SoT": ("Buts par tir cadré", "Performance offensive"),
        "Performance_Off": ("Hors-jeux signalés", "Performance offensive"),
        "Performance_Crs": ("Centres vers la surface", "Performance offensive"),
        "Performance_PKwon": ("Pénaltys obtenus", "Performance offensive"),
        "Per 90 Minutes_Gls": ("Buts par 90 min", "Performance offensive"),
        "Per 90 Minutes_Ast": ( "Passes décisives par 90 min", "Performance offensive", ),
        "Per 90 Minutes_G+A": ( "Buts + Assists par 90 min", "Performance offensive", ),
        "Performance_Fld": ("Fautes subies", "Performance offensive"),
        "indice_danger_tirs": ( "Indice de danger des tirs (xG par tir)", "Performance offensive" ),
        "efficacite_devant_but": ( "Efficacité devant le but", "Performance offensive" ),
        "contribution_offensive_equipe": ( "Contribution offensive à l'équipe (Buts + Assists / Total équipe)",
                                          "Performance offensive" ),
        "rentabilite_buts_minutes": ( "Rentabilité des buts par minute jouée", "Performance offensive" ),
        "impact_buts_points": ( "Impact des buts sur les points gagnés par l'équipe", "Performance offensive" ),
        "ratio_buts_hors_penalty": ( "Ratio de buts hors pénaltys", "Performance offensive" ),
        "ratio_buts_penalty": ( "Ratio de buts sur pénaltys", "Performance offensive" ),
        # Famille : Statistiques avancées (xG, xA, Création)
        "xg": ("Expected Goals (xG)", "Statistiques avancées"),
        "delta_xg": ( "Différence de xG par rapport à la saison précédente", "Statistiques avancées" ),
        "xa": ("Expected Assists (xa)", "Statistiques avancées"),
        "xg_chain": ( "Chaîne Expected Goals (xG Chain)", "Statistiques avancées", ),
        "xg_buildup": ( "Construction Expected Goals", "Statistiques avancées", ),
        # Famille : Discipline et performance défensive
        "Performance_CrdY": ("Cartons jaunes reçus", "Discipline et défense"),
        "Performance_CrdR": ("Cartons rouges reçus", "Discipline et défense"),
        "ratio_agressivite": ( "Ratio agressivité", "Discipline et défense" ),
        "Performance_2CrdY": ( "Expulsions suite à 2 jaunes", "Discipline et défense", ),
        "Performance_Fls": ("Fautes commises", "Discipline et défense"),
        "Performance_Int": ("Interceptions de passes", "Discipline et défense"),
        "Performance_TklW": ("Tacles réussis", "Discipline et défense"),
        "Performance_PKcon": ("Pénaltys concédés", "Discipline et défense"),
        "Performance_OG": ( "Buts contre son camp (OG)", "Discipline et défense", ),
        "indiscipline_par_90": ( "Indiscipline par 90 min", "Discipline et défense" ),
        # Famille : Collectif et succès équipe
        "classement": ("Classement de l'équipe", "Succès équipe et collectif"),
        "Team Success_PPM": ( "Points par match gagnés par l'équipe", "Succès équipe et collectif", ),
        "Team Success_onG": ( "Buts marqués par l'équipe (si présent)", "Succès équipe et collectif", ),
        "Team Success_onGA": ( "Buts encaissés par l'équipe (si présent)", "Succès équipe et collectif", ),
        "Team Success_On-Off": ( "Impact On-Off de la présence du joueur", "Succès équipe et collectif", ),
        # Famille : Spécifique Gardien de but
        "Performance_GA": ("Buts encaissés", "Spécifique gardien"),
        "Performance_GA90": ( "Buts encaissés par 90 min", "Spécifique gardien", ),
        "Performance_Saves": ("Arrêts effectués", "Spécifique gardien"),
        "Performance_Save%": ("Pourcentage d'arrêts", "Spécifique gardien"),
        "Performance_W": ( "Victoires de l'équipe (si présent)", "Spécifique gardien", ),
        "Performance_D": ( "Matchs nuls de l'équipe (si présent)", "Spécifique gardien", ),
        "Performance_L": ( "Défaites de l'équipe (si présent)", "Spécifique gardien", ),
        "Performance_CS": ( "Clean Sheets (Matchs sans but)", "Spécifique gardien", ),
        "Performance_CS%": ( "Pourcentage de Clean Sheets", "Spécifique gardien", ),
        "Penalty Kicks_PKA": ("Pénaltys encaissés", "Spécifique gardien"),
        "Penalty Kicks_PKsv": ("Pénaltys arrêtés", "Spécifique gardien"),
        "Penalty Kicks_PKm": ( "Pénaltys ratés par l'adversaire", "Spécifique gardien", ),
        "Penalty Kicks_Save%": ( "Pourcentage de pénaltys arrêtés", "Spécifique gardien", ),
        "taux_arretXtaille_gardien": ( "Taux d'arrêts par rapport à la taille du gardien", "Spécifique gardien" ),
        # Famille : Historique médical (Blessures)
        "injury_nb_total": ("Nombre total de blessures", "Historique médical"),
        "injury_days_total": ( "Total des jours d'absence", "Historique médical", ),
        "injury_matches_max_single": ( "Max de matchs manqués sur une blessure", "Historique médical", ),
        "injury_musculaire": ( "Blessure musculaire (Présence 0/1)", "Historique médical", ),
        "injury_musculaire_nb_d": ( "Jours d'absence - Muscle", "Historique médical", ),
        "injury_musculaire_nb_m": ( "Matchs manqués - Muscle", "Historique médical", ),
        "injury_genou": ( "Blessure au genou (Présence 0/1)", "Historique médical", ),
        "injury_genou_nb_d": ("Jours d'absence - Genou", "Historique médical"),
        "injury_genou_nb_m": ("Matchs manqués - Genou", "Historique médical"),
        "injury_cheville_pied": ( "Blessure cheville/pied (Présence 0/1)", "Historique médical", ),
        "injury_cheville_pied_nb_d": ( "Jours d'absence - Cheville/Pied", "Historique médical", ),
        "injury_cheville_pied_nb_m": ( "Matchs manqués - Cheville/Pied", "Historique médical", ),
        "injury_mollet_tibia": ( "Blessure mollet/tibia (Présence 0/1)", "Historique médical", ),
        "injury_mollet_tibia_nb_d": ( "Jours d'absence - Mollet/Tibia", "Historique médical", ),
        "injury_mollet_tibia_nb_m": ( "Matchs manqués - Mollet/Tibia", "Historique médical", ),
        "injury_dos_bassin": ( "Blessure dos/bassin (Présence 0/1)", "Historique médical", ),
        "injury_dos_bassin_nb_d": ( "Jours d'absence - Dos/Bassin", "Historique médical", ),
        "injury_dos_bassin_nb_m": ( "Matchs manqués - Dos/Bassin", "Historique médical", ),
        "injury_trauma_severe": ( "Traumatisme sévère/Opération (Présence 0/1)", "Historique médical", ),
        "injury_trauma_severe_nb_d": ( "Jours d'absence - Traumatisme Sévère", "Historique médical", ),
        "injury_trauma_severe_nb_m": ( "Matchs manqués - Traumatisme Sévère", "Historique médical", ),
        "injury_medical_repos": ( "Maladie / Repos obligatoire (Présence 0/1)", "Historique médical", ),
        "injury_medical_repos_nb_d": ( "Jours d'absence - Maladie/Repos", "Historique médical", ),
        "injury_medical_repos_nb_m": ( "Matchs manqués - Maladie/Repos", "Historique médical", ),
        "injury_minor_unknown": ( "Blessure mineure/inconnue (Présence 0/1)", "Historique médical", ),
        "injury_minor_unknown_nb_d": ( "Jours d'absence - Blessure mineure", "Historique médical", ),
        "injury_minor_unknown_nb_m": ( "Matchs manqués - Blessure mineure", "Historique médical", ),
        "taux_indisponibilite": ( "Taux d'indisponibilité (Jours d'absence / 365)", "Historique médical", ),
        "fragilite_chronique": ( "Fragilité chronique (0/1)", "Historique médical", ),
    }

DICT_VARIABLES["log_prev_value"] = (
       "Valeur marchande (log) - Saison précédente",
       "Temps de jeu et contrat",
   )

PALETTE_PRO = {
    "Identité et physique": "#2B3A4A",             # Bleu nuit / Ardoise
    "Postes": "#0066FF",                            # Bleu électrique vif
    "Ligues et contexte international": "#00A896",   # Teal / Vert d'eau foncé
    "Temps de jeu et contrat": "#00B4D8",            # Cyan / Bleu ciel vif
    "Performance offensive": "#FF5A5F",              # Coral / Rouge vif
    "Statistiques avancées": "#8E44AD",             # Violet profond
    "Discipline et défense": "#E67E22",             # Orange cuivré
    "Succès équipe et collectif": "#2ECC71",        # Vert émeraude
    "Spécifique gardien": "#F1C40F",                 # Jaune ambre
    "Historique médical": "#E056FD",               # Rose néon / Magenta
    "Non Classifié": "#95A5A6",                     # Gris neutre
}

class StackingModel(BaseEstimator, RegressorMixin):

    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model

    def fit(self, X, y):
        return self

    def predict(self, X):
        X_meta = np.column_stack([
            np.expm1(model.predict(X)) for model in self.models
        ])

        # Le méta-modèle prédit directement en euros, pas besoin d'expm1 ensuite
        prediction_euros = self.meta_model.predict(X_meta)

        return prediction_euros


def calculer_permutation_importance(modele, X_val, y_val, scoring="r2", n_repeats=1, random_state=1308,
                                    n_jobs=-1, afficher_resultat=True, ):
    """Calcule et affiche l'importance des variables par permutation.
    """
    print("Permutation feature importance :")

    result_perm = permutation_importance(modele, X_val, y_val, scoring=scoring, n_repeats=n_repeats,
                                         random_state=random_state, n_jobs=n_jobs, )

    # Tri des indices par ordre décroissant d'importance
    sorted_idx = result_perm.importances_mean.argsort()[::-1]

    # Construction du DataFrame de résultats
    perm_importances = pd.DataFrame(
        {
            "Feature": X_val.columns[sorted_idx],
            "Importance_Mean": result_perm.importances_mean[sorted_idx],
            "Importance_Std": result_perm.importances_std[sorted_idx],
        }
    )

    if afficher_resultat:
        print(perm_importances)

    return perm_importances


def afficher_top_features_importance(df_importances, dict_variables=None, top_n=15, figsize=(10, 6),
                                     color="steelblue", titre=None):
    """Affiche un barplot horizontal du Top N des variables les plus importantes."""
    # Extraction du Top N
    top_df = df_importances.head(top_n).copy()

    # Mappage des noms si le dictionnaire est fourni
    if dict_variables is not None:
        mapping_noms = {k: v[0] for k, v in dict_variables.items()}
        top_df["Feature_Label"] = top_df["Feature"].map(mapping_noms)
        top_df["Feature_Label"] = top_df["Feature_Label"].fillna(
            top_df["Feature"]
        )
    else:
        top_df["Feature_Label"] = top_df["Feature"]

    # Création du graphique
    plt.figure(figsize=figsize)

    plt.barh(
        top_df["Feature_Label"][::-1],
        top_df["Importance_Mean"][::-1],
        xerr=top_df["Importance_Std"][::-1],
        color=color,
        capsize=3,
    )

    plt.xlabel("Baisse du R² après permutation")
    if titre:
        plt.title(titre)

    plt.tight_layout()
    plt.show()

    return top_df


def analyser_shap_stacking(meta, modeles_finaux, X_val, max_display=10, index_cascade=0):
    """Calcule et affiche l'analyse SHAP pour le méta-modèle de stacking.

    Parameters:
    -----------
    meta : estimator
        Le méta-modèle entraîné (ex: LinearRegression).
    modeles_finaux : dict
        Dictionnaire contenant les modèles de base entraînés.
    X_val : pd.DataFrame
        Variables explicatives du jeu de validation.
    max_display : int, default=10
        Nombre maximal de variables à afficher sur les graphiques SHAP.
    index_cascade : int, default=0
        Index de l'observation à analyser pour le graphique waterfall.

    Returns:
    --------
    shap.Explanation : Les valeurs SHAP calculées.
    """
    print("\nShap du stacking final :")

    # Prédictions directes via le dictionnaire modeles_finaux
    X_meta_val = pd.DataFrame(
        {
            f"Prediction_{nom.split()[0]}": modele.predict(X_val)
            for nom, modele in modeles_finaux.items()
        }
    )

    # Explainer adapté à LinearRegression
    explainer = shap.LinearExplainer(meta, X_meta_val)
    shap_values = explainer(X_meta_val)

    # Graphique Beeswarm
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()

    # Graphique
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("Importance moyenne absolue SHAP - Stacking final")
    plt.tight_layout()
    plt.show()

    # Graphique explication locale
    print( f"\nExplication locale de la prédiction à l'index {index_cascade} du jeu de validation" )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap_values[index_cascade], max_display=max_display, show=False
    )
    plt.tight_layout()
    plt.show()

    return shap_values


def regrouper_shap(shap_values, groupes):
    """
    groupes : dict {"Nom affiché": ["col1", "col2", ...]}
    Additionne les valeurs SHAP des colonnes de chaque groupe en une seule variable.
    """
    feature_names = list(shap_values.feature_names)
    values = shap_values.values.copy()
    data = shap_values.data.copy()

    colonnes_groupees = [c for cols in groupes.values() for c in cols]
    colonnes_gardees = [c for c in feature_names if c not in colonnes_groupees]
    idx_gardees = [feature_names.index(c) for c in colonnes_gardees]

    nouvelles_values = values[:, idx_gardees]
    nouvelles_data   = data[:, idx_gardees]
    nouveaux_noms    = colonnes_gardees.copy()

    for nom_groupe, cols in groupes.items():
        idx_cols = [feature_names.index(c) for c in cols if c in feature_names]
        somme_values = values[:, idx_cols].sum(axis=1, keepdims=True)
        # On garde la valeur brute de la 1ère colonne du groupe pour l'affichage (ex: age brut)
        donnee_repr = data[:, idx_cols[0]:idx_cols[0]+1]

        nouvelles_values = np.hstack([nouvelles_values, somme_values])
        nouvelles_data   = np.hstack([nouvelles_data, donnee_repr])
        nouveaux_noms.append(nom_groupe)

    return shap.Explanation(
        values=nouvelles_values,
        base_values=shap_values.base_values,
        data=nouvelles_data,
        feature_names=nouveaux_noms
    )


def renommer_features_shap(shap_values, dict_variables=DICT_VARIABLES, palette=PALETTE_PRO):
    """Remplace les feature_names techniques par les noms métier,
    et retourne {nom_affiché: couleur_famille} pour colorer les labels ensuite."""
    noms_originaux = list(shap_values.feature_names)

    noms_affiches = []
    couleur_par_label = {}
    for nom in noms_originaux:
        description, famille = dict_variables.get(nom, (nom, "Non Classifié"))
        noms_affiches.append(description)
        couleur_par_label[description] = palette.get(famille, "#2e3440")

    shap_values.feature_names = noms_affiches
    return couleur_par_label


def colorer_labels_waterfall(fig_ou_ax, couleur_par_label, couleur_defaut="#2e3440"):
    """Colore les labels du waterfall SHAP selon la famille de la variable."""
    fig = fig_ou_ax.figure if hasattr(fig_ou_ax, "figure") else fig_ou_ax
    ax = fig.axes[0]  # le premier axe contient les 2 couches de texte réellement visibles
    for label in ax.get_yticklabels():
        texte = label.get_text().strip()
        if texte in couleur_par_label:
            label.set_color(couleur_par_label[texte])


def afficher_waterfall_shap_joueur(modeles_finaux, X_val, df_val, predict_m_euro, groupes,
                                   palette_pro, indice_joueur=90, nom_modele="XGBoost (log)",
                                   colonne_joueur="player", n_samples_bg=100, random_state=42):
    """Calcule et affiche le graphique Waterfall SHAP personnalisé pour un joueur donné."""
    nom_joueur = df_val[colonne_joueur].iloc[indice_joueur]

    # Calcul des valeurs SHAP
    background_data = shap.sample(
        X_val, n_samples_bg, random_state=random_state
    )
    explainer_m_euro = shap.Explainer(predict_m_euro, background_data)
    shap_values_m_euro = explainer_m_euro(X_val.iloc[[indice_joueur]])

    # Regroupement et renommage
    shap_values_m_euro_grouped = regrouper_shap(
        shap_values_m_euro, groupes
    )
    shap_joueur = shap_values_m_euro_grouped[0]
    couleur_par_label = renommer_features_shap(shap_joueur)

    # Affichage du waterfall SHAP
    plt.figure(figsize=(14, 6))
    shap.plots.waterfall(shap_joueur, show=False)

    fig = plt.gcf()
    ax_reel = fig.axes[0]

    colorer_labels_waterfall(fig, couleur_par_label)

    # Remplacement de la ligne E[f(X)] par des pointillés
    e_fx = float(
        shap_joueur.base_values
        if np.isscalar(shap_joueur.base_values)
        else shap_joueur.base_values[0]
    )

    x_min, x_max = ax_reel.get_xlim()
    tolerance = 0.01 * abs(x_max - x_min)  # 1% de la largeur du graphe

    # Nettoyage des anciennes lignes E[f(X)]
    for ax in fig.axes:
        for line in list(ax.lines):
            xdata = line.get_xdata()
            if (
                len(xdata) >= 2
                and np.isclose(xdata[0], xdata[-1])
                and abs(xdata[0] - e_fx) < tolerance
            ):
                line.remove()

    # Ligne pointillée personnalisée
    ax_reel.axvline(x=e_fx, color="#23ad41", linestyle=":", linewidth=1.5, zorder=10,
                    clip_on=False, ymin=-0.05, ymax=1.05)

    # Légende des familles de variables
    familles_utilisees = {c: f for f, c in palette_pro.items() if c in couleur_par_label.values()}
    legend_elements = [Patch(facecolor=c, label=f) for c, f in familles_utilisees.items()]

    if legend_elements:
        ax_reel.legend(handles=legend_elements, title="Familles de variables",
                       bbox_to_anchor=(1.15, 1), loc="upper left", frameon=True)

    plt.title( f"Explication SHAP — {nom_joueur}", fontsize=13, fontweight="bold" )
    plt.xlabel("Valeur marchande (en Millions d'euros)")
    plt.tight_layout()
    plt.show()

    return shap_joueur


# Dictionnaire par défaut des familles de variables "one-hot"
FAMILLES_ONE_HOT_DEFAULT = {
    "confederation_": "Confédération",
    "league_": "Ligue",
    "pos_": "Poste",
    "sub_position_": "Sous-poste",
    "foot_": "Pied fort",
    "classement_FIFA": "Classement FIFA",
}


def trouver_famille_one_hot(nom_colonne, familles):
    """Retourne (prefixe, nom_famille) si la colonne appartient à une famille

    one-hot connue, sinon None.
    """
    for prefixe, nom_famille in familles.items():
        if nom_colonne.startswith(prefixe):
            return prefixe, nom_famille
    return None


def valeur_reelle_categorie(joueur_row, prefixe):
    """Cherche, parmi les colonnes de ce préfixe, celle qui vaut 1 pour ce joueur,

    et retourne son nom nettoyé.
    """
    colonnes_famille = [c for c in joueur_row.index if c.startswith(prefixe)]
    for col in colonnes_famille:
        if joueur_row[col] == 1:
            return col[len(prefixe) :].replace("_", " ").strip().capitalize()
    return "Inconnu"


def clean_feature_label_lime(condition, joueur_row, familles):
    """Nettoie le libellé LIME.

    Pour les variables one-hot, affiche la VRAIE catégorie du joueur plutôt
    qu'un simple Oui/Non.
    """
    if "<= 0.00" in condition:
        feat = condition.replace(" <= 0.00", "").strip()
        est_positif = False
    elif "> 0.00" in condition:
        feat = condition.replace(" > 0.00", "").strip()
        est_positif = True
    else:
        return condition  # Variable numérique continue

    famille = trouver_famille_one_hot(feat, familles)

    if famille is not None:
        prefixe, nom_famille = famille
        valeur_reelle = valeur_reelle_categorie(joueur_row, prefixe)
        categorie_testee = (
            feat[len(prefixe) :].replace("_", " ").strip().capitalize()
        )

        if est_positif:
            return f"{nom_famille} : {valeur_reelle}"
        else:
            return f"{nom_famille} : {valeur_reelle} (≠ {categorie_testee})"

    return f"Non : {feat}" if not est_positif else f"Oui : {feat}"


def explications_lime_joueur(model, X_train, X_val, df_val, indice_joueur=90, colonne_joueur="player",
                             num_features=10, random_state=1308, familles_one_hot=None,
                             afficher_graphique=True):
    """Génère l'explication LIME pour un joueur donné et affiche le barplot des impacts."""
    if familles_one_hot is None:
        familles_one_hot = FAMILLES_ONE_HOT_DEFAULT

    nom_joueur = df_val[colonne_joueur].iloc[indice_joueur]

    # Imputation des NaN pour LIME
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    # Fonction interne de prédiction en Millions d'Euros
    def predict_m_euro(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=X_train.columns)
        log_preds = model.predict(X)
        return np.exp(log_preds) / 1_000_000

    # Création de l'explicateur LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_imputed),
        feature_names=list(X_train.columns),
        class_names=["Valeur_M_Euro"],
        mode="regression",
        random_state=random_state,
        feature_selection="highest_weights",
    )

    # Calcul de l'explication
    joueur_data = X_val_imputed.iloc[indice_joueur].values
    exp = explainer.explain_instance(
        data_row=joueur_data,
        predict_fn=predict_m_euro,
        num_features=num_features,
    )

    # Formattage du DataFrame
    df_lime = pd.DataFrame(exp.as_list(), columns=["Feature_Condition", "Impact"])
    df_lime = df_lime.sort_values(by="Impact", ascending=False)

    df_lime["Pretty_Label"] = df_lime["Feature_Condition"].apply(
        lambda cond: clean_feature_label_lime(
            cond, X_val_imputed.iloc[indice_joueur], familles_one_hot
        )
    )
    df_lime["Color"] = df_lime["Impact"].apply(
        lambda x: "#2ecc71" if x > 0 else "#e74c3c"
    )

    # Graphique
    if afficher_graphique:
        plt.figure(figsize=(10, 6), dpi=120)
        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if "seaborn-v0_8-whitegrid" in plt.style.available
            else "default"
        )

        bars = plt.barh(
            y=df_lime["Pretty_Label"],
            width=df_lime["Impact"],
            color=df_lime["Color"],
            edgecolor="none",
            height=0.65,
        )

        plt.axvline(x=0, color="#333333", linestyle="--", linewidth=0.8, alpha=0.7)

        for bar in bars:
            width = bar.get_width()
            x_pos = width + (0.15 if width > 0 else -0.15)
            ha = "left" if width > 0 else "right"

            plt.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{width:+.2f} M€",
                va="center",
                ha=ha,
                fontsize=9,
                fontweight="bold",
                color="#2c3e50",
            )

        plt.title( f"Explication LIME — {nom_joueur}", fontsize=13, pad=15, fontweight="bold", )
        plt.xlabel( "Impact sur la valeur marchande (en Millions d'Euros)", fontsize=10, labelpad=10, )
        plt.ylabel("Caractéristiques du joueur", fontsize=10)

        x_max = ( max(abs(df_lime["Impact"].min()), abs(df_lime["Impact"].max())) + 1.5 )
        plt.xlim(-x_max, x_max)

        plt.tight_layout()
        plt.show()

    return df_lime


# Mapping par défaut si n_clusters = 4
NOMS_CLUSTERS_DEFAULT = {
    0: "Rotation / valeur modeste",
    1: "Cadres en progression",
    2: "Stars post-pic en repli",
    3: "Superstars en forte hausse",
}


def segmenter_trajectoires_joueurs( df_historique, colonne_cible="market_value_in_eur",
                                   colonne_joueur="player", n_clusters=4, n_min_variations=3,
                                   percentile_clip=0.01, random_state=42, noms_clusters=None,
                                   afficher_resume=True):
    """Calcule les indicateurs de trajectoire d'un jeu de données joueur x saison,
    effectue un clustering KMeans standardisé et renvoie les résultats ainsi que
    le scaler et le modèle KMeans.
    """
    if noms_clusters is None:
        noms_clusters = (
            NOMS_CLUSTERS_DEFAULT
            if n_clusters == 4
            else {i: f"Cluster {i}" for i in range(n_clusters)}
        )

    # Construction de la table pivot (Joueur x Saison = Variation en €)
    pivot_abs = df_historique.pivot_table(
        index=colonne_joueur, columns="season_year", values="delta_vm_eur"
    )

    joueurs_retenus = pivot_abs.dropna(thresh=n_min_variations).index
    pivot_abs_complet = pivot_abs.loc[joueurs_retenus].copy()

    if afficher_resume:
        print(
            f"Nombre de joueurs retenus pour la classification : {len(pivot_abs_complet)} "
            f"(sur {len(pivot_abs)} au total).\n"
        )

    # Calcul des indicateurs de trajectoire
    stats_joueurs = pd.DataFrame(index=pivot_abs_complet.index)
    stats_joueurs["mean_delta_eur"] = pivot_abs_complet.mean(axis=1)
    stats_joueurs["std_delta_eur"] = pivot_abs_complet.std(axis=1).fillna(0)
    stats_joueurs["nb_seasons"] = pivot_abs_complet.notna().sum(axis=1)

    # Récupération de la dernière saison connue par joueur
    colonnes_a_recuperer = [colonne_cible, "pic_distance"]
    dernieres_infos = (
        df_historique.dropna(subset=[colonne_cible])
        .sort_values("season_year")
        .groupby(colonne_joueur)[colonnes_a_recuperer]
        .last()
    )

    stats_joueurs["market_value_in_eur"] = dernieres_infos[colonne_cible]
    stats_joueurs["pic_distance"] = dernieres_infos["pic_distance"]

    stats_joueurs = stats_joueurs.dropna( subset=["market_value_in_eur", "pic_distance"] )

    # Traitement des outliers et mise à l'échelle
    stats_joueurs["log_market_value"] = np.log1p( stats_joueurs["market_value_in_eur"] )

    for col in ["mean_delta_eur", "std_delta_eur"]:
        borne_basse = stats_joueurs[col].quantile(percentile_clip)
        borne_haute = stats_joueurs[col].quantile(1 - percentile_clip)
        stats_joueurs[col + "_clip"] = stats_joueurs[col].clip(
            borne_basse, borne_haute
        )

    # Clustering KMeans
    features_cluster = [
        "mean_delta_eur_clip",
        "std_delta_eur_clip",
        "log_market_value",
        "pic_distance",
    ]

    scaler = RobustScaler()
    X_cluster = scaler.fit_transform(stats_joueurs[features_cluster])

    # Silhouette score pour k = 2 à 8
    scores_silhouette = {}
    for k in range(2, 9):
        km_test = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_test = km_test.fit_predict(X_cluster)
        scores_silhouette[k] = silhouette_score(X_cluster, labels_test)

    if afficher_resume:
        print("Silhouette score par nombre de clusters :")
        for k, s in scores_silhouette.items():
            marqueur = "  <-- nombre de clusters actuel" if k == n_clusters else ""
            print(f"  k={k} : {s:.3f}{marqueur}")

    kmeans = KMeans( n_clusters=n_clusters, random_state=random_state, n_init=10 )
    stats_joueurs["cluster"] = kmeans.fit_predict(X_cluster)
    stats_joueurs["profil_metier"] = stats_joueurs["cluster"].map(noms_clusters)

    # Affichage des synthèses
    if afficher_resume:
        ordre_profils = [
            noms_clusters.get(i, f"Cluster {i}") for i in range(n_clusters)
        ]

        print("\nRépartition des joueurs par cluster")
        effectifs = (
            stats_joueurs["profil_metier"]
            .value_counts()
            .reindex(ordre_profils, fill_value=0)
        )
        pourcentages = (
            stats_joueurs["profil_metier"]
            .value_counts(normalize=True)
            .reindex(ordre_profils, fill_value=0)
            * 100
        )
        df_synthese = pd.DataFrame( {"Effectif": effectifs, "Proportion (%)": pourcentages.round(1)} )
        print(df_synthese)

        print(
            "\nStatistiques de trajectoire par cluster"
        )
        resume_groupes = (
            stats_joueurs.groupby("profil_metier")
            .agg(
                mean_delta_moyen=("mean_delta_eur", "mean"),
                std_delta_moyen=("std_delta_eur", "mean"),
                vm_moyenne=("market_value_in_eur", "mean"),
                pic_distance_moyen=("pic_distance", "mean"),
            )
            .reindex(ordre_profils)
        )

        resume_groupes["mean_delta_moyen"] = (
            resume_groupes["mean_delta_moyen"].round(0).astype(str) + " €"
        )
        resume_groupes["std_delta_moyen"] = (
            resume_groupes["std_delta_moyen"].round(0).astype(str) + " €"
        )
        resume_groupes["vm_moyenne"] = resume_groupes["vm_moyenne"].round(0)
        resume_groupes["pic_distance_moyen"] = resume_groupes[
            "pic_distance_moyen"
        ].round(1)
        print(resume_groupes)

    return stats_joueurs, kmeans, scaler, scores_silhouette


def visualiser_clusters_trajectoires(stats_joueurs, X_cluster, features_plot=None, col_cluster="cluster",
                                     col_profil="profil_metier", random_state=42, palette="tab10"):
    """Génère les visualisations d'analyse de clusters :

    1. Projection 2D via PCA avec variance expliquée.
    2. Pairplot (nuage de points 2 à 2) sur les variables brutes.
    3. Boxplots comparatifs par variable et par cluster.

    """
    if features_plot is None:
        features_plot = [
            "mean_delta_eur",
            "std_delta_eur",
            "log_market_value",
            "pic_distance",
        ]

    # Utilisation des libellés métier si disponibles, sinon des identifiants de clusters
    col_affichage = (
        col_profil if col_profil in stats_joueurs.columns else col_cluster
    )

 
    # Projection en 2D de l'ACP

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_cluster)
    var_expliquee = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=stats_joueurs[col_cluster],
        cmap=palette,
        alpha=0.6,
        s=25,
    )
    ax.set_xlabel(f"PC1 ({var_expliquee[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_expliquee[1]*100:.1f}% variance)")
    ax.set_title("Clusters projetés en 2D (PCA sur variables standardisées)")

    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    plt.tight_layout()
    plt.show()

    print(
        f"Variance totale expliquée par PC1+PC2 : {var_expliquee.sum()*100:.1f}%\n"
    )


    # Plots 2 à 2 sur variables brutes

    cols_existantes = [c for c in features_plot if c in stats_joueurs.columns]
    df_plot = stats_joueurs[cols_existantes + [col_affichage]].copy()

    g = sns.pairplot(
        df_plot,
        hue=col_affichage,
        diag_kind="kde",
        palette=palette,
        plot_kws={"alpha": 0.5, "s": 20},
    )
    g.fig.suptitle("Distribution croisée des variables par cluster", y=1.02)
    plt.show()


    # Boxplots par variable

    n_cols = len(cols_existantes)
    if n_cols > 0:
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

        for ax, col in zip(axes, cols_existantes):
            stats_joueurs.boxplot(
                column=col, by=col_affichage, ax=ax, grid=False
            )
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)

        fig.suptitle("Distribution des variables par cluster", y=1.03)
        plt.tight_layout()
        plt.show()

    return {
        "pca": pca,
        "variance_expliquee_cumulee": var_expliquee.sum(),
    }


def tracer_trajectoires_moyennes_vm(df_historique, stats_joueurs, colonne_cible="market_value_in_eur",
                                    colonne_joueur="player", colonne_profil="profil_metier",
                                    seuil_min_joueurs=30, afficher_graphique=True):
    """Calcule et affiche l'évolution moyenne de la valeur marchande (indice base 100)
    par profil métier en fonction du rang de saison.
    """
    # Fusion des profils métier avec l'historique
    df_clusters = stats_joueurs.reset_index()[[colonne_joueur, colonne_profil]]
    df_traj_full = df_historique.merge(
        df_clusters, on=colonne_joueur, how="inner"
    )

    # Tri et calcul de l'indice base 100
    df_traj_full = df_traj_full.sort_values(by=[colonne_joueur, "season_year"])
    df_traj_full["vm_premiere_saison"] = df_traj_full.groupby(colonne_joueur)[
        colonne_cible
    ].transform("first")

    df_traj_full["indice_vm"] = ( df_traj_full[colonne_cible] / df_traj_full["vm_premiere_saison"] ) * 100

    # Calcul du rang de saison
    df_traj_full["rang_saison"] = (
        df_traj_full.groupby(colonne_joueur)["season_year"]
        .rank(method="first")
        .astype(int)
    )

    # Calcul des moyennes non-filtrées et des effectifs
    trajectoires_moyennes = (
        df_traj_full.groupby([colonne_profil, "rang_saison"])["indice_vm"]
        .mean()
        .unstack(level=0)
    )

    counts_grid = (
        df_traj_full.groupby(["rang_saison", colonne_profil])
        .size()
        .unstack(fill_value=0)
    )

    # Filtrage selon le seuil minimal de joueurs par point
    trajectoires_filtrees = trajectoires_moyennes.copy()

    for col in trajectoires_filtrees.columns:
        saisons_valides = counts_grid[col][
            counts_grid[col] >= seuil_min_joueurs
        ].index
        trajectoires_filtrees.loc[
            ~trajectoires_filtrees.index.isin(saisons_valides), col
        ] = None

    # Graphique
    if afficher_graphique:
        plt.figure(figsize=(10, 6))

        for profil in trajectoires_filtrees.columns:
            effectif_total = (stats_joueurs[colonne_profil] == profil).sum()
            plt.plot(
                trajectoires_filtrees.index,
                trajectoires_filtrees[profil],
                marker="o",
                label=f"{profil} (n={effectif_total})",
            )

        plt.axhline(100, color="grey", linestyle="--", linewidth=0.8)
        plt.xlabel("Ancienneté (rang de saison depuis la 1ère saison connue)")
        plt.ylabel("Valeur marchande (indice base 100)")
        plt.title(
            "Trajectoires moyennes de valeur marchande par profil métier"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    return trajectoires_filtrees


def analyser_explications_shap_joueur(df_historique, datasets_par_split, modele_stack, X_train_bg, groupes,
                                      palette_pro, joueur_etudie=None, groupe_etudie=None, df_clusters=None,
                                      index_joueur=0, colonne_joueur="player", colonne_profil="profil_metier",
                                      colonne_saison="season_year", random_state=42, n_samples_bg=100,
                                      afficher_graphique=True):
    """Initialise l'explainer SHAP, extrait les données d'un joueur à travers ses saisons

    et génère les waterfall plots explicatifs locaux.

    Si `joueur_etudie` est None, sélectionne le joueur situé à `index_joueur` dans `groupe_etudie`.
    """

    # Gestion du choix automatique du joueur par groupe si joueur_etudie est None
    if joueur_etudie is None:
        if groupe_etudie is None or df_clusters is None:
            raise ValueError(
                "Vous devez fournir 'joueur_etudie' OU à la fois 'groupe_etudie' et 'df_clusters'."
            )

        joueurs_du_groupe = df_clusters.loc[
            df_clusters[colonne_profil] == groupe_etudie, colonne_joueur
        ]

        if joueurs_du_groupe.empty:
            raise ValueError(
                f"Aucun joueur trouvé pour le groupe métier : '{groupe_etudie}'."
            )

        # Récupération du N-ième joueur du groupe
        joueur_etudie = joueurs_du_groupe.iloc[index_joueur]
        print(
            f"Aucun joueur spécifié. Sélection automatique du joueur index {index_joueur} du groupe « {groupe_etudie} » : {joueur_etudie}"
        )
    elif groupe_etudie is not None:
        print( f"Étude du joueur : {joueur_etudie} (groupe : « {groupe_etudie} »)" )
    else:
        print(f"Étude du joueur : {joueur_etudie}")

    # Sous-fonction de recherche d'un joueur dans les splits
    def _recuperer_ligne_joueur(nom_joueur, annee):
        for split, (df_split, X_split) in datasets_par_split.items():
            mask = (df_split[colonne_joueur] == nom_joueur) & (
                df_split[colonne_saison] == annee
            )
            if mask.any():
                idx = df_split[mask].index[0]
                return X_split.loc[[idx]], split
        return None, None

    # Fonction de prédiction du Stacking (résultat en M€)
    def _predict_m_euro_stack(X):
        return modele_stack.predict(X) / 1_000_000

    # Initialisation de l'explainer SHAP model-agnostic
    background_data = shap.sample(
        X_train_bg, n_samples_bg, random_state=random_state
    )
    explainer_stack = shap.Explainer(_predict_m_euro_stack, background_data)

    # Identification des saisons du joueur
    annees_joueur = sorted(
        df_historique.loc[
            df_historique[colonne_joueur] == joueur_etudie, colonne_saison
        ].unique()
    )

    print(f"Saisons disponibles pour {joueur_etudie} : {annees_joueur}")

    resultats_shap = {}

    # Boucle sur chaque saison
    for annee in annees_joueur:
        X_joueur_annee, split_trouve = _recuperer_ligne_joueur(
            joueur_etudie, annee
        )
        if X_joueur_annee is None:
            continue

        # Calcul et regroupement SHAP
        shap_values_annee = explainer_stack(X_joueur_annee)
        shap_values_annee_grouped = regrouper_shap(
            shap_values_annee, groupes
        )
        shap_joueur_annee = shap_values_annee_grouped[0]
        couleur_par_label = renommer_features_shap(shap_joueur_annee)

        resultats_shap[annee] = shap_joueur_annee

        if not afficher_graphique:
            continue

        # Affichage du Waterfall Plot
        plt.figure(figsize=(24, 6))
        shap.plots.waterfall(shap_joueur_annee, show=False)

        fig = plt.gcf()
        ax_reel = fig.axes[0]

        # Coloration dynamique des labels
        colorer_labels_waterfall(fig, couleur_par_label)

        # Remplacement de la ligne E[f(X)] par une ligne personnalisée
        e_fx = float(
            shap_joueur_annee.base_values
            if np.isscalar(shap_joueur_annee.base_values)
            else shap_joueur_annee.base_values[0]
        )

        x_min, x_max = ax_reel.get_xlim()
        tolerance = 0.01 * abs(x_max - x_min)

        for ax in fig.axes:
            for line in list(ax.lines):
                xdata = line.get_xdata()
                if (
                    len(xdata) >= 2
                    and np.isclose(xdata[0], xdata[-1])
                    and abs(xdata[0] - e_fx) < tolerance
                ):
                    line.remove()

        ax_reel.axvline(x=e_fx, color="#23ad41", linestyle=":", linewidth=1.5, zorder=10, clip_on=False,
                        ymin=-0.05, ymax=1.05)

        # Construction de la légende des familles de variables
        familles_utilisees = {
            c: f
            for f, c in palette_pro.items()
            if c in couleur_par_label.values()
        }
        legend_elements = [
            Patch(facecolor=c, label=f) for c, f in familles_utilisees.items()
        ]

        if legend_elements:
            ax_reel.legend(
                handles=legend_elements,
                title="Familles de variables",
                bbox_to_anchor=(1.15, 1),
                loc="upper left",
                frameon=True,
            )

        titre_groupe = f" (profil « {groupe_etudie} »)" if groupe_etudie else ""
        plt.title(
            f"Explication SHAP locale — {joueur_etudie}{titre_groupe} ({annee} - {split_trouve})",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel("Valeur marchande (en Millions d'euros)")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    return resultats_shap


def analyser_explicabilite_globale_groupe(groupe_etudie, df_clusters, df_historique, datasets_par_split,
                                          modele_stack, X_train_bg,
                                          groupes, colonne_joueur="player", colonne_profil="profil_metier",
                                          colonne_saison="season_year", random_state=42, n_samples_bg=100,
                                          max_display=15, afficher_graphique=True):
    """Calcule et affiche le beeswarm plot SHAP global pour un groupe métier donné

    en sélectionnant la saison la plus récente disponible pour chaque joueur.

    """

    # Recherche d'une ligne joueur dans les splits
    def _recuperer_ligne_joueur(nom_joueur, annee):
        for split, (df_split, X_split) in datasets_par_split.items():
            mask = (df_split[colonne_joueur] == nom_joueur) & (
                df_split[colonne_saison] == annee
            )
            if mask.any():
                idx = df_split[mask].index[0]
                return X_split.loc[[idx]], split
        return None, None

    # Fonction de prédiction du Stacking (en M€)
    def _predict_m_euro_stack(X):
        return modele_stack.predict(X) / 1_000_000

    # Extraction des joueurs du groupe
    joueurs_du_groupe = df_clusters.loc[
        df_clusters[colonne_profil] == groupe_etudie, colonne_joueur
    ].tolist()

    if not joueurs_du_groupe:
        raise ValueError(
            f"Aucun joueur trouvé pour le groupe métier : '{groupe_etudie}'."
        )

    # Récupération de la dernière saison connue pour chaque joueur
    lignes_groupe = []
    for joueur in joueurs_du_groupe:
        annees = sorted(
            df_historique.loc[
                df_historique[colonne_joueur] == joueur, colonne_saison
            ].unique()
        )
        if not annees:
            continue
        X_ligne, _ = _recuperer_ligne_joueur(joueur, annees[-1])
        if X_ligne is not None:
            lignes_groupe.append(X_ligne)

    if not lignes_groupe:
        raise ValueError(
            f"Impossible de récupérer des données de features pour le groupe '{groupe_etudie}'."
        )

    X_groupe = pd.concat(lignes_groupe)
    print( f"{len(X_groupe)} joueurs inclus dans l'explicabilité globale du profil « {groupe_etudie} »." )

    # Initialisation SHAP & calcul
    background_data = shap.sample(X_train_bg, n_samples_bg, random_state=random_state)
    explainer_stack = shap.Explainer(_predict_m_euro_stack, background_data)

    shap_values_groupe = explainer_stack(X_groupe)
    shap_values_groupe_grouped = regrouper_shap(shap_values_groupe, groupes)
    _ = renommer_features_shap(shap_values_groupe_grouped)

    # Affichage du Beeswarm Plot
    if afficher_graphique:
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(
            shap_values_groupe_grouped, max_display=max_display, show=False
        )
        plt.title(
            f"SHAP Global — Profil « {groupe_etudie} » (dernière saison connue de chaque joueur)"
        )
        plt.tight_layout()
        plt.show()

    return shap_values_groupe_grouped