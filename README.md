# Prédiction et explication de la valeur marchande de joueurs de football masculin

Ce projet a pour but de prédire et d'expliquer la **valeur marchande Transfermarkt** des joueurs de football masculin, à partir de leurs statistiques de performance (FBref/Understat), de leur contexte club (classements, force du championnat) et de leur historique de blessures.

Périmètre : les **5 grands championnats européens** (Big 5 : Angleterre, Espagne, Allemagne, Italie, France), les saisons **2020/2021 à 2025/2026**.

## Sommaire

- [Contexte](#contexte)
- [Pipeline du projet](#pipeline-du-projet)
- [Sources de données](#sources-de-données)
- [Construction de la base d'apprentissage](#construction-de-la-base-dapprentissage)
- [Nettoyage et préparation](#nettoyage-et-préparation)
- [Feature engineering](#feature-engineering)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Modélisation baseline](#modélisation-baseline)
- [Approche de modélisation](#approche-de-modélisation)
- [Modèle final retenu](#modèle-final-retenu)
- [Explicabilité et segmentation des joueurs](#explicabilité-et-segmentation-des-joueurs)
- [Technologies](#technologies)

## Contexte

Ce projet a été développé dans le cadre d'un stage. Il vise à construire un modèle de machine learning capable d'estimer la valeur marchande d'un joueur de football à partir de ses statistiques de performance, de son contexte collectif (classement du club, niveau du championnat) et de son historique médical (blessures).


## Pipeline du projet

Les notebooks du dossier notebooks/ composent le pipeline de traitement du projet et doivent être exécutés dans l'ordre suivant :

| # | Notebook | Rôle |
|---|----------|------|
| 01 | `acquisition_donnees` | Récupération des données brutes depuis toutes les sources (Kaggle, API soccerdata, scraping, téléchargements) |
| 02 | `exploration_qualite` | Audit qualité de chaque source (fraîcheur, valeurs manquantes, doublons, intégrité référentielle) avant intégration |
| 03 | `base_apprentissage` | Réconciliation multi-sources : fusion en cascade (mapping exact puis fuzzy matching) pour obtenir une ligne par joueur/saison |
| 04 | `nettoyage_base` | Nettoyage de la base consolidée, imputation, encodage, et split train/val/test/en_cours |
| 05 | `exploration_approfondie` | Analyse exploratoire détaillée **sur le train uniquement** (répartitions, corrélations, effet âge/poste sur la VM) |
| 06 | `feature_engineering` | Création des variables utilisées par les modèles |
| 07 | `modelisation_baseline` | Premier modèle de référence |
| 08 | `tests_modelisation` | Expérimentations de modélisation (comparaison d'algorithmes, stacking, transformations de la cible, retrait des outliers, réduction de dimensionnalité, modèles par poste) |
| 09 | `modelisation_finale` | Modèle final retenu et évaluation |

Les fonctions communes utilisées par ces notebooks sont centralisées dans `fonctions/` (un fichier ou sous-dossier par étape du pipeline), pour éviter la répétition de code dans les différents notebooks.

## Sources de données

| Source | Contenu | Usage |
|---|---|---|
| **Transfermarkt** ([`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores) sur Kaggle) | Profils joueurs, historique des valeurs marchandes | Fournit la **variable cible** (valeur marchande) et les caractéristiques personnelles (âge, taille, pied fort...) |
| **soccerdata** (API [`soccerdata`](https://soccerdata.readthedocs.io/en/latest/reference/index.html) — FBref + Understat) | Statistiques de performance saison par saison (volume de jeu, buts/passes, xG/xA, métriques défensives) | Variables explicatives de performance sportive |
| **Blessures Transfermarkt** (scraping) | Date, nature de la blessure, matchs manqués, par joueur et par saison | Variables liées à l'historique médical (jours de blessure, gravité) |
| [**football-data.co.uk**] (https://www.football-data.co.uk/data.php) | Résultats de matchs et statistiques collectives | Contexte collectif : force et dynamique de l'équipe du joueur |
| **Classement FIFA** ([Kaggle](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)) | Historique du classement FIFA (jusqu'au 1er avril 2026), top 10 par fin de saison | Contexte des sélections/compétitions internationales |
| **mapping_worldfootballR** (package R [`worldfootballR`](https://github.com/JaseZiv/worldFootballR)) | Table de correspondance entre identifiants FBref et Transfermarkt | Sert de clé de jointure prioritaire lors de la fusion multi-sources |


## Construction de la base d'apprentissage

Le notebook `03_base_apprentissage.ipynb` réalise la **réconciliation multi-sources** pour obtenir une base avec une ligne par joueur et par saison, en appariant les statistiques de performance (soccerdata) avec les valeurs marchandes (Transfermarkt). La stratégie de fusion se fait **en cascade**, liant *matching direct* et *fuzzy matching* pour maximiser le taux de correspondance tout en garantissant la fiabilité des appariements. Les données de blessures et les résultats collectifs sont ensuite fusionnés à cette base appariée.


Les joueurs non retrouvés à l'issue des 4 étapes, appelés « **orphelins** », représentent environ 4,4 % du dataset et sont majoritairement des joueurs ayant changé de club au mercato d'hiver hors périmètre Big 5, ou des jeunes joueurs ayant trop peu de temps de jeu pour être analysés.

## Nettoyage et préparation

Le notebook `04_nettoyage_base.ipynb` prépare la base consolidée pour la modélisation :

- **Doublons** : distinction entre doublons techniques (mêmes joueur/saison/club, fusionnés) et doublons de mercato (même joueur etsaison, mais 2 clubs : statistiques agrégées, ratios recalculés)
- **Valeurs manquantes** : imputation par 0 ou par la médiane selon le contexte
- **Encodage** : One-Hot Encoding des variables catégorielles (ex. pied fort)
- **Réduction de colinéarité** : suppression des variables redondantes
- **Split des données** :
  - **Temporel (par défaut)** : Train = 2020–2022, Validation = 2023, Test = 2024, Saison en cours = 2025 (exclue de l'évaluation)
  - **Aléatoire (alternative)** : 70 % / 15 % / 15 %, avec graine fixe
- **Traitement des outliers** : taille des joueurs bornée à [155 cm, 210 cm], anomalies remplacées par la médiane du poste
- **Imputation croisée** : médiane par Poste × Ligue, apprise uniquement sur le Train pour éviter toute fuite de données
- **Variable de valeur historique** : valeur marchande log de la saison précédente

Les jeux finaux (`train.csv`, `val.csv`, `test.csv`, `en_cours.csv`) sont exportés dans `data_finale/`, et les pipelines de nettoyage/prétraitement sont sauvegardées pour être réappliquées de façon identique (notamment à la saison en cours).

> L'exploration approfondie (`05_exploration_approfondie.ipynb`) est menée **exclusivement sur `train.csv`**, pour garantir qu'aucune décision de prétraitement ou de feature engineering ne fuite d'information depuis la validation ou le test.

## Feature engineering

Le notebook `06_feature_engineering.ipynb` applique une fonction unique (`generer_feature_engineering`) aux quatre jeux (`train`, `val`, `test`, `en_cours`) issus du nettoyage, afin de créer de nouvelles variables susceptibles d'améliorer la prédiction. La base passe à **151 colonnes** au total. Les jeux enrichis sont exportés dans `data_finale/featuring/` (`train_featured.csv`, `val_featured.csv`, `test_featured.csv`, `en_cours_featured.csv`).


## Installation

### Prérequis

- Python 3.x
- PowerShell pour `install.ps1`
- Un compte [Kaggle](https://www.kaggle.com) (API) pour télécharger les datasets Transfermarkt et FIFA ranking

### Étapes

```bash
# Cloner le repo
git clone <https://github.com/Quadratic-Labs/predict_vm_football.git>
cd predict_vm_football

# Créer et activer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

Sous Windows, le script `install.ps1` peut automatiser cette installation :

```powershell
.\install.ps1
```

### Configuration de l'API Kaggle

Créer un fichier `.env` à la racine du projet (non versionné) avec :

```
KAGGLE_USERNAME=votre_nom_d_utilisateur
KAGGLE_API_TOKEN=votre_cle_api
```

Le token s'obtient depuis le profil Kaggle > **Settings** > section **API** > **Generate New API Token**.

## Utilisation

1. Activer l'environnement virtuel (`.venv`).
2. Renseigner le fichier `.env` (identifiants Kaggle).
3. Exécuter les notebooks dans l'ordre numéroté (`01_...` à `09_...`) depuis le dossier `notebooks/`.
4. Les données intermédiaires et finales sont stockées dans `data_finale/`, les résultats et figures dans `outputs/`.

## Modélisation baseline

Le notebook `07_modelisation_baseline.ipynb` établit un **modèle naïf de référence** (régression linéaire simple) avec seulement 3 variables : `Performance_Gls` (nombre de buts)  `Playing Time_MP` (nombre de matchs joués) et `log_prev_value` (valeur marchande log de la saison précédente), pour cible la valeur marchande brute (`market_value_in_eur`).

## Approche de modélisation

Le notebook `08_tests_modelisation.ipynb` documente les expérimentations menées pour dépasser cette baseline et choisir l'approche à retenir :

- **Algorithmes comparés** : XGBoost, LightGBM, CatBoost, SVR, ElasticNet (tuning d'hyperparamètres via Optuna)
- **Transformation de la cible** : comparaison cible brute vs cible log, et test de plusieurs transformations (Power, Quantile, MinMax) pour stabiliser la variance et améliorer les performances
- **Stacking OOF** : combinaison des modèles de base (XGBoost, LightGBM, CatBoost) via un méta-modèle entraîné sur des prédictions out-of-fold
- **Retrait des valeurs extrêmes** : ré-entraînement sur une base sans outliers pour évaluer l'impact sur la performance
- **Réduction de dimensionnalité** : ACP, sélection de features par importance (gain XGBoost) et permutation importance
- **Modèles par groupe** : comparaison d'un modèle global vs des modèles spécifiques par poste (gardiens / joueurs de champ)

## Modèle final retenu

Le notebook `09_modelisation_finale.ipynb` reprend les enseignements de `08` pour produire le modèle final : **CatBoost, XGBoost et LightGBM entraînés sur la cible `log1p`**, combinés par **Stacking OOF**. L'optimisation se fait sur la **MAPE**. Le split reste temporel : Train 2020-2022 / Validation 2023 / Test 2024 (Saison en cours 2025 utilisée comme jeu d'évaluation supplémentaire, hors optimisation).

**Performance des modèles de base sur le jeu de test :**

| Modèle | MAPE | MAE | R² |
|---|---|---|---|
| LightGBM (log) | 40.17 % | 3 321 653 € | 0.877 |
| XGBoost (log) | 40.69 % | 3 231 307 € | 0.891 |
| CatBoost (log) | 41.11 % | 3 433 511 € | 0.880 |

**Stacking OOF (modèle final)** : le Stacking OOF combine les 3 modèles via un méta-modèle linéaire à coefficients positifs (combinaison convexe), entraîné sur des prédictions out-of-fold pour éviter que le méta-modèle ne surapprenne les prédictions déjà vues par les modèles de base. Poids appris : LightGBM **A COMPLETER**, CatBoost **A COMPLETER**, XGBoost **A COMPLETER** (intercept ≈ **A COMPLETER** €). C'est cette prédiction combinée qui est retenue comme prédiction finale du projet.

## Explicabilité et segmentation des joueurs

Au-delà de la prédiction, le notebook `09` propose deux volets d'analyse orientés aide à la décision pour un club :

- **Explicabilité (feature importance + SHAP)** : importance globale des variables, puis analyses SHAP globales (profil moyen) et locales (un joueur donné) pour comprendre ce qui pousse une prédiction à la hausse ou à la baisse.
- **Trajectoires de valeur marchande** : reconstitution de l'historique complet de chaque joueur (toutes saisons, tous splits confondus), calcul des écarts de valeur d'une saison à l'autre (€ et %), puis **clustering** des joueurs selon le niveau et la pente de leur trajectoire de valeur. Quatre profils sont identifiés et interprétés par SHAP :

  | Profil | Effectif | VM moyenne | Évolution / saison | Lecture |
  |---|---|---|---|---|
  | Rotation / valeur modeste | 694 (49.1 %) | ≈ 3.9 M€ | -1.1 M€ | Fortement pénalisé par l'âge ; cible pour du volume ou de la plus-value sur profils jeunes |
  | Cadres en progression | 421 (29.8 %) | ≈ 21.3 M€ | +1.9 M€ | Portés par le niveau collectif du club ; cible pour la performance immédiate à risque maîtrisé |
  | Stars post-pic en repli | 203 (14.4 %) | ≈ 14.7 M€ | -7.0 M€ | Le plus pénalisé par l'âge/l'éloignement du pic ; à éviter en optique plus-value |
  | Superstars en forte hausse | 96 (6.8 %) | ≈ 68.3 M€ | +12.1 M€ | Amplitude d'impact hors norme de la VM N-1 ; cible premium performance immédiate |

  Cette segmentation sert de **support de storytelling** pour orienter une stratégie de recrutement : *plus-value* (dénicher un profil qui va prendre de la valeur) vs *performance immédiate* (recruter un profil déjà performant, quitte à payer plus cher). Ces lectures restent à croiser avec le poste du joueur recherché.

## Technologies

- **Traitement de données** : pandas, numpy
- **Modélisation** : scikit-learn, XGBoost, LightGBM, CatBoost
- **Optimisation d'hyperparamètres** : Optuna
- **Acquisition de données** : API Kaggle, `soccerdata`, `worldfootballR` (R), scraping Python (blessures Transfermarkt)

Voir `requirements.txt` pour la liste complète et les versions exactes des dépendances.

---
