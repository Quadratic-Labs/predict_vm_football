# predict_vm_football

Projet de prédiction de la **valeur marchande (VM)** des joueurs de football à partir de leurs statistiques de performance, données de club, classements et historique de blessures.

> ⚠️ Certaines sections ci-dessous (contexte, sources de données, licence) sont rédigées à partir de la structure du repo — à relire et corriger si besoin.

## Sommaire

- [Contexte](#contexte)
- [Structure du repo](#structure-du-repo)
- [Pipeline du projet](#pipeline-du-projet)
- [Sources de données](#sources-de-données)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Approche de modélisation](#approche-de-modélisation)
- [Technologies](#technologies)

## Contexte

Ce projet a été développé dans le cadre d'un stage. Il vise à construire un modèle de machine learning capable d'estimer la valeur marchande d'un joueur de football à partir de ses statistiques de performance (données FBref/Transfermarkt), de son contexte club (classement, compétition) et d'un historique de blessures.

## Structure du repo

```
predict_vm_football/
├── data/                        # Données brutes
│   ├── classement_fifa/
│   ├── soccerdata/
│   ├── transfermarkt_datasets/
│   ├── classement_fin_saison.csv
│   ├── dataset_blessures.csv
│   └── football_data.csv
├── data_finale/                 # Données transformées / prêtes pour la modélisation
│   ├── featuring/
│   ├── mapping_worldfootballR/
│   ├── pipelines/
│   ├── base_apprentissage.csv
│   ├── train.csv / val.csv / test.csv
│   └── en_cours.csv
├── fonctions/                    # Fonctions Python réutilisées dans les notebooks
│   ├── exploration/
│   ├── modelisations/
│   ├── cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── imports.py
│   ├── merging.py
│   └── utils.py
├── modelisation/                 # Artefacts liés à la modélisation (modèles, résultats, etc.)
├── notebooks/                    # Pipeline du projet, exécuté dans l'ordre
│   ├── 01_acquisition_donnees.ipynb
│   ├── 02_exploration_qualite.ipynb
│   ├── 03_base_apprentissage.ipynb
│   ├── 04_nettoyage_base.ipynb
│   ├── 05_exploration_approfondie.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_modelisation_baseline.ipynb
│   ├── 08_tests_modelisation.ipynb
│   └── 09_modelisation_finale.ipynb
├── outputs/                      # Résultats, figures, exports générés
├── .env                          # Variables d'environnement (non versionné)
├── .gitignore
├── install.ps1                   # Script d'installation (Windows)
├── requirements.txt
└── README.md
```

## Pipeline du projet

Les notebooks du dossier `notebooks/` forment la chaîne de traitement du projet, à exécuter dans l'ordre :

| # | Notebook | Rôle |
|---|----------|------|
| 01 | `acquisition_donnees` | Récupération des données brutes (scraping / import des sources) |
| 02 | `exploration_qualite` | Contrôle qualité des données (valeurs manquantes, doublons, cohérence) |
| 03 | `base_apprentissage` | Constitution de la base d'apprentissage à partir des sources brutes |
| 04 | `nettoyage_base` | Nettoyage de la base (traitement des valeurs manquantes, incohérences) |
| 05 | `exploration_approfondie` | Analyse exploratoire détaillée (distributions, corrélations) |
| 06 | `feature_engineering` | Création des variables utilisées par les modèles |
| 07 | `modelisation_baseline` | Premiers modèles de référence |
| 08 | `tests_modelisation` | Expérimentations de modélisation (comparaison d'algorithmes, stacking, transformations de la cible, retrait des outliers, réduction de dimensionnalité, modèles par groupe de postes) |
| 09 | `modelisation_finale` | Modèle final retenu et évaluation |

Les fonctions communes utilisées par ces notebooks (nettoyage, EDA, feature engineering, modélisation) sont centralisées dans `fonctions/`, pour éviter la duplication de code d'un notebook à l'autre.

## Sources de données

- **`data/soccerdata/`** — données de performance issues du package [`soccerdata`](https://github.com/probberechts/soccerdata) (FBref, etc.)
- **`data/transfermarkt_datasets/`** — valeurs marchandes et données de transfert (Transfermarkt)
- **`data/classement_fifa/`** et **`classement_fin_saison.csv`** — classements de clubs / compétitions
- **`dataset_blessures.csv`** — historique de blessures des joueurs
- **`data_finale/mapping_worldfootballR/`** — table de correspondance construite avec le package R [`worldfootballR`](https://github.com/JaseZiv/worldFootballR) pour faire le lien entre les différentes sources

## Installation

### Étapes

```bash
# Cloner le repo
git clone <url-du-repo>
cd predict_vm_football

# Créer et activer un environnement virtuel
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux

# Installer les dépendances
pip install -r requirements.txt
```

Le script `install.ps1` peut être utilisé pour automatiser cette installation :

```powershell
.\install.ps1
```

Un fichier `.env` est utilisé pour les variables d'environnement (clés d'API, chemins, etc.) — voir `.env` pour la liste des variables attendues (fichier non versionné, à créer localement).

## Utilisation

1. Activer l'environnement virtuel (`.venv`).
2. Exécuter les notebooks dans l'ordre numéroté (`01_...` à `09_...`) depuis le dossier `notebooks/`.
3. Les données intermédiaires et finales sont écrites dans `data_finale/`, les figures dans `outputs/`.

## Approche de modélisation

Le notebook `08_tests_modelisation.ipynb` documente les expérimentations menées pour choisir le modèle final :

- **Algorithmes comparés** : baseline, XGBoost, LightGBM, CatBoost, SVR, ElasticNet (tuning d'hyperparamètres via Optuna)
- **Transformation de la cible** : comparaison cible brute vs cible log, et test de plusieurs transformations (Power, Quantile, MinMax) pour stabiliser la variance et améliorer les performances
- **Stacking OOF** : combinaison des modèles de base (XGBoost, LightGBM, CatBoost) via un méta-modèle entraîné sur des prédictions out-of-fold
- **Retrait des valeurs extrêmes** : ré-entraînement sur une base sans outliers pour évaluer l'impact sur la performance
- **Réduction de dimensionnalité** : ACP, sélection de features par importance (gain XGBoost) et permutation importance
- **Modèles par groupe** : comparaison d'un modèle global vs des modèles spécifiques par poste (gardiens / joueurs de champ)

Le notebook `09_modelisation_finale.ipynb` reprend l'approche retenue à l'issue de ces tests pour produire le modèle final.
---