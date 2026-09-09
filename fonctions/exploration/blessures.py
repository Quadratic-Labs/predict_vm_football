import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def load_and_check_types(filepath):
    """
    Charge un jeu de données CSV, affiche les types de ses colonnes et retourne les structures associées.

    arguments:
        filepath (str): Le chemin d'accès vers le fichier CSV à charger.

    returns:
        tuple[dataframe, dataframe]: Un triplet contenant :
            - df (dataframe) : Le dataframe principal contenant l'intégralité des données chargées.
            - df_info (dataframe) : Un dataframe récapitulatif avec l'index correspondant aux 
              noms des colonnes du dataset et une unique colonne 'Type' indiquant leur type de données.

    """
    df = pd.read_csv(filepath)
    df_info = pd.DataFrame({
        'Type': df.dtypes,
    })
    display(df_info)
    return df, df_info


def check_missing_and_duplicates(df):
    """
    Analyse la présence de valeurs manquantes (NAs) et de lignes doublons dans un DataFrame.

    Cette fonction calcule pour chaque colonne le nombre absolu de valeurs manquantes 
    ainsi que leur proportion en pourcentage par rapport à la taille totale du jeu de données. 
    Elle comptabilise également les doublons parfaits (lignes strictement identiques). 
    Les résultats sont mis en forme dans un tableau récapitulatif et affichés directement 
    pour faciliter le diagnostic de la qualité des données.

    arguments:
        df (dataframe): Le dataframe à analyser.

    returns:
        tuple[dataframe, int]: Un doublet contenant :
            - analysis_df (dataframe) : Un dataframe indexé par le nom des colonnes, 
              contenant deux colonnes : 'Manquants (Absolus)' et 'Manquants (%)'.
            - duplicates (int) : Le nombre total de lignes dupliquées identifiées.
    """
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    duplicates = df.duplicated().sum()
    
    analysis_df = pd.DataFrame({
        'Manquants (Absolus)': missing,
        'Manquants (%)': missing_pct
    })
    print("Taux de NA")
    display(analysis_df)
    print(f"\nNombre de lignes doublons : {duplicates}")
    return analysis_df, duplicates


def analyse_distributions(df):
    """
    Calcule les statistiques descriptives des variables d'indisponibilité et le volume de
    blessures par saison.

    Cette fonction réalise un nettoyage à la volée sur une copie du DataFrame pour isoler 
    les composantes numériques des variables textuelles issues du scraping :
    Elle extrait la partie numérique de la colonne 'Jours' (ex: "24 days" devient 24.0).
    Elle convertit la colonne 'Matchs_Manques' en valeurs numériques, en remplaçant les 
    caractères invalides ou masqués (comme les tirets '-') par des valeurs manquantes (NaN).

    Après ces conversions, elle génère un résumé statistique complet  pour ces variables clés
    et calcule la distribution volumétrique globale (nombre de lignes) par saison sportive.

    arguments:
        df (dataframe): Le dataframe d'origine contenant l'historique des blessures.

    Returns:
        tuple[dataframe, dataframe | None]: Un doublet contenant :
            - stats_globales (dataframe) : Tableau descriptif complet des 
              nouvelles variables calculées ('Jours_num' et 'Matchs_Manques_num').
            - blessures_par_saison (dataframe ou None) : Un dataframe indexé par la colonne 'Saison' 
              contenant une colonne 'Nombre_de_Blessures'. Retourne `None` si la colonne 'Saison' 
              est absente du DataFrame fourni.
    """
    df_temp = df.copy()
    
    # Nettoyage et conversion numérique
    if 'Jours' in df_temp.columns:
        df_temp['Jours_num'] = df_temp['Jours'].astype(str).str.extract('(\d+)').astype(float)
    
    if 'Matchs_Manques' in df_temp.columns:
        df_temp['Matchs_Manques_num'] = pd.to_numeric(df_temp['Matchs_Manques'], errors='coerce')
        
    # Calcul du nombre de blessures par saison
    # On groupe par saison et on compte le nombre total de lignes (blessures)
    if 'Saison' in df_temp.columns:
        blessures_par_saison = df_temp.groupby('Saison').size().to_frame(name='Nombre_de_Blessures')
        print("\nVolume de blessures par saison")
        print(blessures_par_saison)
    else:
        blessures_par_saison = None
        print("\n[Attention] La colonne 'Saison' est introuvable pour analyser les volumes.")

    # Calcul des statistiques descriptives globales
    cols_to_desc = [c for c in ['Jours_num', 'Matchs_Manques_num'] if c in df_temp.columns]
    stats_globales = df_temp[cols_to_desc].describe()
    
    print("\nStatistiques descriptives des variables clés")
    print(stats_globales)
    
    # On retourne les deux éléments sous forme de tuple pour pouvoir les réutiliser au besoin
    return stats_globales, blessures_par_saison

def plot_injury_distributions(df):
   """
    Génère un ensemble de trois graphiques complémentaires pour analyser la distribution des
    blessures.

    Un diagramme en boîte de la variable numérique 'Jours_num' pour identifier 
    la dispersion des indisponibilités et mettre en évidence les valeurs aberrantes.
    Un diagramme en barres horizontales affichant le Top 10 des types de blessures 
    les plus fréquentes.
    Un diagramme en barres horizontales affichant le Top 10 des clubs comptabilisant le plus 
    grand nombre de lignes d'indisponibilité dans le dataset.

    Avant la génération des graphiques, un traitement de nettoyage local est opéré sur les variables 
    de gravité ('Jours' et 'Matchs_Manques') pour les convertir au format numérique.

    arguments:
        df (dataframe): Le dataframe d'origine contenant les colonnes 'Jours', 'Blessure', 
          'Club_Blessure' et éventuellement 'Matchs_Manques'.

    returns:
        figures: L'objet conteneur de la figure Matplotlib (`fig`) contenant 
        les trois axes de graphiques configurés.

    """
   df_temp = df.copy()
   df_temp['Jours_num'] = df_temp['Jours'].astype(str).str.extract('(\d+)').astype(float)
   df_temp['Matchs_Manques_num'] = pd.to_numeric(df_temp['Matchs_Manques'], errors='coerce')
    
   # Passage à 3 sous-graphiques (1 ligne, 3 colonnes) et élargissement de la figure
   fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
   # Boxplot des jours d'indisponibilité
   sns.boxplot(data=df_temp, x='Jours_num', ax=axes[0], color='skyblue')
   axes[0].set_title('Distribution des Jours d\'Indisponibilité\n(Détection des Outliers)')
   axes[0].set_xlabel('Nombre de jours')
    
   # Top 10 des types de blessures
   top_blessures = df_temp['Blessure'].value_counts().iloc[:10].index
   sns.countplot(data=df_temp, y='Blessure', order=top_blessures, ax=axes[1], palette='Blues_r')
   axes[1].set_title('Top 10 des Types de Blessures\nles plus Fréquentes')
   axes[1].set_xlabel('Fréquence')
   axes[1].set_ylabel('Type de Blessure')
   
   # Top 10 des clubs les plus fréquents (les plus touchés par les blessures)
   top_clubs = df_temp['Club_Blessure'].value_counts().iloc[:10].index
   sns.countplot(data=df_temp, y='Club_Blessure', order=top_clubs, ax=axes[2], palette='Oranges_r')
   axes[2].set_title('Top 10 des Clubs les plus Fréquents\ndans le Dataset')
   axes[2].set_xlabel('Nombre de lignes (Blessures)')
   axes[2].set_ylabel('Club')
   
   plt.tight_layout()
   return fig