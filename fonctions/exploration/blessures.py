import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def load_and_check_types(filepath):
    """Charge le dataset et retourne les types de colonnes et la forme."""
    df = pd.read_csv(filepath)
    df_info = pd.DataFrame({
        'Type': df.dtypes,
    })
    display(df_info)
    return df, df_info


def check_missing_and_duplicates(df):
    """Analyse les valeurs manquantes exactes et les doublons."""
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


def analyze_distributions(df):
    """Calcule les statistiques descriptives des variables numériques clés 
    et analyse le volume de blessures par saison.
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
    """Génère les graphiques pour le Notebook incluant les jours d'indisponibilité,
    le top 10 des blessures et le top 10 des clubs les plus touchés.
    """
    df_temp = df.copy()
    df_temp['Jours_num'] = df_temp['Jours'].astype(str).str.extract('(\d+)').astype(float)
    df_temp['Matchs_Manques_num'] = pd.to_numeric(df_temp['Matchs_Manques'], errors='coerce')
    
    # Passage à 3 sous-graphiques (1 ligne, 3 colonnes) et élargissement de la figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    
    # 1. Boxplot des jours d'indisponibilité
    sns.boxplot(data=df_temp, x='Jours_num', ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution des Jours d\'Indisponibilité\n(Détection des Outliers)')
    axes[0].set_xlabel('Nombre de jours')
    
    # 2. Top 10 des types de blessures
    top_blessures = df_temp['Blessure'].value_counts().iloc[:10].index
    sns.countplot(data=df_temp, y='Blessure', order=top_blessures, ax=axes[1], palette='Blues_r')
    axes[1].set_title('Top 10 des Types de Blessures\nles plus Fréquentes')
    axes[1].set_xlabel('Fréquence')
    axes[1].set_ylabel('Type de Blessure')
    
    # 3. AJOUT : Top 10 des clubs les plus fréquents (les plus touchés par les blessures)
    top_clubs = df_temp['Club_Blessure'].value_counts().iloc[:10].index
    sns.countplot(data=df_temp, y='Club_Blessure', order=top_clubs, ax=axes[2], palette='Oranges_r')
    axes[2].set_title('Top 10 des Clubs les plus Fréquents\ndans le Dataset')
    axes[2].set_xlabel('Nombre de lignes (Blessures)')
    axes[2].set_ylabel('Club')
    
    plt.tight_layout()
    return fig