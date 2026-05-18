import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def load_and_check_types(filepath):
    """Charge le dataset et retourne les types de colonnes et la forme."""
    df = pd.read_csv(filepath)
    df_info = pd.DataFrame({
        'Type': df.dtypes,
        'Valeurs Non-Nulles': df.count(),
        'Pourcentage Complétion': (df.count() / len(df)) * 100
    })
    display(df)
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
    """Calcule les statistiques descriptives des variables numériques clés."""
    # Nettoyage rapide de la colonne 'Jours' pour l'analyse numérique ("24 days" -> 24)
    df_temp = df.copy()
    if 'Jours' in df_temp.columns:
        df_temp['Jours_num'] = df_temp['Jours'].astype(str).str.extract('(\d+)').astype(float)
    
    # Remplacement des "-" par NaN dans Matchs_Manques
    if 'Matchs_Manques' in df_temp.columns:
        df_temp['Matchs_Manques_num'] = pd.to_numeric(df_temp['Matchs_Manques'], errors='coerce')
        
    cols_to_desc = [c for c in ['Jours_num', 'Matchs_Manques_num'] if c in df_temp.columns]
    print("\nStatistiques de variables clés")
    display(df_temp[cols_to_desc].describe())
    return df_temp[cols_to_desc].describe()

def plot_injury_distributions(df):
    """Génère les graphiques pour le Notebook."""
    df_temp = df.copy()
    df_temp['Jours_num'] = df_temp['Jours'].astype(str).str.extract('(\d+)').astype(float)
    df_temp['Matchs_Manques_num'] = pd.to_numeric(df_temp['Matchs_Manques'], errors='coerce')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Boxplot des jours d'indisponibilité
    sns.boxplot(data=df_temp, x='Jours_num', ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution des Jours d\'Indisponibilité (Détection des Outliers)')
    axes[0].set_xlabel('Nombre de jours')
    
    # Top 10 des types de blessures
    sns.countplot(data=df_temp, y='Blessure', order=df_temp['Blessure'].value_counts().iloc[:10].index, ax=axes[1], palette='Blues_r')
    axes[1].set_title('Top 10 des Types de Blessures les plus Fréquentes')
    axes[1].set_xlabel('Fréquence')
    
    plt.tight_layout()
    return fig