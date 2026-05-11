import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# Colonnes clés issues de la revue de littérature (joueurs de champ)
KEY_COLS_FIELD = [
    'Playing Time_Min', 'Playing Time_MP',
    'Performance_Gls', 'Performance_Ast', 'Performance_G+A',
    'Per 90 Minutes_Gls', 'Per 90 Minutes_Ast',
    'Standard_Sh', 'Standard_SoT', 'Standard_Sh/90', 'Standard_SoT/90',
    'xg', 'xa', 'np_xg', 'xg_chain', 'xg_buildup',
    'Performance_Fls', 'Performance_Int', 'Performance_TklW',
    'Performance_CrdY', 'Performance_CrdR',
    'age', 'born', 'nation', 'pos', 'league', 'season',
]

KEY_COLS_GK = [
    'Playing Time_Min', 'Playing Time_MP',
    'Performance_GA', 'Performance_GA90', 'Performance_SoTA',
    'Performance_Saves', 'Performance_Save%',
    'Performance_CS', 'Performance_CS%',
    'age', 'born', 'nation', 'league', 'season',
]

LEAGUE_LABELS = {
    'ENG-Premier League': 'Premier League',
    'ESP-La Liga':        'La Liga',
    'FRA-Ligue 1':        'Ligue 1',
    'GER-Bundesliga':     'Bundesliga',
    'ITA-Serie A':        'Serie A',
}


def split_field_gk(df):
    """Sépare joueurs de champ et gardiens."""
    gk    = df[df['pos'].str.contains('GK', na=False)].copy()
    field = df[~df['pos'].str.contains('GK', na=False)].copy()
    return field, gk


def audit_missing_values(df, title="Valeurs manquantes", threshold_pct=20):
    """
    Visualise le taux de valeurs manquantes pour toutes les colonnes contenant des NaN.

    arguments:
        df: DataFrame à analyser.
        title (str): Titre du graphique.
        threshold_pct (float): Seuil d'alerte affiché en rouge (défaut : 20%).

    returns:
        pd.Series: Pourcentage de manquants par colonne, trié décroissant.
    """
    missing_pct = (df.isnull().sum() / len(df) * 100)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if missing_pct.empty:
        print(f"[{title}] Aucune valeur manquante.")
        return missing_pct

    colors = ['crimson' if v > threshold_pct else 'steelblue' for v in missing_pct.values]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(missing_pct.index, missing_pct.values, color=colors)
    ax.axhline(threshold_pct, color='black', linestyle='--', linewidth=1,
               label=f'Seuil {threshold_pct}%')
    ax.set_title(f"{title} — % de NaN par colonne", fontsize=13)
    ax.set_ylabel("% de NaN")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.show()

    above = missing_pct[missing_pct > threshold_pct]
    print(f"{len(above)} colonne(s) au-dessus du seuil {threshold_pct}% :")
    for col, val in above.items():
        print(f"  {col:<45} {val:.1f}%")

    return missing_pct


def audit_key_columns_missing(df, key_cols, title="Couverture des variables clés"):
    """
    Vérifie le taux de complétude uniquement sur les colonnes issues de la revue de littérature.

    arguments:
        df: DataFrame à analyser.
        key_cols (list): Liste des colonnes à inspecter.
        title (str): Titre du graphique.

    returns:
        pd.DataFrame: Table de couverture (colonne, présente, % manquant).
    """
    results = []
    for col in key_cols:
        if col in df.columns:
            pct_missing = df[col].isnull().sum() / len(df) * 100
            results.append({'colonne': col, 'présente': True, '% manquant': round(pct_missing, 2)})
        else:
            results.append({'colonne': col, 'présente': False, '% manquant': 100.0})

    df_result = pd.DataFrame(results).sort_values('% manquant', ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df_result) * 0.35)))
    colors = ['crimson' if not r['présente'] else
              'orange'  if r['% manquant'] > 20 else
              'seagreen'
              for _, r in df_result.iterrows()]
    ax.barh(df_result['colonne'], df_result['% manquant'], color=colors)
    ax.axvline(20, color='black', linestyle='--', linewidth=1, label='Seuil 20%')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("% de NaN")
    ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color='seagreen', label='OK (< 20%)'),
        mpatches.Patch(color='orange',   label='Partiel (> 20%)'),
        mpatches.Patch(color='crimson',  label='Colonne absente'),
    ]
    ax.legend(handles=legend_patches)
    plt.tight_layout()
    plt.show()

    absent = df_result[~df_result['présente']]
    if not absent.empty:
        print(f"Colonnes absentes du DataFrame : {absent['colonne'].tolist()}")

    return df_result


def audit_coverage_by_season(df, title="Couverture par saison et ligue"):
    """
    Affiche le nombre de joueurs par saison et par ligue pour vérifier la couverture temporelle.

    arguments:
        df: DataFrame contenant les colonnes 'season' et 'league'.
        title (str): Titre du graphique.

    returns:
        pd.DataFrame: Table de couverture (saisons × ligues).
    """
    df_plot = df.copy()
    df_plot['league'] = df_plot['league'].map(LEAGUE_LABELS).fillna(df_plot['league'])

    coverage = df_plot.groupby(['season', 'league']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    coverage.plot(kind='bar', ax=ax, colormap='tab10')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Saison")
    ax.set_ylabel("Nombre de joueurs")
    ax.legend(title="Ligue", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(coverage.to_string())
    return coverage


def audit_distributions(df, metrics=None, title="Distributions des métriques clés"):
    if metrics is None:
        metrics = [
            'Playing Time_Min', 'Performance_Gls', 'Performance_Ast',
            'Standard_Sh/90', 'xg', 'xa',
            'Performance_Int', 'Performance_TklW', 'age',
        ]

    # On ne garde que les colonnes présentes ET numériques
    metrics = [
        m for m in metrics
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m])
    ]

    n = len(metrics)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(metrics):
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=axes[i], color='steelblue', bins=30)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
        median_val = data.median()  # ← maintenant garanti numérique
        axes[i].axvline(median_val, color='red', linestyle='--', linewidth=1,
                        label=f'Médiane : {median_val:.1f}')
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def audit_xg_missing_by_season(df, title="Taux de NaN sur xG par saison"):
    """
    Analyse et visualise le taux de valeurs manquantes sur xG et xA par saison.
    Utile pour diagnostiquer si le manque de données xG est structurel ou limité à certaines saisons.

    arguments:
        df: DataFrame contenant les colonnes 'xg', 'xa', 'season'.
        title (str): Titre du graphique.

    returns:
        pd.DataFrame: Taux de NaN sur xg et xa par saison.
    """
    xg_missing = df.groupby('season')[['xg', 'xa', 'np_xg']].apply(
        lambda x: x.isnull().sum() / len(x) * 100
    ).round(1)

    fig, ax = plt.subplots(figsize=(10, 4))
    xg_missing.plot(kind='bar', ax=ax, colormap='Set2')
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("% de NaN")
    ax.set_xlabel("Saison")
    ax.axhline(20, color='black', linestyle='--', linewidth=1, label='Seuil 20%')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(xg_missing.to_string())
    return xg_missing


def run_full_quality_audit(df, verbose=True):
    """
    Lance l'audit qualité complet du dataset SoccerData.

    Enchaîne : séparation GK/field → manquants globaux → couverture variables clés
    → couverture temporelle → distributions → diagnostic xG.

    arguments:
        df: DataFrame brut issu de data_final_soccerdata.csv.
        verbose (bool): Si True, affiche les résultats intermédiaires.

    returns:
        dict: Résultats des audits (clés : 'missing_global', 'key_coverage',
              'coverage_season', 'xg_missing').
    """
    print(f"AUDIT QUALITÉ — SoccerData  ({len(df):,} lignes, {df.shape[1]} colonnes)")

    df_field, df_gk = split_field_gk(df)
    print(f"\nJoueurs de champ : {len(df_field):,}  |  Gardiens : {len(df_gk):,}\n")

    print("\n1. Valeurs manquantes — joueurs de champ")
    missing_global = audit_missing_values(df_field, title="Joueurs de champ")

    print("\n2. Couverture des variables clés")
    key_coverage_field = audit_key_columns_missing(
        df_field, KEY_COLS_FIELD, title="Variables clés — Joueurs de champ"
    )
    key_coverage_gk = audit_key_columns_missing(
        df_gk, KEY_COLS_GK, title="Variables clés — Gardiens"
    )

    print("\n3. Couverture temporelle et géographique")
    coverage_season = audit_coverage_by_season(df)

    print("\n4. Distributions des métriques clés")
    audit_distributions(df_field, title="Distributions — Joueurs de champ")

    print("\n5. Diagnostic des données xG")
    xg_missing = audit_xg_missing_by_season(df_field)

    print("\nAudit terminé.")

    return {
        'missing_global':   missing_global,
        'key_coverage':     key_coverage_field,
        'key_coverage_gk':  key_coverage_gk,
        'coverage_season':  coverage_season,
        'xg_missing':       xg_missing,
    }