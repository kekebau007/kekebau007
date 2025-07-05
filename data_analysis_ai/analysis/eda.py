import pandas as pd
import numpy as np

def get_dataframe_info(df: pd.DataFrame):
    """
    Affiche des informations de base sur le DataFrame: types de données, valeurs manquantes.
    """
    print("Informations générales du DataFrame :")
    df.info()
    print("\n" + "="*30 + "\n")

    print("Nombre de valeurs manquantes par colonne :")
    print(df.isnull().sum())
    print("\n" + "="*30 + "\n")

    print("Pourcentage de valeurs manquantes par colonne :")
    print((df.isnull().sum() / len(df)) * 100)
    print("\n" + "="*30 + "\n")

def get_descriptive_statistics(df: pd.DataFrame, include_dtypes: list = None):
    """
    Calcule et affiche les statistiques descriptives.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        include_dtypes (list, optional): Liste des types de données à inclure (ex: ['number', 'object']).
                                        Si None, utilise le comportement par défaut de df.describe(include='all').
    """
    print("Statistiques descriptives :")
    if include_dtypes:
        print(df.describe(include=include_dtypes))
    else:
        print(df.describe(include='all')) # Inclut numériques et catégorielles (top, freq)
    print("\n" + "="*30 + "\n")

def get_value_counts(df: pd.DataFrame, columns: list = None, limit_per_col: int = 10):
    """
    Affiche le comptage des valeurs pour les colonnes spécifiées (ou colonnes non numériques).

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        columns (list, optional): Liste des colonnes pour lesquelles afficher le comptage.
                                  Si None, cible les colonnes 'object' et 'category'.
        limit_per_col (int): Nombre maximum de valeurs uniques à afficher par colonne.
    """
    print(f"Comptage des valeurs (top {limit_per_col} par colonne) :")
    if columns is None:
        columns_to_process = df.select_dtypes(include=['object', 'category']).columns
    else:
        columns_to_process = [col for col in columns if col in df.columns]

    if not columns_to_process.any(): # .any() pour Series/Index, isEmpty pour listes
        print("Aucune colonne appropriée trouvée ou spécifiée pour le comptage des valeurs.")
        return

    for col in columns_to_process:
        print(f"\n--- Colonne : {col} ---")
        counts = df[col].value_counts(dropna=False) # Inclut les NaN dans le comptage
        print(counts.head(limit_per_col))
        if len(counts) > limit_per_col:
            print(f"... et {len(counts) - limit_per_col} autres valeurs.")
    print("\n" + "="*30 + "\n")


def get_correlation_matrix(df: pd.DataFrame, method: str = 'pearson', columns_to_correlate: list = None):
    """
    Calcule et affiche la matrice de corrélation pour les colonnes numériques.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        method (str): Méthode de corrélation ('pearson', 'kendall', 'spearman').
        columns_to_correlate (list, optional): Liste spécifique de colonnes numériques à utiliser.
                                               Si None, toutes les colonnes numériques sont utilisées.
    """
    print(f"Matrice de corrélation (méthode : {method}) :")

    numeric_df = df.select_dtypes(include=np.number)

    if columns_to_correlate:
        # S'assurer que les colonnes spécifiées sont présentes et numériques
        valid_cols = [col for col in columns_to_correlate if col in numeric_df.columns]
        if not valid_cols:
            print("Aucune colonne numérique valide spécifiée pour la corrélation.")
            return
        numeric_df_subset = numeric_df[valid_cols]
    else:
        numeric_df_subset = numeric_df

    if numeric_df_subset.empty or numeric_df_subset.shape[1] < 2:
        print("Pas assez de colonnes numériques pour calculer la corrélation.")
        return

    corr_matrix = numeric_df_subset.corr(method=method)
    print(corr_matrix)
    print("\n" + "="*30 + "\n")
    return corr_matrix # Retourne la matrice pour une utilisation potentielle (ex: heatmap)


if __name__ == '__main__':
    data_eda = {
        'ID': [1, 2, 3, 4, 5, 6, 7, 8],
        'Age': [25, 30, 22, 45, 33, np.nan, 30, 28],
        'Salaire': [50000, 60000, 45000, 90000, 75000, 50000, 120000, 60000],
        'Departement': ['RH', 'IT', 'Marketing', 'IT', 'RH', 'IT', 'Finance', 'Marketing'],
        'Experience': [2, 5, 1, 10, 7, 2, 15, 5],
        'Satisfaction': ['Haute', 'Moyenne', 'Basse', 'Haute', np.nan, 'Moyenne', 'Haute', 'Basse'],
        'Projets': [3, 5, 2, 8, 6, 3, 10, 4]
    }
    df_test_eda = pd.DataFrame(data_eda)

    print("===== Test get_dataframe_info =====")
    get_dataframe_info(df_test_eda)

    print("===== Test get_descriptive_statistics (tout) =====")
    get_descriptive_statistics(df_test_eda)

    print("===== Test get_descriptive_statistics (numérique seulement) =====")
    get_descriptive_statistics(df_test_eda, include_dtypes=['number'])

    print("===== Test get_descriptive_statistics (objet seulement) =====")
    get_descriptive_statistics(df_test_eda, include_dtypes=['object'])

    print("===== Test get_value_counts (auto) =====")
    get_value_counts(df_test_eda)

    print("===== Test get_value_counts (colonnes spécifiques) =====")
    get_value_counts(df_test_eda, columns=['Departement', 'Age', 'Satisfaction'], limit_per_col=3)

    print("===== Test get_correlation_matrix (pearson, auto) =====")
    correlation_matrix = get_correlation_matrix(df_test_eda)

    print("===== Test get_correlation_matrix (spearman, colonnes spécifiques) =====")
    correlation_matrix_specific = get_correlation_matrix(df_test_eda, method='spearman', columns_to_correlate=['Age', 'Salaire', 'Projets'])

    print("===== Test get_correlation_matrix (pas assez de colonnes) =====")
    df_few_cols = pd.DataFrame({'A': [1,2,3], 'B': ['x','y','z']})
    get_correlation_matrix(df_few_cols)

    print("===== Test get_correlation_matrix (colonnes spécifiées non numériques) =====")
    get_correlation_matrix(df_test_eda, columns_to_correlate=['Departement'])
