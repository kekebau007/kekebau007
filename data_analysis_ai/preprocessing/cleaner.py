import pandas as pd
import numpy as np

def handle_missing_values(df: pd.DataFrame, strategy: str = 'auto', subset_cols: list = None, fill_value_numeric = None, fill_value_categorical = None):
    """
    Gère les valeurs manquantes dans un DataFrame.

    Stratégies :
    - 'auto': Tente de remplir numériquement avec la moyenne et catégoriquement avec le mode.
              Si fill_value_numeric ou fill_value_categorical est fourni, il sera utilisé.
    - 'remove_row': Supprime les lignes contenant au moins une valeur manquante (dans subset_cols si fourni).
    - 'remove_col': Supprime les colonnes contenant au moins une valeur manquante.
    - 'fill_mean': Remplit les valeurs manquantes numériques avec la moyenne de la colonne.
    - 'fill_median': Remplit les valeurs manquantes numériques avec la médiane de la colonne.
    - 'fill_mode': Remplit les valeurs manquantes (numériques ou catégorielles) avec le mode de la colonne.
    - 'fill_custom': Remplit les valeurs manquantes avec `fill_value_numeric` pour les colonnes numériques
                     et `fill_value_categorical` pour les colonnes catégorielles/objets.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        strategy (str): La stratégie à utiliser pour gérer les valeurs manquantes.
        subset_cols (list, optional): Liste des colonnes sur lesquelles appliquer la stratégie.
                                      Si None, s'applique à toutes les colonnes.
        fill_value_numeric (float/int, optional): Valeur personnalisée pour remplir les NaN numériques
                                                 (utilisé avec strategy 'fill_custom' ou 'auto').
        fill_value_categorical (str, optional): Valeur personnalisée pour remplir les NaN catégoriels/objets
                                               (utilisé avec strategy 'fill_custom' ou 'auto').

    Returns:
        pd.DataFrame: DataFrame avec les valeurs manquantes traitées.
    """
    df_processed = df.copy()
    cols_to_process = subset_cols if subset_cols else df_processed.columns

    if strategy == 'remove_row':
        df_processed.dropna(subset=cols_to_process, inplace=True)
        print(f"Lignes avec NaN supprimées pour les colonnes : {cols_to_process}")
    elif strategy == 'remove_col':
        # Si subset_cols est fourni, on ne vérifie que ces colonnes pour suppression
        # Sinon, on vérifie toutes les colonnes.
        # Note: cette logique supprime la colonne entière si elle a un NaN, même si le NaN est hors subset_cols
        # si on voulait supprimer seulement des colonnes de subset_cols qui ont des NaN:
        # for col in cols_to_process:
        #     if df_processed[col].isnull().any():
        #         df_processed.drop(columns=col, inplace=True)
        df_processed.dropna(axis=1, how='any', inplace=True) # Supprime toute colonne avec au moins un NaN
        print(f"Colonnes avec NaN supprimées.")

    elif strategy in ['fill_mean', 'fill_median', 'fill_mode', 'fill_custom', 'auto']:
        for col in cols_to_process:
            if df_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    if strategy == 'fill_mean':
                        fill_val = df_processed[col].mean()
                    elif strategy == 'fill_median':
                        fill_val = df_processed[col].median()
                    elif strategy == 'fill_mode': # Peut aussi s'appliquer aux numériques
                        fill_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else df_processed[col].mean()
                    elif strategy == 'fill_custom' and fill_value_numeric is not None:
                        fill_val = fill_value_numeric
                    elif strategy == 'auto':
                        if fill_value_numeric is not None:
                            fill_val = fill_value_numeric
                        else: # Comportement par défaut de 'auto' pour numérique
                            fill_val = df_processed[col].mean()
                    else: # Fallback si la stratégie numérique n'est pas entièrement couverte ou custom val non fournie
                        fill_val = df_processed[col].mean()
                    df_processed[col].fillna(fill_val, inplace=True)
                    print(f"NaN dans la colonne numérique '{col}' remplis avec {fill_val} (stratégie: {strategy})")

                else: # Colonnes catégorielles ou objets
                    if strategy == 'fill_mode':
                        fill_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Unknown"
                    elif strategy == 'fill_custom' and fill_value_categorical is not None:
                        fill_val = fill_value_categorical
                    elif strategy == 'auto':
                        if fill_value_categorical is not None:
                            fill_val = fill_value_categorical
                        else: # Comportement par défaut de 'auto' pour catégoriel
                            fill_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Unknown"
                    else: # Fallback pour catégoriel
                         fill_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Unknown"

                    df_processed[col].fillna(fill_val, inplace=True)
                    print(f"NaN dans la colonne catégorielle/objet '{col}' remplis avec '{fill_val}' (stratégie: {strategy})")
    else:
        print(f"Stratégie '{strategy}' non reconnue. Aucune action effectuée.")

    return df_processed


if __name__ == '__main__':
    data_missing = {
        'col_num_1': [1, 2, np.nan, 4, 5, np.nan],
        'col_num_2': [np.nan, 2.5, 3.5, 4.5, 5.5, 6.5],
        'col_cat_1': ['A', 'B', 'A', np.nan, 'C', 'B'],
        'col_cat_2': ['X', np.nan, np.nan, 'Y', 'Z', 'X'],
        'col_full': [1,2,3,4,5,6]
    }
    df_test_missing = pd.DataFrame(data_missing)

    print("DataFrame Original:")
    print(df_test_missing)
    print("\n" + "="*30 + "\n")

    print("Test 'remove_row':")
    df_removed_rows = handle_missing_values(df_test_missing.copy(), strategy='remove_row')
    print(df_removed_rows)
    print("\n" + "="*30 + "\n")

    print("Test 'remove_row' sur subset ['col_num_1']:")
    df_removed_rows_subset = handle_missing_values(df_test_missing.copy(), strategy='remove_row', subset_cols=['col_num_1'])
    print(df_removed_rows_subset)
    print("\n" + "="*30 + "\n")

    # print("Test 'remove_col':") # Attention, peut supprimer beaucoup de colonnes
    # df_removed_cols = handle_missing_values(df_test_missing.copy(), strategy='remove_col')
    # print(df_removed_cols)
    # print("\n" + "="*30 + "\n")

    print("Test 'fill_mean' (sur colonnes numériques uniquement par défaut):")
    df_filled_mean = handle_missing_values(df_test_missing.copy(), strategy='fill_mean')
    print(df_filled_mean)
    print("\n" + "="*30 + "\n")

    print("Test 'fill_median':")
    df_filled_median = handle_missing_values(df_test_missing.copy(), strategy='fill_median')
    print(df_filled_median)
    print("\n" + "="*30 + "\n")

    print("Test 'fill_mode':")
    df_filled_mode = handle_missing_values(df_test_missing.copy(), strategy='fill_mode')
    print(df_filled_mode)
    print("\n" + "="*30 + "\n")

    print("Test 'auto':")
    df_filled_auto = handle_missing_values(df_test_missing.copy(), strategy='auto')
    print(df_filled_auto)
    print("\n" + "="*30 + "\n")

    print("Test 'auto' avec fill_value personnalisé:")
    df_filled_auto_custom = handle_missing_values(df_test_missing.copy(), strategy='auto', fill_value_numeric=-1, fill_value_categorical='MISSING')
    print(df_filled_auto_custom)
    print("\n" + "="*30 + "\n")

    print("Test 'fill_custom':")
    df_filled_custom = handle_missing_values(df_test_missing.copy(), strategy='fill_custom', fill_value_numeric=0, fill_value_categorical='N/A')
    print(df_filled_custom)
    print("\n" + "="*30 + "\n")

    print("Test 'fill_custom' sur subset ['col_num_1', 'col_cat_1']:")
    df_filled_custom_subset = handle_missing_values(
        df_test_missing.copy(),
        strategy='fill_custom',
        subset_cols=['col_num_1', 'col_cat_1'],
        fill_value_numeric=999,
        fill_value_categorical='SPECIFIC_NA'
    )
    print(df_filled_custom_subset)
    print("\n" + "="*30 + "\n")
