import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_numerical_features(df: pd.DataFrame, strategy: str = 'standard', columns_to_scale: list = None):
    """
    Met à l'échelle les caractéristiques numériques d'un DataFrame.

    Stratégies :
    - 'standard': Standardisation (moyenne 0, variance 1). Utilise StandardScaler.
    - 'min_max': Mise à l'échelle Min-Max (généralement entre 0 et 1). Utilise MinMaxScaler.
    - 'robust': Mise à l'échelle robuste aux outliers. Utilise RobustScaler.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        strategy (str): La stratégie de mise à l'échelle ('standard', 'min_max', 'robust').
        columns_to_scale (list, optional): Liste des noms de colonnes numériques à mettre à l'échelle.
                                           Si None, met à l'échelle toutes les colonnes numériques.

    Returns:
        pd.DataFrame: DataFrame avec les caractéristiques numériques mises à l'échelle.
    """
    df_scaled = df.copy()

    if columns_to_scale is None:
        columns_to_scale = df_scaled.select_dtypes(include=['number']).columns.tolist()
        if not columns_to_scale:
            print("Aucune colonne numérique trouvée pour la mise à l'échelle.")
            return df_scaled
    else:
        # Vérifier que les colonnes spécifiées sont bien numériques
        valid_columns = []
        for col in columns_to_scale:
            if col in df_scaled.columns:
                if pd.api.types.is_numeric_dtype(df_scaled[col]):
                    valid_columns.append(col)
                else:
                    print(f"Avertissement : La colonne '{col}' n'est pas numérique et sera ignorée pour la mise à l'échelle.")
            else:
                print(f"Avertissement : La colonne '{col}' n'a pas été trouvée dans le DataFrame et sera ignorée.")
        columns_to_scale = valid_columns
        if not columns_to_scale:
            print("Aucune colonne numérique valide spécifiée ou trouvée pour la mise à l'échelle.")
            return df_scaled

    print(f"Colonnes sélectionnées pour la mise à l'échelle ({strategy}): {columns_to_scale}")

    if strategy == 'standard':
        scaler = StandardScaler()
    elif strategy == 'min_max':
        scaler = MinMaxScaler()
    elif strategy == 'robust':
        scaler = RobustScaler()
    else:
        print(f"Stratégie de mise à l'échelle '{strategy}' non reconnue. Aucune action effectuée.")
        return df_scaled

    try:
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
        print(f"Colonnes {columns_to_scale} mises à l'échelle avec la stratégie '{strategy}'.")
    except Exception as e:
        print(f"Erreur lors de la mise à l'échelle des colonnes {columns_to_scale}: {e}")
        return df # Retourne l'original en cas d'erreur

    return df_scaled

if __name__ == '__main__':
    data_num = {
        'ID': [1, 2, 3, 4, 5],
        'Age': [25, 30, 22, 45, 33],
        'Salaire': [50000, 60000, 45000, 90000, 75000],
        'Experience': [2, 5, 1, 10, 7],
        'NonNum': ['A', 'B', 'C', 'D', 'E']
    }
    df_test_num = pd.DataFrame(data_num)

    print("DataFrame Original:")
    print(df_test_num)
    print("\n" + "="*30 + "\n")

    print("Test StandardScaler (auto-détection des colonnes numériques):")
    df_standard = scale_numerical_features(df_test_num.copy(), strategy='standard')
    print(df_standard)
    print(df_standard.describe())
    print("\n" + "="*30 + "\n")

    print("Test MinMaxScaler (colonnes spécifiées ['Age', 'Salaire']):")
    df_min_max = scale_numerical_features(df_test_num.copy(), strategy='min_max', columns_to_scale=['Age', 'Salaire'])
    print(df_min_max)
    print(df_min_max.describe())
    print("\n" + "="*30 + "\n")

    print("Test RobustScaler (colonne spécifiée ['Experience'] et une non numérique ['NonNum']):")
    df_robust = scale_numerical_features(df_test_num.copy(), strategy='robust', columns_to_scale=['Experience', 'NonNum', 'Inexistante'])
    print(df_robust)
    print(df_robust.describe())
    print("\n" + "="*30 + "\n")

    print("Test StandardScaler (aucune colonne numérique):")
    df_non_num_only = pd.DataFrame({'Texte': ['a', 'b', 'c']})
    df_scaled_non_num = scale_numerical_features(df_non_num_only.copy(), strategy='standard')
    print(df_scaled_non_num)
    print("\n" + "="*30 + "\n")

    print("Test MinMaxScaler (colonnes spécifiées mais aucune n'est numérique):")
    df_scaled_spec_non_num = scale_numerical_features(df_test_num.copy(), strategy='min_max', columns_to_scale=['NonNum'])
    print(df_scaled_spec_non_num)
    print("\n" + "="*30 + "\n")
