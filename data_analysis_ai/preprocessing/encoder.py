import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def encode_categorical_features(df: pd.DataFrame, strategy: str = 'one_hot', columns_to_encode: list = None):
    """
    Encode les caractéristiques catégorielles d'un DataFrame.

    Stratégies :
    - 'one_hot': Utilise le One-Hot Encoding. Crée de nouvelles colonnes pour chaque catégorie.
                 Gère automatiquement les colonnes à encoder si non spécifiées (celles de type 'object' ou 'category').
    - 'label': Utilise Label Encoding. Convertit les catégories en entiers.
               Attention: peut introduire un ordre non désiré si utilisé avec des algorithmes sensibles à l'ordre.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        strategy (str): La stratégie d'encodage ('one_hot' ou 'label').
        columns_to_encode (list, optional): Liste des noms de colonnes à encoder.
                                            Si None et strategy='one_hot', encode toutes les colonnes 'object' et 'category'.
                                            Si None et strategy='label', l'utilisateur doit spécifier les colonnes.

    Returns:
        pd.DataFrame: DataFrame avec les caractéristiques catégorielles encodées.
                      Retourne le DataFrame original si aucune colonne n'est encodée ou en cas d'erreur.
    """
    df_encoded = df.copy()

    if columns_to_encode is None:
        if strategy == 'one_hot':
            columns_to_encode = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
            if not columns_to_encode:
                print("Aucune colonne de type 'object' ou 'category' trouvée pour le One-Hot Encoding.")
                return df_encoded
        elif strategy == 'label':
            print("Pour Label Encoding, veuillez spécifier les 'columns_to_encode'.")
            return df_encoded # ou lever une erreur

    if not columns_to_encode: # Si toujours vide après la logique ci-dessus
        print("Aucune colonne spécifiée ou trouvée pour l'encodage.")
        return df_encoded

    print(f"Colonnes sélectionnées pour l'encodage ({strategy}): {columns_to_encode}")

    if strategy == 'one_hot':
        try:
            # S'assurer que les colonnes existent bien dans le DataFrame
            valid_columns_to_encode = [col for col in columns_to_encode if col in df_encoded.columns]
            if not valid_columns_to_encode:
                print("Aucune des colonnes spécifiées pour le One-Hot Encoding n'existe dans le DataFrame.")
                return df_encoded

            ct = ColumnTransformer(
                [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), valid_columns_to_encode)],
                remainder='passthrough'  # Garde les autres colonnes
            )

            # Pour obtenir les noms des nouvelles colonnes après OHE
            # Il faut fitter d'abord pour que get_feature_names_out soit disponible
            ct.fit(df_encoded)
            feature_names = ct.get_feature_names_out()

            df_transformed_np = ct.transform(df_encoded)
            df_encoded = pd.DataFrame(df_transformed_np, columns=feature_names, index=df_encoded.index)

            # Nettoyer les noms de colonnes générés par ColumnTransformer
            df_encoded.columns = [col.replace('one_hot_encoder__', '').replace('remainder__', '') for col in df_encoded.columns]
            print(f"Colonnes {valid_columns_to_encode} encodées avec One-Hot Encoding.")

        except Exception as e:
            print(f"Erreur pendant le One-Hot Encoding: {e}")
            return df # Retourne l'original en cas d'erreur

    elif strategy == 'label':
        for col in columns_to_encode:
            if col in df_encoded.columns:
                if df_encoded[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_encoded[col]):
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str)) # Convertit en str pour gérer les mixtes/NaN
                    print(f"Colonne '{col}' encodée avec Label Encoding.")
                else:
                    print(f"Colonne '{col}' n'est pas de type catégoriel ou objet. Label Encoding ignoré pour cette colonne.")
            else:
                print(f"Colonne '{col}' non trouvée dans le DataFrame.")
    else:
        print(f"Stratégie d'encodage '{strategy}' non reconnue.")

    return df_encoded


if __name__ == '__main__':
    data_cat = {
        'ID': [1, 2, 3, 4, 5, 6],
        'Couleur': ['Rouge', 'Bleu', 'Vert', 'Rouge', 'Bleu', 'Jaune'],
        'Taille': ['Petit', 'Moyen', 'Grand', 'Moyen', 'Petit', 'Grand'],
        'ValeurNum': [10, 20, 15, 25, 10, 30],
        'Forme': ['Carré', 'Cercle', 'Triangle', 'Cercle', 'Carré', np.nan] # Ajout de NaN
    }
    df_test_cat = pd.DataFrame(data_cat)

    print("DataFrame Original:")
    print(df_test_cat)
    print("\n" + "="*30 + "\n")

    print("Test One-Hot Encoding (auto-détection des colonnes):")
    df_one_hot_auto = encode_categorical_features(df_test_cat.copy(), strategy='one_hot')
    print(df_one_hot_auto)
    print("\n" + "="*30 + "\n")

    print("Test One-Hot Encoding (colonnes spécifiées):")
    df_one_hot_specific = encode_categorical_features(df_test_cat.copy(), strategy='one_hot', columns_to_encode=['Couleur', 'Forme'])
    print(df_one_hot_specific)
    print("\n" + "="*30 + "\n")

    print("Test One-Hot Encoding (colonne inexistante):")
    df_one_hot_nonexistent = encode_categorical_features(df_test_cat.copy(), strategy='one_hot', columns_to_encode=['ExistePas'])
    print(df_one_hot_nonexistent)
    print("\n" + "="*30 + "\n")

    print("Test Label Encoding (colonnes spécifiées):")
    df_label_encoded = encode_categorical_features(df_test_cat.copy(), strategy='label', columns_to_encode=['Taille', 'Couleur'])
    print(df_label_encoded)
    print("\n" + "="*30 + "\n")

    print("Test Label Encoding (sans spécifier les colonnes - attendu: pas de changement):")
    df_label_no_cols = encode_categorical_features(df_test_cat.copy(), strategy='label')
    print(df_label_no_cols)
    print("\n" + "="*30 + "\n")

    # Cas avec une seule colonne catégorielle pour OHE
    data_single_cat = {'Cat': ['A', 'B', 'A'], 'Num': [1,2,3]}
    df_single_cat = pd.DataFrame(data_single_cat)
    print("DataFrame avec une seule colonne cat pour OHE:")
    print(df_single_cat)
    df_single_ohe = encode_categorical_features(df_single_cat.copy(), strategy='one_hot')
    print(df_single_ohe)
    print("\n" + "="*30 + "\n")

    # Cas où aucune colonne n'est catégorielle pour OHE
    data_no_cat = {'Num1': [1,2,3], 'Num2': [4,5,6]}
    df_no_cat = pd.DataFrame(data_no_cat)
    print("DataFrame sans colonne cat pour OHE:")
    print(df_no_cat)
    df_no_cat_ohe = encode_categorical_features(df_no_cat.copy(), strategy='one_hot')
    print(df_no_cat_ohe)
    print("\n" + "="*30 + "\n")
