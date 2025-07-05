import pandas as pd
import os
import json

def export_dataframe_to_csv(df: pd.DataFrame, file_path: str, index: bool = False, **kwargs):
    """
    Exporte un DataFrame vers un fichier CSV.

    Args:
        df (pd.DataFrame): Le DataFrame à exporter.
        file_path (str): Le chemin complet du fichier CSV de sortie (incluant le nom du fichier).
        index (bool): Écrit l'index du DataFrame dans le fichier. Par défaut à False.
        **kwargs: Arguments supplémentaires à passer à `pd.DataFrame.to_csv()`.
    """
    if not isinstance(df, pd.DataFrame):
        print("Erreur: L'entrée n'est pas un DataFrame pandas.")
        return False

    try:
        # S'assurer que le répertoire de destination existe
        directory = os.path.dirname(file_path)
        if directory: # Si le chemin n'est pas juste un nom de fichier local
            os.makedirs(directory, exist_ok=True)

        df.to_csv(file_path, index=index, **kwargs)
        print(f"DataFrame exporté avec succès vers : {file_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de l'exportation du DataFrame vers CSV ({file_path}): {e}")
        return False

def export_dataframe_to_json(df: pd.DataFrame, file_path: str, orient: str = 'records', **kwargs):
    """
    Exporte un DataFrame vers un fichier JSON.

    Args:
        df (pd.DataFrame): Le DataFrame à exporter.
        file_path (str): Le chemin complet du fichier JSON de sortie (incluant le nom du fichier).
        orient (str): Format de la chaîne JSON. Options communes : 'records', 'columns', 'values', 'index', 'table'.
        **kwargs: Arguments supplémentaires à passer à `pd.DataFrame.to_json()`.
    """
    if not isinstance(df, pd.DataFrame):
        print("Erreur: L'entrée n'est pas un DataFrame pandas.")
        return False

    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        df.to_json(file_path, orient=orient, indent=4, **kwargs) # indent pour une meilleure lisibilité
        print(f"DataFrame exporté avec succès vers : {file_path} (orient='{orient}')")
        return True
    except Exception as e:
        print(f"Erreur lors de l'exportation du DataFrame vers JSON ({file_path}): {e}")
        return False

def export_dict_to_json(data_dict: dict, file_path: str, **kwargs):
    """
    Exporte un dictionnaire Python vers un fichier JSON.

    Args:
        data_dict (dict): Le dictionnaire à exporter.
        file_path (str): Le chemin complet du fichier JSON de sortie.
        **kwargs: Arguments supplémentaires à passer à `json.dump()`.
    """
    if not isinstance(data_dict, dict):
        print("Erreur: L'entrée n'est pas un dictionnaire.")
        return False

    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(data_dict, f, indent=4, **kwargs) # indent pour une meilleure lisibilité
        print(f"Dictionnaire exporté avec succès vers : {file_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de l'exportation du dictionnaire vers JSON ({file_path}): {e}")
        return False

if __name__ == '__main__':
    # Création d'un répertoire de sortie pour les tests
    output_export_dir = "test_exports"
    os.makedirs(output_export_dir, exist_ok=True)

    # Données de test
    sample_data = {
        'colA': [1, 2, 3, 4],
        'colB': ['apple', 'banana', 'cherry', 'date'],
        'colC': [0.1, 0.2, 0.3, 0.4]
    }
    df_sample = pd.DataFrame(sample_data)

    sample_dict = {
        "name": "Test AI Project",
        "version": "1.0",
        "parameters": {"param1": 10, "param2": "value"},
        "results": [
            {"id": 1, "score": 0.95},
            {"id": 2, "score": 0.89}
        ]
    }

    print("===== Test export_dataframe_to_csv =====")
    csv_file_path = os.path.join(output_export_dir, "sample_export.csv")
    export_dataframe_to_csv(df_sample, csv_file_path, index=True, sep=';')
    # Vérification (lecture simple)
    if os.path.exists(csv_file_path):
        df_read_csv = pd.read_csv(csv_file_path, sep=';', index_col=0)
        print("CSV lu pour vérification:")
        print(df_read_csv.head())
    print("\n" + "="*30 + "\n")

    print("===== Test export_dataframe_to_json (orient='records') =====")
    json_file_path_records = os.path.join(output_export_dir, "sample_export_records.json")
    export_dataframe_to_json(df_sample, json_file_path_records, orient='records')
    # Vérification
    if os.path.exists(json_file_path_records):
        with open(json_file_path_records, 'r') as f:
            print(f"Contenu de {json_file_path_records} (premiers 100 chars): {f.read(100)}...")
    print("\n" + "="*30 + "\n")

    print("===== Test export_dataframe_to_json (orient='columns') =====")
    json_file_path_columns = os.path.join(output_export_dir, "sample_export_columns.json")
    export_dataframe_to_json(df_sample, json_file_path_columns, orient='columns')
    if os.path.exists(json_file_path_columns):
        with open(json_file_path_columns, 'r') as f:
            print(f"Contenu de {json_file_path_columns} (premiers 100 chars): {f.read(100)}...")
    print("\n" + "="*30 + "\n")

    print("===== Test export_dict_to_json =====")
    dict_json_file_path = os.path.join(output_export_dir, "sample_dict_export.json")
    export_dict_to_json(sample_dict, dict_json_file_path)
    # Vérification
    if os.path.exists(dict_json_file_path):
        with open(dict_json_file_path, 'r') as f:
            loaded_dict = json.load(f)
        print(f"Dictionnaire lu depuis {dict_json_file_path}:")
        print(loaded_dict)
    print("\n" + "="*30 + "\n")

    print(f"NOTE: Les fichiers exportés pour test sont dans le dossier '{output_export_dir}'.")

    # Test avec une entrée non-DataFrame/non-dict
    print("===== Test export_dataframe_to_csv (mauvais type) =====")
    export_dataframe_to_csv([1,2,3], "bad_input.csv")
    print("\n" + "="*30 + "\n")

    print("===== Test export_dict_to_json (mauvais type) =====")
    export_dict_to_json([1,2,3], "bad_input_dict.json")
