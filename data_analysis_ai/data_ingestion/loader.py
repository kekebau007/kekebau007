import pandas as pd
import json

def load_csv_data(file_path: str, separator: str = ','):
    """
    Charge des données à partir d'un fichier CSV.

    Args:
        file_path (str): Le chemin vers le fichier CSV.
        separator (str): Le séparateur utilisé dans le fichier CSV. Par défaut, ','.

    Returns:
        pandas.DataFrame: Un DataFrame contenant les données chargées,
                         ou None si une erreur survient.
    """
    try:
        df = pd.read_csv(file_path, sep=separator)
        print(f"Données chargées avec succès depuis {file_path}")
        return df
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier {file_path} est vide.")
        return None
    except pd.errors.ParserError:
        print(f"Erreur : Impossible de parser le fichier {file_path}. Vérifiez le format et le séparateur.")
        return None
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors du chargement de {file_path}: {e}")
        return None

def load_json_data(file_path: str):
    """
    Charge des données à partir d'un fichier JSON.
    Attend un fichier JSON contenant un objet ou une liste d'objets.

    Args:
        file_path (str): Le chemin vers le fichier JSON.

    Returns:
        pandas.DataFrame: Un DataFrame contenant les données chargées,
                         ou None si une erreur survient.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Données chargées avec succès depuis {file_path}")
        return df
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        return None
    except json.JSONDecodeError:
        print(f"Erreur : Impossible de décoder le fichier JSON {file_path}. Vérifiez le format.")
        return None
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors du chargement de {file_path}: {e}")
        return None

if __name__ == '__main__':
    # Créer des fichiers de test pour la démonstration
    # Fichier CSV de test
    sample_csv_data = "col1,col2,col3\n1,a,True\n2,b,False\n3,c,True"
    with open("sample.csv", "w") as f:
        f.write(sample_csv_data)

    # Fichier JSON de test
    sample_json_data = [{"col1": 1, "col2": "a", "col3": True},
                        {"col1": 2, "col2": "b", "col3": False}]
    with open("sample.json", "w") as f:
        json.dump(sample_json_data, f)

    print("--- Test du chargement CSV ---")
    csv_df = load_csv_data("sample.csv")
    if csv_df is not None:
        print("Aperçu des données CSV :")
        print(csv_df.head())

    print("\n--- Test du chargement CSV avec mauvais séparateur (attendu : erreur) ---")
    csv_df_wrong_sep = load_csv_data("sample.csv", separator=';')
    if csv_df_wrong_sep is not None:
        print(csv_df_wrong_sep.head())


    print("\n--- Test du chargement JSON ---")
    json_df = load_json_data("sample.json")
    if json_df is not None:
        print("Aperçu des données JSON :")
        print(json_df.head())

    print("\n--- Test avec un fichier inexistant (attendu : erreur) ---")
    non_existent_df = load_csv_data("non_existent.csv")

    # Nettoyage des fichiers de test
    import os
    os.remove("sample.csv")
    os.remove("sample.json")
