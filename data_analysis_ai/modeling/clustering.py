import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def find_optimal_k_elbow(df_numeric: pd.DataFrame, max_k: int = 10, plot_path: str = None):
    """
    Trouve le nombre optimal de clusters (k) en utilisant la méthode du coude (Elbow method).
    Nécessite des données numériques prétraitées (mises à l'échelle).

    Args:
        df_numeric (pd.DataFrame): DataFrame contenant uniquement les features numériques pour le clustering.
        max_k (int): Nombre maximum de clusters à tester.
        plot_path (str, optional): Chemin pour sauvegarder le graphique du coude. Si None, affiche seulement.

    Returns:
        int: Le nombre optimal de k suggéré par la méthode (peut nécessiter une inspection visuelle).
             Retourne None si une erreur se produit ou si les données ne sont pas adaptées.
    """
    if not isinstance(df_numeric, pd.DataFrame) or df_numeric.empty:
        print("Erreur: df_numeric doit être un DataFrame pandas non vide.")
        return None

    if not all(pd.api.types.is_numeric_dtype(df_numeric[col]) for col in df_numeric.columns):
        print("Erreur: Toutes les colonnes dans df_numeric doivent être numériques.")
        return None

    if df_numeric.isnull().values.any():
        print("Avertissement: df_numeric contient des valeurs NaN. Le clustering peut échouer ou donner des résultats incorrects. Veuillez les imputer.")
        # Optionnellement, on pourrait imputer ici avec une stratégie simple, mais il vaut mieux le faire en amont.
        # df_numeric = df_numeric.fillna(df_numeric.mean())


    inertia = []
    k_values = range(1, max_k + 1)

    if len(df_numeric) < max_k : # Si moins d'échantillons que max_k
        print(f"Avertissement: Moins d'échantillons ({len(df_numeric)}) que max_k ({max_k}). max_k sera ajusté à {len(df_numeric)}.")
        max_k = len(df_numeric)
        if max_k < 2: # K-Means nécessite au moins 1 cluster, mais pour trouver un "coude", on a besoin d'au moins 2 points.
             print("Erreur: Pas assez d'échantillons pour la méthode du coude (besoin d'au moins 2).")
             return None
        k_values = range(1, max_k +1)


    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(df_numeric)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia, marker='o', linestyle='--')
    plt.title('Méthode du Coude pour déterminer k optimal')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie (WCSS)')
    plt.xticks(k_values)
    plt.grid(True)

    if plot_path:
        try:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            print(f"Graphique du coude sauvegardé dans : {plot_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du graphique du coude : {e}")
    plt.show()

    # Logique simple pour suggérer un k (peut être améliorée)
    # On cherche un point où la diminution de l'inertie ralentit significativement.
    # Ceci est une heuristique et l'inspection visuelle est souvent meilleure.
    if len(inertia) > 2:
        # Différences des inerties
        diffs = np.diff(inertia)
        # Différences des différences (accélération)
        diffs2 = np.diff(diffs)
        # Le "coude" est souvent là où la deuxième différence est la plus grande (changement de pente)
        # On ajoute +2 car diffs2 a une longueur de len(inertia)-2 et les k commencent à 1.
        # (index de diffs2) + 1 (pour k) + 1 (car diffs2 commence à k=2 vs k=1)
        # Exemple: inertia pour k=1,2,3,4. diffs pour k=2,3,4. diffs2 pour k=3,4.
        # np.argmax(diffs2) = 0 correspond à k=3.
        try:
            suggested_k = np.argmax(diffs2) + 2
        except ValueError: # Si diffs2 est vide (max_k trop petit)
            suggested_k = k_values[-1] if k_values else 1 # Fallback
    elif k_values:
        suggested_k = k_values[-1] # Si pas assez de points pour diffs2
    else:
        suggested_k = 1


    print(f"Suggestion basée sur la méthode du coude (inspectez le graphique) : k ~ {suggested_k}")
    return suggested_k


def apply_kmeans_clustering(df_numeric: pd.DataFrame, n_clusters: int, original_df: pd.DataFrame = None):
    """
    Applique l'algorithme K-Means sur des données numériques prétraitées.
    Ajoute les étiquettes de cluster au DataFrame original si fourni.

    Args:
        df_numeric (pd.DataFrame): DataFrame contenant les features numériques et mises à l'échelle.
        n_clusters (int): Nombre de clusters à former.
        original_df (pd.DataFrame, optional): DataFrame original auquel ajouter les étiquettes de cluster.
                                              Si None, retourne un DataFrame avec seulement les données et les clusters.

    Returns:
        pd.DataFrame: DataFrame (original ou une copie de df_numeric) avec une nouvelle colonne 'cluster'.
                      Retourne None si une erreur se produit.
    """
    if not isinstance(df_numeric, pd.DataFrame) or df_numeric.empty:
        print("Erreur: df_numeric doit être un DataFrame pandas non vide.")
        return None
    if not all(pd.api.types.is_numeric_dtype(df_numeric[col]) for col in df_numeric.columns):
        print("Erreur: Toutes les colonnes dans df_numeric doivent être numériques.")
        return None
    if df_numeric.isnull().values.any():
        print("Avertissement: df_numeric contient des valeurs NaN. Le clustering peut échouer. Imputez-les.")
        # df_numeric = df_numeric.fillna(df_numeric.mean())

    if n_clusters <= 0:
        print("Erreur: n_clusters doit être un entier positif.")
        return None
    if len(df_numeric) < n_clusters:
        print(f"Erreur: Nombre d'échantillons ({len(df_numeric)}) est inférieur au nombre de clusters ({n_clusters}).")
        return None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    try:
        cluster_labels = kmeans.fit_predict(df_numeric)
    except Exception as e:
        print(f"Erreur durant l'exécution de K-Means: {e}")
        return None

    if original_df is not None:
        if len(original_df) != len(df_numeric):
            print("Avertissement: Le DataFrame original et le DataFrame numérique n'ont pas la même longueur. Les clusters seront ajoutés à une copie de df_numeric.")
            result_df = df_numeric.copy()
        else:
            result_df = original_df.copy()
        result_df['cluster'] = cluster_labels
    else:
        result_df = df_numeric.copy()
        result_df['cluster'] = cluster_labels

    print(f"K-Means appliqué avec {n_clusters} clusters. Les étiquettes ont été ajoutées à la colonne 'cluster'.")
    return result_df


if __name__ == '__main__':
    # Création d'un répertoire de sortie pour les graphiques de test
    output_model_dir = "test_model_plots"
    os.makedirs(output_model_dir, exist_ok=True)

    # Données de test
    data_clus = {
        'Feature1': np.random.rand(100) * 10,
        'Feature2': np.random.rand(100) * 10,
        'Feature3': np.random.rand(100) * 5,
        'NonNumeric': ['A'] * 50 + ['B'] * 50 # Pour tester la sélection des colonnes numériques
    }
    df_test_clus_orig = pd.DataFrame(data_clus)

    # 1. Prétraitement (sélectionner numérique et mise à l'échelle)
    df_numeric_features = df_test_clus_orig.select_dtypes(include=np.number)

    if df_numeric_features.empty:
        print("Aucune feature numérique pour le clustering.")
    else:
        scaler = StandardScaler()
        df_scaled_features = pd.DataFrame(scaler.fit_transform(df_numeric_features), columns=df_numeric_features.columns)

        print("DataFrame mis à l'échelle pour le clustering:")
        print(df_scaled_features.head())
        print("\n" + "="*30 + "\n")

        # 2. Trouver k optimal avec la méthode du coude
        print("===== Test find_optimal_k_elbow =====")
        elbow_plot_path = os.path.join(output_model_dir, "elbow_plot.png")
        suggested_k = find_optimal_k_elbow(df_scaled_features, max_k=8, plot_path=elbow_plot_path)
        print(f"K suggéré (à vérifier visuellement) : {suggested_k}")

        # Pour le test, utilisons une valeur fixe ou celle suggérée si elle est raisonnable
        k_to_use = suggested_k if suggested_k and suggested_k > 1 else 3
        print(f"Utilisation de k = {k_to_use} pour K-Means.")
        print("\n" + "="*30 + "\n")

        # 3. Appliquer K-Means
        print("===== Test apply_kmeans_clustering =====")
        df_clustered = apply_kmeans_clustering(df_scaled_features, n_clusters=k_to_use, original_df=df_test_clus_orig)

        if df_clustered is not None:
            print("DataFrame avec clusters :")
            print(df_clustered.head())
            print("\nDistribution des clusters :")
            print(df_clustered['cluster'].value_counts())

            # Visualisation simple des clusters (si 2 features principales)
            if 'Feature1' in df_scaled_features.columns and 'Feature2' in df_scaled_features.columns and 'cluster' in df_clustered.columns:
                plt.figure(figsize=(10, 7))
                # Utiliser les données originales pour le scatter plot pour une meilleure interprétabilité de l'échelle
                # ou les données scalées si on veut voir la séparation dans l'espace de clustering
                sns.scatterplot(x=df_test_clus_orig['Feature1'], y=df_test_clus_orig['Feature2'], hue=df_clustered['cluster'], palette='viridis', s=100, alpha=0.7)
                plt.title(f'Visualisation des Clusters K-Means (k={k_to_use})')
                plt.xlabel('Feature1 (Original)')
                plt.ylabel('Feature2 (Original)')
                cluster_plot_path = os.path.join(output_model_dir, "kmeans_clusters_scatter.png")
                try:
                    plt.savefig(cluster_plot_path)
                    print(f"Graphique des clusters sauvegardé dans : {cluster_plot_path}")
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde du graphique des clusters : {e}")
                plt.show()
        else:
            print("Le clustering K-Means a échoué ou n'a pas retourné de DataFrame.")

        # Test avec un k trop grand
        print("\n===== Test K-Means avec k > nombre d'échantillons (attendu: erreur/avertissement) =====")
        df_small_sample = df_scaled_features.head(3)
        apply_kmeans_clustering(df_small_sample, n_clusters=5, original_df=df_test_clus_orig.head(3))

        print("\n===== Test K-Means avec données non numériques (attendu: erreur) =====")
        apply_kmeans_clustering(df_test_clus_orig[['Feature1', 'NonNumeric']], n_clusters=2)

        print("\n===== Test méthode du coude avec peu d'échantillons =====")
        find_optimal_k_elbow(df_small_sample, max_k=5)

        print("\n===== Test méthode du coude avec 1 seul échantillon (attendu: erreur) =====")
        find_optimal_k_elbow(df_scaled_features.head(1), max_k=5)


    plt.close('all') # Ferme toutes les figures
    print(f"\nNOTE: Les graphiques de test de modélisation sont sauvegardés dans '{output_model_dir}'.")
