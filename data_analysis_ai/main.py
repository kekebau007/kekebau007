import pandas as pd
import numpy as np
import os

# Import des modules du projet
from data_ingestion import loader
from preprocessing import cleaner, encoder, scaler
from analysis import eda, visualization
from modeling import clustering, classification, regression
from utils import exporter

# Configuration (chemins, etc.) - à adapter par l'utilisateur
BASE_OUTPUT_DIR = "output_analysis_results"
# Créer le répertoire de sortie principal s'il n'existe pas
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def main_data_analysis_pipeline(data_path: str, data_format: str = 'csv'):
    """
    Pipeline principal pour l'analyse de données.
    Charge, prétraite, analyse et modélise les données.
    Les résultats et graphiques sont sauvegardés dans BASE_OUTPUT_DIR.
    """
    print(f"--- DÉBUT DU PIPELINE D'ANALYSE DE DONNÉES POUR : {data_path} ---")

    # 1. Ingestion des données
    print("\n--- 1. Ingestion des données ---")
    df_raw = None
    if data_format == 'csv':
        df_raw = loader.load_csv_data(data_path)
    elif data_format == 'json':
        df_raw = loader.load_json_data(data_path)
    else:
        print(f"Format de données '{data_format}' non supporté.")
        return

    if df_raw is None:
        print("Échec du chargement des données. Arrêt du pipeline.")
        return

    print(f"Données brutes chargées. Dimensions : {df_raw.shape}")
    exporter.export_dataframe_to_csv(df_raw, os.path.join(BASE_OUTPUT_DIR, "0_raw_data.csv"))

    # 2. Prétraitement
    print("\n--- 2. Prétraitement des données ---")
    df_processed = df_raw.copy()

    # Gestion des valeurs manquantes (exemple : stratégie auto)
    # L'utilisateur devrait choisir la meilleure stratégie pour ses données
    df_processed = cleaner.handle_missing_values(df_processed, strategy='auto',
                                                 fill_value_numeric=df_processed.select_dtypes(include=np.number).mean(), # Exemple
                                                 fill_value_categorical='Unknown') # Exemple
    print(f"Dimensions après gestion des NaN : {df_processed.shape}")

    # Encodage des catégorielles (exemple : one-hot pour toutes les colonnes object/category)
    # L'utilisateur devrait spécifier les colonnes et la stratégie
    df_processed = encoder.encode_categorical_features(df_processed, strategy='one_hot')
    print(f"Dimensions après encodage : {df_processed.shape}")

    # Mise à l'échelle des numériques (exemple : standard scaler)
    # L'utilisateur devrait spécifier les colonnes si nécessaire
    df_processed = scaler.scale_numerical_features(df_processed, strategy='standard')
    print("Données numériques mises à l'échelle.")

    exporter.export_dataframe_to_csv(df_processed, os.path.join(BASE_OUTPUT_DIR, "1_processed_data.csv"))
    print("Données prétraitées sauvegardées.")

    # 3. Analyse Exploratoire des Données (EDA)
    print("\n--- 3. Analyse Exploratoire des Données (EDA) ---")
    eda_output_dir = os.path.join(BASE_OUTPUT_DIR, "eda_analysis")
    os.makedirs(eda_output_dir, exist_ok=True)

    eda.get_dataframe_info(df_processed) # Affiche dans la console
    stats_desc = df_processed.describe(include='all')
    print("Statistiques descriptives (extrait) :\n", stats_desc.head())
    exporter.export_dataframe_to_csv(stats_desc, os.path.join(eda_output_dir, "descriptive_statistics.csv"))

    # Visualisations (exemples - à adapter aux colonnes réelles du dataset)
    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        # Histogramme pour la première colonne numérique
        visualization.plot_histogram(df_processed, column=numeric_cols[0],
                                     save_path=os.path.join(eda_output_dir, f"histogram_{numeric_cols[0]}.png"))
        if len(numeric_cols) > 1:
            # Nuage de points pour les deux premières colonnes numériques
            visualization.plot_scatterplot(df_processed, x_column=numeric_cols[0], y_column=numeric_cols[1],
                                           save_path=os.path.join(eda_output_dir, f"scatter_{numeric_cols[0]}_vs_{numeric_cols[1]}.png"))
        # Heatmap de corrélation
        visualization.plot_correlation_heatmap(df_processed,
                                               save_path=os.path.join(eda_output_dir, "correlation_heatmap.png"))
    else:
        print("Pas de colonnes numériques pour les visualisations EDA typiques (histogramme, scatter, heatmap).")

    # 4. Modélisation (Exemples - L'UTILISATEUR DOIT ADAPTER CETTE SECTION)
    print("\n--- 4. Modélisation (Exemples - À ADAPTER) ---")
    modeling_output_dir = os.path.join(BASE_OUTPUT_DIR, "modeling_results")
    os.makedirs(modeling_output_dir, exist_ok=True)

    # Séparer les features pour la modélisation (supposons que toutes les colonnes sont des features après OHE)
    # Ceci est une simplification. L'utilisateur doit définir ses features et sa cible.
    df_for_modeling = df_processed.copy()

    # Exemple de Clustering (sur les données numériques uniquement)
    # Il faut s'assurer que les données pour le clustering sont bien numériques et sans NaN après prétraitement.
    # Si df_for_modeling contient des colonnes non-numériques après OHE (ce qui ne devrait pas être le cas si tout est bien encodé)
    # ou si des NaN persistent, le clustering peut échouer.
    df_numeric_for_clustering = df_for_modeling.select_dtypes(include=np.number).dropna()

    if not df_numeric_for_clustering.empty and len(df_numeric_for_clustering) > 3 : # K-means a besoin d'assez d'échantillons
        print("\n--- Exemple de Clustering K-Means ---")
        k_elbow_path = os.path.join(modeling_output_dir, "clustering_elbow_plot.png")
        # Note: find_optimal_k_elbow s'attend à des données déjà mises à l'échelle. df_processed l'est déjà.
        suggested_k = clustering.find_optimal_k_elbow(df_numeric_for_clustering, max_k=min(8, len(df_numeric_for_clustering)-1), plot_path=k_elbow_path)

        chosen_k = suggested_k if suggested_k and suggested_k > 1 else 3 # Fallback à 3 clusters
        print(f"Application du K-Means avec k={chosen_k}...")

        # On applique le clustering sur les colonnes numériques, et on l'ajoute au df_for_modeling original
        # Assurez-vous que les index correspondent si vous fusionnez.
        # Ici, on va créer une copie pour ajouter le cluster.
        df_clustered_output = clustering.apply_kmeans_clustering(df_numeric_for_clustering,
                                                               n_clusters=chosen_k,
                                                               original_df=df_for_modeling.loc[df_numeric_for_clustering.index]) # Utiliser les mêmes index
        if df_clustered_output is not None:
            exporter.export_dataframe_to_csv(df_clustered_output, os.path.join(modeling_output_dir, "data_with_clusters.csv"))
            print("Données avec clusters sauvegardées.")

            # Visualisation des clusters (si possible et pertinent, ex: 2 premières features)
            if len(df_numeric_for_clustering.columns) >= 2 and 'cluster' in df_clustered_output.columns:
                plt.figure(figsize=(10,7))
                sns.scatterplot(x=df_clustered_output.iloc[:,0], y=df_clustered_output.iloc[:,1], hue='cluster', data=df_clustered_output, palette='viridis')
                plt.title(f'Clusters K-Means (k={chosen_k}) sur les 2 premières features')
                plt.xlabel(df_clustered_output.columns[0])
                plt.ylabel(df_clustered_output.columns[1])
                cluster_scatter_path = os.path.join(modeling_output_dir, "kmeans_scatter_plot.png")
                plt.savefig(cluster_scatter_path)
                plt.show()
                print(f"Scatter plot des clusters sauvegardé: {cluster_scatter_path}")
        else:
            print("Échec du clustering.")
    else:
        print("Pas assez de données numériques ou d'échantillons pour le clustering K-Means.")


    # Exemple de Classification (L'UTILISATEUR DOIT DÉFINIR FEATURES ET CIBLE)
    # Supposons qu'une colonne 'target_classification' existe ou est dérivée.
    # Pour cet exemple, nous allons essayer de prédire le premier cluster si le clustering a fonctionné.
    print("\n--- Exemple de Classification (Prédiction du premier cluster) ---")
    if 'cluster' in df_clustered_output.columns and len(df_clustered_output['cluster'].unique()) > 1:
        # Utiliser les features de df_numeric_for_clustering (avant ajout de 'cluster')
        # et la colonne 'cluster' comme cible.
        # Cela revient à voir si les features peuvent prédire les clusters trouvés.

        # S'assurer que X ne contient pas la colonne 'cluster' si elle y a été ajoutée par erreur
        classification_features = [col for col in df_numeric_for_clustering.columns if col != 'cluster']
        classification_target = 'cluster'

        # Utiliser df_clustered_output qui contient les features et la cible 'cluster'
        # S'assurer que classification_features sont bien dans df_clustered_output et sont numériques
        valid_clf_features = [f for f in classification_features if f in df_clustered_output.columns and pd.api.types.is_numeric_dtype(df_clustered_output[f])]

        if valid_clf_features and classification_target in df_clustered_output.columns:
            print(f"Features pour classification: {valid_clf_features}")
            print(f"Cible pour classification: {classification_target}")

            class_model, X_test_c, y_test_c, acc_c, report_c = classification.train_classification_model(
                df_clustered_output, # Utiliser le df qui contient la cible 'cluster'
                feature_columns=valid_clf_features,
                target_column=classification_target,
                model_type='random_forest'
            )
            if class_model:
                print("Rapport de Classification (prédiction du cluster) :\n", report_c)
                exporter.export_dict_to_json({"accuracy": acc_c, "classification_report": report_c},
                                             os.path.join(modeling_output_dir, "classification_report_cluster.json"))

                # Matrice de confusion
                unique_labels_c = sorted(df_clustered_output[classification_target].unique())
                cm_c_path = os.path.join(modeling_output_dir, "classification_confusion_matrix.png")
                classification.plot_confusion_matrix_classification(y_test_c, class_model.predict(X_test_c),
                                                                    classes=[str(x) for x in unique_labels_c],
                                                                    plot_path=cm_c_path)
        else:
            print("Pas assez de features valides ou cible non trouvée pour l'exemple de classification.")
    else:
        print("Pas de cible de cluster disponible pour l'exemple de classification.")

    # Exemple de Régression (L'UTILISATEUR DOIT DÉFINIR FEATURES ET CIBLE NUMÉRIQUE)
    # Supposons qu'une colonne 'target_regression' (numérique) existe.
    # Pour cet exemple, nous allons essayer de prédire la première feature numérique à partir des autres.
    print("\n--- Exemple de Régression (Prédiction de la première feature numérique) ---")
    if len(numeric_cols) > 1:
        regression_target = numeric_cols[0]
        regression_features = [col for col in numeric_cols if col != regression_target and col in df_for_modeling.columns]

        # S'assurer que les features sont bien dans df_for_modeling et sont numériques
        valid_reg_features = [f for f in regression_features if f in df_for_modeling.columns and pd.api.types.is_numeric_dtype(df_for_modeling[f])]

        if valid_reg_features and regression_target in df_for_modeling.columns and pd.api.types.is_numeric_dtype(df_for_modeling[regression_target]):
            print(f"Features pour régression: {valid_reg_features}")
            print(f"Cible pour régression: {regression_target}")

            reg_model, X_test_r, y_test_r, mse_r, r2_r = regression.train_regression_model(
                df_for_modeling, # Utilise df_for_modeling qui contient les features et la cible
                feature_columns=valid_reg_features,
                target_column=regression_target,
                model_type='linear_regression'
            )
            if reg_model:
                exporter.export_dict_to_json({"mse": mse_r, "r2_score": r2_r},
                                             os.path.join(modeling_output_dir, "regression_metrics.json"))
                reg_plot_path = os.path.join(modeling_output_dir, "regression_real_vs_pred.png")
                regression.plot_regression_results(y_test_r, reg_model.predict(X_test_r), plot_path=reg_plot_path)
        else:
            print("Pas assez de features valides ou cible non valide pour l'exemple de régression.")
    else:
        print("Pas assez de colonnes numériques pour l'exemple de régression.")

    print(f"\n--- FIN DU PIPELINE D'ANALYSE DE DONNÉES ---")
    print(f"Les résultats sont sauvegardés dans le répertoire : {BASE_OUTPUT_DIR}")
    plt.close('all') # Ferme tous les graphiques matplotlib ouverts

if __name__ == '__main__':
    # Créer un fichier CSV de démo pour tester le pipeline
    demo_data_file = "demo_data.csv"
    num_samples = 100
    data = {
        'Age': np.random.randint(20, 60, num_samples),
        'Salaire': np.random.normal(50000, 15000, num_samples).astype(int),
        'Experience': np.random.uniform(0, 30, num_samples).round(1),
        'Departement': np.random.choice(['Ventes', 'Marketing', 'IT', 'RH', None], num_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'Satisfaction': np.random.choice([1, 2, 3, 4, 5, np.nan], num_samples, p=[0.1,0.15,0.3,0.25,0.1,0.1]),
        'ProjetsActuels': np.random.randint(0,5,num_samples)
    }
    df_demo = pd.DataFrame(data)
    # Introduire quelques NaN supplémentaires
    df_demo.loc[df_demo.sample(frac=0.05).index, 'Age'] = np.nan
    df_demo.loc[df_demo.sample(frac=0.08).index, 'Salaire'] = np.nan

    loader.export_dataframe_to_csv(df_demo, demo_data_file, index=False) # Utiliser notre propre exportateur pour créer le fichier
    print(f"Fichier de données de démonstration '{demo_data_file}' créé.")

    # Exécuter le pipeline principal avec les données de démo
    # Vous pouvez changer 'demo_data.csv' par le chemin vers votre propre fichier de données.
    main_data_analysis_pipeline(data_path=demo_data_file, data_format='csv')

    # Nettoyage du fichier de démo après exécution
    # if os.path.exists(demo_data_file):
    #     os.remove(demo_data_file)
    #     print(f"Fichier de données de démonstration '{demo_data_file}' supprimé.")
