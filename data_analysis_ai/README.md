# Projet d'Analyse de Données avec IA

Ce projet fournit une structure et des modules de base pour effectuer une analyse de données complète, depuis l'ingestion jusqu'à la modélisation (clustering, classification, régression) et l'exportation des résultats.

## Structure du Projet

```
data_analysis_ai/
├── main.py                   # Script principal pour exécuter le pipeline d'analyse
├── requirements.txt          # Dépendances du projet (à générer)
├── data_ingestion/
│   ├── __init__.py
│   └── loader.py             # Modules pour charger les données (CSV, JSON)
├── preprocessing/
│   ├── __init__.py
│   ├── cleaner.py            # Gestion des valeurs manquantes
│   ├── encoder.py            # Encodage des variables catégorielles
│   └── scaler.py             # Mise à l'échelle des variables numériques
├── analysis/
│   ├── __init__.py
│   ├── eda.py                # Fonctions pour l'analyse exploratoire des données (statistiques)
│   └── visualization.py      # Fonctions pour la visualisation (histogrammes, boxplots, etc.)
├── modeling/
│   ├── __init__.py
│   ├── clustering.py         # Algorithmes de clustering (K-Means)
│   ├── classification.py     # Algorithmes de classification (Random Forest, Logistic Regression)
│   └── regression.py         # Algorithmes de régression (Random Forest, Linear Regression)
├── utils/
│   ├── __init__.py
│   └── exporter.py           # Fonctions pour exporter les DataFrames et dictionnaires
└── README.md                 # Ce fichier
```

## Fonctionnalités

- **Ingestion de Données :** Chargement de fichiers CSV et JSON.
- **Prétraitement :**
    - Gestion configurable des valeurs manquantes.
    - Encodage des variables catégorielles (One-Hot, Label Encoding).
    - Mise à l'échelle des variables numériques (StandardScaler, MinMaxScaler, RobustScaler).
- **Analyse Exploratoire des Données (EDA) :**
    - Informations générales sur le DataFrame.
    - Statistiques descriptives.
    - Comptage des valeurs.
    - Matrice de corrélation.
- **Visualisation :**
    - Histogrammes, Boxplots, Nuages de points.
    - Heatmap de corrélation.
    - Sauvegarde des graphiques en fichiers image.
- **Modélisation :**
    - **Clustering :** K-Means avec méthode du coude pour estimation de `k`.
    - **Classification :** Random Forest Classifier, Logistic Regression avec évaluation (accuracy, rapport de classification, matrice de confusion).
    - **Régression :** Random Forest Regressor, Linear Regression avec évaluation (MSE, R², graphique réel vs. prédit).
- **Exportation :** Sauvegarde des DataFrames (CSV, JSON) et des résultats d'analyse/modélisation (JSON, images).

## Utilisation

1.  **Installation des dépendances :**
    Créez un environnement virtuel (recommandé) et installez les bibliothèques nécessaires. Un fichier `requirements.txt` sera fourni.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

2.  **Préparer vos données :**
    Assurez-vous que votre fichier de données (par exemple, `my_data.csv`) est accessible.

3.  **Configurer et exécuter le pipeline principal :**
    Ouvrez le fichier `main.py`.
    - Modifiez la variable `data_path` dans la section `if __name__ == '__main__':` pour pointer vers votre fichier de données.
    - Ajustez le `data_format` si vos données ne sont pas en CSV.
    - **IMPORTANT :** Adaptez les sections de prétraitement et surtout de modélisation dans la fonction `main_data_analysis_pipeline` pour qu'elles correspondent à vos données spécifiques (sélection des features, colonnes cibles, stratégies appropriées). Les exemples fournis dans `main.py` sont génériques.

    Exécutez le script :
    ```bash
    python main.py
    ```

4.  **Consulter les résultats :**
    Les résultats, incluant les données traitées, les graphiques EDA, et les rapports de modélisation, seront sauvegardés dans le répertoire `output_analysis_results` (ou celui que vous aurez configuré dans `BASE_OUTPUT_DIR` dans `main.py`).

## Personnalisation

- **Stratégies de Prétraitement :** Modifiez les appels aux fonctions dans `main.py` (par exemple, `cleaner.handle_missing_values`, `encoder.encode_categorical_features`, `scaler.scale_numerical_features`) pour utiliser les stratégies, colonnes, et paramètres les plus adaptés à votre jeu de données.
- **Sélection de Features et Cibles :** La partie la plus cruciale à adapter est la section de modélisation dans `main.py`. Vous devez explicitement définir quelles colonnes utiliser comme features (`feature_columns`) et quelle colonne est votre cible (`target_column`) pour les tâches de classification et de régression.
- **Choix des Modèles :** Vous pouvez changer les types de modèles utilisés (`model_type` dans les fonctions de `classification.py` et `regression.py`) ou ajouter d'autres algorithmes.
- **Paramètres des Modèles :** Les modèles sont initialisés avec des paramètres par défaut. Pour optimiser les performances, vous pourriez vouloir implémenter une recherche d'hyperparamètres (par exemple, avec `GridSearchCV` de Scikit-learn), ce qui n'est pas inclus dans cette version de base.

## Modules Clés et Leurs Fonctions Principales

-   `loader.py`: `load_csv_data()`, `load_json_data()`
-   `cleaner.py`: `handle_missing_values()`
-   `encoder.py`: `encode_categorical_features()`
-   `scaler.py`: `scale_numerical_features()`
-   `eda.py`: `get_dataframe_info()`, `get_descriptive_statistics()`, `get_value_counts()`, `get_correlation_matrix()`
-   `visualization.py`: `plot_histogram()`, `plot_boxplot()`, `plot_scatterplot()`, `plot_correlation_heatmap()`
-   `clustering.py`: `find_optimal_k_elbow()`, `apply_kmeans_clustering()`
-   `classification.py`: `train_classification_model()`, `plot_confusion_matrix_classification()`
-   `regression.py`: `train_regression_model()`, `plot_regression_results()`
-   `exporter.py`: `export_dataframe_to_csv()`, `export_dataframe_to_json()`, `export_dict_to_json()`

Ce projet sert de point de départ. Une analyse de données approfondie et la construction de modèles performants nécessitent une compréhension fine des données et des itérations d'expérimentation.
