import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_classification_model(df: pd.DataFrame,
                               feature_columns: list,
                               target_column: str,
                               model_type: str = 'random_forest',
                               test_size: float = 0.2,
                               random_state: int = 42):
    """
    Entraîne un modèle de classification (Random Forest ou Logistic Regression).
    Nécessite des données où les features sont numériques et la cible encodée numériquement si besoin.

    Args:
        df (pd.DataFrame): DataFrame contenant les features et la cible.
        feature_columns (list): Liste des noms de colonnes à utiliser comme features.
        target_column (str): Nom de la colonne cible.
        model_type (str): Type de modèle ('random_forest' ou 'logistic_regression').
        test_size (float): Proportion du dataset à inclure dans le split de test.
        random_state (int): Graine pour la reproductibilité.

    Returns:
        tuple: (model, X_test, y_test, accuracy, report)
               - model: Le modèle entraîné.
               - X_test: Features de l'ensemble de test.
               - y_test: Cible de l'ensemble de test.
               - accuracy (float): Précision du modèle sur l'ensemble de test.
               - report (str): Rapport de classification.
               Retourne (None, None, None, None, None) si une erreur survient.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Erreur: df doit être un DataFrame pandas non vide.")
        return None, None, None, None, None

    if target_column not in df.columns:
        print(f"Erreur: La colonne cible '{target_column}' n'est pas dans le DataFrame.")
        return None, None, None, None, None

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"Erreur: Les colonnes features suivantes sont manquantes : {missing_features}")
        return None, None, None, None, None

    X = df[feature_columns]
    y = df[target_column]

    # Vérifier si les features sont numériques
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        print("Erreur: Toutes les colonnes features doivent être numériques. Veuillez les encoder/prétraiter.")
        # On pourrait ajouter un encodage automatique ici, mais c'est mieux de le faire en amont.
        return None, None, None, None, None

    # Vérifier si la cible est numérique (LabelEncoder peut être nécessaire en amont)
    if not pd.api.types.is_numeric_dtype(y):
        print(f"Avertissement: La colonne cible '{target_column}' n'est pas numérique. Une tentative d'encodage avec LabelEncoder sera faite.")
        le = LabelEncoder()
        try:
            y = le.fit_transform(y)
            print(f"Colonne cible '{target_column}' encodée avec succès. Classes: {le.classes_}")
        except Exception as e:
            print(f"Erreur lors de l'encodage de la cible : {e}")
            return None, None, None, None, None

    if X.isnull().values.any() or pd.Series(y).isnull().values.any(): # y peut être un array numpy après LabelEncoder
        print("Erreur: Les features ou la cible contiennent des valeurs NaN. Veuillez les imputer.")
        return None, None, None, None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None)
    except ValueError as e:
        # Souvent dû à un nombre insuffisant d'échantillons pour une classe pour stratify
        print(f"Erreur lors du train_test_split (possiblement due à la stratification avec peu de données par classe): {e}")
        print("Tentative de split sans stratification.")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        except Exception as e_split:
            print(f"Échec du split même sans stratification: {e_split}")
            return None, None, None, None, None


    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state, n_estimators=100)
        print("Entraînement du modèle Random Forest Classifier...")
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=random_state, max_iter=1000) # max_iter augmenté pour convergence
        print("Entraînement du modèle Logistic Regression...")
    else:
        print(f"Erreur: Type de modèle '{model_type}' non supporté. Choisissez 'random_forest' ou 'logistic_regression'.")
        return None, None, None, None, None

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0) # zero_division pour éviter les warnings

        print(f"Modèle '{model_type}' entraîné avec succès.")
        print(f"Précision sur l'ensemble de test : {accuracy:.4f}")
        # print("Rapport de classification :\n", report) # Déplacé hors de la fonction pour plus de flexibilité

        return model, X_test, y_test, accuracy, report
    except Exception as e:
        print(f"Erreur lors de l'entraînement ou de la prédiction du modèle '{model_type}': {e}")
        return None, None, None, None, None


def plot_confusion_matrix_classification(y_true, y_pred, classes: list = None, title: str = 'Matrice de Confusion', plot_path: str = None):
    """
    Affiche la matrice de confusion.

    Args:
        y_true (array-like): Vraies étiquettes.
        y_pred (array-like): Étiquettes prédites.
        classes (list, optional): Noms des classes pour les axes.
        title (str): Titre du graphique.
        plot_path (str, optional): Chemin pour sauvegarder le graphique. Si None, affiche seulement.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes if classes else 'auto',
                yticklabels=classes if classes else 'auto')
    plt.title(title)
    plt.ylabel('Vraie étiquette')
    plt.xlabel('Étiquette prédite')

    if plot_path:
        try:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            print(f"Matrice de confusion sauvegardée dans : {plot_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la matrice de confusion : {e}")
    plt.show()


if __name__ == '__main__':
    output_model_dir = "test_model_plots" # Réutiliser le dossier créé par clustering.py
    os.makedirs(output_model_dir, exist_ok=True)

    # Données de test synthétiques pour la classification
    # Features: Age, Salaire | Cible: Promotion (0 ou 1)
    np.random.seed(42)
    data_class = {
        'Age': np.random.randint(22, 60, size=200),
        'Salaire': np.random.randint(30000, 120000, size=200),
        'Experience': np.random.randint(1, 30, size=200),
        'Promotion_Num': np.random.choice([0, 1], size=200, p=[0.7, 0.3]), # Cible numérique
        'Departement': np.random.choice(['IT', 'RH', 'Finance'], size=200) # Cible catégorielle
    }
    df_test_class = pd.DataFrame(data_class)

    # Simuler une relation pour rendre la classification un peu significative
    df_test_class['Salaire'] = df_test_class['Salaire'] + df_test_class['Promotion_Num'] * 20000 + df_test_class['Experience'] * 500
    df_test_class['Age'] = df_test_class['Age'] + df_test_class['Promotion_Num'] * 5


    feature_cols = ['Age', 'Salaire', 'Experience']
    target_col_num = 'Promotion_Num'
    target_col_cat = 'Departement' # Pour tester l'encodage automatique de la cible

    print("===== Test Classification Random Forest (cible numérique) =====")
    rf_model, X_test_rf, y_test_rf, acc_rf, report_rf = train_classification_model(
        df_test_class.copy(),
        feature_columns=feature_cols,
        target_column=target_col_num,
        model_type='random_forest'
    )

    if rf_model:
        print("Rapport de classification (Random Forest):\n", report_rf)
        cm_path_rf = os.path.join(output_model_dir, "confusion_matrix_rf.png")
        plot_confusion_matrix_classification(y_test_rf, rf_model.predict(X_test_rf), classes=['Non Promu', 'Promu'], plot_path=cm_path_rf)
    print("\n" + "="*30 + "\n")


    print("===== Test Classification Logistic Regression (cible catégorielle 'Departement') =====")
    # Pour Logistic Regression, il est bon d'avoir des features mises à l'échelle, mais on teste sans pour l'instant
    # La fonction gère l'encodage de la cible catégorielle

    # Copier pour ne pas modifier l'original, car la cible 'Departement' sera encodée en place dans y
    df_test_logreg = df_test_class.copy()

    log_model, X_test_log, y_test_log, acc_log, report_log = train_classification_model(
        df_test_logreg, # Passe la copie ici
        feature_columns=feature_cols,
        target_column=target_col_cat,
        model_type='logistic_regression'
    )

    if log_model:
        print("Rapport de classification (Logistic Regression - Departement):\n", report_log)
        # Récupérer les classes de LabelEncoder si possible (pas directement retourné par la fonction actuelle)
        # Pour l'exemple, on les hardcode ou on les déduit de y_test_log si toutes sont présentes
        unique_classes_encoded = sorted(np.unique(y_test_log))
        # Si LabelEncoder a été utilisé dans la fonction, on aurait besoin de ses classes_
        # Ici, on suppose qu'on les connaît ou qu'on peut les reconstruire
        # Pour ce test: 'Finance' (0), 'IT' (1), 'RH' (2) après LabelEncoding par ordre alphabétique
        class_names_dept = ['Finance', 'IT', 'RH'] if len(unique_classes_encoded) == 3 else [str(i) for i in unique_classes_encoded]

        cm_path_log = os.path.join(output_model_dir, "confusion_matrix_logreg_dept.png")
        plot_confusion_matrix_classification(y_test_log, log_model.predict(X_test_log), classes=class_names_dept, plot_path=cm_path_log)
    print("\n" + "="*30 + "\n")

    print("===== Test avec features non numériques (attendu: erreur) =====")
    df_test_class_with_cat_features = df_test_class.copy()
    df_test_class_with_cat_features['CatFeature'] = np.random.choice(['X', 'Y', 'Z'], size=200)
    train_classification_model(
        df_test_class_with_cat_features,
        feature_columns=feature_cols + ['CatFeature'],
        target_column=target_col_num
    )
    print("\n" + "="*30 + "\n")

    print("===== Test avec NaN dans les features (attendu: erreur) =====")
    df_test_with_nan = df_test_class.copy()
    df_test_with_nan.loc[0, 'Age'] = np.nan
    train_classification_model(
        df_test_with_nan,
        feature_columns=feature_cols,
        target_column=target_col_num
    )
    plt.close('all')
    print(f"\nNOTE: Les graphiques de test de modélisation sont sauvegardés dans '{output_model_dir}'.")
