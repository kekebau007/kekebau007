import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_regression_model(df: pd.DataFrame,
                           feature_columns: list,
                           target_column: str,
                           model_type: str = 'random_forest',
                           test_size: float = 0.2,
                           random_state: int = 42):
    """
    Entraîne un modèle de régression (Random Forest Regressor ou Linear Regression).
    Nécessite des données où les features et la cible sont numériques.

    Args:
        df (pd.DataFrame): DataFrame contenant les features et la cible.
        feature_columns (list): Liste des noms de colonnes à utiliser comme features.
        target_column (str): Nom de la colonne cible (doit être numérique).
        model_type (str): Type de modèle ('random_forest' ou 'linear_regression').
        test_size (float): Proportion du dataset à inclure dans le split de test.
        random_state (int): Graine pour la reproductibilité.

    Returns:
        tuple: (model, X_test, y_test, mse, r2)
               - model: Le modèle entraîné.
               - X_test: Features de l'ensemble de test.
               - y_test: Cible de l'ensemble de test.
               - mse (float): Erreur quadratique moyenne sur l'ensemble de test.
               - r2 (float): Score R² sur l'ensemble de test.
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

    # Vérifier si les features et la cible sont numériques
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        print("Erreur: Toutes les colonnes features doivent être numériques. Veuillez les encoder/prétraiter.")
        return None, None, None, None, None

    if not pd.api.types.is_numeric_dtype(y):
        print(f"Erreur: La colonne cible '{target_column}' doit être numérique pour la régression.")
        return None, None, None, None, None

    if X.isnull().values.any() or y.isnull().values.any():
        print("Erreur: Les features ou la cible contiennent des valeurs NaN. Veuillez les imputer.")
        return None, None, None, None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    except Exception as e_split:
            print(f"Échec du split: {e_split}")
            return None, None, None, None, None

    if model_type == 'random_forest':
        model = RandomForestRegressor(random_state=random_state, n_estimators=100)
        print("Entraînement du modèle Random Forest Regressor...")
    elif model_type == 'linear_regression':
        model = LinearRegression()
        print("Entraînement du modèle Linear Regression...")
    else:
        print(f"Erreur: Type de modèle '{model_type}' non supporté. Choisissez 'random_forest' ou 'linear_regression'.")
        return None, None, None, None, None

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Modèle '{model_type}' entraîné avec succès.")
        print(f"Erreur Quadratique Moyenne (MSE) sur l'ensemble de test : {mse:.4f}")
        print(f"Score R² sur l'ensemble de test : {r2:.4f}")

        return model, X_test, y_test, mse, r2
    except Exception as e:
        print(f"Erreur lors de l'entraînement ou de la prédiction du modèle '{model_type}': {e}")
        return None, None, None, None, None


def plot_regression_results(y_true, y_pred, title: str = 'Résultats de Régression', plot_path: str = None):
    """
    Affiche un nuage de points des valeurs réelles vs. prédites.

    Args:
        y_true (array-like): Valeurs réelles.
        y_pred (array-like): Valeurs prédites.
        title (str): Titre du graphique.
        plot_path (str, optional): Chemin pour sauvegarder le graphique. Si None, affiche seulement.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--k', lw=2) # Ligne y=x
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Valeurs Prédites')
    plt.title(title)
    plt.grid(True)

    if plot_path:
        try:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            print(f"Graphique de régression sauvegardé dans : {plot_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du graphique de régression : {e}")
    plt.show()


if __name__ == '__main__':
    output_model_dir = "test_model_plots" # Réutiliser le dossier
    os.makedirs(output_model_dir, exist_ok=True)

    # Données de test synthétiques pour la régression
    # Features: Experience, NbProjets | Cible: Salaire
    np.random.seed(42)
    data_reg = {
        'Experience': np.random.rand(200) * 20 + 1, # 1 à 21 ans
        'NbProjets': np.random.randint(1, 10, size=200),
        'NiveauEducation': np.random.randint(1, 5, size=200) # 1:Bac, 2:Bac+2, ..., 4:Bac+5
    }
    df_test_reg = pd.DataFrame(data_reg)
    # Simuler une relation pour le salaire
    df_test_reg['Salaire'] = 30000 + df_test_reg['Experience'] * 1500 + df_test_reg['NbProjets'] * 500 + df_test_reg['NiveauEducation'] * 2000 + np.random.normal(0, 5000, size=200)
    df_test_reg['Salaire'] = df_test_reg['Salaire'].astype(int)

    feature_cols_reg = ['Experience', 'NbProjets', 'NiveauEducation']
    target_col_reg = 'Salaire'

    print("===== Test Régression Random Forest =====")
    rf_reg_model, X_test_rf_reg, y_test_rf_reg, mse_rf_reg, r2_rf_reg = train_regression_model(
        df_test_reg.copy(),
        feature_columns=feature_cols_reg,
        target_column=target_col_reg,
        model_type='random_forest'
    )

    if rf_reg_model:
        plot_path_rf = os.path.join(output_model_dir, "regression_plot_rf.png")
        plot_regression_results(y_test_rf_reg, rf_reg_model.predict(X_test_rf_reg),
                                title='Random Forest Regression: Réel vs. Prédit',
                                plot_path=plot_path_rf)
    print("\n" + "="*30 + "\n")


    print("===== Test Régression Linéaire =====")
    lr_model, X_test_lr, y_test_lr, mse_lr, r2_lr = train_regression_model(
        df_test_reg.copy(),
        feature_columns=feature_cols_reg,
        target_column=target_col_reg,
        model_type='linear_regression'
    )

    if lr_model:
        plot_path_lr = os.path.join(output_model_dir, "regression_plot_lr.png")
        plot_regression_results(y_test_lr, lr_model.predict(X_test_lr),
                                title='Linear Regression: Réel vs. Prédit',
                                plot_path=plot_path_lr)

        # Afficher les coefficients pour la régression linéaire
        if hasattr(lr_model, 'coef_'):
            print("\nCoefficients de la Régression Linéaire:")
            for feature, coef in zip(feature_cols_reg, lr_model.coef_):
                print(f"  {feature}: {coef:.2f}")
            print(f"  Intercept: {lr_model.intercept_:.2f}")

    print("\n" + "="*30 + "\n")

    print("===== Test avec cible non numérique (attendu: erreur) =====")
    df_test_reg_bad_target = df_test_reg.copy()
    df_test_reg_bad_target['Salaire_Cat'] = pd.cut(df_test_reg_bad_target['Salaire'], bins=3, labels=['Bas', 'Moyen', 'Haut'])
    train_regression_model(
        df_test_reg_bad_target,
        feature_columns=feature_cols_reg,
        target_column='Salaire_Cat'
    )
    print("\n" + "="*30 + "\n")

    plt.close('all')
    print(f"\nNOTE: Les graphiques de test de modélisation sont sauvegardés dans '{output_model_dir}'.")
