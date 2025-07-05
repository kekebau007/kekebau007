import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_histogram(df: pd.DataFrame, column: str, bins: int = 10, save_path: str = None):
    """
    Génère et affiche un histogramme pour une colonne spécifiée.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        column (str): Nom de la colonne pour laquelle générer l'histogramme.
        bins (int): Nombre de bins pour l'histogramme.
        save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, affiche seulement.
    """
    if column not in df.columns:
        print(f"Erreur : La colonne '{column}' n'existe pas dans le DataFrame.")
        return
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Erreur : La colonne '{column}' n'est pas numérique. L'histogramme est pour les données numériques.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(f'Histogramme de {column}')
    plt.xlabel(column)
    plt.ylabel('Fréquence')
    plt.grid(axis='y', alpha=0.75)

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Histogramme sauvegardé dans : {save_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du graphique : {e}")
    plt.show()


def plot_boxplot(df: pd.DataFrame, y_column: str, x_column: str = None, save_path: str = None):
    """
    Génère et affiche un boxplot.
    Si x_column est fourni, génère des boxplots de y_column pour chaque catégorie de x_column.
    Sinon, génère un boxplot simple pour y_column.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        y_column (str): Nom de la colonne numérique pour l'axe Y.
        x_column (str, optional): Nom de la colonne catégorielle pour l'axe X.
        save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, affiche seulement.
    """
    if y_column not in df.columns:
        print(f"Erreur : La colonne Y '{y_column}' n'existe pas.")
        return
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        print(f"Erreur : La colonne Y '{y_column}' doit être numérique pour un boxplot.")
        return
    if x_column and x_column not in df.columns:
        print(f"Erreur : La colonne X '{x_column}' n'existe pas.")
        return

    plt.figure(figsize=(12, 7))
    if x_column:
        sns.boxplot(x=df[x_column], y=df[y_column])
        plt.title(f'Boxplot de {y_column} par {x_column}')
    else:
        sns.boxplot(y=df[y_column])
        plt.title(f'Boxplot de {y_column}')

    plt.ylabel(y_column)
    if x_column:
        plt.xlabel(x_column)
        plt.xticks(rotation=45, ha='right') # Améliore la lisibilité pour de nombreux groupes
    plt.tight_layout() # Ajuste pour éviter que les labels se chevauchent

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Boxplot sauvegardé dans : {save_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du graphique : {e}")
    plt.show()


def plot_scatterplot(df: pd.DataFrame, x_column: str, y_column: str, hue_column: str = None, save_path: str = None):
    """
    Génère et affiche un nuage de points entre deux colonnes numériques.
    Peut utiliser une troisième colonne pour la coloration (hue).

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        x_column (str): Nom de la colonne pour l'axe X.
        y_column (str): Nom de la colonne pour l'axe Y.
        hue_column (str, optional): Nom de la colonne pour la coloration des points.
        save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, affiche seulement.
    """
    if x_column not in df.columns or y_column not in df.columns:
        print(f"Erreur : Les colonnes X ('{x_column}') ou Y ('{y_column}') n'existent pas.")
        return
    if not pd.api.types.is_numeric_dtype(df[x_column]) or not pd.api.types.is_numeric_dtype(df[y_column]):
        print(f"Erreur : Les colonnes X ('{x_column}') et Y ('{y_column}') doivent être numériques pour un nuage de points.")
        return
    if hue_column and hue_column not in df.columns:
        print(f"Avertissement : La colonne Hue '{hue_column}' n'existe pas. Le graphique sera généré sans coloration.")
        hue_column = None

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=df[x_column], y=df[y_column], hue=df[hue_column] if hue_column else None)
    plt.title(f'Nuage de points : {y_column} vs {x_column}' + (f' (Coloré par {hue_column})' if hue_column else ''))
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True, alpha=0.5)

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Nuage de points sauvegardé dans : {save_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du graphique : {e}")
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, method: str = 'pearson', columns_to_correlate: list = None, save_path: str = None):
    """
    Génère et affiche une heatmap de la matrice de corrélation.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        method (str): Méthode de corrélation ('pearson', 'kendall', 'spearman').
        columns_to_correlate (list, optional): Liste spécifique de colonnes numériques à utiliser.
                                               Si None, toutes les colonnes numériques sont utilisées.
        save_path (str, optional): Chemin pour sauvegarder le graphique. Si None, affiche seulement.
    """
    numeric_df = df.select_dtypes(include=np.number)

    if columns_to_correlate:
        valid_cols = [col for col in columns_to_correlate if col in numeric_df.columns]
        if not valid_cols:
            print("Aucune colonne numérique valide spécifiée pour la heatmap de corrélation.")
            return
        numeric_df_subset = numeric_df[valid_cols]
    else:
        numeric_df_subset = numeric_df

    if numeric_df_subset.empty or numeric_df_subset.shape[1] < 2:
        print("Pas assez de colonnes numériques pour générer une heatmap de corrélation.")
        return

    corr_matrix = numeric_df_subset.corr(method=method)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Heatmap de Corrélation (Méthode: {method.capitalize()})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Heatmap de corrélation sauvegardée dans : {save_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du graphique : {e}")
    plt.show()


if __name__ == '__main__':
    # Création d'un répertoire de sortie pour les graphiques de test
    output_plot_dir = "test_plots"
    os.makedirs(output_plot_dir, exist_ok=True)

    data_vis = {
        'Age': [25, 30, 22, 45, 33, 30, 28, 50, 35, 40, 23, 29],
        'Salaire': [50000, 60000, 45000, 90000, 75000, 55000, 120000, 65000, 80000, 85000, 48000, 62000],
        'Departement': ['RH', 'IT', 'Marketing', 'IT', 'RH', 'IT', 'Finance', 'Marketing', 'Finance', 'IT', 'Marketing', 'RH'],
        'Experience': [2, 5, 1, 10, 7, 4, 15, 6, 8, 9, 2, 5],
        'Satisfaction': ['Haute', 'Moyenne', 'Basse', 'Haute', 'Moyenne', 'Moyenne', 'Haute', 'Basse', 'Haute', 'Moyenne', 'Basse', 'Haute']
    }
    df_test_vis = pd.DataFrame(data_vis)

    print("===== Test plot_histogram (Age) =====")
    plot_histogram(df_test_vis, 'Age', bins=5, save_path=os.path.join(output_plot_dir, "histogram_age.png"))

    print("\n===== Test plot_histogram (colonne non numérique - attendu: erreur) =====")
    plot_histogram(df_test_vis, 'Departement', save_path=os.path.join(output_plot_dir, "histogram_dept_fail.png"))

    print("\n===== Test plot_boxplot (Salaire) =====")
    plot_boxplot(df_test_vis, y_column='Salaire', save_path=os.path.join(output_plot_dir, "boxplot_salaire.png"))

    print("\n===== Test plot_boxplot (Salaire par Departement) =====")
    plot_boxplot(df_test_vis, y_column='Salaire', x_column='Departement', save_path=os.path.join(output_plot_dir, "boxplot_salaire_par_dept.png"))

    print("\n===== Test plot_scatterplot (Age vs Salaire) =====")
    plot_scatterplot(df_test_vis, x_column='Age', y_column='Salaire', save_path=os.path.join(output_plot_dir, "scatter_age_salaire.png"))

    print("\n===== Test plot_scatterplot (Experience vs Salaire, coloré par Departement) =====")
    plot_scatterplot(df_test_vis, x_column='Experience', y_column='Salaire', hue_column='Departement', save_path=os.path.join(output_plot_dir, "scatter_exp_salaire_hue_dept.png"))

    print("\n===== Test plot_scatterplot (Experience vs Salaire, hue inexistante) =====")
    plot_scatterplot(df_test_vis, x_column='Experience', y_column='Salaire', hue_column='Inexistant', save_path=os.path.join(output_plot_dir, "scatter_exp_salaire_hue_inexistant.png"))

    print("\n===== Test plot_correlation_heatmap (auto) =====")
    plot_correlation_heatmap(df_test_vis, save_path=os.path.join(output_plot_dir, "heatmap_correlation.png"))

    print("\n===== Test plot_correlation_heatmap (spearman, colonnes spécifiques) =====")
    plot_correlation_heatmap(df_test_vis, method='spearman', columns_to_correlate=['Age', 'Salaire', 'Experience'], save_path=os.path.join(output_plot_dir, "heatmap_correlation_specific.png"))

    print(f"\nNOTE: Les graphiques de test sont sauvegardés dans le dossier '{output_plot_dir}'.")
    print("Dans un environnement interactif (comme Jupyter), plt.show() les afficherait directement.")
    print("Pour cet environnement de script, les fichiers sont sauvegardés et plt.show() est appelé, ce qui peut ou pas afficher une fenêtre selon la config du backend matplotlib.")
    plt.close('all') # Ferme toutes les figures pour éviter les fuites de mémoire dans les tests en boucle.
