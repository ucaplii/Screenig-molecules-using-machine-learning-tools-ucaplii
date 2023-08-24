from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr  
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utility.generate_data import generate_morgan_fingerprints,calculate_ligand_protein_distances,combine_csv_files
from models.model import RandomForest, XGBoost, KernelRidgeRegression,SupportVectorRegression,MLP

class ModelTrainer:
    """
    A class for training and evaluating machine learning models.

    Parameters:
    ----------
    model : object
        An instance of the machine learning model.
    data : DataFrame
        The input data for training and evaluation.
    num_folds : int, optional
        Number of cross-validation folds, by default 5.
    num_iterations : int, optional
        Number of iterations for repeated training, by default 10.
    """
    def __init__(self, model, data, num_folds=5, num_iterations=10):
        self.model = model
        self.data = data
        self.num_folds = num_folds
        self.num_iterations = num_iterations
        self.kf = KFold(n_splits=num_folds, shuffle=True)
        self.y_pred_all = np.zeros(len(data))

    def train(self):
        """
        Train the model using cross-validation and calculate evaluation metrics.

        Returns:
        ----------
        avg_mse : float
            Average Mean Squared Error across folds.
        avg_pearson_coeff : float
            Mean Pearson correlation coefficient across folds.
        """        
        x_data = self.data.drop(['barriers', 'cluster_label', 'files'], axis=1)
        y_data = self.data['barriers']

        fold_mses = []  # To store MSEs for each fold
        fold_pearson_coeffs = []  # To store Pearson coefficients for each fold

        for fold_idx, (train_indices, test_indices) in enumerate(self.kf.split(x_data), 1):
            x_train, x_test = x_data.iloc[train_indices], x_data.iloc[test_indices]
            y_train, y_test = y_data.iloc[train_indices], y_data.iloc[test_indices]

            fold_y_pred = np.zeros(len(test_indices))  # To store predictions for this fold

            for _ in range(self.num_iterations):
                self.model.fit(x_train, y_train)
                y_pred = self.model.predict(x_test)
                fold_y_pred += y_pred

            fold_y_pred /= self.num_iterations
            self.data.loc[test_indices, f'fold_{fold_idx}_y_pred'] = fold_y_pred  # Store predictions in DataFrame

            mse = mean_squared_error(y_test, fold_y_pred)
            fold_mses.append(mse)

            pearson_corr, _ = pearsonr(y_test, fold_y_pred)
            fold_pearson_coeffs.append(pearson_corr)

        self.data['y_pred'] = self.data.filter(like='fold').mean(axis=1)  # Calculate average predictions
        avg_mse = np.mean(fold_mses)  # Calculate average MSE across folds
        avg_pearson_coeff = np.mean(fold_pearson_coeffs)  # Calculate mean Pearson coefficient across folds

        print("Average Mean Squared Error across folds:", avg_mse)
        print("Mean Pearson correlation coefficient across folds:", avg_pearson_coeff)

        return avg_mse, avg_pearson_coeff


    def plot_predictions_vs_actual(self):
        """
        Plot the predicted vs. actual barriers.
        """
        y_true = self.data['barriers']
        y_pred = self.data['y_pred']

        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
        plt.xlabel('Actual Barriers', fontsize=14)
        plt.ylabel('Predicted Barriers', fontsize=14)
        plt.title('Predictions vs Actual', fontsize=16)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20) 



    def plot_residuals(self):
        """
        Plot the distribution of residuals.
        """        
        y_true = self.data['barriers']
        y_pred = self.data['y_pred']
        residuals = y_true - y_pred

        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=20, kde=True)
        plt.xlabel('Residuals', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Residuals Distribution', fontsize=16)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20) 

    

def run_Trainer():
    """
    Run training and evaluation of five machine learning models on three datasets.
    """

    rf = RandomForest()
    xgb = XGBoost()
    krr = KernelRidgeRegression()
    svr = SupportVectorRegression()
    mlp = MLP()

    data_file_full = './data/All_Data_Training_with_classes.csv' 
    data_file_1 = './data/class_0_data.csv'
    data_file_2 = './data/class_1_data.csv'



    models = [
        rf.get_model(),
        xgb.get_model(),
        krr.get_model(),
        svr.get_model(),
        mlp.get_model()
    ]

    data_files = [
        ("full set", data_file_full),
        ("class 1", data_file_1),
        ("class 2", data_file_2)
    ]

    for model in models:
        for dataset_name, datafile in data_files:
            model_name = model.__class__.__name__

            print(f"Training and evaluating {model_name} on {dataset_name}...")
            print("==========================================================================")

            dataset = pd.read_csv(datafile)
            model_trainer = ModelTrainer(model, dataset, num_iterations=10)
            model_trainer.train()

            # Plot predictions vs. actual
            model_trainer.plot_predictions_vs_actual()
            figure_pred_vs_act = f'./figures/{model_name}_{dataset_name}_pred_vs_act_figures.png'
            plt.savefig(figure_pred_vs_act, dpi=300)

            # Plot residuals
            model_trainer.plot_residuals()
            figure_residuals = f'./figures/{model_name}_{dataset_name}_residuals_figures.png'
            plt.savefig(figure_residuals, dpi=300)

            print("")
    

