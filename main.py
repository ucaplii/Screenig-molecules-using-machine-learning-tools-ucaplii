from models.model import RandomForest, XGBoost, KernelRidgeRegression,SupportVectorRegression,MLP
from models.model_trainer import ModelTrainer, run_Trainer
import pandas as pd


# def main():
    
#     # Load your data from a CSV file
#     data_file = './data/All_Data_Training_with_classes.csv'  # Replace with your actual data file path
#     data_file1 = './data/class_0_data.csv'
#     data_file2 = './data/class_1_data.csv'
#     data = pd.read_csv(data_file)
#     data1 = pd.read_csv(data_file1)
#     data2 = pd.read_csv(data_file2)

#     # Create an instance of the RandomForest class
#     model_instance = RandomForest()
#     # Get the actual RandomForest model from the instance
#     rf = model_instance.get_model()

#     xgb = XGBoost().get_model()
#     krr = KernelRidgeRegression().get_model()
#     svr = SupportVectorRegression().get_model()
#     mlp = MLP().get_model()
#     model_trainer = ModelTrainer(mlp, data, num_iterations=1)

#     model_trainer.train()
#     model_trainer.plot_predictions_vs_actual()
#     model_trainer.plot_residuals()

#     model_trainer1 = ModelTrainer(mlp, data1, num_iterations=1)

#     model_trainer1.train()
#     model_trainer1.plot_predictions_vs_actual()
#     model_trainer1.plot_residuals()

#     model_trainer2 = ModelTrainer(mlp, data2, num_iterations=1)

#     model_trainer2.train()
#     model_trainer2.plot_predictions_vs_actual()
#     model_trainer2.plot_residuals()

#     # Evaluate model and print metrics

def main():
    run_Trainer()


if __name__ == "__main__":
    main()


