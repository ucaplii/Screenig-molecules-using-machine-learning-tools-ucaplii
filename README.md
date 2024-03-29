# Screenig-molecules-using-machine-learning-tools-ucaplii

This project aims  to predict energy barrier with ligands bound to K-Ras G12D protein using machine learning tools to screen molecules libraries, including Random Forest, XGBoost, Kernel Ridge Regression, Support Vector Regression, and Multilayer Perceptron. The project includes data preprocessing, feature extraction, model training, and evaluation.

## Setup

1. Clone this repository to your local machine.
2. Install the required additional packages using `pip`:

Make sure to install the following packages:

1. RDKit
2. MDAnalysis
3. natsort
4. scikit-learn
5. xgboost
```
pip install rdkit-pypi
pip install MDAnalysis
pip install natsort
pip install scikit-learn
pip install xgboost
```


## Project Structure
1. `data/`: Directory for input data files.
2. `data/ligands/`: Ligand PDB files.
3. `data/protein/`: Protein PDB file.
4. `figures/`: Directory for saving result plots.
5. `models/`: Directory for ML models and trainer.
6. `models/model.py`: Contain 5 ML models classes where parameters could be modified.
7. `models/model_trainer.py`: Contain ModelTrainer class and run_trainer function.
8. `utility/`: Directory for functions that genrate training data.
9. `utility/generate_data.py`: Contain all data preprocessing functinos.
10. `main.py`: Main program for model training and evaluating.
11. `README.md`: Project documentation.

## Usage
1. Run generate_data.py in utility folder to generate Morgan fingerprints, calculate ligand-protein distances, combine CSV files and split the data into two classes:
```
python utility/generate_data.py
```
After this process, `data/` folder should contain `All_Data_Training_with_classes.csv`, `class_0_data.csv` and `class_1_data.csv`, which will be used in the next step.

2. Run main.py to train and evaluate each machine learning model with different classes data:
```
python main.py
```
The program will train on entire 5 ML models with 10 iterations on full set, class 1 and class 2 data , print the performance and save the figures of output to `figure/` folder as following.

![image](https://github.com/ucaplii/screening-molecules-libraries-using-machine-learning-tools-ucaplii/assets/114681378/0f58bfd3-d3a7-458a-b2be-cc0eb689496e)

Each model has three sets of two figures for a total of six. For example, use XGB on class 2.

![XGBRegressor_class 2_pred_vs_act_figures](https://github.com/ucaplii/screening-molecules-libraries-using-machine-learning-tools-ucaplii/assets/114681378/c1856c65-1356-4b28-8238-2cd255d30a1f)

![XGBRegressor_class 2_residuals_figures](https://github.com/ucaplii/screening-molecules-libraries-using-machine-learning-tools-ucaplii/assets/114681378/ce40b0eb-5d52-4931-af54-64ea775d7765)


## Parameters 
1. Parameters of model can be change in `models/model.py` by modifing the `__init__` parameter in each ML model class.

Example in XGBoost Class.

```
class XGBoost:
    def __init__(self,
                 n_estimators=your_parameter,
                 learning_rate=your_parameter,
                 max_depth=your_parameter,
                 colsample_bytree=your_parameter,
                 reg_alpha=your_parameter,
                 subsample=your_parameter,
                 random_state=your_parameter):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.subsample = subsample
        self.random_state = random_state
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            subsample=subsample,
            random_state=random_state)

    def get_model(self):
        return self.model
```
2. numbers of iterations can be modified in the function `run_trainer` in `models/model_trainer.py`.
```
model_trainer = ModelTrainer(model, dataset, num_iterations=your_iterations)
model_trainer.train()
```
## Notes

1. There is no need to set any paths or parameters to run this programme. The parameters are predefined in the programme.
2. If parameters and paths need to be modified, make sure they are appropriate set by the above instructions.
3. Results may vary based on data quality, preprocessing, and model parameters.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

