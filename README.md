# Screenig-molecules-using-machine-learning-tools-ucaplii

This project aims to screen molecules using machine learning tools to predict energy barrier with ligands bound to K-Ras G12D protein, including Random Forest, XGBoost, Kernel Ridge Regression, Support Vector Regression, and Multilayer Perceptron. The project includes data preprocessing, feature extraction, model training, and evaluation.

## Setup

1. Clone this repository to your local machine.
2. Install the required additional packages using `pip`:

Make sure to install the following packages:

1. RDKit
2. MDAnalysis
3. natsort
4. xgboost
```
pip install rdkit-pypi
pip install MDAnalysis
pip install natsort
pip install xgboost
```

## Project Structure
1. `data/`: Directory for input data files.
2. `data/ligands/`: Ligand PDB files.
3. `data/protein/`: Protein PDB file.
4. `figures/`: Dicectory for saving result plots.
5. `models/`: Dicectory for ML models and trainer.
6. `utility/`: Dicectory for functions that genrate training data.
7. `utility/generate_data.py`: Contain all data preprocessing functinos.
8. `main.py`: Main program for model training and evaluating.
9. `README.md`: Project documentation.

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
The program will train on entire 5 ML models with full set, class 1 and class 2 data, print the performance and save the figures of output to `figure/` folder as following.

![image](https://github.com/ucaplii/screening-molecules-libraries-using-machine-learning-tools-ucaplii/assets/114681378/0f58bfd3-d3a7-458a-b2be-cc0eb689496e)

Each model has three sets of two figures for a total of six. For example, use XGB on class 2.

![XGBRegressor_class 2_pred_vs_act_figures](https://github.com/ucaplii/screening-molecules-libraries-using-machine-learning-tools-ucaplii/assets/114681378/c1856c65-1356-4b28-8238-2cd255d30a1f)

![XGBRegressor_class 2_residuals_figures](https://github.com/ucaplii/screening-molecules-libraries-using-machine-learning-tools-ucaplii/assets/114681378/ce40b0eb-5d52-4931-af54-64ea775d7765)


Parameters of model can be change in `models/model.py` by modifing the `__init__` parameter in each ML model class.

```
class RandomForest:
    def __init__(self,
                 n_estimators=500,
                 random_state=None):
        
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state)

    def get_model(self):
        return self.model


class XGBoost:
    def __init__(self,
                 n_estimators=1000,
                 learning_rate=0.01,
                 max_depth=8,
                 colsample_bytree=0.9,
                 reg_alpha=1,
                 subsample=0.6,
                 random_state=None):

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

class KernelRidgeRegression:
    def __init__(self, alpha=0.1, kernel='laplacian'):
        self.alpha = alpha
        self.kernel = kernel
        self.model = KernelRidge(alpha=alpha, kernel=kernel)
    
    def get_model(self):
        return self.model
    
class SupportVectorRegression:
    def __init__(self, kernel='rbf', C=1, epsilon=0.01):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    
    def get_model(self):
        return self.model
    
class MLP:
    def __init__(self, hidden_layers=(500,100,50), activation='relu', alpha=0.01, random_state=None):
        self.hidden_layer_sizes = hidden_layers
        self.activation = activation
        self.alpha = alpha
        self.random_state = random_state
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            alpha=alpha,
            random_state=random_state
        )  
        
    def get_model(self):
        return self.model
```

## Notes

1. There is no need to set any paths or parameters to run this programme. The parameters are predefined in the programme.
2. If parameters and paths need to be modified, make sure they are appropriate set by the above instructions.
3. Results may vary based on data quality, preprocessing, and model parameters.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

