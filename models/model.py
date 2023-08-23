from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

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
        # self.model = MLPRegressor(
        #     hidden_layers=self.hidden_layers,
        #     activation=self.activation,
        #     alpha=self.alpha,
        #     random_state=self.random_state
        # )
        return self.model