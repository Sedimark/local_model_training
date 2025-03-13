import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

class XGBoostModel:
    def __init__(self, n_estimators, learning_rate, early_stopping_rounds, max_depth):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            max_depth=max_depth,
            n_jobs=-1
        )
        self.eval_results = None  # To store evaluation results

    def train(self, X_train, y_train, X_val, y_val):
        print('Starting XGBoost training...')
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=None)
        # self.eval_results = self.model.evals_result()

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, file_path):
        self.model.save_model(file_path)

    @staticmethod
    def load_model(file_path):
        model = xgb.XGBRegressor() 
        model.load_model(file_path)
        return model