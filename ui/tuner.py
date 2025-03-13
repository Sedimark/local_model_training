import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_percentage_error
from model import XGBoostModel

class Tuner:
    def __init__(self, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, model_name):
        self.X_train_scaled = X_train_scaled
        self.y_train_scaled = y_train_scaled
        self.X_val_scaled = X_val_scaled
        self.y_val_scaled = y_val_scaled
        self.model_name = model_name
        

    def tune(self, n_trials):
        if self.model_name == 'XGBoostRegressor':
            best_params = self.tune_xgboost(n_trials)

        return best_params


    def tune_xgboost(self, n_trials):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 50)
            }
            model = XGBoostModel(**params)
            model.train(self.X_train_scaled,self.y_train_scaled,self.X_val_scaled,self.y_val_scaled)
            predictions = model.predict(self.X_val_scaled)
            
            return mean_absolute_percentage_error(self.y_val_scaled, predictions)
        
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        print(f"Best params for XGBoost: {study.best_params}")

        return study.best_params