from sklearn.model_selection import GridSearchCV
from src.utils.config import param_grid
import mlflow 

class HyperparameterTuner:
    def __init__(self, model):
        self.base_model = model
        self.best_model = None

        self.grid = GridSearchCV(
            estimator=self.base_model,
            param_grid=param_grid,
            cv=5,
            scoring="recall",
            n_jobs=-1
        )

    def fit(self, X, y):
        mlflow.autolog()
        self.grid.fit(X, y)
        self.best_model = self.grid.best_estimator_
        
        print("Selected best model")