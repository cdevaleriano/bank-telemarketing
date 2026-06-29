from sklearn.linear_model import LogisticRegression
import mlflow
from imblearn.over_sampling import SMOTE

class ModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(max_iter = 100)
        print(f"Instantiated logreg model")

    def train(self, X, y):
        mlflow.autolog()

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y) 
        print(f"Resampled data using SMOTE")

        self.model.fit(X, y)
        print(f"Trained logreg model")