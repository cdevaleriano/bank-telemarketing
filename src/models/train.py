from sklearn.linear_model import LogisticRegression
import mlflow
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

class ModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(max_iter = 100)
        print(f"Instantiated logreg model")

    def train(self, X_train, y_train, X_test, y_test):

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X_train, y_train) 
        print(f"Resampled data using SMOTE")

        
        with mlflow.start_run():
            self.model.fit(X, y)
            print(f"Trained logreg model")

            y_pred = self.model.predict(X_test)

            mlflow.sklearn.log_model(self.model, "base logistic regression model")
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
