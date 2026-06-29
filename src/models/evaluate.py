from sklearn.metrics import recall_score

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return recall_score(y_test, y_pred)