from src.data.load import load_dataset
from src.data.preprocess import DataPreprocessor 
from src.utils.config import TRAIN_PATH, TEST_PATH
from src.features.build_features import build_features
from src.models.train import ModelTrainer
from src.models.tune import HyperparameterTuner

train = load_dataset(TRAIN_PATH)
test = load_dataset(TEST_PATH)

dp = DataPreprocessor(train)
X_train, y_train = dp.preprocess(train)
X_test, y_test = dp.preprocess(test)

X_train = build_features(X_train)
X_test = build_features(X_test)

mt = ModelTrainer()
mt.train(X_train, y_train, X_test, y_test)

tuner = HyperparameterTuner(model = mt.model)
tuner.fit(X_train, y_train)












