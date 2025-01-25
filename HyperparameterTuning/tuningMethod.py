import os 
import sys
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.exception import CustomException
from src.logger import logging
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        X_train = np.array(X_train)  # Convert to NumPy array if needed
        y_train = np.array(y_train).ravel()  # Ensure it's a 1D array
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)