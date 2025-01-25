import sys
import os
import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBRFClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,accuracy_score
import pickle 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,find_best_model

class ModelTrainerconfig:
    mode_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    
    def __init__(self):
         self.model_train_path = ModelTrainerconfig()
    def model_trainer(self,train_arr,test_arr):
        X_train,y_train = train_arr[:,:-1],train_arr[:,-1]
        X_test,y_test  = test_arr[:,:-1],test_arr[:,-1]
        
        models = {
            'LogisticRegression':LogisticRegression(),
            'SGDClassifier':SGDClassifier(),
            'RandomForestClassifier':RandomForestClassifier(),
            'AdaBoostClassifier':AdaBoostClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'BaggingClassifier':BaggingClassifier(),
            'CatBoostClassifier':CatBoostClassifier()
        }
        results = find_best_model(X_train,y_train,X_test,y_test,models)
        best_result = max(sorted(results.values()))
        
        best_model_name = list(results.keys())[
            list(results.values()).index(best_result)
        ]
        print('The best model name is : ',best_model_name)
        best_model = models[best_model_name]
        
        save_object(
            path=self.model_train_path.mode_path,
            obj=best_model
        )
        
        best_accuracy  = accuracy_score(y_test,best_model.predict(X_test))
        return best_accuracy