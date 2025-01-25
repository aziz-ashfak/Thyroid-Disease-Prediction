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
#from catboost import CatBoostClassifier
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,accuracy_score
import pickle 
from dataclasses import dataclass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from tuningMethod import evaluate_models

@dataclass
class HyperParameterConfig:
    model_file_path=os.path.join("TuningArtifacts","tuning_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=HyperParameterConfig()


    def model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
            'LogisticRegression':LogisticRegression(),
            'SGDClassifier':SGDClassifier(),
            'RandomForestClassifier':RandomForestClassifier(),
            'AdaBoostClassifier':AdaBoostClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'BaggingClassifier':BaggingClassifier(),
            #'CatBoostClassifier':CatBoostClassifier()
            }
            # Hyperparameter grid for tuning
            param_grids = {
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
                'solver': ['liblinear', 'lbfgs', 'saga'],  # Optimization algorithm
                #'penalty': ['l1', 'l2', 'elasticnet', None],  # Regularization types
                #'max_iter': [100, 500, 1000]  # Number of iterations
            },

            'SGDClassifier': {
                'loss': ['hinge', 'log_loss', 'modified_huber'],  # Loss functions
                'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization types
                #'alpha': [0.0001, 0.001, 0.01, 0.1],  # Regularization strength
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                #'max_iter': [1000, 5000, 10000]
            },

            'RandomForestClassifier': {
                'n_estimators': [100, 200, 500],  # Number of trees
                'max_depth': [None, 10, 20, 50],  # Maximum depth of trees
                #'min_samples_split': [2, 5, 10],  # Minimum samples to split
                #'min_samples_leaf': [1, 2, 5],  # Minimum samples per leaf
                'max_features': ['sqrt', 'log2', None],  # Number of features to consider
                #'bootstrap': [True, False]  # Bootstrapping for samples
            },

            'AdaBoostClassifier': {
                'n_estimators': [50, 100, 200],  # Number of estimators
                'learning_rate': [0.001, 0.01, 0.1, 1],  # Weight applied to classifiers
                'algorithm': ['SAMME', 'SAMME.R']  # Weight boosting algorithm
            },

            'GradientBoostingClassifier': {
                'n_estimators': [100, 200, 500],  # Number of boosting stages
                'learning_rate': [0.001, 0.01, 0.1, 0.2],  # Shrinks contribution of each tree
                'max_depth': [3, 5, 10],  # Maximum depth of each tree
                #'subsample': [0.5, 0.8, 1.0],  # Fraction of samples for fitting
                #'min_samples_split': [2, 5, 10]
            },

            'BaggingClassifier': {
                'n_estimators': [10, 50, 100],  # Number of base estimators
                #'max_samples': [0.5, 0.75, 1.0],  # Fraction of training samples
                'max_features': [0.5, 0.75, 1.0],  # Fraction of features
                #'bootstrap': [True, False],  # Whether samples are drawn with replacement
                'bootstrap_features': [True, False]
            },

            # 'CatBoostClassifier': {
            #     'iterations': [100, 500, 1000],  # Number of boosting rounds
            #     'depth': [4, 6, 8, 10],  # Depth of the trees
            #     'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
            #     'l2_leaf_reg': [3, 5, 7],  # L2 regularization
            #     #'border_count': [32, 64, 128]  # Number of splits
            # }
            }

            model_report:dict=evaluate_models(X_train=X_train,
                                              y_train=y_train,
                                              X_test=X_test,
                                              y_test=y_test,
                                             models=models,
                                             param=param_grids
                                             )
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                path=self.model_trainer_config.model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            



            
        except Exception as e:
            raise CustomException(e,sys)