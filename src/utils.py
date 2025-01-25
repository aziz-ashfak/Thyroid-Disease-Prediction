import pandas as pd
import numpy as np 
import os
import sys
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,accuracy_score
import pickle 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.exception import CustomException
from src.logger import logging


def save_object(path,obj):
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def load_object(path):
    try:
        with open(path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

    
    
def find_best_model(X_train,y_train,X_test,y_test,models):
    try:
        con_matrix = []
        train_acc = []
        test_acc = []
        report_details = {}
        result = {}

        # Iterate through models
        for i in range(len(models)):
            
            # separate keys and values
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            # Fit the model
            model.fit(X_train, y_train)
            
            # Predict once to avoid redundant computations
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            
            # Compute accuracy
            train_accuracy = accuracy_score(y_train, train_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions)
            
            # Generate the classification report as a string
            report = classification_report(y_test, test_predictions)
            report_details[list(models.keys())[i]] = report
            # Compute the confusion matrix
            matrix = confusion_matrix(y_test, test_predictions)
            
            # Store the metrics
            train_acc.append(train_accuracy)
            test_acc.append(test_accuracy)
            con_matrix.append(matrix)
            result[list(models.keys())[i]] = test_accuracy
      
            
        data = {
            "Model":list(models.keys()),
            "Train_accuracy":train_acc,
            "Test_accuracy":test_acc,
        }
        df = pd.DataFrame(data)
        df.to_csv('Result/Details.csv',index=False,header=True)
        np.save('Result/matrix.npy',con_matrix)
        report_df = pd.DataFrame([report_details])
        report_df.to_csv('Result/report_details.csv',index=False,header=True)
        
        return result
    except Exception as e:
        raise CustomException(e,sys)