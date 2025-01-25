import pandas as pd
import numpy as np
import os
import sys
import json
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.component.data_transformation import DataTransformConfig
from src.component.data_transformation import DataTransform
from parameter_trainner import ModelTrainer
if __name__ == "__main__":
    
    train_path = os.path.join('artifacts','train_df.csv')
    test_path = os.path.join('artifacts','test_df.csv')

    data_preprocesses = DataTransform()
    train_arr,test_arr = data_preprocesses.data_preprocessing(train_path,test_path)
        
    model_train = ModelTrainer()
    print('The accuracy best accuracy is : ',model_train.model_trainer(train_arr,test_arr))
    # Assuming this is your original code
    accuracy = model_train.model_trainer(train_arr, test_arr)
    print('The best accuracy is:', accuracy)

    # Create a dictionary to store the accuracy result
    result_data = {
        "best_accuracy": accuracy
    }

    # Save the dictionary as a JSON file
    with open('TuningArtifacts/model_accuracy.json', 'w') as json_file:
        json.dump(result_data, json_file, indent=4)

    print("Accuracy saved to model_accuracy.json")