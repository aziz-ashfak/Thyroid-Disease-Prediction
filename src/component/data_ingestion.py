import pandas as pd
import numpy as np
import plotly.express as px 
import matplotlib.pyplot as plt
import os 
import sys

from sklearn.model_selection  import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.exception import CustomException
from src.logger import logging
from src.component.data_transformation import DataTransformConfig
from src.component.data_transformation  import DataTransform
from src.component.model_trainner import ModelTrainer

class DataIngestionConfig:
    train_df_path = os.path.join('artifacts','train_df.csv')
    test_df_path = os.path.join('artifacts','test_df.csv')
    raw_df_path  = os.path.join('artifacts','raw_df.csv')
    logging.info('All the data path has been created!!')
class DataIngestion:
    
    def __init__(self):
       self.data_ingestion_config = DataIngestionConfig()
    
    def data_spliter(self):
        
        try:
            
            # read the main dataset 
            
            df = pd.read_csv('Dataset/Thyroid_Diff.csv')
            logging.info('Dataset has been read!!!')
            
            df.to_csv(self.data_ingestion_config.raw_df_path)
            logging.info('Raw dataset has been saved!!!')
            
            df.rename(columns={'T':'Tumor','N':'Nodal','M':'Metastasis'},inplace=True)
            logging.info('Rename some columns!!!')
            
            # split the dataset into train and test
            
            train_df,test_df = train_test_split(df,test_size=.20,random_state=42)
            logging.info("Split the dataset!!!")
            
            train_df.to_csv(self.data_ingestion_config.train_df_path)
            logging.info("Train dataset  has been created!!!")
            
            test_df.to_csv(self.data_ingestion_config.test_df_path)
            logging.info("Test dataset s has been created!!!")
            
            return (
                self.data_ingestion_config.train_df_path,
                self.data_ingestion_config.test_df_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
    
if __name__ == "__main__":
    
    obj= DataIngestion()
    train_path,test_path = obj.data_spliter()
    logging.info("Train and test path has been created!!!")
    
    data_preprocesses = DataTransform()
    train_arr,test_arr = data_preprocesses.data_preprocessing(train_path,test_path)
    
    model_train = ModelTrainer()
    print('The accuracy best accuracy is : ',model_train.model_trainer(train_arr,test_arr))