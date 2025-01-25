import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
class DataTransformConfig:
    processor_obj_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransform:
    
    def __init__(self):
        self.data_tranform_config = DataTransformConfig()
        self.inputs_cols = [
                            'Age', 
                            'Gender',
                            'Smoking',
                            'Hx Smoking',
                            'Hx Radiothreapy',
                            'Thyroid Function',
                            'Physical Examination',
                            'Adenopathy',
                            'Pathology',
                            'Focality',
                            'Risk',
                            'Tumor',
                            'Nodal',
                            'Metastasis',
                            'Stage',
                            'Response'
                            ]
        self.target_col ='Recurred'
    
    def data_processor(self):
        
        try:
            numeric_features = ['Age']
            categorical_cols = [
                                'Gender',
                                'Smoking',
                                'Hx Smoking',
                                'Hx Radiothreapy',
                                'Thyroid Function',
                                'Physical Examination',
                                'Adenopathy',
                                'Pathology',
                                'Focality',
                                'Risk',
                                'Tumor',
                                'Nodal',
                                'Metastasis',
                                'Stage',
                                'Response'
                                ]
            numeric_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder(sparse_output=False,handle_unknown='ignore')),
                ('scaler',StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer(
                [
                ('numeric_pipeline',numeric_pipeline,numeric_features),
                ('categorical_pipeline',categorical_pipeline,categorical_cols)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def data_preprocessing(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test dataframe!!!')
            
            inputs_train_df = train_df.drop(columns=[self.target_col],axis=1)
            logging.info("Resize the train dataset!!!")
            target_train_df = train_df[[self.target_col]]
            logging.info('Target train df!!!')
            
            inputs_test_df = test_df.drop(columns=[self.target_col],axis=1)
            logging.info("Resize the test dataset!!!")
            target_test_df = test_df[[self.target_col]]
            logging.info('Target test df!!!')
            
            # encodded the target column
            # label = LabelEncoder()
            # target_train = label.fit_transform(target_train_df['Recurred'])
            # target_test = label.transform(target_test_df['Recurred'])
            # logging.info("Labeling targeted train and test column!!")
            
            
            processing_obj = self.data_processor()
            logging.info('Load data preprocessing object!!!')
            
            
            logging.info("save preprocessor!!!")
            train_preprosses_arr = processing_obj.fit_transform(inputs_train_df[self.inputs_cols])
            test_preprosses_arr = processing_obj.transform(inputs_test_df[self.inputs_cols])
            logging.info("Preprocesse train test dataset!!!")
            
            train_arr = np.c_[
                train_preprosses_arr,np.array(target_train_df)
                  ]
            
            test_arr = np.c_[
                test_preprosses_arr,np.array(target_test_df)
                 ]
            
            save_object(
                path=self.data_tranform_config.processor_obj_path,
                obj=processing_obj
            )
            
            return  (
                train_arr,
                test_arr
                )
                 
        except Exception as e:
            raise CustomException(e,sys)
