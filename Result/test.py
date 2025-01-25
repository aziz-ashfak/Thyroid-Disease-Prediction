import pandas as pd
import  numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))
from src.exception import CustomException
from src.logger import logging
from src.utils import  load_object
from src.pipeline.prediction_pipeline import create_prediction
new_input ={
    'Age':30, 
    'Gender':'F',
    'Smoking':'No', 
    'Hx Smoking':'No', 
    'Hx Radiothreapy':'No',
    'Thyroid Function':'Euthyroid', 
    'Physical Examination':'Multinodular goiter', 
    'Adenopathy':'No', 
    'Pathology':'Micropapillary',
    'Focality':'Uni-Focal',
    'Risk':'Low', 
    'Tumor':'T1a',
    'Nodal':'N0', 
    'Metastasis':'M0', 
    'Stage':'I', 
    'Response':'Excellent'
}


create_prediction(new_input)

