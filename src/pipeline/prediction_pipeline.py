import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import save_object,load_object
from src.logger import logging
from src.exception import CustomException

def create_prediction(data):
    
    try:
        model_path = os.path.join('artifacts','model.pkl')
        preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
        model = load_object(path = model_path)
        preprocessor = load_object(path = preprocessor_path)
        # create a  data frame 
        df = pd.DataFrame([data])
        X = preprocessor.transform(df)
        pred = model.predict(X)[0]
        if pred == 'Yes':
            return str("The patient has a high risk of recurrence")
        else:
            return str("The patient has a low risk of recurrence")
    except Exception as e:
        raise CustomException(e,sys)


    