import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import save_object,load_object
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import save_object, load_object
from src.logger import logging
from src.exception import CustomException

def create_prediction(data):
    try:
        # Load model and preprocessor
        model_path = os.path.join('artifacts', 'model.pkl')
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model = load_object(path=model_path)
        preprocessor = load_object(path=preprocessor_path)

        # Expected columns in the model
        expected_cols = [
            'Pathology', 'Physical Examination', 'Smoking', 'Age', 'Adenopathy', 
            'Nodal', 'Metastasis', 'Response', 'Tumor', 'Hx Radiothreapy', 
            'Focality', 'Gender', 'Thyroid Function', 'Risk', 'Hx Smoking', 'Stage'
        ]

        # Map lowercase input keys to expected columns
        key_mapping = {
            'age': 'Age',
            'gender': 'Gender',
            'smoking': 'Smoking',
            'hx_smoking': 'Hx Smoking',
            'hx_radiotherapy': 'Hx Radiothreapy',
            'thyroid_function': 'Thyroid Function',
            'physical_examination': 'Physical Examination',
            'adenopathy': 'Adenopathy',
            'pathology': 'Pathology',
            'response': 'Response',
            'tumor': 'Tumor',
            'nodal': 'Nodal',
            'metastasis': 'Metastasis',
            'stage': 'Stage',
            'risk': 'Risk',
            'focality': 'Focality'
        }

        # Create DataFrame with correct column names
        mapped_data = {}
        for k in expected_cols:
            # find input key that maps to this expected column
            input_key = [ik for ik, ek in key_mapping.items() if ek == k]
            if input_key and input_key[0] in data:
                mapped_data[k] = data[input_key[0]]
            else:
                # default values if missing
                if k in ['Age', 'Nodal', 'Metastasis', 'Tumor']:
                    mapped_data[k] = 0
                else:
                    mapped_data[k] = 'Unknown'

        df = pd.DataFrame([mapped_data])

        # Transform and predict
        X = preprocessor.transform(df)
        pred = model.predict(X)[0]

        if pred == 'Yes':
            return "The patient is at high risk for disease recurrence"
        else:
            return "The patient is at low risk for disease recurrence"

    except Exception as e:
        raise CustomException(e, sys)



    