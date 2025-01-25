from flask import Flask,request,render_template,jsonify
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))
from src.exception import CustomException
from src.logger import logging
from src.pipeline.prediction_pipeline import create_prediction

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    
    # Validate input
    if 'data' not in data:
        return jsonify({'error': 'Invalid input, missing "data" key'}), 400

    # Predict and return result
    output = create_prediction(data['data'])
    return jsonify({'Result' : output})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        data = {
                'Age': float(request.form.get('Age',0)),
                'Gender': request.form.get('Gender'),
                'Smoking': request.form.get('Smoking'),  
                'Hx Smoking': request.form.get('Hx Smoking'),  
                'Hx Radiothreapy': request.form.get('Hx Radiothreapy'),  
                'Thyroid Function': request.form.get('Thyroid Function'),  
                'Physical Examination': request.form.get('Physical Examination'),  
                'Adenopathy': request.form.get('Adenopathy'),  
                'Pathology': request.form.get('Pathology'),  
                'Focality': request.form.get('Focality'),  
                'Risk': request.form.get('Risk'),  
                'Tumor': request.form.get('Tumor'),
                'Nodal': request.form.get('Nodal'),  
                'Metastasis': request.form.get('Metastasis'),  
                'Stage': request.form.get('Stage'),  
                'Response': request.form.get('Response')  
                
            }
        # Validate input
        # Predict
        output = create_prediction(data)
        return render_template("home.html", prediction_text=f"Result : {output}")
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
