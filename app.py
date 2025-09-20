from flask import Flask, request, render_template, jsonify
import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.exception import CustomException
from src.logger import logging
from src.pipeline.prediction_pipeline import create_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     """API endpoint for JSON-based predictions"""
#     data = request.get_json()
#     if 'data' not in data:
#         return jsonify({'error': 'Invalid input, missing "data" key'}), 400
#     output = create_prediction(data['data'])
#     return jsonify({'Result': output})

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        # Collect form data (keys match your HTML form names!)
        data = {
            'age': float(request.form.get('age', 0)),
            'gender': request.form.get('gender'),
            'smoking': request.form.get('smoking'),
            'hx_smoking': request.form.get('hx_smoking'),
            'hx_radiotherapy': request.form.get('hx_radiotherapy'),
            'thyroid_function': request.form.get('thyroid_function'),
            'physical_examination': request.form.get('physical_examination'),
            'adenopathy': request.form.get('adenopathy'),
            'pathology': request.form.get('pathology'),
            'response': request.form.get('response')
        }

        # Run prediction
        output = create_prediction(data)

        # Return JSON for JavaScript to update DOM
        #return jsonify({"Prediction ": f" {output}"})
        return jsonify({"prediction": output})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"prediction": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)

