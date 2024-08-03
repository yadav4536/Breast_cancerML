import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
model_path = '/Users/himanshuyadav/breast_project/breast_cancer_detector.pickle'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]

        # Define feature names
        features_name = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]

        # Create DataFrame for model input
        df = pd.DataFrame(features_value, columns=features_name)

        # Make prediction
        output = model.predict(df)[0]
        
        # Interpret prediction result
        if output == 0:
            res_val = "breast cancer"
        else:
            res_val = "no breast cancer"

        return render_template('index.html', prediction_text='Patient has {}'.format(res_val))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
