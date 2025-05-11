from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('rftm_model.pkl')

@app.route('/')
def home():
    return render_template('index_rftm.html')

@app.route('/predict', methods=['POST'])
def predict():
    recency = float(request.form['Recency'])
    frequency = float(request.form['Frequency'])
    monetary = float(request.form['Monetary'])
    
    # Create DataFrame with the input values
    input_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Change numeric prediction to meaningful labels
    clv_classes = {
        0: "Low Engagement Customer",
        1: "Occasional Customer",
        2: "VIP Regular Customer",
    }
    
    prediction_label = clv_classes.get(prediction, "Unknown")
    
    return render_template('result_rftm.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)