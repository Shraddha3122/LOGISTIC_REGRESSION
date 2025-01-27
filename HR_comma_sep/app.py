# Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from flask import Flask, request, jsonify
import json

# Load the digits dataset
digits = load_digits()
X = digits.data  
y = digits.target  

# Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=10000) 
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Create Flask application
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()  
    input_array = np.array(input_data['image']).reshape(1, -1)  
    prediction = model.predict(input_array) 
    return jsonify({'prediction': int(prediction[0])})  

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)