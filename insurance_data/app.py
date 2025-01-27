#  We are given data like age and bought insurance. 
# Apply Logistic Regression Model and predict whether a person takes 
# insurance or not based on his age.




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
# Load the data
data = pd.read_csv('D:/WebiSoftTech/LOGISTIC REGRESSION/insurance_data/insurance_data (1).csv')

# Prepare features
X = data[['age']] 
y = data['bought_insurance']  

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Load the image
img = Image.open('D:/WebiSoftTech/LOGISTIC REGRESSION/insurance_data/five.png').convert('L')

# Invert the image colors
img = Image.eval(img, lambda x: 255 - x)

# Convert the image to a 2D array
img_array_2d = np.array(img)

# Flatten the 2D array into a 1D array
img_array_1d = img_array_2d.flatten()

# Ex. age for prediction 
age_for_prediction = np.array([[30]]) 

# Predict using the model
prediction = model.predict(age_for_prediction)

# Output the prediction
print("Prediction (1 = bought insurance, 0 = did not buy insurance):", prediction[0])

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = np.array([[data['age']]])
    prediction = model.predict(age)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
    
    