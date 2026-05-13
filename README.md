# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the program.
2.Data preprocessing
3.Cleanse data,handle missing values,encode categorical variables.
4.Model Training:Fit logistic regression model on preprocessed data.
5.Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall. 
6.Prediction: Predict placement status for new student data using trained model.
7.End the program.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SATHISH S
RegisterNumber:212225040390  
*/
*/
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("Placement_Data.csv")

# Copy dataset
data1 = data.copy()

# Drop unwanted columns
data1 = data1.drop(['sl_no', 'salary'], axis=1)

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Features and target
x = data1.iloc[:, :-1].values
y = data1["status"].values

# Initialize theta
theta = np.random.randn(x.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def loss(theta, x, y):
    h = sigmoid(x.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient Descent
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)

    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h - y) / m
        theta -= alpha * gradient

    return theta

# Train model
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

# Prediction function
def predict(theta, x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# Predictions
y_pred = predict(theta, x)

# Accuracy
accuracy = np.mean(y_pred.flatten() == y)

print("Accuracy:", accuracy)

print("\nPredicted:\n", y_pred)

print("\nActual:\n", y)

# New sample prediction
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])

y_prednew = predict(theta, xnew)

print("\nPredicted Result:", y_prednew)
```

## Output:
<img width="754" height="407" alt="590859680-ed748449-02c4-4c82-af50-78a9dc3b28cf" src="https://github.com/user-attachments/assets/256448ce-d176-4da1-ae04-e9e8a15685f2" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

