SentimentFlow-AI (Learning Project)

A simple machine-learning web app I built to learn how sentiment analysis, NLP, Flask, and ML deployment work.
This project takes any text and predicts whether the sentiment is Positive, Negative, or Neutral.
It also shows a probability graph and allows users to add new data to improve the model.

Why I Made This

I created this project for learning purposes.
I wanted to understand:
How machine learning models are trained
How NLP converts text into numbers
How Flask sends data between backend and frontend
How a model can be deployed in a real web page
How user data can update the dataset in real time
How to show results with charts and animations
This project helped me understand end-to-end ML development.

 Features

Real-time sentiment prediction
Probability graph using Chart.js
Clean and attractive UI
User contribution page
Auto-updates dataset and retrains model
Fully working Flask backend
Saved ML model using joblib

 Tech Stack

Python
Flask
Machine Learning (Naive Bayes / TF-IDF)
Pandas
Chart.js
HTML / CSS / JS

How It Works
User enters text
Model predicts sentiment
Probabilities are shown as a graph
Users can add new labeled text
Dataset grows and model retrains
This app learns and improves with new data.

 Installation
pip install -r requirements.txt
python app.py


Then open:
👉 http://127.0.0.1:5000/
