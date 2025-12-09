from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from trained_model import train_model
import os

os.environ["FLASK_SKIP_DOTENV"] = "1"

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    text = ""

    if request.method == "POST":
        text = request.form.get("user_text")

        if text and text.strip():
            transformed = vectorizer.transform([text])
            prediction = model.predict(transformed)[0]

            if prediction == "positive":
                sentiment = {"label": "Positive 😊", "color": "#4CAF50"}
            elif prediction == "negative":
                sentiment = {"label": "Negative 😞", "color": "#F44336"}
            else:
                sentiment = {"label": "Neutral 😐", "color": "#FFC107"}

    return render_template("index.html", sentiment=sentiment, text=text)


@app.route("/contribute", methods=["GET", "POST"])
def contribute():
    message = None

    if request.method == "POST":
        new_text = request.form.get("new_text")
        new_label = request.form.get("label")

        if not new_text or new_text.strip() == "":
            return render_template("contribute.html", message="Text cannot be empty!")

        if new_label not in ["positive", "negative", "neutral"]:
            return render_template("contribute.html", message="Invalid label selected!")

        df = pd.read_csv("dataset.csv")
        df.loc[len(df)] = [new_text, new_label]
        df.to_csv("dataset.csv", index=False)

        train_model()

        global model, vectorizer
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")

        message = "Your contribution is added and model updated successfully!"

    return render_template("contribute.html", message=message)


@app.route("/dataset_stats")
def dataset_stats():
    df = pd.read_csv("dataset.csv")
    total = len(df)

    label_counts = df["label"].value_counts(normalize=True).to_dict()
    for k in label_counts:
        label_counts[k] = round(label_counts[k], 4)

    return jsonify({"total": total, "stats": label_counts})

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
