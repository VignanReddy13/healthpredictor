from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("health_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    
    # Predict risk
    prediction = model.predict(final_features)[0]
    
    # Map prediction to risk level
    risk_levels = {0: "Low Risk", 1: "High Risk"}
    output = risk_levels[prediction]
    
    return render_template(
        "index.html", prediction_text=f"Your Health Risk Level: {output}"
    )

if __name__ == "__main__":
    app.run(debug=True)