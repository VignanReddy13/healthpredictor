from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)

# ------------------ DATABASE SETUP ------------------

def init_db():
    conn = sqlite3.connect("health.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input1 REAL,
            input2 REAL,
            input3 REAL,
            result TEXT
        )
    """)

    conn.commit()
    conn.close()

# call once
init_db()

def save_data(inputs, result):
    conn = sqlite3.connect("health.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (input1, input2, input3, result)
        VALUES (?, ?, ?, ?)
    """, (inputs[0], inputs[1], inputs[2], result))

    conn.commit()
    conn.close()

# ------------------ MODEL ------------------

with open("health_model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------ ROUTES ------------------

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
    
    # Map prediction
    risk_levels = {0: "Low Risk", 1: "High Risk"}
    output = risk_levels[prediction]

    # ✅ SAVE TO DATABASE
    save_data(features, output)

    return render_template(
        "index.html", prediction_text=f"Your Health Risk Level: {output}"
    )

# 🔍 VIEW STORED DATA
@app.route("/data")
def view_data():
    conn = sqlite3.connect("health.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions")
    data = cursor.fetchall()

    conn.close()
    return str(data)

# ------------------ RUN ------------------

if __name__ == "__main__":
    app.run(debug=True)