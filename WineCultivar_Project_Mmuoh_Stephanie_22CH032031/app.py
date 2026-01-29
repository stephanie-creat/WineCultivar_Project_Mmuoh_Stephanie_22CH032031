from flask import Flask, render_template, request
import joblib
import numpy as np
import os  

app = Flask(__name__)

# Load model and scaler
model, scaler = joblib.load("model/wine_cultivar_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["alcalinity_of_ash"]),
            float(request.form["magnesium"]),
            float(request.form["flavanoids"]),
            float(request.form["color_intensity"])
        ]

        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        pred = model.predict(features)[0]
        prediction = f"Cultivar {pred + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
