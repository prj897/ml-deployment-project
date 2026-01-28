from flask import Flask, render_template, request
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

with open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb") as f:
    model = pickle.load(f)

FEATURES = [
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epi_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None

    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]
            X = np.array(values).reshape(1, -1)
            pred = model.predict(X)[0]
            prediction_text = "Malignant (1)" if pred == 1 else "Benign (0)"
        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template("index.html", prediction_text=prediction_text, features=FEATURES)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
