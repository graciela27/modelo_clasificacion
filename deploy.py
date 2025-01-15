from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)

pipeline = joblib.load("modelo_entrenado.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Activado"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    
    features = data.get("text", None)
    if features is None:
        return jsonify({"error": "No se encontraron 'features' en el JSON"}), 400

    X = [features]

    probs = pipeline.predict_proba(X)[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    prediction = pipeline.predict(X)

    return jsonify({"prediction": prediction[0],
        "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)