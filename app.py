from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

# === Load model and mappings ===
model = pickle.load(open("models/rf_model.pkl", "rb"))
disease_mapping = pickle.load(open("models/disease_mapping.pkl", "rb"))

# Load symptom index from training data
symptoms_list = pd.read_csv("Datasets/Training.csv").columns[:-1].tolist()
symptoms_dict = {symptom.lower().strip(): idx for idx, symptom in enumerate(symptoms_list)}

# Load description, precaution, medication, diet, workout
description = pd.read_csv("Datasets/description.csv")
precaution = pd.read_csv("Datasets/precautions_df.csv")
medications = pd.read_csv("Datasets/medications.csv")
diets = pd.read_csv("Datasets/diets.csv")
workout = pd.read_csv("Datasets/workout_df.csv")

# === Helper ===
def helper(disease):
    desc = description[description['Disease'] == disease]['Description'].values
    pre = precaution[precaution['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
    med = medications[medications['Disease'] == disease]['Medication'].values.tolist()
    diet = diets[diets['Disease'] == disease]['Diet'].values.tolist()
    wrk = workout[workout['disease'] == disease]['workout'].values.tolist()

    return (
        desc[0] if len(desc) > 0 else "",
        pre[0] if pre else [],
        med,
        diet,
        wrk
    )

# === App Setup ===
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get("symptoms", [])

        input_vector = np.zeros(len(symptoms_list))
        for symptom in user_input:
            normalized = symptom.strip().lower().replace(" ", "_")
            if normalized in symptoms_dict:
                input_vector[symptoms_dict[normalized]] = 1

        prediction = model.predict([input_vector])[0]
        disease = disease_mapping[prediction]

        desc, pre, med, diet, wrk = helper(disease)

        return jsonify({
            "disease": disease,
            "description": desc,
            "precautions": pre,
            "medications": med,
            "diets": diet,
            "workout": wrk
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Prediction failed."}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"symptoms": symptoms_list})

@app.route('/diseases', methods=['GET'])
def get_diseases():
    return jsonify({"diseases": list(disease_mapping.values())})

if __name__ == '__main__':
    app.run(debug=True)
