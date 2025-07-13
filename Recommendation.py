# === Step 1: Train and Save Model ===
# File: train_model.py

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load training data (must contain 132 symptom columns + prognosis column)
df = pd.read_csv('Datasets/Training.csv')

X = df.drop(columns=['prognosis'])
y = df['prognosis']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save the encoder for future decoding
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42)
model.fit(X_train, y_train)

# Accuracy check
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# === Step 2: Flask API ===
# File: app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

# Load trained model and encoder
rf_model = pickle.load(open("models/rf_model.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))

# Load CSV files for additional info
description = pd.read_csv("Datasets/description.csv")
precaution = pd.read_csv("Datasets/precautions_df.csv")
medications = pd.read_csv("Datasets/medications.csv")
diets = pd.read_csv("Datasets/diets.csv")
workout = pd.read_csv("Datasets/workout_df.csv")

# Define symptoms dict (ordered)
symptoms_dict = {symptom: idx for idx, symptom in enumerate(pd.read_csv("Datasets/Training.csv").columns[:-1])}

# Helper to extract extra info from CSVs
def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'].values)
    pre = precaution[precaution['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].values.tolist()
    return desc, pre, med, die, wrkout

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get("symptoms", [])

        if not isinstance(user_input, list):
            return jsonify({"error": "Invalid symptoms format. Must be a list."}), 400

        input_vector = np.zeros(len(symptoms_dict))
        for symptom in user_input:
            normalized = symptom.strip().lower().replace(" ", "_")
            if normalized in symptoms_dict:
                input_vector[symptoms_dict[normalized]] = 1

        prediction_code = rf_model.predict([input_vector])[0]
        disease = le.inverse_transform([prediction_code])[0]

        desc, pre, med, die, wrkout = helper(disease)

        return jsonify({
            "disease": disease,
            "description": desc,
            "precautions": pre[0] if pre else [],
            "medications": med,
            "diets": die,
            "workout": wrkout
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Prediction failed."}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"symptoms": sorted(symptoms_dict.keys())})

@app.route('/diseases', methods=['GET'])
def get_diseases():
    return jsonify({"diseases": list(le.classes_)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)