# === retrain_model.py ===
# This file trains a better model using RandomForestClassifier and saves it.

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load training data
data = pd.read_csv("Datasets/Training.csv")

X = data.drop(columns=["prognosis"])
y = data["prognosis"]

# Encode target labels
diseases = sorted(y.unique())
disease_to_index = {disease: i for i, disease in enumerate(diseases)}
index_to_disease = {i: disease for disease, i in disease_to_index.items()}
y = y.map(disease_to_index)

# Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Optional: evaluate
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model and disease mapping
pickle.dump(model, open("models/rf_model.pkl", "wb"))
pickle.dump(index_to_disease, open("models/disease_mapping.pkl", "wb"))
