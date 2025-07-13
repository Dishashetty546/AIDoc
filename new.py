import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the training dataset
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Desktop\AI doctor\Datasets\Training.csv")

# Encode target labels
le = LabelEncoder()
le.fit(df["prognosis"])

# Create disease_dict
disease_dict = {i: label for i, label in enumerate(le.classes_)}

# Print or save it
print(disease_dict)
