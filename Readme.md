List of disease and symptoms

1.  http://localhost:5000/symptoms GET

{
"symptoms": [
"acidity",
"anxiety",
"breathlessness",
"burning_micturition",
"chills",
"cold_hands_and_feets",
"continuous_sneezing",
"cough",
"dehydration",
"fatigue",
"headache",
"high_fever",
"indigestion",
"irregular_sugar_level",
"itching",
"joint_pain",
"lethargy",
"mood_swings",
"muscle_wasting",
"nausea",
"nodal_skin_eruptions",
"patches_in_throat",
"restlessness",
"shivering",
"skin_rash",
"stomach_pain",
"sunken_eyes",
"sweating",
"ulcers_on_tongue",
"vomiting",
"weight_gain",
"weight_loss"
]
}

2.  http://localhost:5000/diseases GET

{
"diseases": [
"AIDS",
"Allergy",
"Bronchial Asthma",
"Chronic cholestasis",
"Diabetes ",
"Drug Reaction",
"Fungal infection",
"GERD",
"Gastroenteritis",
"Hypertension ",
"Migraine",
"Peptic ulcer disease"
]
}

3.  Predict http://localhost:5000/predict Post
    Content-Type  
     application/json
    {
    "symptoms": ["cough", "fatigue", "high_fever", "headache"]
    }
