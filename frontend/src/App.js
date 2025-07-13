import React, { useState } from "react";

function App() {
  const [symptoms, setSymptoms] = useState("");
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ symptoms }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Could not fetch prediction." });
    }
  };

  return (
    <div
      style={{
        maxWidth: "600px",
        margin: "50px auto",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <h2>Disease Predictor</h2>
      <p>Enter symptoms separated by commas (e.g. headache, nausea):</p>
      <input
        type="text"
        value={symptoms}
        onChange={(e) => setSymptoms(e.target.value)}
        style={{ width: "100%", padding: "10px", marginBottom: "10px" }}
      />
      <button onClick={handlePredict} style={{ padding: "10px 20px" }}>
        Predict
      </button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          {result.error ? (
            <p style={{ color: "red" }}>{result.error}</p>
          ) : (
            <>
              <h3>Predicted Disease: {result.disease}</h3>
              <p>
                <strong>Description:</strong> {result.description}
              </p>
              <p>
                <strong>Precautions:</strong>
              </p>
              <ul>
                {result.precautions?.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
              <p>
                <strong>Recommended Medications:</strong>{" "}
                {result.medications?.join(", ")}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
