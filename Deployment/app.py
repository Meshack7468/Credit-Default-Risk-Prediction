import os
import pickle
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load full pipeline (preprocessing + model)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "random_forest_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

numeric_features = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "MonthsEmployed", "NumCreditLines", "InterestRate",
    "LoanTerm", "DTIRatio"
]

categorical_features = [
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
]

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None

    if request.method == "POST":
        data = {}

        # Collect numeric inputs
        for feature in numeric_features:
            data[feature] = float(request.form[feature])

        # Collect categorical inputs
        for feature in categorical_features:
            data[feature] = request.form[feature]

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)
