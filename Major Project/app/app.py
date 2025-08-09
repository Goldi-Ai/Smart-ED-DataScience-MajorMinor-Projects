from flask import Flask, render_template, request
import os
import joblib
import pandas as pd

# ==== FIXED PATHS ====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# ==== LOAD MODEL & PREPROCESSOR ====
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ==== INIT FLASK ====
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Get form data
            form_data = request.form.to_dict()

            # Convert to DataFrame (1 row)
            input_df = pd.DataFrame([form_data])

            # Convert numeric fields from string to float
            for col in input_df.columns:
                try:
                    input_df[col] = input_df[col].astype(float)
                except:
                    pass  # keep as string if categorical

            # Preprocess
            X_transformed = preprocessor.transform(input_df)

            # Predict + Probability
            prediction = model.predict(X_transformed)[0]
            probability = model.predict_proba(X_transformed)[0][prediction] * 100  # percentage

            # Prepare output
            if prediction == 1:
                prediction_text = "Loan will DEFAULT"
                badge_color = "red"
            else:
                prediction_text = "Loan will NOT default"
                badge_color = "green"

            return render_template(
                "result.html",
                prediction=prediction_text,
                probability=round(probability, 2),
                badge_color=badge_color
            )

        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
