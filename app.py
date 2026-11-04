
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
app = Flask(__name__)
# LOAD MODEL
BASE_DIR = Path(__file__).resolve().parent
artifact = joblib.load(BASE_DIR / 'model_xgb_new.pkl')
model = artifact['model']
scaler = artifact['scaler']
le = artifact['label_encoder']
feature_names = artifact['feature_names']
# HOME PAGE
@app.route('/')
def home():
    return render_template(
        'home.html',
        features=feature_names
    )

# PREDICTION

@app.route('/predict', methods=['POST'])
def predict():
    try:

        input_data = {}
        for feature in feature_names:
            value = request.form.get(feature, '').strip()
            if not value:
                raise ValueError(f"Please enter {feature}.")
            input_data[feature] = float(value)

        # Create DataFrame in training feature order
        input_df = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # Apply fitted training scaler
        scaled_input = scaler.transform(input_df)

        # Predict encoded class
        prediction_encoded = model.predict(scaled_input)

        # Decode prediction
        prediction = le.inverse_transform(
            np.asarray(prediction_encoded, dtype=int)
        )[0]

        return render_template(
            'home.html',
            features=feature_names,
            prediction=str(prediction)
        )

    except ValueError as e:

        return render_template(
            'home.html',
            features=feature_names,
            error=str(e)
        )

    except Exception:

        app.logger.exception("Prediction error")

        return render_template(
            'home.html',
            features=feature_names,
            error="Something went wrong while generating the prediction."
        )


# RUN APPLICATION

if __name__ == '__main__':

    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000
    )
