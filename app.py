import gradio as gr
import joblib
import pandas as pd

# Load model and features
model = joblib.load("final_model.pkl")
features = joblib.load("features_used.pkl")

# Accepts positional args instead of kwargs
def predict_customer_satisfaction(*args):
    input_df = pd.DataFrame([args], columns=features)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    label = "Satisfied" if prediction == 0 else "Unsatisfied"
    return f"Prediction: {label}\nConfidence: {round(probability * 100, 2)}%"

inputs = [gr.Number(label=feat) for feat in features]

demo = gr.Interface(
    fn=predict_customer_satisfaction,
    inputs=inputs,
    outputs="text",
    title="Flipkart Customer Satisfaction Classifier",
    description="Predicts customer satisfaction using structured, sentiment, and BERT features"
)

demo.launch()
