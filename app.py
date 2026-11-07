
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib

# Load the trained model and preprocessing files
model = tf.keras.models.load_model("spam_classifier.keras")
tfidf = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

# Create FastAPI application
app = FastAPI(title="Spam Classifier API")

# Define input format
class MessageInput(BaseModel):
    message: str

# Home endpoint
@app.get("/")
def home():
    return {"message": "Spam Classifier API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: MessageInput):
    message_vector = tfidf.transform([data.message]).toarray()

    spam_probability = float(model.predict(message_vector, verbose=0)[0][0])
    ham_probability = 1.0 - spam_probability

    prediction = "spam" if spam_probability >= 0.5 else "ham"

    return {
        "message": data.message,
        "prediction": prediction,
        "probabilities": {
            "ham": ham_probability,
            "spam": spam_probability
        }
    }
