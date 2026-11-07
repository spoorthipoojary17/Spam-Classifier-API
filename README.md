
# Spam Classifier API

## Project Overview
This project uses Deep Learning and Natural Language Processing (NLP) to classify SMS messages as Spam or Ham.

It provides a REST API using FastAPI.

## Technologies Used
- Python 3.13
- TensorFlow
- Scikit-learn
- TF-IDF Vectorization
- FastAPI
- Docker
- Pytest

## Project Files
- app.py - FastAPI application
- spam_classifier.keras - Trained deep learning model
- tfidf_vectorizer.pkl - TF-IDF vectorizer
- label_encoder.pkl - Label encoder
- test_app.py - Automated tests
- requirements.txt - Python dependencies
- Dockerfile - Docker configuration

## API Endpoints

### Home
GET /

### Predict
POST /predict

Example request:

```json
{
  "message": "Congratulations! You won a free iPhone!"
}
