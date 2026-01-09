from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import re

app = FastAPI(title="FinSmart API", description="Bitcoin Fiyat ve Duygu Analizi Servisi", version="1.0")

try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Tokenizer yüklenirken hata: {e}")

try:
    sentiment_model = tf.keras.models.load_model('finsmart_sentiment_v1.keras')
except Exception as e:
    print(f"Duygu Modeli hatası: {e}")

try:
    price_model = tf.keras.models.load_model('finsmart_model_v1.keras')
except Exception as e:
    print(f"Fiyat Modeli hatası: {e}")


class SentimentRequest(BaseModel):
    text: str

class PriceRequest(BaseModel):
    prices: list[float]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    return text


@app.get("/")
def home():
    return {"message": "FinSmart AI Online"}

@app.post("/predict/sentiment")
def predict_sentiment(request: SentimentRequest):
    
    cleaned_text = clean_text(request.text)
    
    seq = tokenizer.texts_to_sequences([cleaned_text])
    
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    
    prediction = sentiment_model.predict(padded)
    
    
    sentiment_label = "Bilinmiyor"
    confidence = 0.0

    if prediction.shape[1] == 1:
        score = prediction[0][0]
        if score > 0.5:
            sentiment_label = "POZİTİF"
            confidence = float(score)
        else:
            sentiment_label = "NEGATİF"
            confidence = float(1 - score)
            
    else:
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        if class_idx == 0:
            sentiment_label = "NEGATİF"
        elif class_idx == 1:
            sentiment_label = "NÖTR"
        else:
            sentiment_label = "POZİTİF"

    return {
        "text": request.text,
        "sentiment": sentiment_label,
        "confidence_score": f"%{confidence*100:.2f}"
    }

@app.post("/predict/price")
def predict_price(request: PriceRequest):
    
    input_data = np.array(request.prices)
    
    if len(input_data) != 60:
        raise HTTPException(status_code=400, detail="Lütfen tam olarak son 60 günün fiyatını gönderin.")
    
    input_reshaped = input_data.reshape(1, 60, 1)
    
    prediction = price_model.predict(input_reshaped)
    
    
    return {
        "predicted_scaled_price": float(prediction[0][0]),
        "note": "Bu değer 0-1 arası normalize edilmiş fiyattır."
    }