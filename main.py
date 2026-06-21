from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import requests as req
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

@app.get("/predict/price/live")
def predict_live_price():
    try:
        btc = yf.download("BTC-USD", period="7d", interval="1h")
        
        df_60 = btc[['Close', 'High', 'Low', 'Volume']].tail(60)
        
        input_data = df_60.values
        
        min_vals = input_data.min(axis=0)
        max_vals = input_data.max(axis=0)
        
        scaled_data = (input_data - min_vals) / (max_vals - min_vals + 1e-8)
        

        input_reshaped = scaled_data.reshape(1, 60, 4)
        
        prediction = price_model.predict(input_reshaped)
        predicted_scaled = float(prediction[0][0])
        
        predicted_real_price = predicted_scaled * (max_vals[0] - min_vals[0]) + min_vals[0]
        
        current_price = float(df_60['Close'].iloc[-1])

        return {
            "status": "success",
            "current_price": current_price,
            "predicted_price": predicted_real_price,
            "note": "Veriler anlık olarak Yahoo Finance üzerinden çekilmiştir."
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/predict/sentiment/live")
def predict_live_sentiment():
    try:
        import re
        
        url = "https://api.rss2json.com/v1/api.json?rss_url=https://cointelegraph.com/rss"
        
        raw_response = req.get(url, timeout=10)
        api_response = raw_response.json()

        if api_response.get("status") != "ok":
            return {"status": "error", "message": "Haber kaynağına ulaşılamadı. Daha sonra tekrar deneyin."}

        items = api_response.get("items", [])
        news_list = items[:5]
        
        if not news_list:
            return {"status": "error", "message": "Sitede şu an gösterilecek haber bulunamadı."}

        analyzed_news = []
        sentiment_scores = [] 
        confidence_sum = 0.0
        
        for article in news_list:
            title = article.get('title', '')
            body_html = article.get('description', '')
            
            body = re.sub(r'<[^>]+>', ' ', body_html)
            
            full_text = f"{title}. {body}"

            cleaned = clean_text(full_text)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100, padding='post', truncating='post')
            
            prediction = sentiment_model.predict(padded)
            class_idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            
            if class_idx == 0:
                label = "NEGATİF"
                sentiment_scores.append(-1)
            elif class_idx == 1:
                label = "NÖTR"
                sentiment_scores.append(0)
            else:
                label = "POZİTİF"
                sentiment_scores.append(1)
                
            confidence_sum += confidence
            
            analyzed_news.append({
                "title": title,
                "sentiment": label,
                "confidence": confidence
            })
            
        avg_polarity = sum(sentiment_scores) / len(sentiment_scores)
        avg_confidence = confidence_sum / len(sentiment_scores)
        
        if avg_polarity > 0.2:
            overall_sentiment = "POZİTİF"
        elif avg_polarity < -0.2:
            overall_sentiment = "NEGATİF"
        else:
            overall_sentiment = "NÖTR"
            
        return {
            "status": "success",
            "overall_sentiment": overall_sentiment,
            "overall_confidence": avg_confidence,
            "news": analyzed_news
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Sistem Hatası: {str(e)}"}