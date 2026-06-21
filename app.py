import streamlit as st
import requests
import numpy as np

st.set_page_config(page_title="FinSmart AI", layout="wide")

st.title("FinSmart: Hibrit Yapay Zeka Asistanı")
st.markdown("""
Bu sistem, geçmiş fiyat hareketlerini (**Sayısal Beyin**) ve güncel haber akışını (**Sözel Beyin**) 
analiz ederek Bitcoin piyasası hakkında öngörüler sunar.
""")
st.divider()

st.sidebar.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=100)
st.sidebar.header("Hakkında")
st.sidebar.info("Güncel haberleri ve geçmiş verileri yapay zeka ile analiz ederek, Bitcoin'in yönünü sizin için tahmin eden akıllı finans asistanınız.")

col1, col2 = st.columns(2)

with col1:
    st.header("📰 NLP: Duygu Analizi")
    
    tab1, tab2 = st.tabs(["🌐 Canlı Haber Akışı", "✍️ Manuel Test"])
    
    with tab1:
        st.write("Sistem CoinTelegraph RSS kaynağına bağlanır ve son 5 haberi yapay zeka ile okur.")
        if st.button("🌐 Canlı Haberleri Çek ve Analiz Et", type="primary"):
            with st.spinner('İnternetten haberler toplanıyor...'):
                try:
                    response = requests.get("http://127.0.0.1:8000/predict/sentiment/live")
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("status") == "success":
                            overall = result["overall_sentiment"]
                            conf = result["overall_confidence"] * 100
                            
                            st.subheader("Genel Piyasa Duyarlılığı:")
                            if "POZİTİF" in overall:
                                st.success(f"**{overall}** (Ortalama Güven: %{conf:.2f})")
                                st.balloons()
                            elif "NEGATİF" in overall:
                                st.error(f"**{overall}** (Ortalama Güven: %{conf:.2f})")
                            else:
                                st.warning(f"**{overall}** (Ortalama Güven: %{conf:.2f})")
                                
                            st.divider()
                            st.write("**İncelenen Son 5 Haber:**")
                            for n in result["news"]:
                                with st.expander(f"{n['sentiment']} - {n['title']}"):
                                    st.write(f"Modelin Karar Güveni: %{n['confidence']*100:.2f}")
                        else:
                            st.error("API Hatası: " + result.get("message", ""))
                    else:
                        st.error("Sunucu yanıt vermedi.")
                except Exception as e:
                    st.error(f"Bağlantı hatası: {e}")

    with tab2:
        st.write("Modeli test etmek için manuel bir İngilizce haber metni veya başlığı girin.")
        news_text = st.text_area("İngilizce Haber Metni:", height=100, placeholder="Örn: Bitcoin crashes heavily as panic selling continues...")
        
        if st.button("🧠 Metni Analiz Et"):
            if news_text:
                try:
                    response = requests.post("http://127.0.0.1:8000/predict/sentiment", json={"text": news_text})
                    if response.status_code == 200:
                        result = response.json()
                        sentiment = result["sentiment"]
                        score = result["confidence_score"]
                        
                        st.subheader("Yapay Zeka Kararı:")
                        if "POZİTİF" in sentiment:
                            st.success(f"**{sentiment}** (Güven Skoru: {score})")
                        elif "NEGATİF" in sentiment:
                            st.error(f"**{sentiment}** (Güven Skoru: {score})")
                        else:
                            st.warning(f"**{sentiment}** (Güven Skoru: {score})")
                    else:
                        st.error("API Hatası!")
                except Exception as e:
                    st.error(f"Hata: {e}")
            else:
                st.warning("Lütfen bir metin girin.")

with col2:
    st.header("📈 LSTM: Canlı Fiyat Tahmini")
    st.write("Sistem canlı olarak Yahoo Finance'e bağlanır, son 60 saatin (Close, High, Low, Volume) verisini çeker ve gelecek saati tahmin eder.")
    
    if st.button("📡 Fiyat Yönünü Tahmin Et", type="primary"):
        with st.spinner('Borsaya bağlanılıyor ve son 60 saat analiz ediliyor...'):
            try:
                response = requests.get("http://127.0.0.1:8000/predict/price/live")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("status") == "success":
                        current = result["current_price"]
                        predicted = result["predicted_price"]
                        
                        diff = predicted - current
                        
                        metric_col1, metric_col2 = st.columns(2)
                        metric_col1.metric("Anlık Bitcoin Fiyatı", f"${current:,.2f}")
                        metric_col2.metric("Gelecek Saat Tahmini", f"${predicted:,.2f}", f"{diff:,.2f} USD", delta_color="normal")
                        
                        st.caption("✅ " + result["note"])
                    else:
                        st.error("API'de bir hata oluştu: " + result.get("message", "Bilinmeyen hata"))
                else:
                    st.error("Sunucudan geçerli bir yanıt alınamadı.")
            except Exception as e:
                st.error(f"Bağlantı hatası: Arkada FastAPI'nin (main.py) çalıştığından emin olun! Detay: {e}")