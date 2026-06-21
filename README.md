# FinSmart: Hibrit Yapay Zeka Finans Asistanı

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)

FinSmart, Bitcoin piyasasını hem nicel hem de nitel olarak analiz edebilen uçtan uca hibrit bir yapay zeka asistanıdır. Özel olarak eğitilmiş derin öğrenme modelleri sayesinde; bir yandan canlı borsa verileriyle teknik fiyat tahminleri üretirken, diğer yandan güncel kripto haberlerini okuyup anlamlandırarak piyasanın genel duygu durumunu (sentiment) anlık olarak ölçer. Bu iki farklı analitik güç, modern bir web arayüzünde birleşerek veri odaklı bir karar destek mekanizması sunar.

---

## Temel Özellikler

* **📈 Canlı Fiyat Tahmini (Sayısal Beyin - LSTM):** Sistem canlı olarak Yahoo Finance'e bağlanır, son 60 saatin (Close, High, Low, Volume) verisini çeker ve özel eğitilmiş LSTM modeli ile gelecek saatin fiyat yönünü tahmin eder.
* **📰 Canlı Haber Duygu Analizi (Sözel Beyin - NLP):** CoinTelegraph canlı RSS akışına bağlanarak en güncel 5 İngilizce kripto haberini çeker. Haber metinlerini NLP modeline sokarak piyasanın yönünü 3 sınıfta (*Pozitif 🚀, Nötr 😐, Negatif 📉*) analiz eder.
* **✍️ Manuel Test Modülü:** Sistemin hiç görmediği veya kullanıcının o an gireceği spesifik bir İngilizce haber başlığını/metnini test etmek için anlık analiz yeteneği.
* **🐳 Docker Konteynerizasyonu:** "Benim bilgisayarımda çalışıyordu" sorununu ortadan kaldıran, işletim sistemi bağımsız tam izole Docker altyapısı.
* **⚡ Mikroservis Mimarisi:** Arka planda model çıkarımlarını ve veri çekme işlemlerini yöneten yüksek performanslı FastAPI sunucusu.

---

## Kullanılan Teknolojiler

* **Yapay Zeka & Veri Bilimi:** `TensorFlow`, `Keras`, `Scikit-learn`, `Pandas`, `NumPy`, `NLTK`
* **Arka Plan (Backend):** `FastAPI`, `Uvicorn`, `Requests`
* **Ön Yüz (Frontend):** `Streamlit`
* **Veri Kaynakları:** `yfinance` (Canlı Fiyat), `CoinTelegraph RSS` (Canlı Haber)
* **DevOps & Dağıtım:** `Docker`, `Git`

---

## Kurulum ve Çalıştırma (Docker ile)

Proje, hiçbir Python kütüphanesi kurmanıza gerek kalmadan Docker üzerinden tek tıkla çalıştırılabilir. Bilgisayarınızda **Docker**'ın yüklü ve açık olduğundan emin olun.

**1. Projeyi Klonlayın**
```bash
git clone [https://github.com/KULLANICI_ADIN/finsmart-ai.git](https://github.com/KULLANICI_ADIN/finsmart-ai.git)
cd finsmart-ai
