import os
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
from decimal import Decimal

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
db_uri = os.getenv("DATABASE_URI")

client = Client(api_key, api_secret)

conn = None
try:
    print("Veritabanına bağlanılıyor")
    conn = psycopg2.connect(db_uri)
    cur = conn.cursor()
    print("Veritabanı bağlantısı başarılı")

    create_table_query = """
    CREATE TABLE IF NOT EXISTS btc_usdt_hourly (
        open_time TIMESTAMP PRIMARY KEY,
        open_price NUMERIC(20, 8),
        high_price NUMERIC(20, 8),
        low_price NUMERIC(20, 8),
        close_price NUMERIC(20, 8),
        volume NUMERIC(20, 8)
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    print("Saatlik veri tablosu başarıyla kontrol edildi/oluşturuldu.")

    print("Binance'ten geçmiş saatlik veriler çekiliyor...")

    klines = client.get_historical_klines(
        symbol='BTCUSDT',
        interval=Client.KLINE_INTERVAL_1HOUR,
        start_str="1 Jan, 2020"
    )
    print(f"{len(klines)} adet saatlik veri çekildi")

    data_to_insert = []
    for kline in klines:
        record = (
            datetime.fromtimestamp(kline[0] / 1000),
            Decimal(kline[1]),
            Decimal(kline[2]),
            Decimal(kline[3]),
            Decimal(kline[4]),
            Decimal(kline[5]),
        )
        data_to_insert.append(record)
    
    if data_to_insert:
        print(f"{len(data_to_insert)} adet kayıt veritabanına ekleniyor...")

        insert_query = """
        INSERT INTO btc_usdt_hourly (open_time, open_price, high_price, low_price, close_price, volume)
        VALUES %s
        ON CONFLICT (open_time) DO NOTHING;
        """
        psycopg2.extras.execute_values(cur, insert_query, data_to_insert)
        conn.commit()
        print(f"{cur.rowcount} adet yeni kayıt başarıyla eklendi")

except BinanceAPIException as e:
    print(f"Binance API Hatası: {e}")
except psycopg2.Error as e:
    print(f"Veritabanı Hatası: {e}")
finally:
    if conn:
        cur.close()
        conn.close()
        print("Veritabanı bağlantısı kapatıldı")