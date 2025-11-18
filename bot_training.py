import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")
import glob
import os
import logging
import argparse
import threading
import time
import itertools
import yfinance as yf
import numpy as np
import pandas as pd
import polars as pl
import pandas_ta as ta
import asyncio
import aiohttp
import uuid
import sys
from typing import Dict, List, Tuple
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import gymnasium as gym
import talib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pdfplumber
import multiprocessing
import shap
import pickle
import cvxpy as cp
from scipy.stats import norm
from datetime import timedelta

# Plotly kontrolü
try:
    import plotly.graph_objects as go
    PLOTLY_YUKLU = True
except ImportError:
    PLOTLY_YUKLU = False
    print("Plotly kütüphanesi yüklü değil. Dashboard yerine CSV raporu üretilecek.")

# Eksik modülleri buradan import edin (projenize göre dosya adlarını düzeltin)
from utils import linear_schedule, TrainingProgressCallback, MetricsCallback, simulate_snowball  # Diğer fonksiyonlar/sınıflar

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_log.txt", mode='a'),
        logging.StreamHandler()
    ]
)
BOT_ADI = "JTTWS Ticaret Botu"
BOT_KULLANICI_ADI = "jttws_ticaret_bot"
TELEGRAM_TOKEN = "8008545474:AAHansC5Xag1b9N96bMAGE0YLTfykXoOPyY"
TELEGRAM_KULLANICI_ID = 1590841427
CSV_FILES = {
    'EURUSD': [
        os.path.expanduser('~/Desktop/JTTWS/data/EURUSD2003-2024/EURUSD_Candlestick_15_M_BID_01.01.2021-01.01.2024.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/EURUSD2003-2024/EURUSD_Candlestick_15_M_BID_01.01.2012-01.01.2015.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/EURUSD2003-2024/EURUSD_Candlestick_15_M_BID_01.01.2009-01.01.2012.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/EURUSD2003-2024/EURUSD_Candlestick_15_M_BID_01.01.2024-31.12.2024.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/EURUSD2003-2024/EURUSD_Candlestick_15_M_BID_05.05.2003-01.01.2006.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/EURUSD2003-2024/EURUSD_Candlestick_15_M_BID_01.01.2018-01.01.2021.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/EURUSD2003-2024/EURUSD_Candlestick_15_M_BID_01.01.2006-01.01.2009.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/EURUSD2003-2024/EURUSD_Candlestick_15_M_BID_01.01.2015-01.01.2018.csv'),
    ],
    'GBPUSD': [
        os.path.expanduser('~/Desktop/JTTWS/data/GBPUSD2003-2024/GBPUSD_Candlestick_15_M_BID_01.01.2015-01.01.2018.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/GBPUSD2003-2024/GBPUSD_Candlestick_15_M_BID_01.01.2006-01.01.2009.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/GBPUSD2003-2024/GBPUSD_Candlestick_15_M_BID_01.01.2018-01.01.2021.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/GBPUSD2003-2024/GBPUSD_Candlestick_15_M_BID_01.01.2024-31.12.2024.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/GBPUSD2003-2024/GBPUSD_Candlestick_15_M_BID_05.05.2003-31.12.2005.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/GBPUSD2003-2024/GBPUSD_Candlestick_15_M_BID_01.01.2009-01.01.2012.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/GBPUSD2003-2024/GBPUSD_Candlestick_15_M_BID_01.01.2012-01.01.2015.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/GBPUSD2003-2024/GBPUSD_Candlestick_15_M_BID_01.01.2021-01.01.2024.csv'),
    ],
    'USDJPY': [
        os.path.expanduser('~/Desktop/JTTWS/data/USDJPY2003-2024/USDJPY_Candlestick_15_M_BID_05.05.2003-01.01.2006.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/USDJPY2003-2024/USDJPY_Candlestick_15_M_BID_01.01.2023-31.12.2024.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/USDJPY2003-2024/USDJPY_Candlestick_15_M_BID_01.01.2021-01.01.2023.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/USDJPY2003-2024/USDJPY_Candlestick_15_M_BID_01.01.2009-01.01.2012.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/USDJPY2003-2024/USDJPY_Candlestick_15_M_BID_01.01.2012-01.01.2015.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/USDJPY2003-2024/USDJPY_Candlestick_15_M_BID_01.01.2006-01.01.2009.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/USDJPY2003-2024/USDJPY_Candlestick_15_M_BID_01.01.2015-01.01.2018.csv'),
        os.path.expanduser('~/Desktop/JTTWS/data/USDJPY2003-2024/USDJPY_Candlestick_15_M_BID_01.01.2018-01.01.2021.csv'),
    ],
}
CSV_YOLLARI = CSV_FILES
SEMBOLLER = ['EURUSD', 'USDJPY', 'GBPUSD']
BASLANGIC_BAKIYESI = 25000.0
TEMEL_SPREAD = 0.0002
PRIME_ZAMANI = range(0, 24)
ISLEM_BASINA_MAKS_RISK = 0.02
ORIJINAL_MAKS_ACIK_POZISYON = 6
MIN_TICARET_GUNU = 1
MAX_LOT = 0.25
TARGET_PAYOUT = 2500.0
MAX_ACCOUNTS = 3
NEWS_API_KEY = None  # Kullanıcıdan sağlanmalı
KRIZ_DONEMLERI = {
    2008: slice(0, 10000),        # 2008 finansal krizi
    2020: slice(50000, 60000),    # COVID-19 krizi
    'normal': slice(20000, 30000)
}

GUNLUK_ISLEM_HAKLARI = {"USDJPY": 25, "EURUSD": 21, "GBPUSD": 18}
GUNLUK_PARITE_BUTCE = {"USDJPY": 250, "EURUSD": 250, "GBPUSD": 250}
GUN_BASLANGIC_SAATI = 8
GUN_BITIS_SAATI = 23

args = None
analizci = SentimentIntensityAnalyzer()
duygu_onbellek = {'skor': 0.0, 'zaman_damgasi': 0, 'metin': '', 'history': deque(maxlen=3)}
ozellik_onbellek = {}
volatilite_tarihi = deque(maxlen=100)
shap_degerleri = {}

global vix_cache
vix_cache = {}
def fetch_vix_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    cache_key = f"{start_date}_{end_date}"
    if cache_key in vix_cache:
        return vix_cache[cache_key]
    try:
        if start_date is None or pd.to_datetime(start_date, errors='coerce') is pd.NaT:
            start_date = '2008-01-01'
        if end_date is None or pd.to_datetime(end_date, errors='coerce') is pd.NaT:
            end_date = '2008-12-31'
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False)
        if vix.empty:
            logging.warning(f"VIX verisi boş: {start_date} - {end_date}, varsayılan değer kullanılıyor")
            return pd.DataFrame({'vix': [20.0]}, index=[pd.to_datetime('2008-01-01', utc=True)])
        vix = vix[['Close']].rename(columns={'Close': 'vix'})
        vix.index = pd.to_datetime(vix.index, utc=True)
        vix = vix.resample('15min').ffill()
        vix_cache[cache_key] = vix
        return vix
    except Exception as e:
        logging.error(f"VIX çekme hatası: {e}")
        return pd.DataFrame({'vix': [20.0]}, index=[pd.to_datetime('2008-01-01', utc=True)])
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )    
    
def linear_schedule(initial_value: float, final_value: float) -> callable:
    def schedule(progress: float) -> float:
        if progress >= 1.0:
            return final_value
        return initial_value + (final_value - initial_value) * progress
    return schedule

class TelegramBot:
    def __init__(self, token: str, kullanici_id: int):
        self.token = token
        self.kullanici_id = kullanici_id
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def send_message(self, text: str):
        async with aiohttp.ClientSession() as session:
            payload = {"chat_id": self.kullanici_id, "text": text, "parse_mode": "HTML"}
            async with session.post(self.api_url, data=payload) as response:
                return await response.json()

    def sync_send_message(self, text: str):
        future = asyncio.run_coroutine_threadsafe(self.send_message(text), self.loop)
        return future.result()

def csv_verisini_yukle(path: str) -> pd.DataFrame:
    try:
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, '*.csv'))
            files = [f for f in files if 'economic_calendar.csv' not in f]
            if not files:
                raise ValueError(f"{path} dizininde uygun CSV dosyası bulunamadı")
            file_starts = []
            for f in files:
                try:
                    with open(f, 'r') as fh:
                        header = fh.readline().strip().split(',')
                        first = fh.readline().strip().split(',')
                    time_col = next((col for col in ['Local time', 'Local Time', 'time', 'Date', 'date'] if col in header), None)
                    if not time_col:
                        raise ValueError(f"'time' veya 'Local time' sütunu bulunamadı: {f}")
                    idx = header.index(time_col)
                    dt = pd.to_datetime(first[idx], utc=True, errors='coerce')
                    if pd.isna(dt):
                        logging.warning(f"Geçersiz ilk zaman damgası: {f}, {first[idx]}")
                    file_starts.append((dt if not pd.isna(dt) else pd.Timestamp.max, f))
                except Exception as e:
                    logging.warning(f"Dosya okuma hatası: {f}, {e}")
                    file_starts.append((pd.Timestamp.max, f))
            sorted_files = [f for _, f in sorted(file_starts, key=lambda x: x[0])]
            dfs = []
            for f in sorted_files:
                df = pd.read_csv(f)
                column_map = {
                    'Local time': 'time', 'Local Time': 'time', 'time': 'time', 'Date': 'time', 'date': 'time',
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                }
                df = df.rename(columns=column_map)
                required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Eksik sütunlar: {f}, eksik: {missing_cols}")
                df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
                if df['time'].isna().any():
                    logging.warning(f"Geçersiz zaman damgaları bulundu: {f}, satır indeksleri: {df[df['time'].isna()].index.tolist()}")
                    df = df.dropna(subset=['time'])
                
                # Hafta sonu filtreleme
                df['is_weekend'] = df['time'].dt.weekday.isin([5, 6]).astype(bool)
                df = df[~df['is_weekend']].drop(columns=['is_weekend'])
                
                # Volume sıfır veya NaN olan satırları filtrele
                df = df[df['volume'] > 0].reset_index(drop=True)
                if df.empty:
                    logging.warning(f"{f} dosyasında geçerli veri bulunamadı (volume sıfır)")
                    continue
                dfs.append(df[required_cols])
            if not dfs:
                raise ValueError(f"{path} dizininde geçerli veri bulunamadı")
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values('time').reset_index(drop=True)
        else:
            df = pd.read_csv(path)
            column_map = {
                'Local time': 'time', 'Local Time': 'time', 'time': 'time', 'Date': 'time', 'date': 'time',
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }
            df = df.rename(columns=column_map)
            required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Eksik sütunlar: {path}, eksik: {missing_cols}")
            df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
            if df['time'].isna().any():
                logging.warning(f"Geçersiz zaman damgaları bulundu: {path}, satır indeksleri: {df[df['time'].isna()].index.tolist()}")
                df = df.dropna(subset=['time'])
            
            # Hafta sonu filtreleme
            df['is_weekend'] = df['time'].dt.weekday.isin([5, 6]).astype(bool)
            df = df[~df['is_weekend']].drop(columns=['is_weekend'])
            
            # Volume sıfır veya NaN olan satırları filtrele
            df = df[df['volume'] > 0].reset_index(drop=True)
            if df.empty:
                raise ValueError(f"{path} dosyasında geçerli veri bulunamadı (volume sıfır)")
            df = df[required_cols].sort_values('time').reset_index(drop=True)

        # Tatil kontrolü için ekonomik takvimle eşleştirme
        try:
            takvim_df = takvim_verisini_yukle()
            if not takvim_df.empty:
                takvim_df['date'] = pd.to_datetime(takvim_df['date'], utc=True, errors='coerce')
                if takvim_df['date'].isna().any():
                    logging.warning(f"Takvimde geçersiz tarihler bulundu: {takvim_df['date'].isna().sum()} satır")
                    takvim_df = takvim_df.dropna(subset=['date'])
                holidays = takvim_df[takvim_df['olay'].str.contains('Holiday|Closed', case=False, na=False)]['date'].dt.date
                df['is_holiday'] = df['time'].dt.date.isin(holidays).astype(bool)
                df = df[~df['is_holiday']].drop(columns=['is_holiday'])
            else:
                logging.warning("Ekonomik takvim boş, tatil filtrelemesi atlanıyor.")
        except Exception as e:
            logging.warning(f"Tatil filtreleme hatası: {e}")

        if df.empty:
            raise ValueError(f"CSV boş: {path}")
        if df['time'].isna().any():
            raise ValueError(f"Geçersiz zaman damgaları sonrası veri boş: {path}")
        logging.info(f"CSV yüklendi: {path}, sütunlar: {df.columns.tolist()}, satır sayısı: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"CSV verisi yükleme hatası: {path}, {e}")
        raise

def takvim_verisini_yukle(csv_yolu: str = os.path.expanduser('~/Desktop/JTTWS/data/economic_calendar.csv')) -> pd.DataFrame:
    try:
        takvim_df = pd.DataFrame()
        if not os.path.exists(csv_yolu):
            pdf_path = os.path.join(os.path.dirname(csv_yolu), 'Ekonomik Takvim - Investing.com.pdf')
            if os.path.exists(pdf_path):
                logging.info(f"PDF dosyası bulundu, işleniyor: {pdf_path}")
                with pdfplumber.open(pdf_path) as pdf:
                    all_tables = []
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        for table in tables:
                            if not table or len(table) < 2:
                                continue
                            header = table[0]
                            en_cols = ['Date', 'Time', 'Currency', 'Impact', 'Event', 'Previous', 'Actual', 'Forecast']
                            tr_cols = ['Tarih', 'Saat', 'Ülke', 'Önemi', 'Olay', 'Önceki', 'Gerçekleşen', 'Beklenti']
                            map_tr2en = {
                                'Tarih': 'Date', 'Saat': 'Time', 'Ülke': 'Currency', 'Önemi': 'Impact',
                                'Olay': 'Event', 'Önceki': 'Previous', 'Gerçekleşen': 'Actual', 'Beklenti': 'Forecast'
                            }
                            if header[:len(en_cols)] == en_cols or header[:len(tr_cols)] == tr_cols:
                                df = pd.DataFrame(table[1:], columns=header)
                                if header[:len(tr_cols)] == tr_cols:
                                    df = df.rename(columns=map_tr2en)
                                all_tables.append(df)
                    if all_tables:
                        takvim_df = pd.concat(all_tables, ignore_index=True)
                        takvim_df['datetime'] = pd.to_datetime(
                            takvim_df['Date'] + ' ' + takvim_df['Time'],
                            errors='coerce', utc=True
                        )
                        if takvim_df['datetime'].isna().any():
                            logging.warning(f"Geçersiz tarih formatları bulundu: {takvim_df['datetime'].isna().sum()} satır")
                            takvim_df = takvim_df.dropna(subset=['datetime'])
                        takvim_df = takvim_df.rename(
                            columns={'Currency': 'ülke', 'Impact': 'önem', 'Event': 'olay'}
                        )[['datetime', 'olay', 'önem', 'ülke']]
                        takvim_df.to_csv(csv_yolu, index=False)
                        logging.info(f"Takvim CSV'si oluşturuldu: {csv_yolu}, satır sayısı: {len(takvim_df)}")
                    else:
                        logging.warning("PDF'den tablo çıkarılamadı.")
            else:
                logging.warning(f"Takvim PDF dosyası bulunamadı: {pdf_path}")
        if os.path.exists(csv_yolu):
            logging.info(f"Takvim CSV'si bulundu, yükleniyor: {csv_yolu}")
            takvim_df = pd.read_csv(csv_yolu)
            if 'datetime' not in takvim_df.columns:
                takvim_df['datetime'] = pd.to_datetime(
                    takvim_df.get('Date', '') + ' ' + takvim_df.get('Time', ''),
                    errors='coerce', utc=True
                )
                if takvim_df['datetime'].isna().any():
                    logging.warning(f"CSV'de geçersiz tarih formatları bulundu: {takvim_df['datetime'].isna().sum()} satır")
                    takvim_df = takvim_df.dropna(subset=['datetime'])
            takvim_df = takvim_df.rename(columns={'datetime': 'date'})
            logging.info(f"Takvim CSV'si yüklendi: {takvim_df.shape[0]} satır, sütunlar: {takvim_df.columns.tolist()}")
        else:
            logging.warning(f"Takvim CSV dosyası bulunamadı: {csv_yolu}")
        return takvim_df
    except Exception as e:
        logging.warning(f"Takvim yükleme hatası: {e}")
        return pd.DataFrame()

async def fetch_news(keywords: List[str] = ['crisis', 'crash', 'emergency', 'recession', 'geopolitical', 'rate hike', 'EUR', 'USD', 'GBP']) -> bool:
    if not NEWS_API_KEY:
        logging.warning("NewsAPI anahtarı sağlanmadı.")
        return False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://newsapi.org/v2/everything?q={' '.join(keywords)}&apiKey={NEWS_API_KEY}") as response:
                data = await response.json()
                articles = data.get('articles', [])
                for article in articles:
                    title_desc = article.get('title', '') + " " + article.get('description', '')
                    sentiment = analizci.polarity_scores(title_desc)
                    if any(currency in title_desc for currency in ['EUR', 'USD', 'GBP']) and sentiment['compound'] < -0.7:
                        logging.info(f"Varlık bazlı kriz haberi: {article.get('title', '')}, sentiment: {sentiment['compound']:.2f}")
                        return True
                return False
    except Exception as e:
        logging.warning(f"Haber çekme hatası: {e}")
        return False

def duygu_skoru_al(metin: str | None = None, yuksek_etki: bool = False) -> float:
    current_time = time.time()
    if metin == duygu_onbellek['metin'] and current_time - duygu_onbellek['zaman_damgasi'] < 300:
        return float(duygu_onbellek['skor'])
    if not metin:
        metin = "Piyasa nötr"
    try:
        scores = analizci.polarity_scores(metin)
        compound_score = scores['compound']
        if yuksek_etki and any(kw in metin.lower() for kw in ['faiz', 'crisis', 'recession']):
            compound_score = -0.95
        duygu_onbellek['history'].append(compound_score)
        weighted_score = np.average(list(duygu_onbellek['history']), weights=[0.5, 0.3, 0.2])
        duygu_onbellek['skor'] = np.clip(weighted_score, -1.0, 1.0)
        duygu_onbellek['zaman_damgasi'] = current_time
        duygu_onbellek['metin'] = metin
        return float(duygu_onbellek['skor'])
    except Exception:
        return 0.0

def ozellikleri_sec(ozellikler: pd.DataFrame) -> pd.DataFrame:
    """
    Yüksek korelasyona sahip sütunlardan kurtul, ama 'atr', 'adx', 'senkou_span_a', 'senkou_span_b' koru.
    """
    if ozellikler.empty:
        logging.warning("Özellik DataFrame'i boş!")
        return ozellikler
    corr = ozellikler.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if col not in ['atr', 'adx', 'senkou_span_a', 'senkou_span_b'] and any(upper[col] > 0.8)]
    logging.info(f"Düşürülecek sütunlar: {to_drop}")
    ozellikler = ozellikler.drop(columns=to_drop, errors='ignore')
    logging.info(f"Özellik seçimi sonrası sütunlar: {ozellikler.columns.tolist()}")
    if 'senkou_span_a' not in ozellikler.columns or 'senkou_span_b' not in ozellikler.columns:
        logging.warning("Ichimoku sütunları eksik: senkou_span_a veya senkou_span_b bulunamadı!")
    return ozellikler

def rejim_ozellikleri_ekle(ozellikler: pd.DataFrame) -> pd.DataFrame:
    if ozellikler.empty or len(ozellikler) < 1:
        ozellikler['piyasa_rejimi'] = 0
        return ozellikler
    if all(col in ozellikler.columns for col in ['rsi', 'adx', 'atr', 'volatility']):
        regime_data = ozellikler[['rsi', 'adx', 'atr', 'volatility']].dropna()
        if len(regime_data) < 3 or regime_data.nunique().min() < 2:
            ozellikler['piyasa_rejimi'] = 0
            return ozellikler
        try:
            gmm = GaussianMixture(n_components=5, random_state=42).fit(regime_data)
            ozellikler['piyasa_rejimi'] = gmm.predict(regime_data)
        except ValueError:
            ozellikler['piyasa_rejimi'] = 0
    else:
        ozellikler['piyasa_rejimi'] = 0
    return ozellikler

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    freq = '1h' if timeframe == 'H1' else '4h' if timeframe == 'H4' else timeframe
    try:
        resampled_df = df.set_index('time').resample(freq).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).reset_index().interpolate()
        return resampled_df.dropna()
    except Exception:
        return pd.DataFrame()

def ozellik_muhandisligi(df: pd.DataFrame, takvim_df: pd.DataFrame, sembol: str) -> Tuple[pd.DataFrame, pd.Series]:
    import numpy as np
    import pandas as pd
    import pandas_ta as ta
    import logging

    # Veri kontrolü
    if df.empty or len(df) < 52:
        logging.error(f"{sembol} için veri boş veya yetersiz uzunlukta: {len(df)}")
        return pd.DataFrame(), pd.Series()

    # Özellik DataFrame'ini oluştur
    features = df[['open', 'high', 'low', 'close', 'volume']].copy().astype(np.float32)
    cache_key = f"{sembol}_{df['time'].iloc[0]}_{df['time'].iloc[-1]}"

    # Hafta sonu ve tatil bayrakları
    features['is_weekend'] = df['time'].dt.weekday.isin([5, 6]).astype(bool)  # bool olarak tut
    try:
        if not takvim_df.empty:
            takvim_df['date'] = pd.to_datetime(takvim_df['date'], utc=True)
            holidays = takvim_df[takvim_df['olay'].str.contains('Holiday|Closed', case=False, na=False)]['date'].dt.date
            features['is_holiday'] = df['time'].dt.date.isin(holidays).astype(bool)  # bool olarak tut
        else:
            features['is_holiday'] = pd.Series(False, index=features.index, dtype=bool)  # bool olarak varsayılan
    except Exception as e:
        logging.warning(f"Tatil bayrağı eklenirken hata ({sembol}): {e}")
        features['is_holiday'] = pd.Series(False, index=features.index, dtype=bool)  # bool olarak varsayılan

    # İşlem günü bayrağı
    features['is_trading_day'] = (~features['is_weekend'] & ~features['is_holiday']).astype(np.float32)
    # Alternatif olarak, daha güvenli bir hesaplama:
    # features['is_trading_day'] = ((features['is_weekend'] == False) & (features['is_holiday'] == False)).astype(np.float32)

    # Veri tiplerini logla
    logging.info(f"is_weekend dtype: {features['is_weekend'].dtype}")
    logging.info(f"is_holiday dtype: {features['is_holiday'].dtype}")
    logging.info(f"is_trading_day dtype: {features['is_trading_day'].dtype}")
    logging.info(f"is_weekend sample: {features['is_weekend'].head().to_list()}")
    logging.info(f"is_holiday sample: {features['is_holiday'].head().to_list()}")
    logging.info(f"is_trading_day sample: {features['is_trading_day'].head().to_list()}")
    # Tanımlı indikatörler
    secilen_indikatorler = [
        ('rsi', 14, ['rsi']),
        ('ema', 20, ['ema_20']),
        ('macd', 26, ['macd', 'macd_signal']),
        ('stoch', 14, ['stoch_k', 'stoch_d']),
        ('adx', 14, ['ADX_14']),
        ('obv', 1, ['obv']),
        ('ad', 1, ['ad']),
        ('bbands', 20, ['boll_ub', 'boll_mb', 'boll_lb']),
        ('ichimoku', 52, ['tenkan_9', 'kijun_26', 'senkou_span_a', 'senkou_span_b', 'chikou_span']),
        ('cci', 20, ['cci']),
        ('candles', 1, ['doji_star', 'engulfing'])
    ]

    # Veri dizilerini hazırla
    open_price = df['open'].to_numpy(dtype=np.float64)
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    close = df['close'].to_numpy(dtype=np.float64)
    volume = df['volume'].to_numpy(dtype=np.float64)

    def is_valid_array(arr: np.ndarray, min_length: int) -> bool:
        return (arr is not None and 
                isinstance(arr, np.ndarray) and 
                len(arr) >= min_length and 
                not np.any(np.isnan(arr)))

    def interpolate_feature(arr: np.ndarray, length: int) -> np.ndarray:
        if not is_valid_array(arr, length):
            valid_indices = np.where(~np.isnan(arr) & (arr > 0))[0]
            if len(valid_indices) < 2:
                return np.zeros(len(arr), dtype=np.float32)
            return np.interp(np.arange(len(arr)), valid_indices, arr[valid_indices])
        return arr

    # Indikatör hesaplamaları (işlem günlerinde)
    for indicator, length, columns in secilen_indikatorler:
        try:
            # İşlem günü olmayanlarda indikatörleri sıfırla
            if features['is_trading_day'].iloc[-1] == 0:
                for col in columns:
                    features[col] = np.zeros(len(features), dtype=np.float32)
                continue

            interp_data = [interpolate_feature(close, length)]
            if indicator in ['stoch', 'adx', 'bbands', 'ichimoku', 'ad', 'cci']:
                interp_data.extend([interpolate_feature(high, length), interpolate_feature(low, length)])
            if indicator in ['obv', 'ad']:
                volume_data = interpolate_feature(volume, length)
                # Hacim sıfır veya negatifse küçük bir pozitif değer ata
                volume_data[volume_data <= 0] = 1e-6
                interp_data.append(volume_data)
            if indicator == 'candles':
                interp_data = [interpolate_feature(open_price, length), 
                             interpolate_feature(high, length),
                             interpolate_feature(low, length), 
                             interpolate_feature(close, length)]

            # Veri kontrolü
            if not all(is_valid_array(data, length) for data in interp_data):
                logging.warning(f"{sembol} için {indicator} indikatörü: Geçersiz veri, sıfır atanıyor")
                for col in columns:
                    features[col] = np.zeros(len(features), dtype=np.float32)
                continue

            if indicator == 'rsi':
                result = ta.rsi(interp_data[0], length=length)
                features['rsi'] = pd.Series(result, index=features.index).ffill().fillna(50.0).astype(np.float32)
            elif indicator == 'ema':
                result = ta.ema(interp_data[0], length=length)
                features['ema_20'] = pd.Series(result, index=features.index).ffill().fillna(0.0).astype(np.float32)
            elif indicator == 'macd':
                result = ta.macd(interp_data[0], fast=12, slow=26, signal=9)
                features['macd'] = result['MACD_12_26_9'].reindex(features.index).ffill().fillna(0.0).astype(np.float32)
                features['macd_signal'] = result['MACDs_12_26_9'].reindex(features.index).ffill().fillna(0.0).astype(np.float32)
            elif indicator == 'stoch':
                result = ta.stoch(interp_data[0], interp_data[1], interp_data[2], k=14, d=3)
                features['stoch_k'] = result['STOCHk_14_3_3'].reindex(features.index).ffill().fillna(50.0).astype(np.float32)
                features['stoch_d'] = result['STOCHd_14_3_3'].reindex(features.index).ffill().fillna(50.0).astype(np.float32)
            elif indicator == 'adx':
                result = ta.adx(interp_data[0], interp_data[1], interp_data[2], length=length)
                features['ADX_14'] = result['ADX_14'].reindex(features.index).ffill().fillna(25.0).astype(np.float32)
            elif indicator == 'obv':
                obv_series = ta.obv(close=interp_data[0], volume=interp_data[1])
                features['obv'] = pd.Series(obv_series, index=features.index).ffill().fillna(0.0).astype(np.float32)
            elif indicator == 'ad':
                ad_series = ta.ad(high=interp_data[0], low=interp_data[1], close=interp_data[2], volume=interp_data[3])
                features['ad'] = pd.Series(ad_series, index=features.index).ffill().fillna(0.0).astype(np.float32)
            elif indicator == 'bbands':
                result = ta.bbands(interp_data[0], length=length, std=2)
                features['boll_ub'] = result['BBU_20_2.0'].reindex(features.index).ffill().fillna(0.0).astype(np.float32)
                features['boll_mb'] = result['BBM_20_2.0'].reindex(features.index).ffill().fillna(0.0).astype(np.float32)
                features['boll_lb'] = result['BBL_20_2.0'].reindex(features.index).ffill().fillna(0.0).astype(np.float32)
            elif indicator == 'ichimoku':
                ich = ta.ichimoku(high=interp_data[0], low=interp_data[1], close=interp_data[2], tenkan=9, kijun=26, senkou=52)[0]
                features['tenkan_9'] = ich['ITS_9'].reindex(features.index).bfill().fillna(0.0).astype(np.float32)
                features['kijun_26'] = ich['IKS_26'].reindex(features.index).bfill().fillna(0.0).astype(np.float32)
                features['senkou_span_a'] = ich['ISA_9'].reindex(features.index).bfill().fillna(0.0).astype(np.float32)
                features['senkou_span_b'] = ich['ISB_26'].reindex(features.index).bfill().fillna(0.0).astype(np.float32)
                features['chikou_span'] = ich['ICS_26'].reindex(features.index).ffill().fillna(0.0).astype(np.float32)
            elif indicator == 'cci':
                result = ta.cci(interp_data[0], interp_data[1], interp_data[2], length=length)
                features['cci'] = pd.Series(result, index=features.index).ffill().fillna(0.0).astype(np.float32)
            elif indicator == 'candles':
                open_p, high_p, low_p, close_p = interp_data
                features['doji_star'] = ta.cdl_doji_star(open_p, high_p, low_p, close_p).astype(np.float32)
                features['engulfing'] = ta.cdl_engulfing(open_p, high_p, low_p, close_p).astype(np.float32)

        except Exception as e:
            logging.warning(f"{sembol} için {indicator} hesaplama hatası: {e}")
            for col in columns:
                features[col] = np.zeros(len(features), dtype=np.float32)

    # ATR (tekrar kontrol için)
    try:
        features['atr'] = ta.atr(high=high, low=low, close=close, length=14)
        features['atr'] = features['atr'].ffill().fillna(1.0).replace(0, 1.0).astype(np.float32)
    except Exception as e:
        logging.warning(f"ATR hesaplama hatası ({sembol}): {e}")
        features['atr'] = np.ones(len(features), dtype=np.float32)

    # Pivot ve Fibonacci seviyeleri
    try:
        pivot_high = df['high'].rolling(window=20).max().interpolate()
        pivot_low = df['low'].rolling(window=20).min().interpolate()
        for level in [0.236, 0.382, 0.618]:
            fib = pivot_low + (pivot_high - pivot_low) * level
            col_name = f'fib_{int(level*100)}'
            features[col_name] = fib.fillna(0.0).astype(np.float32)
    except Exception as e:
        logging.warning(f"Fibonacci hesaplama hatası ({sembol}): {e}")
        for level in [0.236, 0.382, 0.618]:
            features[f'fib_{int(level*100)}'] = np.zeros(len(features), dtype=np.float32)

    # Çoklu zaman dilimi özellikleri
    zaman_dilimleri = ['H1', 'H4']
    for timeframe in zaman_dilimleri:
        try:
            resampled_df = resample_data(df, timeframe)
            if resampled_df.empty:
                logging.warning(f"{sembol} için {timeframe} zaman diliminde veri boş")
                features[f'rsi_{timeframe}'] = np.zeros(len(features), dtype=np.float32)
                features[f'adx_{timeframe}'] = np.zeros(len(features), dtype=np.float32)
                continue
            if len(resampled_df) >= 14:
                rsi = ta.rsi(resampled_df['close'].interpolate(), length=14)
                adx = ta.adx(resampled_df['high'].interpolate(), resampled_df['low'].interpolate(),
                            resampled_df['close'].interpolate(), length=14)
                features[f'rsi_{timeframe}'] = rsi.reindex(features.index, method='ffill').fillna(50.0).astype(np.float32)
                features[f'adx_{timeframe}'] = adx['ADX_14'].reindex(features.index, method='ffill').fillna(25.0).astype(np.float32)
            else:
                features[f'rsi_{timeframe}'] = np.zeros(len(features), dtype=np.float32)
                features[f'adx_{timeframe}'] = np.zeros(len(features), dtype=np.float32)
        except Exception as e:
            logging.warning(f"{sembol} için {timeframe} zaman dilimi hatası: {e}")
            features[f'rsi_{timeframe}'] = np.zeros(len(features), dtype=np.float32)
            features[f'adx_{timeframe}'] = np.zeros(len(features), dtype=np.float32)

    # VIX verisi ekleme
    try:
        vix_data = fetch_vix_data(df['time'].iloc[0], df['time'].iloc[-1])
        vix_data = vix_data.reindex(features.index, method='ffill').fillna(20.0)
        features['vix'] = vix_data['vix'].astype(np.float32)
    except Exception as e:
        logging.warning(f"VIX verisi eklenirken hata ({sembol}): {e}")
        features['vix'] = np.full(len(features), 20.0, dtype=np.float32)

    # Duygu skoru ekleme
    try:
        if not takvim_df.empty and 'olay' in takvim_df.columns:
            features['duygu'] = takvim_df['olay'].apply(lambda x: duygu_skoru_al(x, False)).reindex(features.index, method='ffill').fillna(0.0).astype(np.float32)
        else:
            features['duygu'] = np.zeros(len(features), dtype=np.float32)
    except Exception as e:
        logging.warning(f"Duygu skoru eklenirken hata ({sembol}): {e}")
        features['duygu'] = np.zeros(len(features), dtype=np.float32)

    # Fiyat değişim yüzdesi
    try:
        features['price_change_pct'] = df['close'].pct_change().fillna(0.0).astype(np.float32)
    except Exception as e:
        logging.warning(f"Fiyat değişim yüzdesi hatası ({sembol}): {e}")
        features['price_change_pct'] = np.zeros(len(features), dtype=np.float32)

    # Normalizasyon için hedef seri
    try:
        target = df['close'].pct_change().shift(-1).fillna(0.0).astype(np.float32)
    except Exception as e:
        logging.warning(f"Hedef seri hesaplama hatası ({sembol}): {e}")
        target = pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)

    # Özellik seçimi ve rejim ekleme
    features = ozellikleri_sec(features)
    features = rejim_ozellikleri_ekle(features)

    # İşlem günü olmayanlarda indikatörleri sıfırlama
    for col in features.columns:
        if col not in ['is_trading_day', 'is_holiday', 'is_weekend']:
            features.loc[features['is_trading_day'] == 0, col] = 0.0

    # Temizlik
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    ozellik_onbellek[cache_key] = (features, target)

    logging.info(f"Özellik mühendisliği tamamlandı: {sembol}, şekil: {features.shape}, sütunlar: {list(features.columns)}")
    return features, target

    def is_valid_array(arr: np.ndarray, min_length: int) -> bool:
        return (arr is not None and 
                isinstance(arr, np.ndarray) and 
                len(arr) >= min_length and 
                not np.any(np.isnan(arr)) and 
                not np.any(arr <= 0))  # Sıfır kontrolü eklendi

    def interpolate_feature(arr: np.ndarray, length: int) -> np.ndarray:
        if not is_valid_array(arr, length):
            valid_indices = np.where(~np.isnan(arr))[0]
            if len(valid_indices) < 2:
                return np.zeros(length, dtype=np.float32)
            return np.interp(np.arange(len(arr)), valid_indices, arr[valid_indices])
        return arr

    for indicator, length, columns in indikatorler:
        try:
            interp_data = [interpolate_feature(close, length)]
            if indicator in ['stoch', 'adx', 'bbands', 'ichimoku', 'ad', 'cci']:
                interp_data.extend([interpolate_feature(high, length), interpolate_feature(low, length)])
            if indicator in ['obv', 'ad']:
                interp_data.append(interpolate_feature(volume, length))
            if indicator == 'candles':
                interp_data = [interpolate_feature(open_price, length), interpolate_feature(high, length),
                               interpolate_feature(low, length), interpolate_feature(close, length)]

            # Veri kontrolü
            if not all(is_valid_array(data, length) for data in interp_data):
                logging.warning(f"{sembol} için {indicator} indikatörü: Geçersiz veri, sıfır atanıyor")
                for col in columns:
                    features[col] = np.zeros(len(features), dtype=np.float32)
                continue

            if indicator == 'rsi':
                result = ta.rsi(interp_data[0], length=length)
            elif indicator == 'ema':
                result = ta.ema(interp_data[0], length=length)
            elif indicator == 'macd':
                result = ta.macd(interp_data[0], fast=12, slow=26, signal=9)
            elif indicator == 'stoch':
                result = ta.stoch(interp_data[0], interp_data[1], interp_data[2], k=14, d=3)
            elif indicator == 'adx':
                result = ta.adx(interp_data[0], interp_data[1], interp_data[2], length=length)
                if result is None or 'ADX_14' not in result:
                    features['ADX_14'] = np.zeros(len(features), dtype=np.float32)
                    continue
                adx_series = result['ADX_14'].reindex(features.index, method='ffill').fillna(0.0)
                features['ADX_14'] = adx_series.values
                continue
            elif indicator == 'obv':
                if not all(isinstance(x, np.ndarray) for x in interp_data[:2]):
                    raise ValueError(f"OBV için giriş verileri numpy.ndarray olmalı: {sembol}")
                close, volume = [x.astype(float) for x in interp_data[:2]]
                if len(close) < 1 or len(volume) < 1:
                    raise ValueError(f"OBV için veri uzunluğu yetersiz: {sembol}")
                if np.any(np.isnan(close)) or np.any(np.isnan(volume)):
                    raise ValueError(f"OBV için NaN veri bulundu: {sembol}")
                obv_series = ta.obv(close=close, volume=volume)
                if obv_series is None or np.all(np.isnan(obv_series)):
                    raise ValueError(f"OBV hesaplaması boş veya NaN: {sembol}")
                obv_series = pd.Series(obv_series, index=features.index).ffill().fillna(0.0)
                features['obv'] = obv_series.values.astype(np.float32)
                continue
            elif indicator == 'ad':
                if not all(isinstance(x, np.ndarray) for x in interp_data[:4]):
                    raise ValueError(f"AD için giriş verileri numpy.ndarray olmalı: {sembol}")
                high, low, close, volume = [x.astype(float) for x in interp_data[:4]]
                if len(high) < 1 or len(low) < 1 or len(close) < 1 or len(volume) < 1:
                    raise ValueError(f"AD için veri uzunluğu yetersiz: {sembol}")
                if np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)) or np.any(np.isnan(volume)):
                    raise ValueError(f"AD için NaN veri bulundu: {sembol}")
                ad_series = ta.ad(high=high, low=low, close=close, volume=volume)
                if ad_series is None or np.all(np.isnan(ad_series)):
                    raise ValueError(f"AD hesaplaması boş veya NaN: {sembol}")
                ad_series = pd.Series(ad_series, index=features.index).ffill().fillna(0.0)
                features['ad'] = ad_series.values.astype(np.float32)
                continue
            elif indicator == 'bbands':
                result = ta.bbands(interp_data[0], length=length, std=2)
            elif indicator == 'ichimoku':
                if not all(isinstance(x, np.ndarray) for x in interp_data[:3]):
                    raise ValueError(f"Ichimoku için giriş verileri numpy.ndarray olmalı: {sembol}")
                high, low, close = [x.astype(float) for x in interp_data[:3]]
                if len(high) < 52 or len(low) < 52 or len(close) < 52:
                    raise ValueError(f"Ichimoku için veri uzunluğu yetersiz: {len(high)}, sembol: {sembol}")
                if np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)):
                    raise ValueError(f"Ichimoku için NaN veri bulundu: {sembol}")
                ich = ta.ichimoku(high=high, low=low, close=close, tenkan=9, kijun=26, senkou=52)[0]
                if ich is None or ich.empty:
                    raise ValueError(f"Ichimoku hesaplaması boş veri döndü: {sembol}")
                features['tenkan_9'] = ich['ITS_9'].reindex(features.index).bfill().values
                features['kijun_26'] = ich['IKS_26'].reindex(features.index).bfill().values
                features['senkou_span_a'] = ich['ISA_9'].reindex(features.index).bfill().values
                features['senkou_span_b'] = ich['ISB_26'].reindex(features.index).bfill().values
                features['chikou_span'] = ich['ICS_26'].reindex(features.index).ffill().values
                continue
            elif indicator == 'cci':
                result = ta.cci(interp_data[0], interp_data[1], interp_data[2], length=length)
            elif indicator == 'candles':
                open_p, high_p, low_p, close_p = [x.astype(np.float64) for x in interp_data]
                result = [
                    talib.CDLDOJISTAR(open_p, high_p, low_p, close_p),
                    talib.CDLENGULFING(open_p, high_p, low_p, close_p)
                ]

            for i, col in enumerate(columns):
                if isinstance(result, dict):
                    data = result.get(col, np.nan)
                elif isinstance(result, list):
                    data = result[i] if i < len(result) else np.nan
                else:
                    data = result
                if pd.api.types.is_scalar(data) or isinstance(data, (int, float, bool)):
                    features[col] = np.full(len(features), float(data) if not pd.isna(data) else 0.0, dtype=np.float32)
                elif isinstance(data, np.ndarray):
                    if np.all(pd.isna(data)) or (np.nanmean(data) <= 0 and data.size > 0):
                        features[col] = np.zeros(len(features), dtype=np.float32)
                    else:
                        features[col] = data
                else:
                    features[col] = np.zeros(len(features), dtype=np.float32)
        except Exception as e:
            logging.warning(f"Indikatör hesaplama hatası ({indicator}, sembol: {sembol}): {e}")
            for col in columns:
                features[col] = np.zeros(len(features), dtype=np.float32)

    # Pivot ve Fibonacci seviyeleri
    try:
        pivot_high = df['high'].rolling(window=20).max().interpolate()
        pivot_low = df['low'].rolling(window=20).min().interpolate()
        for level in [0.236, 0.382, 0.618]:
            fib = pivot_low + (pivot_high - pivot_low) * level
            col_name = f'fib_{int(level*100)}'
            features[col_name] = fib if not fib.isnull().all() else np.zeros(len(fib), dtype=np.float32)
    except Exception as e:
        logging.warning(f"Fibonacci hesaplama hatası (sembol: {sembol}): {e}")
        for level in [0.236, 0.382, 0.618]:
            features[f'fib_{int(level*100)}'] = np.zeros(len(features), dtype=np.float32)

    # ATR kontrolü
    if 'atr' not in features.columns or features['atr'].isnull().all() or (features['atr'] == 0).all():
        atr_arr = ta.atr(high, low, close, length=14)
        if isinstance(atr_arr, pd.Series):
            atr_arr = atr_arr.fillna(1.0).replace(0, 1.0)
            features['atr'] = atr_arr.values
        else:
            features['atr'] = np.ones(len(features), dtype=np.float32)
    else:
        features['atr'] = features['atr'].fillna(1.0).replace(0, 1.0)

    # Özellik seçimi ve rejim ekleme
    features = ozellikleri_sec(features)
    features = rejim_ozellikleri_ekle(features)

    # Temizlik
    features = features.replace([np.inf, -np.inf], 0.0)
    features = features.fillna(0.0).infer_objects(copy=False).astype(np.float32)

    ozellik_onbellek[cache_key] = (features, time_series)
    logging.info(f"Özellik mühendisliği tamamlandı, özellik sayısı: {features.shape[1]}, sembol: {sembol}")
    logging.info(f"Özellik sütunları: {features.columns.tolist()}")
    return features, time_series
    
    def is_valid_array(arr: np.ndarray, min_length: int) -> bool:
        return arr is not None and len(arr) >= min_length and not np.any(np.isnan(arr))

    def interpolate_feature(arr: np.ndarray, length: int) -> np.ndarray:
        if not is_valid_array(arr, length):
            return np.interp(np.arange(len(arr)), np.where(~np.isnan(arr))[0], arr[~np.isnan(arr)])
        return arr

    for indicator, length, columns in secilen_indikatorler:
        try:
            interp_data = [interpolate_feature(close, length)]
            if indicator in ['stoch', 'adx', 'bbands', 'ichimoku', 'atr', 'ad', 'cci']:
                interp_data.extend([interpolate_feature(high, length), interpolate_feature(low, length)])
            if indicator in ['obv', 'ad']:
                interp_data.append(interpolate_feature(volume, length))
            if indicator == 'candles':
                interp_data = [interpolate_feature(open_price, length), interpolate_feature(high, length),
                               interpolate_feature(low, length), interpolate_feature(close, length)]

            if not all(is_valid_array(data, length) for data in interp_data):
                for col in columns:
                    features[col] = np.nan
                continue

            if indicator == 'rsi':
                result = ta.rsi(interp_data[0], length=length)
            elif indicator == 'ema':
                result = ta.ema(interp_data[0], length=length)
            elif indicator == 'macd':
                result = ta.macd(interp_data[0], fast=12, slow=26, signal=9)
            elif indicator == 'stoch':
                result = ta.stoch(interp_data[0], interp_data[1], interp_data[2], k=14, d=3)
            elif indicator == 'adx':
                result = ta.adx(interp_data[0], interp_data[1], interp_data[2], length=length)
                # ta.adx bazen None dönebilir, veya ADX_14 yoktur
                if result is None or 'ADX_14' not in result:
                    features['adx'] = np.full(len(features), 0.0)
                else:
                    # eksik değerleri doldurup index’e göre hizala
                    adx_series = result['ADX_14'].reindex(features.index, method='ffill').fillna(0.0)
                    features['adx'] = adx_series.values
            elif indicator == 'obv':
                try:
                    if not all(isinstance(x, np.ndarray) for x in interp_data[:2]):
                        raise ValueError(f"OBV için giriş verileri numpy.ndarray olmalı, alınan: {[type(x) for x in interp_data[:2]]}")
                    close, volume = [x.astype(float) for x in interp_data[:2]]
                    if len(close) < 1 or len(volume) < 1:
                        raise ValueError(f"OBV için veri uzunluğu yetersiz: close={len(close)}, volume={len(volume)}")
                    if np.any(np.isnan(close)) or np.any(np.isnan(volume)):
                        raise ValueError("OBV için NaN veri bulundu")
                    obv_series = ta.obv(close=close, volume=volume)
                    if obv_series is None or np.all(np.isnan(obv_series)):
                        raise ValueError("OBV hesaplaması boş veya NaN veri döndü")
                    obv_series = pd.Series(obv_series, index=features.index).ffill().fillna(0.0)
                    features['obv'] = obv_series.values.astype(np.float32)
                except Exception as e:
                    logging.warning(f"OBV hesaplama hatası (sembol: {sembol}): {e}")
                    features['obv'] = np.zeros(len(features), dtype=np.float32)
                continue
            elif indicator == 'ad':
                try:
                    if not all(isinstance(x, np.ndarray) for x in interp_data[:4]):
                        raise ValueError("AD için giriş verileri numpy.ndarray olmalı")
                    ad_series = ta.ad(
                        high=interp_data[0].astype(float),
                        low=interp_data[1].astype(float),
                        close=interp_data[2].astype(float),
                        volume=interp_data[3].astype(float)
                    )
                    if ad_series is None or np.all(np.isnan(ad_series)):
                        features['ad'] = np.zeros(len(features), dtype=np.float32)
                    else:
                        ad_series = pd.Series(ad_series, index=features.index).ffill().fillna(0.0)
                        features['ad'] = ad_series.values.astype(np.float32)
                except Exception as e:
                    logging.warning(f"AD hesaplama hatası: {e}")
                    features['ad'] = np.zeros(len(features), dtype=np.float32)
                continue
            elif indicator == 'bbands':
                result = ta.bbands(interp_data[0], length=length, std=2)
            elif indicator == 'ichimoku':
                try:
                    if not all(isinstance(x, np.ndarray) for x in interp_data[:3]):
                        raise ValueError(f"Ichimoku için giriş verileri numpy.ndarray olmalı, alınan: {[type(x) for x in interp_data[:3]]}")
                    high, low, close = [x.astype(float) for x in interp_data[:3]]
                    if len(high) < 52 or len(low) < 52 or len(close) < 52:
                        raise ValueError(f"Ichimoku için veri uzunluğu yetersiz: {len(high)}")
                    if np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)):
                        raise ValueError("Ichimoku için NaN veri bulundu")
                    ich = ta.ichimoku(high=high, low=low, close=close, tenkan=9, kijun=26, senkou=52)[0]
                    if ich is None or ich.empty:
                        raise ValueError("Ichimoku hesaplaması boş veri döndü")
                    features['tenkan_9'] = ich['ITS_9'].reindex(features.index).bfill().values
                    features['kijun_26'] = ich['IKS_26'].reindex(features.index).bfill().values
                    features['chikou_span'] = ich['ICS_26'].reindex(features.index).ffill().values
                    features['senkou_span_a'] = ich['ISA_9'].reindex(features.index).bfill().values
                    features['senkou_span_b'] = ich['ISB_26'].reindex(features.index).bfill().values
                except Exception as e:
                    logging.warning(f"Ichimoku hesaplama hatası (sembol: {sembol}): {e}")
                    features['tenkan_9'] = np.zeros(len(features), dtype=np.float32)
                    features['kijun_26'] = np.zeros(len(features), dtype=np.float32)
                    features['chikou_span'] = np.zeros(len(features), dtype=np.float32)
                    features['senkou_span_a'] = np.zeros(len(features), dtype=np.float32)
                    features['senkou_span_b'] = np.zeros(len(features), dtype=np.float32)
                continue
               
            elif indicator == 'cci':
                    result = ta.cci(interp_data[0], interp_data[1], interp_data[2], length=length)
            elif indicator == 'candles':
                try:
                    open_p, high_p, low_p, close_p = [x.astype(np.float64) for x in interp_data]
                    result = [
                        talib.CDLDOJISTAR(open_p, high_p, low_p, close_p),
                        talib.CDLENGULFING(open_p, high_p, low_p, close_p)
                    ]
                except Exception as e:
                    logging.warning(f"TA-Lib mum çubuğu hatası: {e}")
                    result = [np.zeros(len(interp_data[0])), np.zeros(len(interp_data[0]))]

            for i, col in enumerate(columns):
                if isinstance(result, dict):
                    data = result.get(col, np.nan)
                elif isinstance(result, list):
                    data = result[i] if i < len(result) else np.nan
                else:
                    data = result
                if pd.api.types.is_scalar(data) or isinstance(data, (int, float, bool)):
                    features[col] = np.full(len(features), float(data) if not pd.isna(data) else np.nan)
                elif isinstance(data, np.ndarray):
                    if np.all(pd.isna(data)) or (np.nanmean(data) <= 0 and data.size > 0):
                        features[col] = np.zeros(len(data))
                    else:
                        features[col] = data
                else:
                    features[col] = np.nan
        except (ValueError, KeyError) as e:
            logging.warning(f"Indikatör hesaplama hatası ({indicator}): {e}")
            for col in columns:
                features[col] = np.nan

    try:
        pivot_high = df['high'].rolling(window=20).max().interpolate()
        pivot_low = df['low'].rolling(window=20).min().interpolate()
        for level in [0.236, 0.382, 0.618]:
            fib = pivot_low + (pivot_high - pivot_low) * level
            col_name = f'fib_{int(level*100)}'
            features[col_name] = fib if not fib.isnull().all() else np.zeros(len(fib))
    except Exception:
        for level in [0.236, 0.382, 0.618]:
            features[f'fib_{int(level*100)}'] = np.nan

    for timeframe in zaman_dilimleri[1:]:
        resampled_df = resample_data(df, timeframe)
        if resampled_df.empty:
            features[f'rsi_{timeframe}'] = np.nan
            features[f'adx_{timeframe}'] = np.nan
            continue
        try:
            if len(resampled_df) >= 14:
                rsi = ta.rsi(resampled_df['close'].interpolate(), length=14)
                adx = ta.adx(resampled_df['high'].interpolate(), resampled_df['low'].interpolate(),
                            resampled_df['close'].interpolate(), length=14)
                features[f'rsi_{timeframe}'] = rsi.reindex(df.index, method='ffill').fillna(0)
                features[f'adx_{timeframe}'] = adx['ADX_14'].reindex(df.index, method='ffill').fillna(0)
            else:
                features[f'rsi_{timeframe}'] = np.nan
                features[f'adx_{timeframe}'] = np.nan
        except Exception:
            features[f'rsi_{timeframe}'] = np.nan
            features[f'adx_{timeframe}'] = np.nan

    if takvim_df is not None and not takvim_df.empty:
        try:
            relevant_currencies = ['USD', 'EUR', 'GBP']
            takvim_df = takvim_df[takvim_df['ülke'].isin(relevant_currencies)]
            takvim_pl = pl.from_pandas(takvim_df).with_columns(
                pl.col("date").cast(pl.Datetime).dt.replace_time_zone("UTC")
            )
            features_pl = pl.from_pandas(features.assign(time=df['time'])).with_columns(
                pl.col("time").cast(pl.Datetime).dt.replace_time_zone("UTC")
            )
            features_pl = features_pl.join_asof(
                takvim_pl.select(['date', 'olay', 'önem']),
                left_on="time",
                right_on="date",
                tolerance=pd.Timedelta(minutes=30)
            )
            features_pl = features_pl.with_columns(
                pl.col('olay').fill_null("Önemli haber yok").map_elements(
                    lambda x: duygu_skoru_al(x, False if pl.col('önem') is None else pl.col('önem') == 'yüksek'),
                    return_dtype=pl.Float32
                ).alias('duygu')
            )
            features = features_pl.to_pandas().drop(['time', 'olay', 'önem'], axis=1, errors='ignore')
        except Exception:
            features['duygu'] = 0.0
    else:
        features['duygu'] = 0.0

    # ATR kontrolü (güvenli ekleme)
    if 'atr' not in features.columns or features['atr'].isnull().all() or (features['atr'] == 0).all():
        atr_arr = ta.atr(high, low, close, length=14)
        if isinstance(atr_arr, pd.Series):
            atr_arr = atr_arr.fillna(1.0).replace(0, 1.0)
            features['atr'] = atr_arr.values
        else:
            features['atr'] = np.ones(len(features))
    else:
        features['atr'] = features['atr'].fillna(1.0).replace(0, 1.0)

    features = ozellikleri_sec(features)
    features = rejim_ozellikleri_ekle(features)

    # Temizlik
    if 'time' in features.columns:
        features = features[~features['time'].isna()]
        features = features[features['time'].apply(lambda x: str(x) != 'NaT')]
    if isinstance(time_series, np.ndarray) and np.issubdtype(time_series.dtype, np.datetime64):
        mask = ~pd.isnull(time_series)
        time_series = time_series[mask]
        features = features.iloc[-len(time_series):].reset_index(drop=True)
    features = features.replace([np.inf, -np.inf], 0.0)
    features = features.fillna(0.0).infer_objects(copy=False).astype(np.float32)

    ozellik_onbellek[cache_key] = (features, time_series)
    logging.info(f"Özellik mühendisliği tamamlandı, özellik sayısı: {features.shape[1]}")
    logging.info(f"Özellik mühendisliği sonrası sütunlar: {features.columns.tolist()}")
    return features, time_series

def simulate_snowball(env, model, initial_balance=BASLANGIC_BAKIYESI, target_payout=TARGET_PAYOUT, max_accounts=MAX_ACCOUNTS):
    crisis_type = np.random.choice(['volatility', 'spread', 'liquidity'], p=[0.6, 0.3, 0.1])
    if np.random.random() < 0.1:
        if crisis_type == 'volatility':
            env.volatility = np.random.uniform(2.5, 4.0)
            logging.info(f"Rastgele volatilite şoku: volatilite={env.volatility:.2f}")
        elif crisis_type == 'spread':
            env.temel_spread *= np.random.uniform(2.0, 3.0)
            logging.info(f"Rastgele spread şoku: spread={env.temel_spread:.5f}")
        elif crisis_type == 'liquidity':
            env.maks_acik_pozisyon = max(1, env.maks_acik_pozisyon // 2)
            logging.info(f"Rastgele likidite şoku: maks_acik_pozisyon={env.maks_acik_pozisyon}")
    accounts = [{'balance': initial_balance, 'active': True, 'payouts': 0, 'semboller': env.semboller}]
    total_profit = 0.0
    fail_count = 0
    while any(acc['active'] for acc in accounts) and len(accounts) < max_accounts + 1:
        for acc in accounts:
            if not acc['active']:
                continue
            obs = env.reset()[0]
            acc_balance = acc['balance']
            while not env.bitti:
                action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, truncated, info = env.step(action)
                acc_balance += info['toplam_kar_zarar']
                if info['kayip_nedenleri']:
                    fail_count += 1
                    acc['active'] = False
                    break
                if acc_balance >= acc['balance'] + target_payout:
                    acc['payouts'] += 1
                    total_profit += target_payout
                    acc['balance'] = acc_balance - target_payout
                    if len(accounts) < max_accounts:
                        new_account = {'balance': 100000.0, 'active': True, 'payouts': 0, 'semboller': env.semboller}
                        accounts.append(new_account)
                        for s in env.semboller:
                            env.alt_bakiyeler[s] += 10000.0 / len(env.semboller)
                    break
    return total_profit, fail_count, len(accounts) - 1

class TrainingProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % 10000 == 0:
            logging.info(f"Adım: {self.num_timesteps}, Epizod: {self.episode_count}")
        return True

    def _on_rollout_end(self) -> None:
        self.episode_count += 1

from stable_baselines3.common.callbacks import BaseCallback
import logging
import numpy as np

class MetricsCallback(BaseCallback):
    def __init__(self, log_interval=10, summary_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.summary_freq = summary_freq
        self.step_count = 0
        # Yeni özellikler tanımlanıyor
        self.last_avg_pnl = 0.0
        self.last_avg_opened = 0.0
        self.last_avg_lot = 0.0
        self.metrics_history = []  # Metrikleri saklamak için liste

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            metrics = self.training_env.env_method("get_metrics")
            opened = [m["trades_opened"] for m in metrics]
            closed = [m["trades_closed"] for m in metrics]
            pnls = [m["pnl"] for m in metrics]
            lots = [m["lot_size"] for m in metrics]
            sharpe = [m["sharpe_ratio"] for m in metrics]

            avg_opened = float(np.mean(opened)) if opened else 0.0
            avg_closed = float(np.mean(closed)) if closed else 0.0
            avg_pnl = float(np.mean(pnls)) if pnls else 0.0
            avg_lot = float(np.mean(lots)) if lots else 0.0
            avg_sharpe = float(np.mean(sharpe)) if sharpe else 0.0

            # Özellikleri güncelle
            self.last_avg_pnl = avg_pnl
            self.last_avg_opened = avg_opened
            self.last_avg_lot = avg_lot

            # Metrikleri kaydet
            self.metrics_history.append({
                "step": self.num_timesteps,
                "avg_opened": avg_opened,
                "avg_closed": avg_closed,
                "avg_pnl": avg_pnl,
                "avg_lot": avg_lot,
                "avg_sharpe": avg_sharpe
            })

            self.logger.record("train/avg_opened", avg_opened)
            self.logger.record("train/avg_closed", avg_closed)
            self.logger.record("train/avg_pnl", avg_pnl)
            self.logger.record("train/avg_lot", avg_lot)
            self.logger.record("train/avg_sharpe", avg_sharpe)

            logging.info(
                f"| Adım: {self.num_timesteps:>6} | "
                f"avg_opened: {avg_opened:>6.1f} | "
                f"avg_closed: {avg_closed:>6.1f} | "
                f"avg_pnl: {avg_pnl:>8.2f} | "
                f"avg_lot: {avg_lot:>7.3f} | "
                f"avg_sharpe: {avg_sharpe:>7.2f} |"
            )

        if self.step_count % self.summary_freq == 0:
            tm = self.logger.name_to_value
            fps = tm.get("fps", "")
            iterations = tm.get("iterations", "")
            time_elapsed = tm.get("time_elapsed", "")
            total_timesteps = tm.get("total_timesteps", self.num_timesteps)

            actor_loss = tm.get("train/actor_loss", "")
            critic_loss = tm.get("train/critic_loss", "")
            ent_coef = tm.get("train/ent_coef", "")
            ent_loss = tm.get("train/ent_loss", "")
            lr = tm.get("train/learning_rate", "")
            policy_loss = tm.get("train/policy_loss", "")

            logging.info("\n=== Training Summary @ step %d ===", self.num_timesteps)
            logging.info("| %-15s | %s", "time/", "")
            logging.info("| %-15s | %s", "fps", fps)
            logging.info("| %-15s | %s", "iterations", iterations)
            logging.info("| %-15s | %s", "time_elapsed", time_elapsed)
            logging.info("| %-15s | %s", "total_timesteps", total_timesteps)
            logging.info("| %-15s | %s", "train/", "")
            logging.info("| %-15s | %s", "actor_loss", actor_loss)
            logging.info("| %-15s | %s", "critic_loss", critic_loss)
            logging.info("| %-15s | %s", "ent_coef", ent_coef)
            logging.info("| %-15s | %s", "ent_loss", ent_loss)
            logging.info("| %-15s | %s", "learning_rate", lr)
            logging.info("| %-15s | %s", "policy_loss", policy_loss)
            logging.info("====================================\n")

        return True

    def _on_rollout_end(self) -> None:
        metrics = self.training_env.env_method("get_metrics")
        for i, m in enumerate(metrics):
            logging.info(f"Rollout metrics (Env {i}): {m}")

    def _on_training_end(self) -> None:
        tm = self.logger.name_to_value
        fps = tm.get("fps", "")
        iterations = tm.get("iterations", "")
        time_elapsed = tm.get("time_elapsed", "")
        total_timesteps = tm.get("total_timesteps", self.num_timesteps)

        actor_loss = tm.get("train/actor_loss", "")
        critic_loss = tm.get("train/critic_loss", "")
        ent_coef = tm.get("train/ent_coef", "")
        ent_loss = tm.get("train/ent_loss", "")
        lr = tm.get("train/learning_rate", "")
        policy_loss = tm.get("train/policy_loss", "")

        # Son metriklerden ortalamaları al
        avg_pnl = round(self.last_avg_pnl, 2) if self.last_avg_pnl is not None else 0.0
        avg_trades = round(self.last_avg_opened, 1) if self.last_avg_opened is not None else 0.0
        avg_lot = round(self.last_avg_lot, 3) if self.last_avg_lot is not None else 0.0

        logging.info("=== Final Training Summary ===")
        logging.info(f"| time/          |            |")
        logging.info(f"| fps            | {fps}       |")
        logging.info(f"| iterations     | {iterations}        |")
        logging.info(f"| time_elapsed   | {time_elapsed}      |")
        logging.info(f"| total_timesteps| {total_timesteps}    |")
        logging.info(f"| train/         |            |")
        logging.info(f"| actor_loss     | {actor_loss}    |")
        logging.info(f"| critic_loss    | {critic_loss}    |")
        logging.info(f"| ent_coef       | {ent_coef}    |")
        logging.info(f"| ent_loss       | {ent_loss}   |")
        logging.info(f"| learning_rate  | {lr}    |")
        logging.info(f"| policy_loss    | {policy_loss}   |")
        logging.info(
            f"| avg_pnl       | {avg_pnl}    "
            f"| avg_trades    | {avg_trades}  "
            f"| avg_lot       | {avg_lot}  |"
        )

class TicaretOrtami(gym.Env):
    def __init__(self, config: Dict):
        super().__init__()
        self.TEMINAT_GEREKSINIMLERI = {"USDJPY": 10.0, "EURUSD": 11.73, "GBPUSD": 13.62}
        self.epizod_baslangic_zamani = time.time()
        self.max_epizod_suresi = 3600  # 1 saat
        self.total_timesteps = config.get('total_timesteps', 1000000)  # Toplam eğitim adımları
        self.semboller = config.get('semboller', SEMBOLLER)
        self.ozellikler_dict = config['df']
        self.zaman_serisi_dict = config['zaman_serisi']
        self.df_full_dict = config['df_full']
        self.mevcut_adim = 0
        self.max_adim = config.get('max_adim', len(self.zaman_serisi_dict[self.semboller[0]]))
        self.epizod_sayisi = 0
        self.bitti = False
        self.episode_steps = 0
        self.baslangic_bakiyesi = config['baslangic_bakiyesi']
        self.bakiye = float(self.baslangic_bakiyesi)
        self.gunluk_baslangic_bakiyesi = self.bakiye
        self.gunluk_islem_haklari = dict(GUNLUK_ISLEM_HAKLARI)
        self.kalan_islem_haklari = dict(self.gunluk_islem_haklari)
        self.gunluk_parite_butce = {"USDJPY": 150, "EURUSD": 150, "GBPUSD": 150}  # Daha yüksek bütçe
        self.kalan_parite_butce = dict(self.gunluk_parite_butce)
        self.gun_baslangic_saati = GUN_BASLANGIC_SAATI
        self.gun_bitis_saati = GUN_BITIS_SAATI
        self.son_hak_guncelleme = None
        self.min_gunluk_oz_sermaye = self.bakiye
        self.min_toplam_oz_sermaye = self.bakiye
        self.gunluk_kapali_pnl = 0.0
        self.yuzen_pnl = 0.0
        self.toplam_oz_sermaye = self.bakiye
        self.pozisyonlar: Dict[str, Dict] = {}
        self.lot_buyuklukleri = {s: 0.0 for s in self.semboller}
        self.temel_spread = config['temel_spread']
        self.olceklendirici = config['olceklendirici']
        self.gunluk_kayip = 0.0
        self.toplam_kayip = 0.0
        self.islemler: List[Dict] = []
        self.gunluk_islemler: List[Dict] = []
        self.haftalik_islemler: List[Dict] = []  # Haftalık işlemleri takip et
        self.islem_yok_sayisi = 0
        self.pasif_pnl = 0.0
        self.kayip_nedenleri: List[str] = []
        self.prime_zamani = config.get('prime_zamani', PRIME_ZAMANI)
        self.alt_bakiyeler = {s: self.baslangic_bakiyesi / len(self.semboller) for s in self.semboller}
        self.ticaret_gunleri: set = set()
        self.mevcut_tarih = pd.to_datetime(self.zaman_serisi_dict[self.semboller[0]][0], utc=True)
        self.hafta_baslangic_tarihi = self.mevcut_tarih
        self.hafta_baslangic_bakiyesi = self.bakiye
        self.marjin_seviyesi = 100.0
        self.telegram = TelegramBot(config['telegram_token'], config['telegram_kullanici_id'])
        self.maks_acik_pozisyon = ORIJINAL_MAKS_ACIK_POZISYON
        self.min_ticaret_gunu = MIN_TICARET_GUNU
        self.panik_butonu = False
        self.basarisizlik_sayisi = 0
        self.ardisik_basarisizlik = 0
        self.max_daily_loss = 0.07 if config.get('mod') == 'backtest' else 0.05
        self.max_total_loss = 0.15 if config.get('mod') == 'backtest' else 0.10
        self.open_count = 0
        self.volatility = 1.0
        self.risk_orani = ISLEM_BASINA_MAKS_RISK
        self.sharpe_ratio = 0.0
        self.black_swan_history = deque(maxlen=10)
        self.portfolio_weights = {s: 1.0 / len(self.semboller) for s in self.semboller}
        self.last_portfolio_update = 0
        self.haftalik_rapor_adimi = 0
        self.haber_etkileri: List[Dict] = []  # Haberlerin kâr/zarar etkisini sakla
        self.train_metrics = {
            'fps': 0.0,
            'iterations': 0,
            'time_elapsed': 0.0,
            'total_timesteps': 0,
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'ent_coef': 0.0,
            'ent_loss': 0.0,
            'learning_rate': 0.0,
            'policy_loss': 0.0
        }
        for sembol in self.semboller:
            self.df_full_dict[sembol]['time_tr'] = self.df_full_dict[sembol]['time'].dt.tz_convert('Europe/Istanbul')
        self.scaled_features = {
            s: self.olceklendirici[s].transform(self.ozellikler_dict[s])
            for s in self.semboller
        }
        obs_shape = sum(f.shape[1] for f in self.ozellikler_dict.values())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.semboller),), dtype=np.float32)
    def get_metrics(self) -> Dict[str, float]:
            """
            Stable-Baselines3'ün VecMonitor/MetricCallback
            çağırdığında bulamadığı metodu stub olarak ekliyoruz.
            """
            # isterseniz burada daha zengin bir dict dönebilirsiniz,
            # ben temel olarak _get_info’daki verilere dayanan basit bir örnek verdim:
            current_profit = self.bakiye - self.baslangic_bakiyesi
            return {
                "trades_opened": self.open_count,
                "trades_closed": len(self.islemler),
                "pnl": current_profit,
                "lot_size": sum(p["lot_buyuklugu"] for p in self.islemler),
                "sharpe_ratio": self.sharpe_ratio,
            }    

    def train_and_evaluate(config: dict, total_timesteps: int, n_envs: int, debug: bool, run_id: str) -> Dict[str, float]:
        train_env = SubprocVecEnv([lambda: TicaretOrtami(config) for _ in range(n_envs)])
        train_env = VecMonitor(train_env, filename=f"logs/monitor_{run_id}.csv")

        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=linear_schedule(0.0001, 0.00001),  # Daha düşük öğrenme oranı
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef=0.1,  # Entropi katsayısını artır
            verbose=1 if debug else 0
        )

        callbacks = [
            TrainingProgressCallback(),
            MetricsCallback(log_interval=10, summary_freq=1000)
        ]

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True if debug else False
        )

        eval_env = TicaretOrtami(config)
        total_profit, fail_count, account_count = simulate_snowball(eval_env, model)

        train_env.close()
        return {
            "total_profit": total_profit,
            "fail_count": fail_count,
            "account_count": account_count
        }
    
    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed)  # Gymnasium'un reset metodunu çağır
        self.mevcut_adim = 0
        self.bitti = False
        self.epizod_sayisi += 20000
        self.epizod_baslangic_zamani = time.time()
        self.bakiye = float(self.baslangic_bakiyesi)
        self.gunluk_baslangic_bakiyesi = self.bakiye
        self.kalan_islem_haklari = dict(self.gunluk_islem_haklari)
        self.kalan_parite_butce = dict(self.gunluk_parite_butce)
        self.gunluk_kapali_pnl = 0.0
        self.yuzen_pnl = 0.0
        self.toplam_oz_sermaye = self.bakiye
        self.pozisyonlar = {}
        self.islemler = []
        self.gunluk_islemler = []
        self.haftalik_islemler = []
        self.kayip_nedenleri = []
        self.mevcut_tarih = pd.to_datetime(self.zaman_serisi_dict[self.semboller[0]][0], utc=True)
        self.hafta_baslangic_tarihi = self.mevcut_tarih
        self.hafta_baslangic_bakiyesi = self.bakiye
        logging.info(f"Epizod {self.epizod_sayisi} başladı")
        return self._gozlem_al(), {}    

    def _lot_buyuklugu_hesapla(self, sembol: str, atr: float, sinyal_gucu: float) -> float:
        return 0.01  # Sabit lot büyüklüğü

    def _safe_atr(self, sembol: str, window: int, upper_k: float, default_pips: float) -> float:
        atr = self.ozellikler_dict[sembol].get('atr', pd.Series(np.nan)).iloc[self.mevcut_adim]
        return atr if not pd.isna(atr) else default_pips

    def _yuzen_pnl_hesapla(self) -> float:
        total_pnl = 0.0
        idx = self.mevcut_adim
        pip_sizes = {'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01}
        
        for trade_id, position in list(self.pozisyonlar.items()):
            sembol = position['sembol']
            current_price = self.df_full_dict[sembol]['close'].iloc[idx]
            pip_size = pip_sizes.get(sembol, 0.0001)
            price_diff = (current_price - position['giris_fiyati']) if position['tip'] == 'buy' \
                         else (position['giris_fiyati'] - current_price)
            position['pnl'] = price_diff * position['lot_buyuklugu'] * (1/pip_size) * pip_size * 10000
            total_pnl += position['pnl']

            # Trailing stop
            atr = self.ozellikler_dict[sembol]['atr'].iloc[idx]
            if position['pnl'] > 10 * pip_size:  # 10 pip kârda
                if position['tip'] == 'buy':
                    position['sl'] = max(position['sl'], current_price - atr)
                else:
                    position['sl'] = min(position['sl'], current_price + atr)

            # SL/TP kontrolü
            if (position['tip'] == 'buy' and current_price <= position['sl']) or \
               (position['tip'] == 'sell' and current_price >= position['sl']):
                self._pozisyonu_kapat(trade_id, 'SL', position['sl'])
            elif (position['tip'] == 'buy' and current_price >= position['tp']) or \
                 (position['tip'] == 'sell' and current_price <= position['tp']):
                self._pozisyonu_kapat(trade_id, 'TP', position['tp'])

        return total_pnl

    def _calculate_reward(self, action_idx: int) -> float:
        return self.gunluk_kapali_pnl  # Basit ödül

    def _get_info(self, total_profit: float) -> Dict:
        wins = sum(1 for t in self.islemler if t['pnl'] > 0)
        total_trades = len(self.islemler)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        return {
            'trades_opened': self.open_count,
            'trades_closed': len(self.gunluk_islemler),
            'pnl': total_profit,
            'lot_size': sum(p['lot_buyuklugu'] for p in self.pozisyonlar.values()),
            'sharpe_ratio': self.sharpe_ratio,
            'toplam_kar_zarar': self.bakiye - self.baslangic_bakiyesi,
            'kayip_nedenleri': self.kayip_nedenleri,
            'win_rate': win_rate
        }

    def _zaman_asimi_kontrol_et(self):
        MAX_HOLD_STEPS = 92  # 23 saat ≈ 92 * 15 dakika
        for trade_id, pos in list(self.pozisyonlar.items()):
            if self.mevcut_adim - pos['acilis_adimi'] >= MAX_HOLD_STEPS:
                self._pozisyonu_kapat(trade_id, 'Zaman Aşımı', self.df_full_dict[pos['sembol']]['close'].iloc[self.mevcut_adim])
            elif self.df_full_dict[pos['sembol']]['time_tr'].iloc[self.mevcut_adim].hour == 23:
                self._tum_pozisyonlari_kapat('Gece Pozisyon Kapatma (TR Saati)')

    def _tum_pozisyonlari_kapat(self, reason: str):
        for trade_id in list(self.pozisyonlar.keys()):
            self._pozisyonu_kapat(trade_id, reason, self.df_full_dict[self.pozisyonlar[trade_id]['sembol']]['close'].iloc[self.mevcut_adim])

    def _islem_simule_et(self, sembol: str, lot_buyuklugu: float, islem_tipi: str, sinyal_gucu: float) -> float:
        if lot_buyuklugu != 0.01:
            logging.error(f"{sembol} için lot büyüklüğü sabit 0.01 olmalı, bulundu: {lot_buyuklugu}")
            return 0.0
        if self.kalan_parite_butce.get(sembol, 0) < self.TEMINAT_GEREKSINIMLERI[sembol]:
            logging.info(f"{sembol} için yetersiz teminat: {self.kalan_parite_butce.get(sembol, 0)} < {self.TEMINAT_GEREKSINIMLERI[sembol]}")
            return 0.0
        if self.kalan_islem_haklari.get(sembol, 0) <= 0:
            logging.info(f"{sembol} için işlem hakkı kalmadı")
            return 0.0

        idx = self.mevcut_adim
        price = self.df_full_dict[sembol]['close'].iloc[idx]
        if pd.isna(price):
            raise ValueError(f"{sembol} için fiyat NaN")
        if sembol in ["EURUSD", "GBPUSD"]:
            if price < 0.5 or price > 2.5 or price <= 0:
                raise ValueError(f"{sembol} için fiyat bozuk: {price}")
        elif sembol == "USDJPY":
            if price < 50 or price > 200 or price <= 0:
                raise ValueError(f"{sembol} için fiyat bozuk: {price}")

        # Sabit TP ve SL (15 pip TP, 30 pip SL)
        pip_size = 0.0001 if sembol in ["EURUSD", "GBPUSD"] else 0.01
        tp_pip = 15 * pip_size  # 15 pip
        sl_pip = 30 * pip_size  # 30 pip
        if islem_tipi == 'buy':
            sl = price - sl_pip
            tp = price + tp_pip
        else:
            sl = price + sl_pip
            tp = price - tp_pip

        trade_id = str(uuid.uuid4())
        self.pozisyonlar[trade_id] = {
            'sembol': sembol,
            'tip': islem_tipi,
            'lot_buyuklugu': 0.01,
            'giris_fiyati': price,
            'sl': sl,
            'tp': tp,
            'pnl': 0.0,
            'acilis_zamani': self.mevcut_tarih,
            'acilis_adimi': self.mevcut_adim,
            'sinyal_gucu': sinyal_gucu
        }
        self.open_count += 1
        self.kalan_islem_haklari[sembol] -= 1  # İşlem hakkını azalt
        self.kalan_parite_butce[sembol] -= self.TEMINAT_GEREKSINIMLERI[sembol]
        logging.info(
            f"İşlem açıldı: {sembol}, tip: {islem_tipi}, lot: 0.01, "
            f"giriş: {price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}, "
            f"Kalan teminat: {self.kalan_parite_butce[sembol]:.2f}, Kalan hak: {self.kalan_islem_haklari[sembol]}"
        )
        return 0.0

    def _pozisyonu_kapat(self, trade_id: str, reason: str, exit_price: float):
        position = self.pozisyonlar.pop(trade_id, None)
        if not position:
            return
        sembol = position['sembol']
        pip_sizes = {'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01}
        pip_size = pip_sizes.get(sembol, 0.0001)
        price_diff = (exit_price - position['giris_fiyati']) if position['tip'] == 'buy' \
                     else (position['giris_fiyati'] - exit_price)
        pnl = price_diff * position['lot_buyuklugu'] * (1/pip_size) * pip_size * 10000
        self.gunluk_kapali_pnl += pnl
        self.bakiye += pnl
        self.alt_bakiyeler[sembol] += pnl
        trade_record = {
            'islem_id': trade_id,
            'sembol': sembol,
            'tip': position['tip'],
            'lot_buyuklugu': position['lot_buyuklugu'],
            'giris_fiyati': position['giris_fiyati'],
            'cikis_fiyati': exit_price,
            'pnl': pnl,
            'acilis_zamani': position['acilis_zamani'],
            'kapanis_zamani': self.mevcut_tarih,
            'neden': reason,
            'sinyal_gucu': position['sinyal_gucu']
        }
        self.islemler.append(trade_record)
        self.gunluk_islemler.append(trade_record)
        self.haftalik_islemler.append(trade_record.copy())  # Haftalık işlem takibi
        if pnl < 0:
            self.kalan_islem_haklari[sembol] = max(0, self.kalan_islem_haklari.get(sembol, 0) - 1)
        self.kalan_parite_butce[sembol] += self.TEMINAT_GEREKSINIMLERI[sembol]
        logging.info(
            f"İşlem kapandı: {sembol}, neden: {reason}, PNL: {pnl:.2f}, "
            f"kalan teminat: {self.kalan_parite_butce[sembol]:.2f}, kalan hak: {self.kalan_islem_haklari[sembol]}"
        )

    def _haftalik_rapor_gonder(self):
        # Hafta başlangıç ve bitiş tarihleri
        hafta_bitis_tarihi = self.mevcut_tarih
        hafta_uzunlugu = (hafta_bitis_tarihi - self.hafta_baslangic_tarihi).days

        # Haftalık kâr/zarar ve işlem istatistikleri
        haftalik_pnl = sum(t['pnl'] for t in self.haftalik_islemler if 'pnl' in t)
        hafta_sonu_bakiyesi = self.bakiye
        parite_istatistikleri = {s: {'karli': 0, 'zararli': 0} for s in self.semboller}
        gunluk_pnl = {}
        for t in self.haftalik_islemler:
            if 'pnl' in t:
                sembol = t['sembol']
                if t['pnl'] > 0:
                    parite_istatistikleri[sembol]['karli'] += 1
                elif t['pnl'] < 0:
                    parite_istatistikleri[sembol]['zararli'] += 1
                gun = t['kapanis_zamani'].date()
                gunluk_pnl[gun] = gunluk_pnl.get(gun, 0.0) + t['pnl']

        # En başarılı gün
        en_basarili_gun = max(gunluk_pnl.items(), key=lambda x: x[1], default=(None, 0.0))[0] if gunluk_pnl else None

        # Haber etkileri
        haber_istatistikleri = []
        for haber in self.haber_etkileri:
            haber_tarihi = pd.to_datetime(haber['tarih']).date()
            if self.hafta_baslangic_tarihi.date() <= haber_tarihi <= hafta_bitis_tarihi.date():
                haber_istatistikleri.append({
                    'tarih': haber_tarihi,
                    'baslik': haber['baslik'],
                    'pnl': sum(t['pnl'] for t in self.haftalik_islemler if 'pnl' in t and t['kapanis_zamani'].date() == haber_tarihi)
                })

        # En etkili haber
        en_etkili_haber = max(haber_istatistikleri, key=lambda x: abs(x['pnl']), default={'tarih': None, 'baslik': 'Yok', 'pnl': 0.0})

        # Kalan eğitim süresi
        kalan_adim = self.total_timesteps - self.mevcut_adim
        kalan_sure_saniye = kalan_adim * (15 * 60) / self.train_metrics['fps'] if self.train_metrics['fps'] > 0 else 0
        kalan_sure_gun = kalan_sure_saniye / (24 * 3600)

        # Tablo oluştur
        tablo_veri = {
            'Metrik': [
                'Hafta Başlangıcı', 'Hafta Bitişi', 'Hafta Uzunluğu (Gün)', 'Başlangıç Bakiyesi (USD)', 'Hafta Sonu Bakiyesi (USD)', 
                'Haftalık Kâr/Zarar (USD)', 'En Başarılı Gün', 'EURUSD Kârlı İşlem', 'EURUSD Zararlı İşlem', 
                'USDJPY Kârlı İşlem', 'USDJPY Zararlı İşlem', 'GBPUSD Kârlı İşlem', 'GBPUSD Zararlı İşlem', 
                'En Etkili Haber Tarihi', 'En Etkili Haber Başlığı', 'En Etkili Haber Kâr/Zarar (USD)', 
                'FPS', 'İterasyonlar', 'Geçen Süre (s)', 'Toplam Adımlar', 'Aktör Kaybı', 'Kritik Kaybı', 
                'Entropi Katsayısı', 'Entropi Kaybı', 'Öğrenme Oranı', 'Politika Kaybı', 'Kalan Eğitim Süresi (Gün)'
            ],
            'Değer': [
                self.hafta_baslangic_tarihi.strftime('%Y-%m-%d'), hafta_bitis_tarihi.strftime('%Y-%m-%d'), hafta_uzunlugu, 
                f"{self.hafta_baslangic_bakiyesi:.2f}", f"{hafta_sonu_bakiyesi:.2f}", f"{haftalik_pnl:.2f}", 
                en_basarili_gun.strftime('%Y-%m-%d') if en_basarili_gun else 'Yok', 
                parite_istatistikleri['EURUSD']['karli'], parite_istatistikleri['EURUSD']['zararli'], 
                parite_istatistikleri['USDJPY']['karli'], parite_istatistikleri['USDJPY']['zararli'], 
                parite_istatistikleri['GBPUSD']['karli'], parite_istatistikleri['GBPUSD']['zararli'], 
                en_etkili_haber['tarih'].strftime('%Y-%m-%d') if en_etkili_haber['tarih'] else 'Yok', 
                en_etkili_haber['baslik'], f"{en_etkili_haber['pnl']:.2f}", 
                f"{self.train_metrics['fps']:.2f}", self.train_metrics['iterations'], 
                f"{self.train_metrics['time_elapsed']:.2f}", self.train_metrics['total_timesteps'], 
                f"{self.train_metrics['actor_loss']:.4f}", f"{self.train_metrics['critic_loss']:.4f}", 
                f"{self.train_metrics['ent_coef']:.4f}", f"{self.train_metrics['ent_loss']:.4f}", 
                f"{self.train_metrics['learning_rate']:.6f}", f"{self.train_metrics['policy_loss']:.4f}", 
                f"{kalan_sure_gun:.2f}"
            ],
            'Açıklama': [
                'Haftanın başlangıç tarihi', 'Haftanın bitiş tarihi', 'Haftanın kaç gün sürdüğü', 
                'Haftaya başlarken toplam para', 'Haftayı bitirirken toplam para', 
                'Hafta boyunca kazanılan veya kaybedilen para', 'Haftanın en çok kazanç getiren günü', 
                'EURUSD için kazançlı işlem sayısı', 'EURUSD için zarar eden işlem sayısı', 
                'USDJPY için kazançlı işlem sayısı', 'USDJPY için zarar eden işlem sayısı', 
                'GBPUSD için kazançlı işlem sayısı', 'GBPUSD için zarar eden işlem sayısı', 
                'En etkili haberin tarihi', 'En etkili haberin başlığı', 
                'O haber gününde kazanılan/kaybedilen para', 
                'Saniyede işlenen adım sayısı (eğitim hızı)', 'Eğitimde tamamlanan döngü sayısı', 
                'Eğitimde geçen toplam süre (saniye)', 'Toplam hedef adım sayısı', 
                'Botun hareket seçimindeki hata (düşük olmalı)', 'Botun değer tahminindeki hata (düşük olmalı)', 
                'Keşif ve denge ayarı (0.0-0.1 arası iyi)', 'Hareket çeşitliliği hatası (sıfıra yakın iyi)', 
                'Botun öğrenme hızı (0.0001-0.001 arası iyi)', 'Botun politika hatası (sıfıra yakın iyi)', 
                'Eğitimin bitmesi için kalan gün sayısı'
            ],
            'Kabul Sınırı': [
                '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 
                '-', '-', '-', '50-500', 'Eğitim süresine bağlı', 'Binlerce saniye', '100,000-10,000,000', 
                '-1.0 ila 1.0', '0.0-1.0', '0.0-0.1', '-0.5 ila 0.5', '0.0001-0.001', '-1.0 ila 1.0', 
                'Eğitim hedefine bağlı'
            ]
        }
        tablo = pd.DataFrame(tablo_veri)
        mesaj = f"📊 **Haftalık Rapor ({self.hafta_baslangic_tarihi.strftime('%Y-%m-%d')} - {hafta_bitis_tarihi.strftime('%Y-%m-%d')})**\n\n"
        mesaj += "```\n" + tablo.to_markdown(index=False) + "\n```"
        self.telegram.sync_send_message(mesaj)
        logging.info(f"Haftalık rapor gönderildi: {self.hafta_baslangic_tarihi} - {hafta_bitis_tarihi}")

        # Haftalık verileri sıfırla
        self.haftalik_islemler = []
        self.hafta_baslangic_tarihi = hafta_bitis_tarihi
        self.hafta_baslangic_bakiyesi = hafta_sonu_bakiyesi

    async def fetch_news(self) -> List[Dict]:
        # Örnek haber verisi (gerçek API kullanılmalı)
        haberler = [
            {'tarih': self.mevcut_tarih, 'baslik': 'Örnek: Faiz Artışı', 'etki': 'negatif'},
            {'tarih': self.mevcut_tarih, 'baslik': 'Örnek: PMI Verisi', 'etki': 'pozitif'}
        ]
        self.haber_etkileri.extend(haberler)
        return haberler

    def update_train_metrics(self, metrics: Dict):
        self.train_metrics.update(metrics)

    def harmonik_sinyal(self, sembol: str) -> Tuple[str, float]:
        features = self.ozellikler_dict[sembol]
        idx = self.mevcut_adim
        current_price = self.df_full_dict[sembol]['close'].iloc[idx]
        sinyaller = []
        gucler = []
        
        # Ichimoku
        ichimoku_a = features['senkou_span_a'].iloc[idx]
        ichimoku_b = features['senkou_span_b'].iloc[idx]
        if not pd.isna(ichimoku_a) and not pd.isna(ichimoku_b):
            if current_price > ichimoku_a and current_price > ichimoku_b:
                sinyaller.append('buy')
                gucler.append(0.8)
            elif current_price < ichimoku_a and current_price < ichimoku_b:
                sinyaller.append('sell')
                gucler.append(0.8)
        
        # Breakout
        window = 20
        high = self.df_full_dict[sembol]['high'].iloc[max(0, idx - window):idx].max()
        low = self.df_full_dict[sembol]['low'].iloc[max(0, idx - window):idx].min()
        if current_price > high:
            sinyaller.append('buy')
            gucler.append(0.7)
        elif current_price < low:
            sinyaller.append('sell')
            gucler.append(0.7)
        
        # RSI
        rsi = features['rsi'].iloc[idx]
        if not pd.isna(rsi):
            if rsi > 70:
                sinyaller.append('sell')
                gucler.append(0.6)
            elif rsi < 30:
                sinyaller.append('buy')
                gucler.append(0.6)
        
        # MACD
        macd = features.get('macd', pd.Series(np.nan)).iloc[idx]
        macd_signal = features.get('macd_signal', pd.Series(np.nan)).iloc[idx]
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal and macd > 0:
                sinyaller.append('buy')
                gucler.append(0.6)
            elif macd < macd_signal and macd < 0:
                sinyaller.append('sell')
                gucler.append(0.6)
        
        # ADX
        adx = features.get('ADX_14', pd.Series(np.nan)).iloc[idx]
        if not pd.isna(adx) and adx > 25:
            gucler = [g * 1.2 if s in ['buy', 'sell'] else g for s, g in zip(sinyaller, gucler)]
        
        # Bollinger Bands
        boll_ub = features.get('boll_ub', pd.Series(np.nan)).iloc[idx]
        boll_lb = features.get('boll_lb', pd.Series(np.nan)).iloc[idx]
        if not pd.isna(boll_ub) and not pd.isna(boll_lb):
            if current_price > boll_ub:
                sinyaller.append('sell')
                gucler.append(0.5)
            elif current_price < boll_lb:
                sinyaller.append('buy')
                gucler.append(0.5)
        
        if not sinyaller:
            return None, 0.0
        
        buy_score = sum(g for s, g in zip(sinyaller, gucler) if s == 'buy')
        sell_score = sum(g for s, g in zip(sinyaller, gucler) if s == 'sell')
        total_score = buy_score + sell_score
        
        if total_score == 0:
            return None, 0.0
        
        sinyal = 'buy' if buy_score > sell_score else 'sell'
        sinyal_gucu = max(buy_score, sell_score) / total_score
        
        return sinyal, np.clip(sinyal_gucu, 0.0, 1.0)

    def step(self, actions, train_metrics: Dict = None):
        if train_metrics:
            self.update_train_metrics(train_metrics)

        if self.mevcut_adim >= 100000:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Maksimum Adım Sınırı')
            logging.warning("Maksimum adım sınırına ulaşıldı")
            return self._gozlem_al(), 0.0, self.bitti, False, self._get_info(0.0)

        if time.time() - self.epizod_baslangic_zamani > self.max_epizod_suresi:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Epizod Zaman Aşımı')
            logging.warning("Epizod zaman aşımına ulaşıldı")
            return self._gozlem_al(), 0.0, self.bitti, False, self._get_info(0.0)

        self.mevcut_adim += 1
        self._zaman_asimi_kontrol_et()

        info = {
            'trades_opened': 0,
            'trades_closed': 0,
            'pnl': 0.0,
            'lot_size': 0.0,
            'sharpe_ratio': 0.0,
            'toplam_kar_zarar': 0.0,
            'kayip_nedenleri': []
        }

        mevcut_zaman = self.df_full_dict[self.semboller[0]]['time'].iloc[self.mevcut_adim]
        self.mevcut_tarih = mevcut_zaman
        turkey_time = self.df_full_dict[self.semboller[0]]['time_tr'].iloc[self.mevcut_adim]

        if pd.isna(turkey_time):
            logging.warning(f"Türkiye saati NaT, varsayılan saat kontrolü atlanıyor: adım {self.mevcut_adim}")
            return self._gozlem_al(), 0.0, self.bitti, False, info

        # Gece yasağı: 23:00-08:00 arası işlem açma ve pozisyon tutma engeli
        if turkey_time.hour >= 23 or turkey_time.hour < 8:
            self._tum_pozisyonlari_kapat("Gece Pozisyon Kapatma (TR Saati)")
            logging.info(f"Gece yasağı: Pozisyonlar kapatıldı, adım: {self.mevcut_adim}, saat: {turkey_time}")
            return self._gozlem_al(), 0.0, self.bitti, False, info

        # Gün değişimi: Sadece işlem haklarını sıfırla
        if self.son_hak_guncelleme is None or mevcut_zaman.date() != self.son_hak_guncelleme:
            self.kalan_islem_haklari = dict(self.gunluk_islem_haklari)  # Sadece hakları sıfırla
            self.son_hak_guncelleme = mevcut_zaman.date()
            self._tum_pozisyonlari_kapat('Gün Sonu Zorunlu Kapatma')
            logging.info(f"Gün değişimi: İşlem hakları sıfırlandı, tarih: {mevcut_zaman.date()}")
            
        # Haftalık rapor kontrolü
        if self.mevcut_adim - self.haftalik_rapor_adimi >= 672:  # Yaklaşık 1 hafta (672 * 15dk)
            self._haftalik_rapor_gonder()
            self.haftalik_rapor_adimi = self.mevcut_adim

        total_profit = 0.0
        for sembol_idx, sembol in enumerate(self.semboller):
            try:
                action_value = float(actions[sembol_idx])
                signal, signal_strength = self.harmonik_sinyal(sembol)
                atr = self._safe_atr(
                    sembol=sembol,
                    window=50,
                    upper_k=3.0,
                    default_pips=0.0010 if sembol in ["EURUSD", "GBPUSD"] else 0.10
                )
                mean_atr = np.nanmean(self.ozellikler_dict[sembol]['atr'].iloc[max(0, self.mevcut_adim-20):self.mevcut_adim+1])
                self.volatility = atr / mean_atr if mean_atr > 0 else 1.0

                if self.volatility > 2.0:
                    self.telegram.sync_send_message(f"⚠️ Yüksek Volatilite: {sembol}, volatilite: {self.volatility:.2f}")
                    logging.info(f"Yüksek volatilite: {sembol}, volatilite: {self.volatility:.2f}")
                    continue  # Yüksek volatilitede işlem açma

                # Volatilite filtresi: ATR > 10 pip
                pip_size = 0.0001 if sembol in ["EURUSD", "GBPUSD"] else 0.01
                if atr < 10 * pip_size:
                    logging.info(f"{sembol} için düşük volatilite: ATR={atr:.5f}, işlem açılmadı")
                    continue

                if abs(action_value) >= 0.01 and signal in ('buy', 'sell') and signal_strength > 0.80:
                    lot_size = self._lot_buyuklugu_hesapla(sembol, atr, signal_strength)
                    if self.kalan_islem_haklari.get(sembol, 0) > 0 and self.kalan_parite_butce.get(sembol, 0) >= lot_size:
                        try:
                            pnl = self._islem_simule_et(sembol, lot_size, signal, signal_strength)
                            info['trades_opened'] += 1
                            info['lot_size'] += lot_size
                            total_profit += pnl
                            logging.info(
                                f"İşlem açıldı: {sembol}, tip: {signal}, lot: {lot_size:.3f}, "
                                f"sinyal_gücü: {signal_strength:.2f}, kalan hak: {self.kalan_islem_haklari[sembol]}, "
                                f"kalan bütçe: {self.kalan_parite_butce[sembol]:.2f}"
                            )
                        except ValueError as e:
                            logging.error(f"İşlem açılamadı: {sembol}, hata: {str(e)}")
                            continue
                    else:
                        logging.info(
                            f"İşlem açılmadı: {sembol}, yetersiz hak: {self.kalan_islem_haklari.get(sembol, 0)}, "
                            f"bütçe: {self.kalan_parite_butce.get(sembol, 0):.2f}"
                        )
                else:
                    logging.info(f"İşlem açılmadı: {sembol}, action_value: {action_value:.2f}, signal: {signal}, signal_strength: {signal_strength:.2f}")
            except Exception as e:
                logging.error(f"Sembol {sembol} için hata: {str(e)}")
                continue

        self.yuzen_pnl = self._yuzen_pnl_hesapla()
        self.toplam_oz_sermaye = self.bakiye + self.yuzen_pnl
        self.gunluk_kayip = self.gunluk_baslangic_bakiyesi - self.toplam_oz_sermaye
        self.toplam_kayip = self.baslangic_bakiyesi - self.toplam_oz_sermaye

        if self.pozisyonlar and self.gunluk_kayip > self.gunluk_baslangic_bakiyesi * self.max_daily_loss * 0.8:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Günlük Kayıp Limiti')
            logging.info("Günlük kayıp limiti aşıldı, epizod bitti")
            return self._gozlem_al(), -1500.0, self.bitti, False, self._get_info(total_profit)

        if self.pozisyonlar and self.toplam_kayip > self.baslangic_bakiyesi * self.max_total_loss * 0.8:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Toplam Kayıp Limiti')
            logging.info("Toplam kayıp limiti aşıldı, epizod bitti")
            return self._gozlem_al(), -1500.0, self.bitti, False, self._get_info(total_profit)

        if self.mevcut_adim >= len(self.zaman_serisi_dict[self.semboller[0]]) - 2:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Veri Sonu')
            logging.info(f"Epizod bitti: Veri sonu, adımlar: {self.mevcut_adim}")
            return self._gozlem_al(), 0.0, self.bitti, False, self._get_info(total_profit)

        info = self._get_info(total_profit)
        reward = self._calculate_reward(0)
        logging.debug(f"Adım Sonu: {self.mevcut_adim}, Pozisyonlar: {len(self.pozisyonlar)}, "
                      f"Toplam Kâr/Zarar: {total_profit:.2f}, Ödül: {reward:.2f}")
        return self._gozlem_al(), reward, self.bitti, False, info

    def _gozlem_al(self):
        return np.concatenate([self.scaled_features[s][self.mevcut_adim] for s in self.semboller])

    def detect_black_swan(self, sembol: str) -> bool:
        try:
            short_term_atr = np.nanmean(self.ozellikler_dict[sembol]['atr'].iloc[-10:])
            long_term_atr = np.nanmean(self.ozellikler_dict[sembol]['atr'].iloc[-100:])
            if long_term_atr is None or long_term_atr <= 0:
                return False
            atr_spike = short_term_atr / long_term_atr > 3.0
            volume_spike = np.nanmean(self.df_full_dict[sembol]['volume'].iloc[-10:]) / \
                           np.nanmean(self.df_full_dict[sembol]['volume'].iloc[-100:]) > 3.0
            sentiment_score = self.ozellikler_dict[sembol].get('duygu', pd.Series(0.0)).iloc[self.mevcut_adim]
            vix_value = self.ozellikler_dict[sembol].get('vix', pd.Series([20.0] * len(self.ozellikler_dict[sembol]))).iloc[self.mevcut_adim]
            vix_series = self.ozellikler_dict[sembol].get('vix', pd.Series([20.0] * len(self.ozellikler_dict[sembol])))
            vix_threshold = np.percentile(vix_series.iloc[-100:], 95) if len(vix_series) >= 100 else 30.0
            vix_spike = vix_value > vix_threshold
            is_black_swan = (atr_spike and volume_spike) or (sentiment_score < -0.5 and vix_spike)
            if is_black_swan:
                self.black_swan_history.append((sembol, self.mevcut_tarih, vix_value))
                logging.info(f"Black Swan tespit edildi: {sembol}, ATR oranı: {short_term_atr/long_term_atr:.2f}, "
                             f"Hacim: {volume_spike}, Sentiment: {sentiment_score:.2f}, VIX: {vix_value:.2f}")
                self.telegram.sync_send_message(f"{sembol} için Black Swan: VIX={vix_value:.2f}, pozisyonlar kapanıyor!")
                self._tum_pozisyonlari_kapat(f"Black Swan: {sembol} (VIX={vix_value:.2f})")
            else:
                logging.info(f"Black Swan yok: {sembol}, ATR oranı: {short_term_atr/long_term_atr:.2f}, "
                 f"Hacim: {volume_spike}, Sentiment: {sentiment_score:.2f}, VIX: {vix_value:.2f}")
                self.telegram.sync_send_message(f"{sembol} için Black Swan: VIX={vix_value:.2f}, pozisyonlar kapanıyor!")
                self._tum_pozisyonlari_kapat(f"Black Swan: {sembol} (VIX={vix_value:.2f})")
            return is_black_swan
        except Exception as e:
            print(f"Black Swan hatası: {sembol}, {e}")
            return False

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        try:
            if len(returns) < 2 or returns.std() == 0:
                return 0.0
            sharpe = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
            return float(np.clip(sharpe, -100.0, 100.0))
        except Exception:
            return 0.0

    def compute_var(self, returns: list[float], confidence: float = 0.95) -> float:
        μ = np.mean(returns)
        σ = np.std(returns)
        z = norm.ppf(1 - confidence)
        return -(μ + z * σ)

##    def optimize_portfolio(self, semboller: List[str], returns: pd.DataFrame) -> Dict[str, float]:
##        if self.mevcut_adim - self.last_portfolio_update < 100:
##            return self.portfolio_weights
##        try:
##            cov_matrix = returns.cov().values
##            expected_returns = returns.mean().values
##            n = len(semboller)
##            weights = cp.Variable(n)
##            risk = cp.quad_form(weights, cov_matrix)
##            expected_return = weights @ expected_returns
##            objective = cp.Minimize(risk - 0.5 * expected_return)
##            constraints = [cp.sum(weights) == 1, weights >= 0.01]
##            for i, sembol in enumerate(semboller):
##                if self.detect_black_swan(sembol):
##                    constraints.append(weights[i] <= 0.1)
##            sharpe_ratios = {s: self.calculate_sharpe_ratio(pd.Series([t['pnl'] for t in self.islemler if t['sembol'] == s])) for s in semboller}
##            max_sharpe = max(sharpe_ratios.values(), default=1.0)
##            if max_sharpe > 0:
##                for i, sembol in enumerate(semboller):
##                    constraints.append(weights[i] >= 0.1 * sharpe_ratios[sembol] / max_sharpe)
##            prob = cp.Problem(objective, constraints)
##            prob.solve()
##            self.portfolio_weights = {sembol: max(0.01, weight.value) for sembol, weight in zip(semboller, weights)}
##            self.last_portfolio_update = self.mevcut_adim
##            logging.info(f"Portföy ağırlıkları güncellendi: {self.portfolio_weights}")
##            return self.portfolio_weights
##        except Exception:
##            logging.warning("Portföy optimizasyonu başarısız, eşit ağırlıklar kullanılıyor")
##            return {s: 1.0 / len(semboller) for s in semboller}

##    def mean_reversion_signal(self, sembol: str) -> str:
##        try:
##            rsi = self.ozellikler_dict[sembol]['rsi'].iloc[self.mevcut_adim]
##            if rsi > 80:
##                return 'sell'
##            elif rsi < 20:
##                return 'buy'
##            return None
##        except Exception:
##            return None

    def trend_signal(self, sembol: str) -> str:
        try:
            if 'senkou_span_a' not in self.ozellikler_dict[sembol].columns or 'senkou_span_b' not in self.ozellikler_dict[sembol].columns:
                logging.warning(f"{sembol} için Ichimoku sütunları eksik: Mevcut sütunlar={self.ozellikler_dict[sembol].columns.tolist()}")
                return None
            ichimoku_a = self.ozellikler_dict[sembol]['senkou_span_a'].iloc[self.mevcut_adim]
            ichimoku_b = self.ozellikler_dict[sembol]['senkou_span_b'].iloc[self.mevcut_adim]
            current_price = self.df_full_dict[sembol]['close'].iloc[self.mevcut_adim]
            if pd.isna(ichimoku_a) or pd.isna(ichimoku_b):
                logging.warning(f"{sembol} için Ichimoku değerleri NaN: senkou_span_a={ichimoku_a}, senkou_span_b={ichimoku_b}")
                return None
            logging.info(f"Trend Signal - sembol: {sembol}, current_price: {current_price:.5f}, ichimoku_a: {ichimoku_a:.5f}, ichimoku_b: {ichimoku_b:.5f}")
            if current_price > ichimoku_a and current_price > ichimoku_b:
                return 'buy'
            elif current_price < ichimoku_a and current_price < ichimoku_b:
                return 'sell'
            return None
        except Exception as e:
            logging.error(f"Trend Signal Error - sembol: {sembol}, error: {str(e)}")
            return None
        
    def detect_breakout(self, sembol: str) -> str:
        try:
            window = 20
            high = self.df_full_dict[sembol]['high'].iloc[max(0, self.mevcut_adim - window):self.mevcut_adim].max()
            low = self.df_full_dict[sembol]['low'].iloc[max(0, self.mevcut_adim - window):self.mevcut_adim].min()
            current_price = self.df_full_dict[sembol]['close'].iloc[self.mevcut_adim]
            if current_price > high:
                return 'buy'
            elif current_price < low:
                return 'sell'
            return None
        except Exception:
            return None

    def harmonik_sinyal(self, sembol: str) -> Tuple[str, float]:
        """
        Tüm indikatörlerden harmonik bir sinyal üretir.
        Returns: (sinyal: str | None, sinyal_gucu: float)
        """
        try:
            # Mevcut özellikler
            features = self.ozellikler_dict[sembol]
            idx = self.mevcut_adim
            current_price = self.df_full_dict[sembol]['close'].iloc[idx]
            
            # Sinyal ve güç listeleri
            sinyaller = []
            gucler = []
            
            # 1. Ichimoku Sinyali
            ichimoku_a = features['senkou_span_a'].iloc[idx]
            ichimoku_b = features['senkou_span_b'].iloc[idx]
            if not pd.isna(ichimoku_a) and not pd.isna(ichimoku_b):
                if current_price > ichimoku_a and current_price > ichimoku_b:
                    sinyaller.append('buy')
                    gucler.append(0.8)  # Ichimoku'ya yüksek ağırlık
                elif current_price < ichimoku_a and current_price < ichimoku_b:
                    sinyaller.append('sell')
                    gucler.append(0.8)
            
            # 2. Breakout Sinyali
            window = 20
            high = self.df_full_dict[sembol]['high'].iloc[max(0, idx - window):idx].max()
            low = self.df_full_dict[sembol]['low'].iloc[max(0, idx - window):idx].min()
            if current_price > high:
                sinyaller.append('buy')
                gucler.append(0.7)
            elif current_price < low:
                sinyaller.append('sell')
                gucler.append(0.7)
            
            # 3. RSI Sinyali
            rsi = features['rsi'].iloc[idx]
            if not pd.isna(rsi):
                if rsi > 70:
                    sinyaller.append('sell')
                    gucler.append(0.6)
                elif rsi < 30:
                    sinyaller.append('buy')
                    gucler.append(0.6)
            
            # 4. MACD Sinyali
            macd = features.get('macd', pd.Series(np.nan)).iloc[idx]
            macd_signal = features.get('macd_signal', pd.Series(np.nan)).iloc[idx]
            if not pd.isna(macd) and not pd.isna(macd_signal):
                if macd > macd_signal and macd > 0:
                    sinyaller.append('buy')
                    gucler.append(0.6)
                elif macd < macd_signal and macd < 0:
                    sinyaller.append('sell')
                    gucler.append(0.6)
            
            # 5. ADX Sinyali (trend gücü)
            adx = features.get('ADX_14', pd.Series(np.nan)).iloc[idx]
            if not pd.isna(adx) and adx > 25:
                # ADX yüksekse, Ichimoku veya breakout sinyaline ek güç
                gucler = [g * 1.2 if s in ['buy', 'sell'] else g for s, g in zip(sinyaller, gucler)]
            
            # 6. Bollinger Bands Sinyali
            boll_ub = features.get('boll_ub', pd.Series(np.nan)).iloc[idx]
            boll_lb = features.get('boll_lb', pd.Series(np.nan)).iloc[idx]
            if not pd.isna(boll_ub) and not pd.isna(boll_lb):
                if current_price > boll_ub:
                    sinyaller.append('sell')
                    gucler.append(0.5)
                elif current_price < boll_lb:
                    sinyaller.append('buy')
                    gucler.append(0.5)
            
            # Harmonik Sinyal Hesaplama
            if not sinyaller:
                return None, 0.0
            
            # Ağırlıklı oy sistemi
            buy_score = sum(g for s, g in zip(sinyaller, gucler) if s == 'buy')
            sell_score = sum(g for s, g in zip(sinyaller, gucler) if s == 'sell')
            total_score = buy_score + sell_score
            
            if total_score == 0:
                return None, 0.0
            
            # Nihai sinyal ve güç
            sinyal = 'buy' if buy_score > sell_score else 'sell'
            sinyal_gucu = max(buy_score, sell_score) / total_score
            
            logging.info(
                f"Harmonik sinyal: {sembol}, sinyal: {sinyal}, güç: {sinyal_gucu:.2f}, "
                f"buy_score: {buy_score:.2f}, sell_score: {sell_score:.2f}, "
                f"indikatörler: {sinyaller}"
            )
            return sinyal, np.clip(sinyal_gucu, 0.0, 1.0)
        
        except Exception as e:
            logging.error(f"Harmonik sinyal hatası: {sembol}, {str(e)}")
            return None, 0.0

    # Mevcut step fonksiyonuna harmonik sinyali entegre etme
    # Orijinal step fonksiyonundan sadece ilgili kısmı güncelleniyor
    def step(self, actions):
        """
        Ortamın bir adımını işler, harmonik sinyali kullanarak işlem kararları alır.
        Args:
            actions: Modelden gelen aksiyon vektörü (her sembol için bir değer).
        Returns:
            observation, reward, done, truncated, info
        """
        # Sert adım sınırı
        if self.mevcut_adim >= 100000:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Maksimum Adım Sınırı')
            logging.warning("Maksimum adım sınırına ulaşıldı")
            return self._gozlem_al(), 0.0, self.bitti, False, self._get_info(0.0)

        # Zaman tabanlı kontrol
        if time.time() - self.epizod_baslangic_zamani > self.max_epizod_suresi:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Epizod Zaman Aşımı')
            logging.warning("Epizod zaman aşımına ulaşıldı")
            return self._gozlem_al(), 0.0, self.bitti, False, self._get_info(0.0)

        # Adım başlangıç logu
        logging.debug(f"Adım: {self.mevcut_adim}, Pozisyonlar: {len(self.pozisyonlar)}, "
                      f"Bakiye: {self.bakiye:.2f}, Yüzen P&L: {self.yuzen_pnl:.2f}, "
                      f"Öz Sermaye: {self.toplam_oz_sermaye:.2f}")

        # Adım sayısını artır
        self.mevcut_adim += 1
        self._zaman_asimi_kontrol_et()

        # Başlangıç bilgi sözlüğü
        info = {
            'trades_opened': 0,
            'trades_closed': 0,
            'pnl': 0.0,
            'lot_size': 0.0,
            'sharpe_ratio': 0.0,
            'toplam_kar_zarar': 0.0,
            'kayip_nedenleri': []
        }

        # Mevcut zamanı al
        mevcut_zaman = self.df_full_dict[self.semboller[0]]['time'].iloc[self.mevcut_adim]
        turkey_time = self.df_full_dict[self.semboller[0]]['time_tr'].iloc[self.mevcut_adim]

        # İşlem günü kontrolü
        is_trading_day = self.ozellikler_dict[self.semboller[0]]['is_trading_day'].iloc[self.mevcut_adim]
        if not is_trading_day:
            self._tum_pozisyonlari_kapat("Tatil veya Hafta Sonu")
            logging.info(f"Tatil/hafta sonu nedeniyle işlem yapılmadı, adım: {self.mevcut_adim}")
            return self._gozlem_al(), 0.0, self.bitti, False, self._get_info(0.0)

        # Gece 23:00 - 08:00 arası işlem yasağı
        if turkey_time.hour >= 23 or turkey_time.hour < 8:
            self._tum_pozisyonlari_kapat("Gece Pozisyon Kapatma (TR Saati)")
            logging.info(f"Gece yasağı: Pozisyonlar kapatıldı, adım: {self.mevcut_adim}")
            return self._gozlem_al(), 0.0, self.bitti, False, self._get_info(0.0)

        # Gün değişimi kontrolü
        if self.son_hak_guncelleme is None or mevcut_zaman.date() != self.son_hak_guncelleme:
            self.kalan_islem_haklari = dict(self.gunluk_islem_haklari)  # Sadece hakları sıfırla
            self.son_hak_guncelleme = mevcut_zaman.date()
            self._tum_pozisyonlari_kapat('Gün Sonu Zorunlu Kapatma')
            logging.info(f"Gün değişimi: İşlem hakları sıfırlandı, tarih: {mevcut_zaman.date()}")

        # Toplam kâr/zarar takibi
        total_profit = 0.0

        # Semboller üzerinde döngü
        for sembol_idx, sembol in enumerate(self.semboller):
            try:
                # Model aksiyonu
                action_value = float(actions[sembol_idx])
                # Harmonik sinyal
                signal, signal_strength = self.harmonik_sinyal(sembol)

                # ATR ve volatilite hesapla
                atr = self._safe_atr(
                    sembol=sembol,
                    window=50,
                    upper_k=3.0,
                    default_pips=0.0010 if sembol in ["EURUSD", "GBPUSD"] else 0.10
                )
                mean_atr = np.nanmean(self.ozellikler_dict[sembol]['atr'].iloc[max(0, self.mevcut_adim-20):self.mevcut_adim+1])
                self.volatility = atr / mean_atr if mean_atr > 0 else 1.0

                # Yüksek volatilite uyarısı
                if self.volatility > 2.0:
                    self.telegram.sync_send_message(f"⚠️ Yüksek Volatilite: {sembol}, volatilite: {self.volatility:.2f}")
                    logging.info(f"Yüksek volatilite: {sembol}, volatilite: {self.volatility:.2f}")

                # İşlem açma şartı
                if abs(action_value) >= 0.01 and signal in ('buy', 'sell'):
                    lot_size = self._lot_buyuklugu_hesapla(sembol, atr, signal_strength)
                    if self.kalan_islem_haklari.get(sembol, 0) > 0 and self.kalan_parite_butce.get(sembol, 0) >= lot_size:
                        try:
                            pnl = self._islem_simule_et(sembol, lot_size, signal, signal_strength)
                            info['trades_opened'] += 1
                            info['lot_size'] += lot_size
                            total_profit += pnl
                            self.kalan_islem_haklari[sembol] -= 1
                            self.kalan_parite_butce[sembol] -= self.TEMINAT_GEREKSINIMLERI[sembol]
                            logging.info(
                                f"İşlem açıldı: {sembol}, tip: {signal}, lot: {lot_size:.3f}, "
                                f"sinyal_gücü: {signal_strength:.2f}, kalan hak: {self.kalan_islem_haklari[sembol]}, "
                                f"kalan bütçe: {self.kalan_parite_butce[sembol]:.2f}"
                            )
                        except ValueError as e:
                            logging.error(f"İşlem açılamadı: {sembol}, hata: {str(e)}")
                            continue
                    else:
                        logging.info(
                            f"İşlem açılmadı: {sembol}, yetersiz hak: {self.kalan_islem_haklari.get(sembol, 0)}, "
                            f"bütçe: {self.kalan_parite_butce.get(sembol, 0):.2f}"
                        )
                else:
                    logging.info(f"İşlem açılmadı: {sembol}, action_value: {action_value:.2f}, signal: {signal}")

            except Exception as e:
                logging.error(f"Sembol {sembol} için hata: {str(e)}")
                continue

        # Yüzen P&L ve öz sermaye güncelle
        self.yuzen_pnl = self._yuzen_pnl_hesapla()
        self.toplam_oz_sermaye = self.bakiye + self.yuzen_pnl
        self.gunluk_kayip = self.gunluk_baslangic_bakiyesi - self.toplam_oz_sermaye
        self.toplam_kayip = self.baslangic_bakiyesi - self.toplam_oz_sermaye

        # Kayıp limitleri kontrolü
        if self.pozisyonlar and self.gunluk_kayip > self.gunluk_baslangic_bakiyesi * self.max_daily_loss * 0.8:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Günlük Kayıp Limiti')
            logging.info("Günlük kayıp limiti aşıldı, epizod bitti")
            return self._gozlem_al(), -1500.0, self.bitti, False, self._get_info(total_profit)

        if self.pozisyonlar and self.toplam_kayip > self.baslangic_bakiyesi * self.max_total_loss * 0.8:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Toplam Kayıp Limiti')
            logging.info("Toplam kayıp limiti aşıldı, epizod bitti")
            return self._gozlem_al(), -1500.0, self.bitti, False, self._get_info(total_profit)

        # Veri sonu kontrolü
        if self.mevcut_adim >= len(self.zaman_serisi_dict[self.semboller[0]]) - 2:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Veri Sonu')
            logging.info(f"Epizod bitti: Veri sonu, adımlar: {self.mevcut_adim}")
            return self._gozlem_al(), 0.0, self.bitti, False, self._get_info(total_profit)

        # Bilgi güncelle ve ödül hesapla
        info = self._get_info(total_profit)
        reward = self._calculate_reward(0)
        logging.debug(f"Adım Sonu: {self.mevcut_adim}, Pozisyonlar: {len(self.pozisyonlar)}, "
                      f"Toplam Kâr/Zarar: {total_profit:.2f}, Ödül: {reward:.2f}")
        return self._gozlem_al(), reward, self.bitti, False, info

    def calculate_win_rate(self, sembol: str) -> float:
        trades = [t for t in self.islemler if t['sembol'] == sembol]
        if not trades:
            return 0.5  # Varsayılan
        wins = sum(1 for t in trades if t['pnl'] > 0)
        return wins / len(trades) if trades else 0.5        

    def calculate_risk_reward_ratio(self, sembol: str, atr: float) -> float:
        sl_pip = atr * 2.0
        tp_pip = atr * 4.0
        return tp_pip / sl_pip if sl_pip > 0 else 2.0  # Varsayılan 2:1

    def calculate_kelly_fraction(self, sembol: str, atr: float) -> float:
        W = self.calculate_win_rate(sembol)
        R = self.calculate_risk_reward_ratio(sembol, atr)
        if R == 0:
            return 0.1  # Güvenli varsayılan
        f_star = W - (1 - W) / R
        return max(0.01, min(f_star, 0.2))  # %1-%20 sınırla

    def calculate_volatility(self, sembol: str) -> float:
        return self.ozellikler_dict[sembol]['atr'].iloc[self.mevcut_adim] if self.mevcut_adim < len(self.ozellikler_dict[sembol]['atr']) else 0.0

    def _update_open_positions(self, env_idx: int):
    # Geçici dummy metot, pozisyonları güncellemek için gerçek mantık eklenmeli
        pass

    def _calculate_reward(self, env_idx: int) -> float:
        reward = self.gunluk_kapali_pnl / self.baslangic_bakiyesi  # Günlük kâr odaklı
        if self.gunluk_kapali_pnl > 0:
            reward *= 1.5  # Kârlı işlemleri ödüllendir
        return np.clip(reward, -1.0, 1.0)

##    def apply_hedging(self, sembol: str, action_value: float = 0.0):
##        try:
##            correlation_matrix = pd.DataFrame({s: self.df_full_dict[s]['close'] for s in self.semboller}).corr()
##            corr_threshold = 0.7 if self.volatility < 1.5 else 0.6
##            for other_sembol in self.semboller:
##                if other_sembol != sembol and correlation_matrix.loc[sembol, other_sembol] > corr_threshold:
##                    hedge_factor = max(0.3, 1.0 - correlation_matrix.loc[sembol, other_sembol]) * \
##                                   (0.5 if self.volatility > 1.5 else 1.0)
##                    hedge_lot = self._lot_buyuklugu_hesapla(other_sembol, 
##                                                           self.ozellikler_dict[other_sembol]['atr'].iloc[self.mevcut_adim], 
##                                                           0.5) * hedge_factor
##                    self._islem_simule_et(other_sembol, hedge_lot, 'sell' if action_value > 0 else 'buy', 0.5)
##                    logging.info(f"Hedging işlemi: {other_sembol}, tip: {'sell' if action_value > 0 else 'buy'}, "
##                                f"lot: {hedge_lot:.3f}, korelasyon: {correlation_matrix.loc[sembol, other_sembol]:.2f}")
##        except Exception:
##            pass

    def _lot_buyuklugu_hesapla(self, sembol: str, atr: float, sinyal_gucu: float) -> float:
        return 0.01  # Sabit lot büyüklüğü

    def _safe_atr(self, sembol: str, window: int = 50, upper_k: float = 3.0, default_pips: float = None) -> float:
        """
        Geçerli bir ATR değeri sağlar, bozuk durumlarda dinamik yedekleme yapar.
        
        Args:
            sembol (str): "EURUSD", "GBPUSD" veya "USDJPY".
            window (int): ATR medyanı için bakılacak mum sayısı (varsayılan: 50).
            upper_k (float): Üst eşik için medyan çarpanı (varsayılan: 3.0).
            default_pips (float): Tüm yedekler başarısız olursa kullanılacak sabit ATR (fiyat biriminde).
        
        Returns:
            float: Geçerli bir ATR değeri.
        """
        # Mevcut ATR ve penceredeki ATR serisi
        atr_series = self.ozellikler_dict[sembol]['atr']
        idx = self.mevcut_adim
        raw_atr = float(atr_series.iat[idx]) if idx < len(atr_series) else np.nan

        # Hareketli medyan hesapla
        start = max(0, idx - window + 1)
        window_vals = atr_series.iloc[start:idx + 1].dropna().values
        med_atr = float(np.median(window_vals)) if len(window_vals) > 0 else np.nan

        # Üst eşik
        upper_thresh = med_atr * upper_k if not np.isnan(med_atr) else np.nan

        # Bozuk ATR kontrolü ve yedekleme
        if np.isnan(raw_atr) or raw_atr <= 0 or (not np.isnan(upper_thresh) and raw_atr > upper_thresh):
            logging.warning(
                f"{sembol} için bozuk ATR: {raw_atr:.6f} > upper_thresh={upper_thresh:.6f} "
                f"=> yedek medyan={med_atr:.6f}"
            )
            safe_atr = med_atr
        else:
            safe_atr = raw_atr

        # Hâlâ geçersizse, sembole özel varsayılan değer
        if np.isnan(safe_atr) or safe_atr <= 0:
            if default_pips is None:
                default_pips = 0.0010 if sembol in ["EURUSD", "GBPUSD"] else 0.10
            logging.warning(f"{sembol} için medyan yedek başarısız, sabit ATR= {default_pips}")
            safe_atr = default_pips

        return float(safe_atr)

    TEMINAT_GEREKSINIMLERI = {"USDJPY": 10.0, "EURUSD": 11.73, "GBPUSD": 13.62}
    
    def _islem_simule_et(self, sembol: str, lot_buyuklugu: float, islem_tipi: str, sinyal_gucu: float) -> float:
        # Sabit lot kontrolü
        if lot_buyuklugu != 0.01:
            logging.error(f"{sembol} için lot büyüklüğü sabit 0.01 olmalı, bulundu: {lot_buyuklugu}")
            return 0.0

        # Teminat ve hak kontrolü
        TEMINAT_GEREKSINIMLERI = {"USDJPY": 10.0, "EURUSD": 11.73, "GBPUSD": 13.62}
        if self.kalan_parite_butce.get(sembol, 0) < TEMINAT_GEREKSINIMLERI[sembol]:
            logging.info(f"{sembol} için yetersiz teminat: {self.kalan_parite_butce.get(sembol, 0)} < {TEMINAT_GEREKSINIMLERI[sembol]}")
            return 0.0
        if self.kalan_islem_haklari.get(sembol, 0) <= 0:
            logging.info(f"{sembol} için işlem hakkı kalmadı")
            return 0.0

        idx = self.mevcut_adim
        price = self.df_full_dict[sembol]['close'].iloc[idx]
        if pd.isna(price):
            raise ValueError(f"{sembol} için fiyat NaN")

        # Fiyat kontrolü (daha sıkı hata yönetimi)
        if sembol in ["EURUSD", "GBPUSD"]:
            if price < 0.5 or price > 2.5 or price <= 0:
                raise ValueError(f"{sembol} için fiyat bozuk: {price}")
        elif sembol == "USDJPY":
            if price < 50 or price > 200 or price <= 0:
                raise ValueError(f"{sembol} için fiyat bozuk: {price}")

        # ATR'yi güvenli şekilde al
        atr = self._safe_atr(
            sembol=sembol,
            window=50,
            upper_k=3.0,
            default_pips=0.0010 if sembol in ["EURUSD", "GBPUSD"] else 0.10
        )

        # Stop loss ve take profit için pip hesaplamaları
        sl_pip = atr * 2.0
        tp_pip = atr * 4.0

        long_term_atr = self._safe_atr(
            sembol=sembol,
            window=200,
            upper_k=3.0,
            default_pips=0.0010 if sembol in ["EURUSD", "GBPUSD"] else 0.10
        )

        sl_pip = np.clip(sl_pip, long_term_atr * 0.5, long_term_atr * 3.0)
        tp_pip = np.clip(tp_pip, long_term_atr * 1.0, long_term_atr * 6.0)

        # Stop loss ve take profit fiyatları
        if islem_tipi == 'buy':
            sl = price - sl_pip
            tp = price + tp_pip
        else:
            sl = price + sl_pip
            tp = price - tp_pip

        # İşlem kaydı
        trade_id = str(uuid.uuid4())
        self.pozisyonlar[trade_id] = {
            'sembol': sembol,
            'tip': islem_tipi,
            'lot_buyuklugu': 0.01,  # Sabit lot
            'giris_fiyati': price,
            'sl': sl,
            'tp': tp,
            'pnl': 0.0,
            'acilis_zamani': self.mevcut_tarih,
            'acilis_adimi': self.mevcut_adim,
            'sinyal_gucu': sinyal_gucu
        }
        self.open_count += 1

        # Teminat ve hak azaltımı
        self.kalan_parite_butce[sembol] -= TEMINAT_GEREKSINIMLERI[sembol]
        logging.info(f"Post-trade: {sembol}, kalan teminat: {self.kalan_parite_butce[sembol]:.2f}, kalan hak: {self.kalan_islem_haklari[sembol]}")

        logging.info(
            f"İşlem açıldı: {sembol}, tip: {islem_tipi}, lot: 0.01, "
            f"giriş: {price:.5f}, ATR: {atr:.5f}, SL pip: {sl_pip:.5f}, TP pip: {tp_pip:.5f}, "
            f"SL: {sl:.5f}, TP: {tp:.5f}, LTA: {long_term_atr:.5f}, "
            f"Kalan teminat: {self.kalan_parite_butce[sembol]:.2f}, Kalan hak: {self.kalan_islem_haklari[sembol]}"
        )
        return 0.0
    
    def _zaman_asimi_kontrol_et(self):
        MAX_HOLD_STEPS = 92  # 15dk * 92 ≈ 23 saat
        mevcut_zaman = self.df_full_dict[self.semboller[0]]['time'].iloc[self.mevcut_adim]
        # Türkiye saatiyle 23:00 ise tüm pozisyonları kapat
        if mevcut_zaman.tz_convert('Europe/Istanbul').hour == 23:
            self._tum_pozisyonlari_kapat('Gün Sonu (23:00) Zorunlu Kapatma')
            return

        for trade_id, position in list(self.pozisyonlar.items()):
            acilis_zamani = position['acilis_zamani']
            if (mevcut_zaman - acilis_zamani).total_seconds() >= 23*3600:
                sembol = position['sembol']
                current_price = self.df_full_dict[sembol]['close'].iloc[self.mevcut_adim]
                self._pozisyonu_kapat(trade_id, 'Max Taşıma Süresi (23 saat)', current_price)

    def _yuzen_pnl_hesapla(self) -> float:
        total_pnl = 0.0
        idx = self.mevcut_adim
        pip_sizes = {'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01}
        
        for trade_id, position in list(self.pozisyonlar.items()):
            sembol = position['sembol']
            current_price = self.df_full_dict[sembol]['close'].iloc[idx]
            pip_size = pip_sizes.get(sembol, 0.0001)

            price_diff = (current_price - position['giris_fiyati']) if position['tip'] == 'buy' \
                         else (position['giris_fiyati'] - current_price)

            position['pnl'] = price_diff * position['lot_buyuklugu'] * (1/pip_size) * pip_size * 10000
            total_pnl += position['pnl']

            atr = self.ozellikler_dict[sembol]['atr'].iloc[idx]
            if position['pnl'] > atr * 2.0:
                trailing_sl = current_price - atr if position['tip'] == 'buy' else current_price + atr
                if position['tip'] == 'buy':
                    position['sl'] = max(position['sl'], trailing_sl)
                else:
                    position['sl'] = min(position['sl'], trailing_sl)

            if (position['tip'] == 'buy' and current_price <= position['sl']) or \
               (position['tip'] == 'sell' and current_price >= position['sl']):
                self._pozisyonu_kapat(trade_id, 'SL', position['sl'])
            elif (position['tip'] == 'buy' and current_price >= position['tp']) or \
                 (position['tip'] == 'sell' and current_price <= position['tp']):
                self._pozisyonu_kapat(trade_id, 'TP', position['tp'])

        return total_pnl

    def _pozisyonu_kapat(self, trade_id: str, reason: str, exit_price: float):
        position = self.pozisyonlar.pop(trade_id, None)
        if not position:
            return
        sembol = position['sembol']
        TEMINAT_GEREKSINIMLERI = {"USDJPY": 10.0, "EURUSD": 11.73, "GBPUSD": 13.62}
        pip_sizes = {'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01}
        pip_size = pip_sizes.get(sembol, 0.0001)
        price_diff = (exit_price - position['giris_fiyati']) if position['tip'] == 'buy' \
                     else (position['giris_fiyati'] - exit_price)
        pnl = price_diff * position['lot_buyuklugu'] * (1/pip_size) * pip_size * 10000
        self.gunluk_kapali_pnl += pnl
        self.bakiye += pnl
        self.alt_bakiyeler[sembol] += pnl
        trade_record = {
            'islem_id': trade_id,
            'sembol': sembol,
            'tip': position['tip'],
            'lot_buyuklugu': position['lot_buyuklugu'],
            'giris_fiyati': position['giris_fiyati'],
            'cikis_fiyati': exit_price,
            'pnl': pnl,
            'acilis_zamani': position['acilis_zamani'],
            'kapanis_zamani': self.mevcut_tarih,
            'neden': reason,
            'sinyal_gucu': position['sinyal_gucu']
        }
        self.islemler.append(trade_record)
        self.gunluk_islemler.append(trade_record)
        self.kalan_parite_butce[sembol] += TEMINAT_GEREKSINIMLERI[sembol]
        if pnl < 0:
            self.kalan_islem_haklari[sembol] = max(0, self.kalan_islem_haklari.get(sembol, 0) - 1)
            self.kalan_parite_butce[sembol] = max(0, self.kalan_parite_butce.get(sembol, 0) - TEMINAT_GEREKSINIMLERI[sembol])
        logging.info(
            f"İşlem kapandı: {sembol}, neden: {reason}, PNL: {pnl:.2f}, "
            f"kalan teminat: {self.kalan_parite_butce[sembol]:.2f}, kalan hak: {self.kalan_islem_haklari[sembol]}"
        )

    def _tum_pozisyonlari_kapat(self, reason: str = 'Zorunlu'):
        for trade_id in list(self.pozisyonlar.keys()):
            self._pozisyonu_kapat(trade_id, reason,
                                 self.df_full_dict[self.pozisyonlar[trade_id]['sembol']]['close'].iloc[self.mevcut_adim])

    def _prime_zamani_mi(self) -> bool:
        return True

    def _sinyal_gucu_hesapla(self, sembol: str) -> float:
        try:
            rsi = self.ozellikler_dict[sembol]['rsi'].iloc[self.mevcut_adim]
            signal = (rsi - 50) / 50  # RSI tabanlı basit sinyal
            return np.clip(signal, -1.0, 1.0)
        except Exception:
            logging.warning(f"{sembol} için sinyal gücü hesaplanamadı, varsayılan: 0.5")
            return 0.5

    def _korelasyonlari_kontrol_et(self) -> bool:
        correlation_matrix = pd.DataFrame({s: self.df_full_dict[s]['close'] for s in self.semboller}).corr()
        for s1, s2 in itertools.combinations(self.semboller, 2):
            if correlation_matrix.loc[s1, s2] > 0.7:
                logging.info(f"Yüksek korelasyon: {s1}-{s2}: {correlation_matrix.loc[s1, s2]:.2f}")
                return False  # Aynı anda işlem açmayı engelle
        return True

    def _marjin_seviyesi_kontrol_et(self) -> bool:
        if not self.pozisyonlar:
            self.marjin_seviyesi = 100.0
            return True
        used_margin = sum(pos['lot_buyuklugu'] * 100000 * 2 for pos in self.pozisyonlar.values())
        self.marjin_seviyesi = (self.toplam_oz_sermaye / used_margin * 100) if used_margin > 0 else 100.0
        if self.marjin_seviyesi < 50:
            self.kayip_nedenleri.append(f"Marjin seviyesi düşük: {self.marjin_seviyesi:.2f}%")
            self._tum_pozisyonlari_kapat('Marjin Çağrısı')
            self.telegram.sync_send_message(f"🚨 Marjin Çağrısı: {self.marjin_seviyesi:.2f}%")
            return False
        return True

        # Birinci döngü: İşlem açma
        for sembol_idx, sembol in enumerate(self.semboller):
            action_value = float(actions[sembol_idx])
            # Trend veya breakout sinyali
            signal = self.trend_signal(sembol) or self.detect_breakout(sembol)

            # ATR ve volatilite hesapla
            atr = self._safe_atr(
                sembol=sembol,
                window=50,
                upper_k=3.0,
                default_pips=0.0010 if sembol in ["EURUSD", "GBPUSD"] else 0.10
            )
            mean_atr = np.nanmean(self.ozellikler_dict[sembol]['atr'].iloc[max(0, self.mevcut_adim-20):self.mevcut_adim+1])
            self.volatility = atr / mean_atr if mean_atr > 0 else 1.0

            # İşlem açma şartı
            if abs(action_value) >= 0.01 and signal in ('buy', 'sell'):
                lot_size = self._lot_buyuklugu_hesapla(
                    sembol,
                    atr,
                    abs(action_value)
                )

                # ---- HAK ve BÜTÇE KONTROLÜ BURADA ----
                if self.kalan_islem_haklari.get(sembol, 0) > 0 and self.kalan_parite_butce.get(sembol, 0) >= lot_size:
                    pnl = self._islem_simule_et(
                        sembol,
                        lot_size,
                        signal,
                        abs(action_value)
                    )
                    self.kalan_islem_haklari[sembol] -= 1
                    logging.info(f"{sembol} kalan hak: {self.kalan_islem_haklari[sembol]}, kalan bütçe: {self.kalan_parite_butce[sembol]:.2f}")
                    self.kalan_parite_butce[sembol] -= TEMINAT_GEREKSINIMLERI[sembol]

                    info['trades_opened'] += 1
                    info['lot_size'] += lot_size
                    logging.info(
                        f"İşlem açıldı: {sembol}, tip: {signal}, "
                        f"lot: {lot_size:.3f}, action_value: {action_value:.2f}, pnl: {pnl:.2f}"
                    )
                else:
                    logging.info(
                        f"{sembol} için yeterli hak veya bütçe yok. "
                        f"Kalan hak: {self.kalan_islem_haklari.get(sembol, 0)}, "
                        f"Kalan bütçe: {self.kalan_parite_butce.get(sembol, 0):.3f}"
                    )
            else:
                logging.info(
                    f"İşlem açılmadı: {sembol}, "
                    f"action_value: {action_value:.2f}, signal: {signal}"
                )

        self._update_open_positions(0)
        reward = self._calculate_reward(0)
        info['pnl'] = sum(
            t['pnl'] for t in self.islemler
            if t.get('env_idx', 0) == 0 and t.get('kapanis_adimi') == self.mevcut_adim
        )
        info['sharpe_ratio'] = self.calculate_sharpe_ratio(0)
        info['toplam_kar_zarar'] = sum(
            t['pnl'] for t in self.islemler if t.get('env_idx', 0) == 0
        )

        logging.info(f"[STEP] mevcut_adim: {self.mevcut_adim}")
        logging.info(
            f"[DATE] mevcut_adim: {self.mevcut_adim} | "
            f"Tarih: {self.df_full_dict[self.semboller[0]]['time'].iloc[self.mevcut_adim]}"
        )

        total_profit = 0.0
        reward = 0.0
        truncated = False

        if NEWS_API_KEY and self.mevcut_adim - self.last_news_check >= 10:
            if asyncio.run_coroutine_threadsafe(fetch_news(), self.telegram.loop).result():
                self.panik_butonu = True
                for sembol in self.semboller:
                    self.apply_hedging(sembol, action_value=0.0)
                    self.telegram.sync_send_message(f"🚨 Negatif Haber Tespit Edildi: {sembol}, hedging uygulanıyor")
                reward = -1000.0
                info = self._get_info(total_profit)
                logging.info("Negatif haber tespit edildi, hedging uygulandı")
                self.last_news_check = self.mevcut_adim
                return self._gozlem_al(), reward, self.bitti, truncated, self._get_info(total_profit)

        if self.mevcut_adim >= len(self.zaman_serisi_dict[self.semboller[0]]) - 2:
            self.bitti = True
            self._tum_pozisyonlari_kapat('Veri Sonu')
            obs = self._gozlem_al()
            info = self._get_info(total_profit)
            logging.info(f"Epizod bitti: Veri sonu, adımlar: {self.episode_steps}")
            return obs, reward, self.bitti, truncated, info

        if self.gunluk_kapali_pnl > self.gunluk_baslangic_bakiyesi * 0.03:
            self._tum_pozisyonlari_kapat('Günlük Kâr Kilidi')
            self.panik_butonu = True
            info = self._get_info(total_profit)
            logging.info("Günlük kâr kilidi tetiklendi")
            return self._gozlem_al(), reward, self.bitti, truncated, self._get_info(total_profit)

        if self.panik_butonu:
            self.islem_yok_sayisi += 1
            info = self._get_info(total_profit)
            logging.info("Panik butonu aktif, işlem yapılmadı")
            return self._gozlem_al(), reward, self.bitti, truncated, self._get_info(total_profit)

        if self.pozisyonlar and not self._marjin_seviyesi_kontrol_et():
            reward = -100.0
            self.bitti = True
            info = self._get_info(total_profit)
            logging.info("Marjin seviyesi düşük, epizod bitti")
            return self._gozlem_al(), reward, self.bitti, truncated, self._get_info(total_profit)

        if len(self.pozisyonlar) >= self.maks_acik_pozisyon:
            self.islem_yok_sayisi += 1
            info = self._get_info(total_profit)
            logging.info("Maksimum açık pozisyon sınırı, işlem yapılmadı")
            return self._gozlem_al(), reward, self.bitti, truncated, self._get_info(total_profit)

        # İkinci döngü: Diğer işlemler ve volatilite kontrolleri
        for sembol_idx, sembol in enumerate(self.semboller):
            action_value = float(actions[sembol_idx])
            signal_strength = self._sinyal_gucu_hesapla(sembol)

            # ATR ve volatilite hesapla
            atr = self._safe_atr(
                sembol=sembol,
                window=50,
                upper_k=3.0,
                default_pips=0.0010 if sembol in ["EURUSD", "GBPUSD"] else 0.10
            )
            mean_atr = np.nanmean(self.ozellikler_dict[sembol]['atr'].iloc[max(0, self.mevcut_adim-20):self.mevcut_adim+1])
            self.volatility = atr / mean_atr if mean_atr > 0 else 1.0

            # Volatilite uyarısı
            if self.volatility > 2.0:
                self.telegram.sync_send_message(f"⚠️ Yüksek Volatilite Tespit Edildi: {sembol}, volatilite: {self.volatility:.2f}")

            if self.detect_black_swan(sembol) or self.z_score_outlier(sembol):
                self.risk_orani = max(0.002, self.risk_orani * 0.3)
                logging.info(f"Risk oranı düşürüldü: {self.risk_orani:.4f}")

    def train_model(
        total_steps: int,
        semboller: List[str],
        n_envs: int,
        debug: bool,
        run_id: str,
        start_year: int,
        end_year: int,
        start_month: int,
        end_month: int,
    ) -> Dict[str, float]:
        df_dict: Dict[str, pd.DataFrame] = {}
        for s in semboller:
            parca_df_list = [csv_verisini_yukle(path) for path in CSV_FILES[s]]
            df = pd.concat(parca_df_list, ignore_index=True).sort_values('time').reset_index(drop=True)
            df_dict[s] = df  # Yıl filtresi kaldırılarak tüm veri kullanılır
        
        for s in semboller:
            df = df_dict[s]
            if start_year == end_year:
                mask = (
                    (df['time'].dt.year == start_year) &
                    (df['time'].dt.month >= start_month) &
                    (df['time'].dt.month <= end_month)
                )
            else:
                mask = (
                    ((df['time'].dt.year == start_year) & (df['time'].dt.month >= start_month)) |
                    ((df['time'].dt.year > start_year) & (df['time'].dt.year < end_year)) |
                    ((df['time'].dt.year == end_year) & (df['time'].dt.month <= end_month))
                )
            df = df.loc[mask].reset_index(drop=True)
            if df.empty:
                raise ValueError(f"{s} için {start_year}/{start_month}-{end_year}/{end_month} arasında veri bulunamadı")
            df_dict[s] = df
        
        takvim_df = takvim_verisini_yukle()
        ozellikler_dict, zaman_serisi_dict, olceklendirici_dict = {}, {}, {}
        for s in semboller:
            feat, times = ozellik_muhandisligi(df_dict[s], takvim_df, sembol=s)
            ozellikler_dict[s] = feat
            zaman_serisi_dict[s] = times
            olceklendirici_dict[s] = StandardScaler().fit(feat)
        
        config = {
            'semboller': semboller,
            'df': ozellikler_dict,
            'zaman_serisi': zaman_serisi_dict,
            'df_full': df_dict,
            'baslangic_bakiyesi': BASLANGIC_BAKIYESI,
            'temel_spread': TEMEL_SPREAD,
            'olceklendirici': olceklendirici_dict,
            'prime_zamani': PRIME_ZAMANI,
            'telegram_token': TELEGRAM_TOKEN,
            'telegram_kullanici_id': TELEGRAM_KULLANICI_ID,
            'total_timesteps': total_steps,
            'mod': 'train'
        }
        
        return train_and_evaluate(config, total_steps, n_envs, debug, run_id)

    def stress_test(
        self,
        start_year: int,
        end_year: int,
        start_month: int,
        end_month: int,
        semboller: List[str],
        adimlar: int,
        n_envs: int,
        debug: bool,
        run_id: str
    ) -> Tuple[float, int, int, int]:
        try:
            df_dict: Dict[str, pd.DataFrame] = {}
            for s in semboller:
                parca_df_list = [csv_verisini_yukle(path) for path in CSV_FILES[s]]
                df = pd.concat(parca_df_list, ignore_index=True).sort_values('time').reset_index(drop=True)
                df_dict[s] = df
            
            total_steps = adimlar
            years = range(start_year, end_year + 1)
            steps_per_year = total_steps // len(years)
            results = {}
            
            for year in years:
                config = {
                    'semboller': semboller,
                    'df': {},
                    'zaman_serisi': {},
                    'df_full': {},
                    'baslangic_bakiyesi': BASLANGIC_BAKIYESI,
                    'temel_spread': TEMEL_SPREAD,
                    'olceklendirici': {},
                    'prime_zamani': PRIME_ZAMANI,
                    'telegram_token': TELEGRAM_TOKEN,
                    'telegram_kullanici_id': TELEGRAM_KULLANICI_ID,
                    'total_timesteps': steps_per_year,
                    'mod': 'backtest'
                }
                for s in semboller:
                    df = df_dict[s]
                    mask = (df['time'].dt.year == year)
                    config['df_full'][s] = df.loc[mask].reset_index(drop=True)
                    if config['df_full'][s].empty:
                        logging.warning(f"{s} için {year} yılında veri bulunamadı, atlanıyor")
                        continue
                    feat, times = ozellik_muhandisligi(config['df_full'][s], takvim_verisini_yukle(), sembol=s)
                    config['df'][s] = feat
                    config['zaman_serisi'][s] = times
                    config['olceklendirici'][s] = StandardScaler().fit(feat)
                
                log_file = f"JTTWS/data/{year}_SAC_backtest.log"
                setup_logging(log_file)
                logging.info(f"{year} için SAC ile backtest başlıyor…")
                
                result = TicaretOrtami.train_and_evaluate(config, steps_per_year, n_envs, debug, f"{run_id}_{year}")
                results[f"{year}_SAC"] = {
                    'total_profit': result['total_profit'],
                    'fail_count': result['fail_count'],
                    'account_count': result['account_count'],
                    'episode_count': steps_per_year // 100  # Yaklaşık tahmin
                }
                logging.info(f"{year} için SAC backtest tamamlandı: Profit={result['total_profit']}, Fails={result['fail_count']}, Accounts={result['account_count']}, Episodes={steps_per_year // 100}")
            
            if not results:
                logging.error("Hiçbir sonuç üretilmedi")
                return 0.0, 0, 0, 0
            
            best_name, best_res = max(
                results.items(),
                key=lambda kv: kv[1]['total_profit'] - kv[1]['fail_count'] * 1000
            )
            logging.info(f"En iyi algoritma: {best_name}, Profit: {best_res['total_profit']}, Fails: {best_res['fail_count']}, Episodes: {best_res['episode_count']}")
            return best_res['total_profit'], best_res['fail_count'], best_res['account_count'], best_res['episode_count']
        
        except Exception as e:
            logging.error(f"Stres testi hatası: {e}")
            return 0.0, 0, 0, 0

def main():
    parser = argparse.ArgumentParser(description="JTTWS Ticaret Botu")
    parser.add_argument('--mod', type=str, default='train', choices=['train', 'backtest', 'live'],
                        help="Botun çalışma modu")
    parser.add_argument('--adimlar', type=int, default=100000,
                        help="Toplam eğitim/backtest adımı")
    parser.add_argument('--yil', type=str, default='2003-2024',
                        help="Tek yıl (örn. 2020) veya aralık (örn. 2003-2024)")
    parser.add_argument('--start-month', type=int, default=1,
                        help="Başlangıç ayı (1-12)")
    parser.add_argument('--end-month', type=int, default=12,
                        help="Bitiş ayı (1-12)")
    parser.add_argument('--debug', action='store_true',
                        help="Hata ayıklama modunu etkinleştir")
    args = parser.parse_args()

    # Yıl/aralık ayrıştırması
    if '-' in args.yil:
        start_year, end_year = map(int, args.yil.split('-', 1))
    else:
        start_year = end_year = int(args.yil)

    run_id = str(uuid.uuid4())
    configure(f"logs/run_{run_id}", ["stdout", "csv", "tensorboard"])

    # Verileri yükle ve filtrele
    df_dict = {}
    for s in SEMBOLLER:
        parca_df_list = [csv_verisini_yukle(path) for path in CSV_FILES[s]]
        df = pd.concat(parca_df_list, ignore_index=True).sort_values('time').reset_index(drop=True)
        if start_year == end_year:
            mask = (
                (df['time'].dt.year == start_year) &
                (df['time'].dt.month >= args.start_month) &
                (df['time'].dt.month <= args.end_month)
            )
        else:
            mask = (
                ((df['time'].dt.year == start_year) & (df['time'].dt.month >= args.start_month)) |
                ((df['time'].dt.year > start_year) & (df['time'].dt.year < end_year)) |
                ((df['time'].dt.year == end_year) & (df['time'].dt.month <= args.end_month))
            )
        df = df.loc[mask].reset_index(drop=True)
        if df.empty:
            logging.error(f"{s} için {start_year}/{args.start_month}-{end_year}/{args.end_month} arasında veri bulunamadı")
            sys.exit(1)
        df_dict[s] = df

    # Ekonomik takvim verisini yükle
    takvim_df = takvim_verisini_yukle()

    # Özellik mühendisliği
    ozellikler_dict, zaman_serisi_dict, olceklendirici_dict = {}, {}, {}
    for s in SEMBOLLER:
        feat, times = ozellik_muhandisligi(df_dict[s], takvim_df, sembol=s)  # sembol=s eklendi
        ozellikler_dict[s] = feat
        zaman_serisi_dict[s] = times
        olceklendirici_dict[s] = StandardScaler().fit(feat)

    # Config oluşturma
    config = {
        'semboller': SEMBOLLER,
        'df': ozellikler_dict,
        'zaman_serisi': zaman_serisi_dict,
        'df_full': df_dict,
        'baslangic_bakiyesi': BASLANGIC_BAKIYESI,
        'temel_spread': TEMEL_SPREAD,
        'olceklendirici': olceklendirici_dict,
        'prime_zamani': PRIME_ZAMANI,
        'telegram_token': TELEGRAM_TOKEN,
        'telegram_kullanici_id': TELEGRAM_KULLANICI_ID,
        'total_timesteps': args.adimlar,
        'mod': args.mod
    }

    if args.mod == 'backtest':
        # TicaretOrtami örneği oluştur
        env = TicaretOrtami(config)
        
        # stress_test metodunu çağır
        profit, fails, accounts, episode_count = env.stress_test(
            start_year=start_year,
            end_year=end_year,
            start_month=args.start_month,
            end_month=args.end_month,
            semboller=SEMBOLLER,
            adimlar=args.adimlar,
            n_envs=4,
            debug=args.debug,
            run_id=run_id
        )
        logging.info(f"Backtest tamamlandı: toplam_kar={profit}, başarısız={fails}, hesap={accounts}, epizod_sayısı={episode_count}")
        logging.info(f"Toplam epizod sayısı: {episode_count}")
        sys.exit(0)

    if args.mod == 'train':
        stats = train_model(
            total_steps=args.adimlar,
            semboller=SEMBOLLER,
            n_envs=4,
            debug=args.debug,
            run_id=run_id,
            start_year=start_year,
            end_year=end_year,
            start_month=args.start_month,
            end_month=args.end_month
        )
        logging.info(f"Eğitim tamamlandı: {stats}")
        sys.exit(0)

    elif args.mod == 'live':
        run_live(
            semboller=SEMBOLLER,
            debug=args.debug,
            run_id=run_id
        )
        sys.exit(0)

if __name__ == "__main__":
    main()
