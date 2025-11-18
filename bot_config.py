#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULTIMATE FTMO TRADING BOT V7.0 PROFESSIONAL - Configuration
Tüm ayarlar bu dosyada merkezi olarak yönetilir
"""

import os
from datetime import time
from pathlib import Path

class BotConfig:
    """Bot için tüm yapılandırma ayarları"""
    
    # ==================== GENEL AYARLAR ====================
    VERSION = "7.0-PROFESSIONAL"
    
    # Data klasörü - kullanıcının MacBook'undaki yol
    BASE_DIR = Path.home() / "Desktop" / "JTTWS"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    
    # ==================== TELEGRAM AYARLARI ====================
    TELEGRAM_TOKEN = "8008545474:AAHansC5Xag1b9N96bMAGE0YLTfykXoOPyY"
    TELEGRAM_CHAT_ID = 1590841427
    TELEGRAM_ENABLED = True  # False yaparsanız telegram bildirimleri kapalı olur
    
    # ==================== CURRENCY PAIRS ====================
    PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
    
    # Her pair için data klasörleri
    PAIR_DATA_PATHS = {
        "EURUSD": DATA_DIR / "EURUSD2003-2024",
        "GBPUSD": DATA_DIR / "GBPUSD2003-2024",
        "USDJPY": DATA_DIR / "USDJPY2003-2024"
    }
    
    # Weekly range CSV dosyaları
    WEEKLY_RANGE_FILES = {
        "EURUSD": DATA_DIR / "EURUSD_weekly_ranges.csv",
        "GBPUSD": DATA_DIR / "GBPUSD_weekly_ranges.csv",
        "USDJPY": DATA_DIR / "USDJPY_weekly_ranges.csv"
    }
    
    # ==================== TRADING HOURS (UTC+3) ====================
    # Yeni pozisyon açma saatleri
    TRADING_START_HOUR = 0  # 00:00
    TRADING_END_HOUR = 22   # 22:30'dan sonra yeni giriş yok
    TRADING_END_MINUTE = 30
    
    # Zorla kapatma saati
    FORCE_CLOSE_HOUR = 23   # 23:00'da tüm pozisyonlar kapanır
    FORCE_CLOSE_MINUTE = 0
    
    # ==================== RISK MANAGEMENT ====================
    # Başlangıç sermayesi
    INITIAL_CAPITAL = 100000.0  # $100,000
    
    # Günlük toplam risk limiti (sermayenin %'si)
    DAILY_RISK_LIMIT = 0.05  # %5
    
    # Pair başına maksimum risk (günlük bütçenin %'si)
    MAX_RISK_PER_PAIR = 0.33  # Her pair günlük bütçenin %33'ü
    
    # Position sizing
    MIN_LOT_SIZE = 0.01
    MAX_LOT_SIZE = 2.0
    DEFAULT_LOT_SIZE = 0.1
    
    # Kelly Criterion için
    KELLY_FRACTION = 0.25  # Kelly'nin 1/4'ünü kullan (güvenli)
    
    # Stop Loss / Take Profit (ATR multiplier)
    SL_ATR_MULTIPLIER = 2.0
    TP_ATR_MULTIPLIER = 3.0
    
    # VaR / CVaR
    VAR_CONFIDENCE = 0.95  # %95 güven aralığı
    CVAR_CONFIDENCE = 0.95
    
    # ==================== VOLATILITY GUARDS ====================
    # RangeGuard: Haftalık range'in p95'inden büyük ise giriş yapma
    RANGE_GUARD_PERCENTILE = 95  # p95
    
    # GapGuard: Açılış farkı ATR'nin kaç katı olursa giriş yapma
    GAP_GUARD_ATR_MULTIPLIER = 1.5
    
    # ShallowHour: Saatlik bar ATR'nin kaç katından küçükse giriş yapma
    SHALLOW_HOUR_ATR_MULTIPLIER = 0.5
    
    
    # ==================== EMAIL NOTIFICATIONS ====================
    EMAIL_ENABLED = True
    EMAIL_ADDRESS = "journeytothewallstreet@gmail.com"
    
    # Gmail App Password (2-factor auth gerektirir)
    # https://myaccount.google.com/apppasswords adresinden alın
    EMAIL_APP_PASSWORD = ""  # Buraya Gmail App Password gireceksiniz
    
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    # ==================== NEWS BLACKOUT ====================
    # Birleştirilmiş ekonomik takvim dosyası
    NEWS_CALENDAR_FILE = DATA_DIR / "combined_economic_calendar.csv"
    
    # Haber kategorilerine göre blackout süreleri (dakika)
    NEWS_BLACKOUT_CRITICAL_BEFORE = 60  # CRITICAL haberler öncesi 60 dk
    NEWS_BLACKOUT_CRITICAL_AFTER = 60   # CRITICAL haberler sonrası 60 dk
    
    NEWS_BLACKOUT_HIGH_BEFORE = 30      # HIGH haberler öncesi 30 dk
    NEWS_BLACKOUT_HIGH_AFTER = 30       # HIGH haberler sonrası 30 dk
    
    NEWS_BLACKOUT_MEDIUM_BEFORE = 15    # MEDIUM haberler öncesi 15 dk
    NEWS_BLACKOUT_MEDIUM_AFTER = 15     # MEDIUM haberler sonrası 15 dk
    
    # LOW impact haberler için blackout YOK
    # ==================== TREND & CORRELATION ====================
    # Trend filtresi için SMA periyotları
    TREND_SMA_FAST = 20
    TREND_SMA_SLOW = 50
    
    # Minimum trend gücü (0-1 arası)
    MIN_TREND_STRENGTH = 0.3
    
    # Distance filtresi: Mevcut fiyat SMA'dan kaç ATR uzakta olabilir?
    MAX_DISTANCE_FROM_SMA = 2.0  # ATR cinsinden
    
    # Korelasyon kontrolü: Maksimum aynı yöndeki pozisyon sayısı
    MAX_CORRELATED_POSITIONS = 2
    
    # ==================== SEQUENTIAL LOSS/PROFIT LOCK ====================
    # Art arda kaç kayıp olursa trading durur?
    SEQUENTIAL_LOSS_LIMIT = 3
    
    # Art arda kaç kar olursa daily profit'in %20'sine ulaşıldığında dur?
    SEQUENTIAL_WIN_PROFIT_THRESHOLD = 0.20  # Günlük profit hedefinin %20'si
    
    # ==================== HOURLY RIGHTS ALLOCATION ====================
    # Her saate tahsis edilecek "hak" sayısı
    HOURLY_RIGHTS = 3  # Her saat 3 işlem hakkı
    
    # ==================== THOMPSON SAMPLING ====================
    # Thompson bandit için alpha/beta başlangıç değerleri
    THOMPSON_ALPHA_INIT = 1.0
    THOMPSON_BETA_INIT = 1.0
    
    # Signal tipleri ve ağırlıkları
    SIGNAL_TYPES = ["TREND", "MEAN_REVERSION", "BREAKOUT", "MOMENTUM"]
    
    # ==================== FEATURE ENGINEERING ====================
    # Teknik göstergeler için periyotlar
    SMA_PERIODS = [20, 50, 200]
    EMA_PERIODS = [12, 26]
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    
    # ==================== REINFORCEMENT LEARNING ====================
    # Rainbow DQN parametreleri
    RL_LEARNING_RATE = 0.0001
    RL_GAMMA = 0.99  # Discount factor
    RL_EPSILON_START = 1.0
    RL_EPSILON_END = 0.01
    RL_EPSILON_DECAY = 0.995
    RL_BATCH_SIZE = 64
    RL_MEMORY_SIZE = 100000
    RL_TARGET_UPDATE = 1000  # Her kaç step'te target network güncellenir
    
    # LSTM için
    RL_LSTM_HIDDEN_SIZE = 128
    RL_LSTM_LAYERS = 2
    RL_SEQUENCE_LENGTH = 50  # Kaç bar geriye bakılır
    
    # ==================== BACKTEST AYARLARI ====================
    # Backtest için yıl aralığı
    BACKTEST_START_YEAR = 2020
    BACKTEST_END_YEAR = 2024
    
    # Training için yıl aralığı
    TRAIN_START_YEAR = 2003
    TRAIN_END_YEAR = 2019
    
    # ==================== LOGGING ====================
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ==================== PAPER TRADING (MT5) ====================
    # MetaTrader 5 bağlantı ayarları (kullanılacaksa)
    MT5_ENABLED = False  # True yaparsanız MT5'e bağlanır
    MT5_LOGIN = None
    MT5_PASSWORD = None
    MT5_SERVER = None
    
    @classmethod
    def validate(cls):
        """Konfigürasyonu doğrula ve gerekli klasörleri oluştur"""
        # Klasörleri oluştur
        for directory in [cls.LOGS_DIR, cls.MODELS_DIR, cls.OUTPUTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Data klasörünü kontrol et
        if not cls.DATA_DIR.exists():
            raise FileNotFoundError(
                f"Data klasörü bulunamadı: {cls.DATA_DIR}\n"
                f"Lütfen KULLANIM_KILAVUZU.md dosyasındaki talimatları takip edin."
            )
        
        # Pair data yollarını kontrol et
        for pair, path in cls.PAIR_DATA_PATHS.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"{pair} için data klasörü bulunamadı: {path}"
                )
        
        # Weekly range dosyalarını kontrol et
        for pair, file in cls.WEEKLY_RANGE_FILES.items():
            if not file.exists():
                raise FileNotFoundError(
                    f"{pair} için weekly range dosyası bulunamadı: {file}"
                )
        
        print("✅ Konfigürasyon doğrulandı ve tüm klasörler hazır!")
        return True


if __name__ == "__main__":
    # Test konfigürasyonu
    print("Bot Configuration V7.0")
    print("=" * 50)
    print(f"Base Directory: {BotConfig.BASE_DIR}")
    print(f"Data Directory: {BotConfig.DATA_DIR}")
    print(f"Trading Pairs: {BotConfig.PAIRS}")
    print(f"Telegram Enabled: {BotConfig.TELEGRAM_ENABLED}")
    print("=" * 50)
    
    try:
        BotConfig.validate()
    except FileNotFoundError as e:
        print(f"❌ Hata: {e}")
