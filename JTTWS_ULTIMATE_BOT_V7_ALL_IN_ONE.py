#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            JTTWS ULTIMATE FTMO TRADING BOT V7.0 PROFESSIONAL                ║
║                    "Clockwork Reliability, Maximum Transparency"             ║
║                          *** ALL-IN-ONE VERSION ***                          ║
║                                                                              ║
║   🎯 12-Point Professional Strategy | 🛡️ Advanced Risk Management          ║
║   📊 Full Data Integration (2003-2024) | 🤖 Rainbow DQN + LSTM              ║
║   📱 Telegram & Email Notifications | 🔒 Volatility Guards                  ║
║   📈 Enhanced Trade Logging | ⚡ News Blackout System                       ║
║                                                                              ║
║   Tüm modüller tek dosyada birleştirilmiştir - Bağımsız çalışır            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

JTTWS VERİ KAYNAKLARI:
=====================
1. EURUSD2003-2024/ klasörü - 21 yıllık EURUSD M1 veri (2003-2024)
2. GBPUSD2003-2024/ klasörü - 21 yıllık GBPUSD M1 veri (2003-2024)
3. USDJPY2003-2024/ klasörü - 21 yıllık USDJPY M1 veri (2003-2024)
4. EURUSD_weekly_ranges.csv - EURUSD haftalık range istatistikleri
5. GBPUSD_weekly_ranges.csv - GBPUSD haftalık range istatistikleri
6. USDJPY_weekly_ranges.csv - USDJPY haftalık range istatistikleri
7. combined_economic_calendar.csv - Birleştirilmiş ekonomik takvim verileri

BOT ÖZELLİKLERİ:
================
✓ 12 Noktalı Profesyonel Trading Stratejisi
✓ Rainbow DQN + LSTM Reinforcement Learning Agent
✓ Gelişmiş Risk Yönetimi (VaR, CVaR, Kelly Criterion)
✓ Haber Bazlı Blackout Sistemi (CRITICAL/HIGH/MEDIUM kategorileri)
✓ Volatilite Koruma Mekanizmaları (RangeGuard, GapGuard, ShallowHour)
✓ Telegram Push Notifications (Türkçe)
✓ Email Bildirimleri (HTML formatted)
✓ Detaylı Trade Logging (indikatörler, risk/reward, lot analizi)
✓ Haftalık Performans Raporları
✓ Backtest, Training ve Paper Trading modları

KULLANIM:
=========
# Backtest modu (geçmiş verilerde test)
python JTTWS_ULTIMATE_BOT_V7_ALL_IN_ONE.py --mode backtest

# Training modu (RL agent eğitimi)
python JTTWS_ULTIMATE_BOT_V7_ALL_IN_ONE.py --mode train --episodes 1000

# Paper trading modu (canlı veri, simülasyon)
python JTTWS_ULTIMATE_BOT_V7_ALL_IN_ONE.py --mode paper

NOT: Bot çalışmadan önce ~/Desktop/JTTWS/ klasörünü ve içindeki
     yukarıdaki veri dosyalarını kontrol edin!

Versiyon: 7.0-PROFESSIONAL-ALL-IN-ONE
Oluşturulma: 2024
"""


import sys
import os
import argparse
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import warnings

# Core libraries
import numpy as np
import pandas as pd
from scipy import stats
import pytz

# PyTorch for RL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Telegram
from telegram import Bot
from telegram.error import TelegramError
import asyncio

# Plotting
import matplotlib
matplotlib.use('Agg')  # Backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# Local config
from bot_config import BotConfig

# Enhanced modules
from email_notifier import EmailNotifier
from enhanced_trade_logger import EnhancedTradeLogger

# Local modules
from news_manager import NewsManager, create_blackout_config
from weekly_reporter import WeeklyReporter

warnings.filterwarnings('ignore')

# ============================================================================
# BOT CONFIGURATION (BotConfig sınıfı - tüm ayarlar)
# ============================================================================

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
    TELEGRAM_CHAT_ID = 1590841427  # @JourneyToTheWallStreet
    TELEGRAM_ENABLED = True  # False yaparsanız telegram bildirimleri kapalı olur
    
    # ==================== EMAIL AYARLARI ====================
    EMAIL_ENABLED = True  # False yaparsanız email bildirimleri kapalı olur
    EMAIL_ADDRESS = "your_email@gmail.com"  # Kullanıcı kendi email'ini girecek
    EMAIL_APP_PASSWORD = "vorw noth yfey efuz"  # Gmail App Password
    EMAIL_SMTP_SERVER = "smtp.gmail.com"
    EMAIL_SMTP_PORT = 587
    EMAIL_TO_ADDRESS = "your_email@gmail.com"  # Bildirimlerin gönderileceği adres
    
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
    LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR (geçici olarak DEBUG)
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



# ============================================================================
# EMAIL NOTIFIER (EmailNotifier sınıfı - email bildirimleri)
# ============================================================================

class EmailNotifier:
    """
    Email bildirimleri için modül
    Gmail SMTP kullanarak trade alerts ve raporlar gönderir
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Args:
            config: BotConfig instance
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = config.EMAIL_ENABLED
        
        if not self.enabled:
            self.logger.info("📧 Email notifications disabled")
            return
            
        # Email ayarlarını kontrol et
        if not config.EMAIL_APP_PASSWORD or config.EMAIL_APP_PASSWORD == "":
            self.logger.warning("⚠️ Email App Password yok! Email bildirimleri devre dışı.")
            self.enabled = False
            return
            
        self.smtp_server = config.EMAIL_SMTP_SERVER
        self.smtp_port = config.EMAIL_SMTP_PORT
        self.from_email = config.EMAIL_ADDRESS
        self.to_email = config.EMAIL_TO_ADDRESS
        self.password = config.EMAIL_APP_PASSWORD
        
        self.logger.info(f"✅ Email Notifier initialized: {self.from_email} -> {self.to_email}")
    
    def _send_email(self, subject: str, body: str, html: bool = True):
        """
        Email gönder
        
        Args:
            subject: Email konusu
            body: Email içeriği
            html: HTML formatında mı?
        """
        if not self.enabled:
            return
            
        try:
            # Email oluştur
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            msg['Subject'] = subject
            
            # Body ekle
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # SMTP ile gönder
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.from_email, self.password)
                server.send_message(msg)
            
            self.logger.debug(f"📧 Email sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"❌ Email send error: {e}")
    
    def send_trade_alert(self, pair: str, direction: str, lot_size: float, 
                        entry_price: float, sl: float, tp: float):
        """
        Yeni trade açıldı email'i
        
        Args:
            pair: Currency pair (EURUSD)
            direction: LONG veya SHORT
            lot_size: Lot büyüklüğü
            entry_price: Giriş fiyatı
            sl: Stop Loss
            tp: Take Profit
        """
        if not self.enabled:
            return
            
        arrow = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
        
        subject = f"🚀 TRADE OPENED: {arrow} {pair}"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: {'green' if direction == 'LONG' else 'red'};">
                {arrow} {pair}
            </h2>
            <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Direction:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{direction}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Lot Size:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{lot_size}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Entry Price:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{entry_price:.5f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Stop Loss:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{sl:.5f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Take Profit:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{tp:.5f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Time:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>
            <p style="color: #666; margin-top: 20px;">
                <i>JTTWS Bot V7.0 Professional</i>
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body, html=True)
    
    def send_trade_closed(self, pair: str, direction: str, profit: float, 
                         pips: float, duration: str):
        """
        Trade kapandı email'i
        
        Args:
            pair: Currency pair
            direction: LONG veya SHORT
            profit: Kar/Zarar ($)
            pips: Pip cinsinden kar/zarar
            duration: Trade süresi
        """
        if not self.enabled:
            return
            
        emoji = "✅" if profit > 0 else "❌"
        color = "green" if profit > 0 else "red"
        
        subject = f"{emoji} TRADE CLOSED: {pair} ({'+' if profit > 0 else ''}{profit:.2f}$)"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: {color};">
                {emoji} {pair} - Trade Closed
            </h2>
            <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Direction:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{direction}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Profit/Loss:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd; color: {color};">
                        <b>{'+' if profit > 0 else ''}{profit:.2f}$</b>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Pips:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{'+' if pips > 0 else ''}{pips:.1f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Duration:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{duration}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Time:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>
            <p style="color: #666; margin-top: 20px;">
                <i>JTTWS Bot V7.0 Professional</i>
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body, html=True)
    
    def send_weekly_report(self, report_data: dict):
        """
        Haftalık performans raporu email'i
        
        Args:
            report_data: Rapor verileri
        """
        if not self.enabled:
            return
            
        subject = f"📊 WEEKLY REPORT - Week {report_data.get('week_number', 'N/A')}"
        
        stats = report_data.get('stats', {})
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #2c3e50;">📊 Weekly Performance Report</h2>
            
            <h3>Overall Statistics</h3>
            <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Total Trades:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{stats.get('total_trades', 0)}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Win Rate:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{stats.get('win_rate', 0):.1f}%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Total Profit:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{stats.get('total_profit', 0):.2f}$</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Average Profit:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{stats.get('avg_profit', 0):.2f}$</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Max Drawdown:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{stats.get('max_drawdown', 0):.2f}$</td>
                </tr>
            </table>
            
            <p style="color: #666; margin-top: 30px;">
                <i>JTTWS Bot V7.0 Professional - Detailed report attached in Telegram</i>
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body, html=True)
    
    def send_error_alert(self, error_type: str, error_message: str):
        """
        Hata bildirimi email'i
        
        Args:
            error_type: Hata tipi
            error_message: Hata mesajı
        """
        if not self.enabled:
            return
            
        subject = f"⚠️ BOT ERROR: {error_type}"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #e74c3c;">⚠️ Bot Error Alert</h2>
            
            <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Error Type:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{error_type}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Error Message:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{error_message}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Time:</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>
            
            <p style="color: #666; margin-top: 20px;">
                <i>JTTWS Bot V7.0 Professional</i>
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body, html=True)


# ============================================================================
# ENHANCED TRADE LOGGER (EnhancedTradeLogger sınıfı - detaylı logging)
# ============================================================================

class EnhancedTradeLogger:
    """
    Detaylı trade logging için modül
    Her trade için şunları loglar:
    - Tüm teknik indikatör değerleri
    - Yakındaki önemli haberler
    - Lot hesaplama detayları
    - Risk/Reward oranı
    - Trend ve momentum analizi
    """
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        Args:
            config: BotConfig instance
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.trade_logs = []
        
        self.logger.info("✅ Enhanced Trade Logger initialized")
    
    def log_trade_open(self, pair: str, direction: str, bar: pd.Series, 
                      lot_size: float, entry_price: float, sl: float, tp: float,
                      indicators: Optional[Dict[str, Any]] = None,
                      nearby_news: Optional[list] = None,
                      risk_calculation: Optional[Dict[str, Any]] = None):
        """
        Trade açılışını detaylı logla
        
        Args:
            pair: Currency pair
            direction: LONG veya SHORT
            bar: Current price bar (pandas Series)
            lot_size: Lot büyüklüğü
            entry_price: Giriş fiyatı
            sl: Stop Loss
            tp: Take Profit
            indicators: Teknik indikatör değerleri
            nearby_news: Yakındaki haberler listesi
            risk_calculation: Risk hesaplama detayları
        """
        trade_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Temel bilgiler
        log_data = {
            'trade_id': trade_id,
            'timestamp': datetime.now(),
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'lot_size': lot_size,
            'status': 'OPEN'
        }
        
        # Price bar bilgileri
        if bar is not None:
            log_data['bar_open'] = bar.get('open', 0)
            log_data['bar_high'] = bar.get('high', 0)
            log_data['bar_low'] = bar.get('low', 0)
            log_data['bar_close'] = bar.get('close', 0)
            log_data['bar_volume'] = bar.get('volume', 0)
        
        # Teknik indikatörler
        if indicators:
            log_data['indicators'] = indicators
        
        # Yakındaki haberler
        if nearby_news:
            log_data['nearby_news'] = nearby_news
        
        # Risk hesaplaması
        if risk_calculation:
            log_data['risk_calc'] = risk_calculation
        
        # Risk/Reward hesapla
        sl_distance = abs(entry_price - sl)
        tp_distance = abs(entry_price - tp)
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        log_data['risk_reward_ratio'] = rr_ratio
        
        # Potential profit/loss
        log_data['potential_profit'] = tp_distance * 100000 * lot_size
        log_data['potential_loss'] = sl_distance * 100000 * lot_size
        
        # Kaydet
        self.trade_logs.append(log_data)
        
        # Console log
        self._print_trade_open(log_data)
        
        return trade_id
    
    def log_trade_close(self, trade_id: str, exit_price: float, profit: float, 
                       pips: float, close_reason: str, duration: str):
        """
        Trade kapanışını logla
        
        Args:
            trade_id: Trade ID
            exit_price: Çıkış fiyatı
            profit: Kar/Zarar ($)
            pips: Pip cinsinden kar/zarar
            close_reason: Kapanma sebebi (SL/TP/Manual)
            duration: Trade süresi
        """
        # Trade'i bul ve güncelle
        for trade in self.trade_logs:
            if trade['trade_id'] == trade_id:
                trade['status'] = 'CLOSED'
                trade['exit_price'] = exit_price
                trade['profit'] = profit
                trade['pips'] = pips
                trade['close_reason'] = close_reason
                trade['duration'] = duration
                trade['close_timestamp'] = datetime.now()
                
                # Console log
                self._print_trade_close(trade)
                break
    
    def _print_trade_open(self, log_data: dict):
        """Trade açılış bilgilerini konsola yazdır"""
        arrow = "🟢" if log_data['direction'] == "LONG" else "🔴"
        
        self.logger.info("=" * 70)
        self.logger.info(f"{arrow} TRADE OPENED - {log_data['pair']} {log_data['direction']}")
        self.logger.info("=" * 70)
        self.logger.info(f"Trade ID: {log_data['trade_id']}")
        self.logger.info(f"Entry: {log_data['entry_price']:.5f}")
        self.logger.info(f"SL: {log_data['sl']:.5f} | TP: {log_data['tp']:.5f}")
        self.logger.info(f"Lot Size: {log_data['lot_size']}")
        self.logger.info(f"Risk/Reward: 1:{log_data['risk_reward_ratio']:.2f}")
        self.logger.info(f"Potential Profit: ${log_data['potential_profit']:.2f}")
        self.logger.info(f"Potential Loss: ${log_data['potential_loss']:.2f}")
        
        # Indikatörler
        if 'indicators' in log_data:
            self.logger.info("--- Indicators ---")
            for key, value in log_data['indicators'].items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        # Haberler
        if 'nearby_news' in log_data and log_data['nearby_news']:
            self.logger.info("--- Nearby News ---")
            for news in log_data['nearby_news']:
                self.logger.info(f"  {news}")
        
        # Risk hesaplaması
        if 'risk_calc' in log_data:
            self.logger.info("--- Risk Calculation ---")
            for key, value in log_data['risk_calc'].items():
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 70)
    
    def _print_trade_close(self, log_data: dict):
        """Trade kapanış bilgilerini konsola yazdır"""
        emoji = "✅" if log_data['profit'] > 0 else "❌"
        
        self.logger.info("=" * 70)
        self.logger.info(f"{emoji} TRADE CLOSED - {log_data['pair']}")
        self.logger.info("=" * 70)
        self.logger.info(f"Trade ID: {log_data['trade_id']}")
        self.logger.info(f"Direction: {log_data['direction']}")
        self.logger.info(f"Entry: {log_data['entry_price']:.5f} | Exit: {log_data['exit_price']:.5f}")
        self.logger.info(f"Close Reason: {log_data['close_reason']}")
        self.logger.info(f"Duration: {log_data['duration']}")
        self.logger.info(f"Profit: ${log_data['profit']:.2f} ({'+' if log_data['pips'] > 0 else ''}{log_data['pips']:.1f} pips)")
        self.logger.info("=" * 70)
    
    def get_trade_stats(self) -> dict:
        """Trade istatistiklerini döndür"""
        if not self.trade_logs:
            return {}
        
        closed_trades = [t for t in self.trade_logs if t['status'] == 'CLOSED']
        
        if not closed_trades:
            return {'total_trades': len(self.trade_logs), 'open_trades': len(self.trade_logs)}
        
        total_profit = sum(t['profit'] for t in closed_trades)
        winning_trades = [t for t in closed_trades if t['profit'] > 0]
        losing_trades = [t for t in closed_trades if t['profit'] <= 0]
        
        return {
            'total_trades': len(self.trade_logs),
            'closed_trades': len(closed_trades),
            'open_trades': len(self.trade_logs) - len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0,
            'total_profit': total_profit,
            'avg_profit': total_profit / len(closed_trades) if closed_trades else 0,
            'avg_win': sum(t['profit'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
        }
    
    def export_to_csv(self, filepath: str):
        """Trade loglarını CSV'ye aktar"""
        if not self.trade_logs:
            self.logger.warning("No trades to export")
            return
        
        # DataFrame'e çevir
        df = pd.DataFrame(self.trade_logs)
        
        # Nested dict'leri string'e çevir
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        df.to_csv(filepath, index=False)
        self.logger.info(f"✅ Trade logs exported to {filepath}")
    
    def get_all_trades(self) -> list:
        """Tüm trade loglarını döndür"""
        return self.trade_logs


# ============================================================================
# NEWS MANAGER (NewsManager sınıfı - haber yönetimi)
# ============================================================================

class NewsManager:
    """
    Gelişmiş haber yönetim sistemi
    - Haber kategorilerine göre farklı blackout süreleri
    - Haber bazlı volatilite profili
    - Detaylı loglama
    """
    
    def __init__(self, calendar_file: Path):
        """
        Args:
            calendar_file: Combined economic calendar CSV file
        """
        self.calendar_file = calendar_file
        self.calendar_df = None
        self.news_stats = {}
        self.load_calendar()
    
    def load_calendar(self):
        """Load and prepare economic calendar"""
        try:
            if not self.calendar_file.exists():
                logger.warning(f"Calendar file not found: {self.calendar_file}")
                logger.warning("NewsBlackout will be DISABLED")
                return
            
            self.calendar_df = pd.read_csv(self.calendar_file)
            
            # Parse datetime column
            if 'datetime' not in self.calendar_df.columns:
                logger.error("Calendar file missing 'datetime' column")
                return
            
            self.calendar_df['datetime'] = pd.to_datetime(self.calendar_df['datetime'])
            
            # Validate required columns
            required_cols = ['datetime', 'Name', 'Impact', 'Currency', 'Category']
            missing_cols = [col for col in required_cols if col not in self.calendar_df.columns]
            if missing_cols:
                logger.error(f"Calendar missing columns: {missing_cols}")
                return
            
            # Statistics
            total_events = len(self.calendar_df)
            categories = self.calendar_df['Category'].value_counts().to_dict()
            
            logger.info("=" * 60)
            logger.info("NEWS MANAGER INITIALIZED")
            logger.info("=" * 60)
            logger.info(f"Total events: {total_events:,}")
            logger.info(f"Date range: {self.calendar_df['datetime'].min()} to {self.calendar_df['datetime'].max()}")
            logger.info(f"Categories:")
            for cat, count in sorted(categories.items()):
                pct = (count / total_events * 100)
                logger.info(f"  {cat:10s}: {count:6,d} events ({pct:5.1f}%)")
            
            # Build news statistics (volatility profiles will be calculated during training)
            self._build_news_stats()
            
            logger.info("✓ News Manager ready!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error loading calendar: {e}")
            self.calendar_df = None
    
    def _build_news_stats(self):
        """Build statistics for each news type"""
        if self.calendar_df is None:
            return
        
        # Group by news name and category
        for name in self.calendar_df['Name'].unique():
            news_events = self.calendar_df[self.calendar_df['Name'] == name]
            category = news_events['Category'].iloc[0]
            currencies = news_events['Currency'].unique().tolist()
            
            self.news_stats[name] = {
                'category': category,
                'currencies': currencies,
                'count': len(news_events),
                'avg_volatility': None,  # Will be calculated during training
                'win_rate_after': None,   # Will be calculated during training
            }
    
    def is_blackout_period(
        self, 
        current_time: datetime, 
        currency: str,
        blackout_config: Dict[str, int]
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if current time is in a news blackout period
        
        Args:
            current_time: Current datetime
            currency: Currency to check (USD, EUR, GBP, JPY)
            blackout_config: Dictionary with blackout minutes for each category
                Example: {
                    'CRITICAL_BEFORE': 60,
                    'CRITICAL_AFTER': 60,
                    'HIGH_BEFORE': 30,
                    'HIGH_AFTER': 30,
                    'MEDIUM_BEFORE': 15,
                    'MEDIUM_AFTER': 15
                }
        
        Returns:
            (is_blackout, event_info)
            - is_blackout: True if in blackout period
            - event_info: Dict with event details if in blackout, else None
        """
        if self.calendar_df is None:
            return False, None
        
        # Filter events for this currency
        currency_events = self.calendar_df[self.calendar_df['Currency'] == currency].copy()
        
        if currency_events.empty:
            return False, None
        
        # Check each category
        for category in ['CRITICAL', 'HIGH', 'MEDIUM']:
            before_key = f'{category}_BEFORE'
            after_key = f'{category}_AFTER'
            
            if before_key not in blackout_config or after_key not in blackout_config:
                continue
            
            before_minutes = blackout_config[before_key]
            after_minutes = blackout_config[after_key]
            
            # Filter events of this category
            cat_events = currency_events[currency_events['Category'] == category]
            
            for _, event in cat_events.iterrows():
                event_time = event['datetime']
                
                # Check if we're in blackout window
                start_blackout = event_time - timedelta(minutes=before_minutes)
                end_blackout = event_time + timedelta(minutes=after_minutes)
                
                if start_blackout <= current_time <= end_blackout:
                    time_to_event = (event_time - current_time).total_seconds() / 60
                    
                    return True, {
                        'category': category,
                        'name': event['Name'],
                        'event_time': event_time,
                        'time_to_event_minutes': time_to_event,
                        'currency': currency,
                        'before_minutes': before_minutes,
                        'after_minutes': after_minutes
                    }
        
        return False, None
    
    def get_upcoming_news(
        self, 
        current_time: datetime, 
        currency: str, 
        lookahead_hours: int = 24
    ) -> List[Dict]:
        """
        Get upcoming news events for a currency
        
        Args:
            current_time: Current datetime
            currency: Currency code
            lookahead_hours: How many hours ahead to look
        
        Returns:
            List of upcoming news events
        """
        if self.calendar_df is None:
            return []
        
        end_time = current_time + timedelta(hours=lookahead_hours)
        
        upcoming = self.calendar_df[
            (self.calendar_df['Currency'] == currency) &
            (self.calendar_df['datetime'] >= current_time) &
            (self.calendar_df['datetime'] <= end_time)
        ].sort_values('datetime')
        
        events = []
        for _, event in upcoming.iterrows():
            events.append({
                'name': event['Name'],
                'datetime': event['datetime'],
                'category': event['Category'],
                'impact': event['Impact'],
                'hours_until': (event['datetime'] - current_time).total_seconds() / 3600
            })
        
        return events
    
    def get_news_at_time(self, target_time: datetime, currency: str, window_minutes: int = 60) -> List[Dict]:
        """
        Get news events around a specific time
        
        Args:
            target_time: Time to check
            currency: Currency code
            window_minutes: Window size (before and after)
        
        Returns:
            List of news events in the window
        """
        if self.calendar_df is None:
            return []
        
        start_time = target_time - timedelta(minutes=window_minutes)
        end_time = target_time + timedelta(minutes=window_minutes)
        
        events = self.calendar_df[
            (self.calendar_df['Currency'] == currency) &
            (self.calendar_df['datetime'] >= start_time) &
            (self.calendar_df['datetime'] <= end_time)
        ]
        
        result = []
        for _, event in events.iterrows():
            result.append({
                'name': event['Name'],
                'datetime': event['datetime'],
                'category': event['Category'],
                'impact': event['Impact'],
                'minutes_diff': (event['datetime'] - target_time).total_seconds() / 60
            })
        
        return result
    
    def log_news_impact(self, trade_time: datetime, currency: str, result: str, pnl: float):
        """
        Log the impact of news on a trade (for learning)
        
        Args:
            trade_time: When the trade was opened/closed
            currency: Currency pair
            result: 'win' or 'loss'
            pnl: Profit/loss amount
        """
        # Get news around this time
        nearby_news = self.get_news_at_time(trade_time, currency, window_minutes=120)
        
        if nearby_news:
            logger.debug(f"Trade at {trade_time} | {currency} | {result} | PnL: ${pnl:.2f}")
            logger.debug(f"  Nearby news events:")
            for news in nearby_news:
                logger.debug(f"    - {news['name']} ({news['category']}) at {news['datetime']} ({news['minutes_diff']:.0f}m)")
    
    def get_statistics_summary(self) -> Dict:
        """Get summary statistics about the calendar"""
        if self.calendar_df is None:
            return {}
        
        return {
            'total_events': len(self.calendar_df),
            'categories': self.calendar_df['Category'].value_counts().to_dict(),
            'currencies': self.calendar_df['Currency'].value_counts().to_dict(),
            'date_range': (
                self.calendar_df['datetime'].min(),
                self.calendar_df['datetime'].max()
            ),
            'unique_news_types': len(self.news_stats)
        }


# Convenience function for creating blackout config
def create_blackout_config(critical_before=60, critical_after=60,
                          high_before=30, high_after=30,
                          medium_before=15, medium_after=15):
    """Helper function to create blackout configuration"""
    return {
        'CRITICAL_BEFORE': critical_before,
        'CRITICAL_AFTER': critical_after,
        'HIGH_BEFORE': high_before,
        'HIGH_AFTER': high_after,
        'MEDIUM_BEFORE': medium_before,
        'MEDIUM_AFTER': medium_after,
    }


# ============================================================================
# WEEKLY REPORTER (WeeklyReporter sınıfı - haftalık raporlar)
# ============================================================================

class WeeklyReporter:
    """
    Haftalık performans raporu oluşturur
    - Parite bazlı kar/zarar
    - Haber bazlı reaksiyon analizi
    - Lot analizi
    - Win/loss pattern'leri
    """
    
    def __init__(self):
        self.trade_history = []
        self.news_impacts = defaultdict(list)
        self.current_week_start = None
    
    def add_trade(self, trade_data: Dict):
        """
        Add a completed trade to history
        
        Args:
            trade_data: Dictionary containing:
                - pair: str
                - entry_time: datetime
                - exit_time: datetime
                - direction: 'LONG' or 'SHORT'
                - lot_size: float
                - entry_price: float
                - exit_price: float
                - pnl: float
                - result: 'WIN' or 'LOSS'
                - strategy_type: str (TREND, BREAKOUT, etc.)
                - nearby_news: List[Dict] (optional)
        """
        self.trade_history.append(trade_data)
        
        # Track news impacts
        if 'nearby_news' in trade_data and trade_data['nearby_news']:
            for news in trade_data['nearby_news']:
                key = f"{news['name']}_{news['category']}"
                self.news_impacts[key].append({
                    'pair': trade_data['pair'],
                    'result': trade_data['result'],
                    'pnl': trade_data['pnl'],
                    'time_to_news': news['minutes_diff']
                })
    
    def generate_weekly_report(self, week_start: datetime = None) -> Dict:
        """
        Generate comprehensive weekly report
        
        Args:
            week_start: Start of the week (if None, uses last 7 days)
        
        Returns:
            Dictionary with report data
        """
        if week_start is None:
            week_start = datetime.now() - timedelta(days=7)
        
        week_end = week_start + timedelta(days=7)
        
        # Filter trades for this week
        week_trades = [
            t for t in self.trade_history
            if week_start <= t['exit_time'] < week_end
        ]
        
        if not week_trades:
            logger.warning(f"No trades found for week starting {week_start.date()}")
            return {}
        
        report = {
            'week_start': week_start,
            'week_end': week_end,
            'total_trades': len(week_trades),
            'pairs': self._analyze_pairs(week_trades),
            'news_reactions': self._analyze_news_reactions(week_trades),
            'lot_analytics': self._analyze_lots(week_trades),
            'time_analytics': self._analyze_time_patterns(week_trades),
            'strategy_performance': self._analyze_strategies(week_trades),
            'overall_metrics': self._calculate_overall_metrics(week_trades)
        }
        
        return report
    
    def _analyze_pairs(self, trades: List[Dict]) -> Dict:
        """Analyze performance by currency pair"""
        pair_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'total_lots': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.0
        })
        
        for trade in trades:
            pair = trade['pair']
            stats = pair_stats[pair]
            
            stats['trades'] += 1
            stats['total_pnl'] += trade['pnl']
            stats['total_lots'] += trade['lot_size']
            
            if trade['result'] == 'WIN':
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            
            if trade['pnl'] > stats['best_trade']:
                stats['best_trade'] = trade['pnl']
            if trade['pnl'] < stats['worst_trade']:
                stats['worst_trade'] = trade['pnl']
        
        # Calculate averages and win rates
        for pair, stats in pair_stats.items():
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
            stats['win_rate'] = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        
        # Sort by total PnL
        sorted_pairs = dict(sorted(pair_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True))
        
        return sorted_pairs
    
    def _analyze_news_reactions(self, trades: List[Dict]) -> Dict:
        """Analyze how news events affected trades"""
        news_stats = defaultdict(lambda: {
            'trades_affected': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'category': 'UNKNOWN'
        })
        
        for trade in trades:
            if 'nearby_news' not in trade or not trade['nearby_news']:
                continue
            
            for news in trade['nearby_news']:
                key = news['name']
                stats = news_stats[key]
                
                stats['trades_affected'] += 1
                stats['category'] = news['category']
                stats['total_pnl'] += trade['pnl']
                
                if trade['result'] == 'WIN':
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1
        
        # Calculate metrics
        for news_name, stats in news_stats.items():
            if stats['trades_affected'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['trades_affected'] * 100)
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades_affected']
        
        # Sort by trades affected
        sorted_news = dict(sorted(news_stats.items(), key=lambda x: x[1]['trades_affected'], reverse=True))
        
        return sorted_news
    
    def _analyze_lots(self, trades: List[Dict]) -> Dict:
        """Analyze lot sizing patterns"""
        lot_sizes = [t['lot_size'] for t in trades]
        pnls = [t['pnl'] for t in trades]
        
        # Correlation between lot size and PnL
        correlation = np.corrcoef(lot_sizes, pnls)[0, 1] if len(lot_sizes) > 1 else 0
        
        # Group by lot size ranges
        lot_ranges = {
            '0.01-0.05': [],
            '0.05-0.10': [],
            '0.10-0.20': [],
            '0.20-0.50': [],
            '0.50+': []
        }
        
        for trade in trades:
            lot = trade['lot_size']
            if lot < 0.05:
                lot_ranges['0.01-0.05'].append(trade)
            elif lot < 0.10:
                lot_ranges['0.05-0.10'].append(trade)
            elif lot < 0.20:
                lot_ranges['0.10-0.20'].append(trade)
            elif lot < 0.50:
                lot_ranges['0.20-0.50'].append(trade)
            else:
                lot_ranges['0.50+'].append(trade)
        
        lot_range_stats = {}
        for range_name, range_trades in lot_ranges.items():
            if range_trades:
                wins = sum(1 for t in range_trades if t['result'] == 'WIN')
                total_pnl = sum(t['pnl'] for t in range_trades)
                lot_range_stats[range_name] = {
                    'trades': len(range_trades),
                    'win_rate': (wins / len(range_trades) * 100),
                    'total_pnl': total_pnl,
                    'avg_pnl': total_pnl / len(range_trades)
                }
        
        return {
            'min_lot': min(lot_sizes),
            'max_lot': max(lot_sizes),
            'avg_lot': np.mean(lot_sizes),
            'median_lot': np.median(lot_sizes),
            'lot_pnl_correlation': correlation,
            'lot_ranges': lot_range_stats
        }
    
    def _analyze_time_patterns(self, trades: List[Dict]) -> Dict:
        """Analyze time-based patterns"""
        hour_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0.0})
        day_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0.0})
        
        for trade in trades:
            hour = trade['entry_time'].hour
            day = trade['entry_time'].strftime('%A')
            
            hour_stats[hour]['trades'] += 1
            hour_stats[hour]['total_pnl'] += trade['pnl']
            if trade['result'] == 'WIN':
                hour_stats[hour]['wins'] += 1
            
            day_stats[day]['trades'] += 1
            day_stats[day]['total_pnl'] += trade['pnl']
            if trade['result'] == 'WIN':
                day_stats[day]['wins'] += 1
        
        # Calculate win rates
        for hour, stats in hour_stats.items():
            stats['win_rate'] = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        
        for day, stats in day_stats.items():
            stats['win_rate'] = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        
        # Find best and worst hours
        best_hour = max(hour_stats.items(), key=lambda x: x[1]['total_pnl'])
        worst_hour = min(hour_stats.items(), key=lambda x: x[1]['total_pnl'])
        
        return {
            'hourly': dict(hour_stats),
            'daily': dict(day_stats),
            'best_hour': {'hour': best_hour[0], **best_hour[1]},
            'worst_hour': {'hour': worst_hour[0], **worst_hour[1]}
        }
    
    def _analyze_strategies(self, trades: List[Dict]) -> Dict:
        """Analyze performance by strategy type"""
        strategy_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_pnl': 0.0
        })
        
        for trade in trades:
            strategy = trade.get('strategy_type', 'UNKNOWN')
            stats = strategy_stats[strategy]
            
            stats['trades'] += 1
            stats['total_pnl'] += trade['pnl']
            if trade['result'] == 'WIN':
                stats['wins'] += 1
        
        # Calculate metrics
        for strategy, stats in strategy_stats.items():
            if stats['trades'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['trades'] * 100)
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
        
        return dict(strategy_stats)
    
    def _calculate_overall_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate overall performance metrics"""
        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        losses = len(trades) - wins
        
        win_pnls = [t['pnl'] for t in trades if t['result'] == 'WIN']
        loss_pnls = [t['pnl'] for t in trades if t['result'] == 'LOSS']
        
        return {
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / len(trades) * 100) if trades else 0,
            'total_pnl': total_pnl,
            'avg_win': np.mean(win_pnls) if win_pnls else 0,
            'avg_loss': np.mean(loss_pnls) if loss_pnls else 0,
            'profit_factor': abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else 0,
            'largest_win': max(win_pnls) if win_pnls else 0,
            'largest_loss': min(loss_pnls) if loss_pnls else 0
        }
    
    def format_report_text(self, report: Dict) -> str:
        """Format report as readable text for Telegram"""
        if not report:
            return "❌ Rapor oluşturulamadı - veri yok"
        
        text = f"""
📊 HAFTALIK PERFORMANS RAPORU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Tarih: {report['week_start'].strftime('%d/%m/%Y')} - {report['week_end'].strftime('%d/%m/%Y')}

💰 GENEL PERFORMANS
  • Toplam Trade: {report['overall_metrics']['total_trades']}
  • Kazanan: {report['overall_metrics']['wins']} ({report['overall_metrics']['win_rate']:.1f}%)
  • Kaybeden: {report['overall_metrics']['losses']}
  • Toplam PnL: ${report['overall_metrics']['total_pnl']:.2f}
  • Profit Factor: {report['overall_metrics']['profit_factor']:.2f}
  • Ortalama Kazanç: ${report['overall_metrics']['avg_win']:.2f}
  • Ortalama Kayıp: ${report['overall_metrics']['avg_loss']:.2f}

📈 PARİTE BAZLI PERFORMANS
"""
        
        for pair, stats in report['pairs'].items():
            emoji = "🟢" if stats['total_pnl'] > 0 else "🔴"
            text += f"""{emoji} {pair}
  • Trade: {stats['trades']} | Win Rate: {stats['win_rate']:.1f}%
  • PnL: ${stats['total_pnl']:.2f} | Avg: ${stats['avg_pnl']:.2f}
  • En İyi: ${stats['best_trade']:.2f} | En Kötü: ${stats['worst_trade']:.2f}
  • Toplam Lot: {stats['total_lots']:.2f}

"""
        
        # Top news reactions
        if report['news_reactions']:
            text += "📰 EN ÇOK ETKİLEYEN HABERLER (Top 5)\n"
            top_news = list(report['news_reactions'].items())[:5]
            for news_name, stats in top_news:
                emoji = "⚠️" if stats['category'] == 'CRITICAL' else "📌"
                text += f"""{emoji} {news_name} ({stats['category']})
  • Etkilenen Trade: {stats['trades_affected']}
  • Win Rate: {stats['win_rate']:.1f}%
  • Avg PnL: ${stats['avg_pnl']:.2f}

"""
        
        # Lot analytics
        lot_stats = report['lot_analytics']
        text += f"""📊 LOT ANALİZİ
  • Min: {lot_stats['min_lot']:.2f} | Max: {lot_stats['max_lot']:.2f}
  • Ortalama: {lot_stats['avg_lot']:.2f} | Medyan: {lot_stats['median_lot']:.2f}
  • Lot-PnL Korelasyon: {lot_stats['lot_pnl_correlation']:.2f}

"""
        
        # Best/worst trading hours
        time_stats = report['time_analytics']
        text += f"""⏰ ZAMAN ANALİZİ
  • En İyi Saat: {time_stats['best_hour']['hour']}:00 
    ({time_stats['best_hour']['trades']} trade, ${time_stats['best_hour']['total_pnl']:.2f})
  • En Kötü Saat: {time_stats['worst_hour']['hour']}:00
    ({time_stats['worst_hour']['trades']} trade, ${time_stats['worst_hour']['total_pnl']:.2f})

"""
        
        text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n✅ Rapor oluşturma: " + datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        
        return text


# ============================================================================
# ANA BOT KODLARI (Trading system, RL agent, environment, vs.)
# ============================================================================

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup structured logging"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"bot_v7_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("FTMO_Bot_V7")
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# 1. RIGHTS MANAGER (Günlük Bütçe & Hak Yönetimi)
# ============================================================================

class RightsManager:
    """
    Günlük risk bütçesini ve trading haklarını yönetir.
    - Toplam günlük risk limiti
    - Pair başına adil dağılım
    - Harcanan/kalan bütçe takibi
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Günlük bütçe
        self.daily_budget = config.INITIAL_CAPITAL * config.DAILY_RISK_LIMIT
        self.remaining_budget = self.daily_budget
        
        # Pair başına bütçe
        self.pair_budgets = {
            pair: self.daily_budget * config.MAX_RISK_PER_PAIR
            for pair in config.PAIRS
        }
        self.pair_remaining = self.pair_budgets.copy()
        
        # Kullanım geçmişi
        self.usage_history = []
        
        self.logger.info(f"💰 RightsManager: Daily budget ${self.daily_budget:.2f}")
    
    def can_trade(self, pair: str, risk_amount: float) -> bool:
        """Bir trade için bütçe var mı?"""
        if self.remaining_budget < risk_amount:
            self.logger.warning(f"⛔ Günlük bütçe aşıldı! Kalan: ${self.remaining_budget:.2f}")
            return False
        
        if self.pair_remaining[pair] < risk_amount:
            self.logger.warning(f"⛔ {pair} bütçesi aşıldı! Kalan: ${self.pair_remaining[pair]:.2f}")
            return False
        
        return True
    
    def allocate(self, pair: str, risk_amount: float):
        """Bütçeden ayır"""
        self.remaining_budget -= risk_amount
        self.pair_remaining[pair] -= risk_amount
        self.usage_history.append({
            'pair': pair,
            'risk': risk_amount,
            'timestamp': datetime.now()
        })
        self.logger.debug(f"💸 {pair} için ${risk_amount:.2f} ayrıldı. Kalan: ${self.remaining_budget:.2f}")
    
    def reset_daily(self):
        """Günlük bütçeyi sıfırla"""
        self.remaining_budget = self.daily_budget
        self.pair_remaining = self.pair_budgets.copy()
        self.usage_history = []
        self.logger.info(f"🔄 Günlük bütçe sıfırlandı: ${self.daily_budget:.2f}")
    
    def get_status(self) -> Dict:
        """Durum özeti"""
        return {
            'daily_budget': self.daily_budget,
            'remaining_budget': self.remaining_budget,
            'usage_pct': (1 - self.remaining_budget / self.daily_budget) * 100,
            'pair_remaining': self.pair_remaining,
            'total_allocated': sum(h['risk'] for h in self.usage_history)
        }


# ============================================================================
# 2. WEEKLY RANGE LEARNER (CSV'den haftalık range öğrenme)
# ============================================================================

class WeeklyRangeLearner:
    """
    _weekly_ranges.csv dosyasından haftalık range verilerini okur.
    Her pair için istatistikler hesaplar (avg, p95, max).
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.range_data = {}
        self.stats = {}
        
        self._load_all_ranges()
    
    def _load_all_ranges(self):
        """Tüm pair'lerin weekly range'lerini yükle"""
        for pair in self.config.PAIRS:
            csv_file = self.config.WEEKLY_RANGE_FILES[pair]
            
            if not csv_file.exists():
                self.logger.warning(f"⚠️ {pair} weekly range dosyası bulunamadı: {csv_file}")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                self.range_data[pair] = df
                
                # İstatistikler
                self.stats[pair] = {
                    'avg_range': df['range'].mean(),
                    'std_range': df['range'].std(),
                    'p95_range': df['range'].quantile(0.95),
                    'p99_range': df['range'].quantile(0.99),
                    'max_range': df['range'].max()
                }
                
                self.logger.info(
                    f"📈 {pair} Weekly Ranges: avg={self.stats[pair]['avg_range']:.1f} pips, "
                    f"p95={self.stats[pair]['p95_range']:.1f} pips"
                )
            
            except Exception as e:
                self.logger.error(f"❌ {pair} weekly range yüklenemedi: {e}")
    
    def get_current_week_range(self, pair: str, current_date: datetime) -> Optional[float]:
        """Belirli bir tarihteki haftalık range'i al"""
        if pair not in self.range_data:
            return None
        
        df = self.range_data[pair]
        # Haftanın başını bul (Pazartesi)
        week_start = current_date - timedelta(days=current_date.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # En yakın haftalık veriyi bul
        try:
            row = df.loc[week_start]
            return row['range']
        except KeyError:
            # Tam eşleşme yoksa en yakın önceki hafta
            before = df[df.index <= week_start]
            if not before.empty:
                return before.iloc[-1]['range']
            return None
    
    def get_p95_threshold(self, pair: str) -> float:
        """RangeGuard için p95 threshold"""
        if pair in self.stats:
            return self.stats[pair]['p95_range']
        return float('inf')  # Veri yoksa sınırsız


# ============================================================================
# 3. NEWS BLACKOUT (Haber vakitleri filtresi)
# ============================================================================

class NewsBlackout:
    """
    Yüksek etkili haber saatlerinde trading yapma.
    - Haber öncesi/sonrası blackout periyodu
    - Opsiyonel: news_calendar.csv entegrasyonu
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.blackout_periods = []
        
        self._load_news_calendar()
    
    def _load_news_calendar(self):
        """Haber takvimini yükle (opsiyonel)"""
        calendar_file = self.config.NEWS_CALENDAR_FILE
        
        if not calendar_file.exists():
            self.logger.info("📰 News calendar dosyası yok, manuel filtre kullanılacak.")
            return
        
        try:
            df = pd.read_csv(calendar_file)
            df['time'] = pd.to_datetime(df['time'])
            
            # Yüksek impact haberler
            high_impact = df[df['impact'] == 'HIGH']
            
            for _, row in high_impact.iterrows():
                news_time = row['time']
                start = news_time - timedelta(minutes=self.config.NEWS_BLACKOUT_BEFORE)
                end = news_time + timedelta(minutes=self.config.NEWS_BLACKOUT_AFTER)
                self.blackout_periods.append((start, end))
            
            self.logger.info(f"📰 {len(self.blackout_periods)} haber blackout periyodu yüklendi.")
        
        except Exception as e:
            self.logger.error(f"❌ News calendar yüklenemedi: {e}")
    
    def is_blackout(self, current_time: datetime) -> bool:
        """Şu an blackout periyodu mu?"""
        for start, end in self.blackout_periods:
            if start <= current_time <= end:
                self.logger.warning(f"🔇 NEWS BLACKOUT: {start} - {end}")
                return True
        return False


# ============================================================================
# 4. VOLATILITY GUARDS (3 koruma mekanizması)
# ============================================================================

class VolatilityGuards:
    """
    3 volatilite koruması:
    1. RangeGuard: Haftalık range > p95
    2. GapGuard: Açılış farkı > 1.5x ATR
    3. ShallowHour: Saatlik bar < 0.5x ATR
    """
    
    def __init__(self, config: BotConfig, range_learner: WeeklyRangeLearner, logger: logging.Logger):
        self.config = config
        self.range_learner = range_learner
        self.logger = logger
    
    def range_guard(self, pair: str, current_week_range: float) -> bool:
        """Haftalık range p95'ten büyükse False döner"""
        threshold = self.range_learner.get_p95_threshold(pair)
        
        if current_week_range > threshold:
            self.logger.warning(
                f"🛡️ RANGE GUARD: {pair} haftalık range ({current_week_range:.1f} pips) "
                f"> p95 ({threshold:.1f} pips)"
            )
            return False
        return True
    
    def gap_guard(self, pair: str, gap_size: float, atr: float) -> bool:
        """Açılış gap'i 1.5x ATR'den büyükse False döner"""
        threshold = atr * self.config.GAP_GUARD_ATR_MULTIPLIER
        
        if abs(gap_size) > threshold:
            self.logger.warning(
                f"🛡️ GAP GUARD: {pair} gap ({gap_size:.5f}) > {self.config.GAP_GUARD_ATR_MULTIPLIER}x ATR ({threshold:.5f})"
            )
            return False
        return True
    
    def shallow_hour_guard(self, pair: str, hourly_range: float, atr: float) -> bool:
        """Saatlik bar range 0.5x ATR'den küçükse False döner"""
        threshold = atr * self.config.SHALLOW_HOUR_ATR_MULTIPLIER
        
        if hourly_range < threshold:
            self.logger.warning(
                f"🛡️ SHALLOW HOUR: {pair} hourly range ({hourly_range:.5f}) < {self.config.SHALLOW_HOUR_ATR_MULTIPLIER}x ATR ({threshold:.5f})"
            )
            return False
        return True


# ============================================================================
# 5. TREND FILTER (Trend yönü & distance)
# ============================================================================

class TrendFilter:
    """
    Trend filtreleme:
    - SMA20 vs SMA50 (fast vs slow)
    - ADX > threshold (trend gücü)
    - Distance: Fiyat SMA'dan max 2 ATR uzakta
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def check_trend(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Trend var mı? Yönü ne?
        Returns: (trend_valid, direction)
        """
        last = df.iloc[-1]
        
        sma_fast = last.get(f'SMA_{self.config.TREND_SMA_FAST}', None)
        sma_slow = last.get(f'SMA_{self.config.TREND_SMA_SLOW}', None)
        adx = last.get('ADX', None)
        
        if sma_fast is None or sma_slow is None or adx is None:
            return False, "NONE"
        
        # ADX kontrolü
        if adx < (self.config.MIN_TREND_STRENGTH * 100):  # ADX 0-100 arası
            self.logger.debug(f"📉 Trend zayıf: ADX={adx:.1f}")
            return False, "NONE"
        
        # Yön kontrolü
        if sma_fast > sma_slow:
            direction = "UP"
        else:
            direction = "DOWN"
        
        return True, direction
    
    def check_distance(self, df: pd.DataFrame) -> bool:
        """Fiyat SMA'dan çok uzakta mı?"""
        last = df.iloc[-1]
        
        close = last['close']
        sma = last.get(f'SMA_{self.config.TREND_SMA_FAST}', close)
        atr = last.get('ATR', 0.0001)
        
        distance = abs(close - sma)
        max_distance = atr * self.config.MAX_DISTANCE_FROM_SMA
        
        if distance > max_distance:
            self.logger.warning(
                f"🎯 DISTANCE: Fiyat SMA'dan çok uzak ({distance:.5f} > {max_distance:.5f})"
            )
            return False
        
        return True


# ============================================================================
# 6. CORRELATION CONTROL (Portföy korelasyon kontrolü)
# ============================================================================

class CorrelationControl:
    """
    Aynı yönde maksimum 2 pozisyon kuralı.
    EURUSD long + GBPUSD long = OK
    EURUSD long + GBPUSD long + USDJPY long = HAYIR
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.open_positions = {}  # {pair: direction}
    
    def update_positions(self, positions: Dict[str, str]):
        """Açık pozisyonları güncelle"""
        self.open_positions = positions
    
    def can_open(self, pair: str, direction: str) -> bool:
        """Yeni pozisyon açılabilir mi?"""
        same_direction_count = sum(
            1 for p, d in self.open_positions.items()
            if d == direction and p != pair
        )
        
        if same_direction_count >= self.config.MAX_CORRELATED_POSITIONS:
            self.logger.warning(
                f"🔗 CORRELATION: Aynı yönde ({direction}) zaten {same_direction_count} pozisyon var!"
            )
            return False
        
        return True
    
    def add_position(self, pair: str, direction: str):
        """Pozisyon ekle"""
        self.open_positions[pair] = direction
    
    def remove_position(self, pair: str):
        """Pozisyon kaldır"""
        if pair in self.open_positions:
            del self.open_positions[pair]


# ============================================================================
# 7. HOURLY ALLOCATOR (Saatlik hak tahsisi)
# ============================================================================

class HourlyAllocator:
    """
    Her saate 3 işlem hakkı tahsis eder.
    Haklar biterse o saatte yeni giriş yok.
    Saat başında haklar yenilenir.
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.hourly_rights = self.config.HOURLY_RIGHTS
        self.current_hour = None
        self.remaining_rights = self.hourly_rights
    
    def check_and_consume(self, current_time: datetime) -> bool:
        """Hak var mı? Varsa kullan."""
        hour = current_time.hour
        
        # Saat değiştiyse hakları yenile
        if hour != self.current_hour:
            self.current_hour = hour
            self.remaining_rights = self.hourly_rights
            self.logger.info(f"⏰ Yeni saat: {hour}:00 - {self.hourly_rights} hak tahsis edildi.")
        
        # Hak kontrolü
        if self.remaining_rights <= 0:
            self.logger.warning(f"⛔ HOURLY LIMIT: Bu saatte hak kalmadı!")
            return False
        
        # Hak kullan
        self.remaining_rights -= 1
        self.logger.debug(f"✅ Hak kullanıldı. Kalan: {self.remaining_rights}")
        return True


# ============================================================================
# 8. THOMPSON BANDIT (Sinyal seçici)
# ============================================================================

class ThompsonBandit:
    """
    Thompson Sampling ile en iyi sinyal tipini seçer.
    4 sinyal tipi: TREND, MEAN_REVERSION, BREAKOUT, MOMENTUM
    Her sinyal başarı/başarısızlık kaydedilir.
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Alpha/Beta parametreleri (Beta distribution)
        self.alpha = {sig: self.config.THOMPSON_ALPHA_INIT for sig in self.config.SIGNAL_TYPES}
        self.beta = {sig: self.config.THOMPSON_BETA_INIT for sig in self.config.SIGNAL_TYPES}
    
    def select_signal(self) -> str:
        """Thompson Sampling ile sinyal seç"""
        samples = {
            sig: np.random.beta(self.alpha[sig], self.beta[sig])
            for sig in self.config.SIGNAL_TYPES
        }
        
        selected = max(samples, key=samples.get)
        self.logger.debug(f"🎲 Thompson: {selected} seçildi (samples: {samples})")
        return selected
    
    def update(self, signal_type: str, success: bool):
        """Sonucu kaydet"""
        if success:
            self.alpha[signal_type] += 1
            self.logger.debug(f"✅ {signal_type} başarılı! Alpha: {self.alpha[signal_type]}")
        else:
            self.beta[signal_type] += 1
            self.logger.debug(f"❌ {signal_type} başarısız! Beta: {self.beta[signal_type]}")
    
    def get_stats(self) -> Dict:
        """İstatistikleri al"""
        return {
            sig: {
                'alpha': self.alpha[sig],
                'beta': self.beta[sig],
                'win_rate': self.alpha[sig] / (self.alpha[sig] + self.beta[sig])
            }
            for sig in self.config.SIGNAL_TYPES
        }


# ============================================================================
# 9. TELEGRAM REPORTER (Detaylı Türkçe raporlama)
# ============================================================================

class TelegramReporter:
    """
    Zengin formatlı Türkçe Telegram bildirimleri:
    - Trade açılışı/kapanışı
    - Günlük özet
    - Performans metrikleri
    - Uyarılar
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.bot = None
        self.chat_ids = []
        
        if self.config.TELEGRAM_ENABLED:
            self._initialize_bot()
    
    def _initialize_bot(self):
        """Telegram bot'u başlat"""
        try:
            self.bot = Bot(token=self.config.TELEGRAM_TOKEN)
            self.logger.info("📱 Telegram bot başlatıldı.")
        except Exception as e:
            self.logger.error(f"❌ Telegram bot hatası: {e}")
            self.config.TELEGRAM_ENABLED = False
    
    async def _send_message(self, message: str):
        """Tüm chat ID'lere mesaj gönder"""
        if not self.config.TELEGRAM_ENABLED or not self.bot:
            return
        
        # Eğer chat_id yoksa, genel broadcast (user /start ile eklenmelidir)
        if not self.chat_ids:
            self.logger.warning("📱 Telegram chat_id yok. /start gönderin.")
            return
        
        for chat_id in self.chat_ids:
            try:
                await self.bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')
            except TelegramError as e:
                self.logger.error(f"❌ Telegram mesaj hatası (chat {chat_id}): {e}")
    
    def send_trade_opened(self, pair: str, direction: str, lot_size: float, entry_price: float, sl: float, tp: float):
        """Yeni trade açıldı bildirimi"""
        arrow = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
        message = (
            f"{arrow} <b>{pair}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📈 Giriş: {entry_price:.5f}\n"
            f"🛡️ SL: {sl:.5f}\n"
            f"🎯 TP: {tp:.5f}\n"
            f"💰 Lot: {lot_size:.2f}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        asyncio.run(self._send_message(message))
    
    def send_trade_closed(self, pair: str, direction: str, profit: float, pips: float, duration: str):
        """Trade kapandı bildirimi"""
        emoji = "✅" if profit > 0 else "❌"
        message = (
            f"{emoji} <b>{pair}</b> Kapatıldı\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💵 Kar/Zarar: ${profit:.2f}\n"
            f"📊 Pip: {pips:.1f}\n"
            f"⏱️ Süre: {duration}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        asyncio.run(self._send_message(message))
    
    def send_daily_summary(self, summary: Dict):
        """Günlük özet"""
        message = (
            f"📊 <b>GÜNLÜK ÖZET</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 Net P&L: ${summary['net_profit']:.2f}\n"
            f"📈 Toplam Trade: {summary['total_trades']}\n"
            f"✅ Kazanan: {summary['winning_trades']}\n"
            f"❌ Kaybeden: {summary['losing_trades']}\n"
            f"🎯 Win Rate: {summary['win_rate']:.1f}%\n"
            f"📉 Max Drawdown: ${summary['max_drawdown']:.2f}\n"
            f"📊 Sharpe Ratio: {summary['sharpe_ratio']:.2f}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        asyncio.run(self._send_message(message))
    
    def send_warning(self, warning_text: str):
        """Uyarı mesajı"""
        message = f"⚠️ <b>UYARI</b>\n{warning_text}"
        asyncio.run(self._send_message(message))
    
    def add_chat_id(self, chat_id: int):
        """Yeni chat ID ekle"""
        if chat_id not in self.chat_ids:
            self.chat_ids.append(chat_id)
            self.logger.info(f"📱 Telegram chat eklendi: {chat_id}")


# ============================================================================
# 10. DATA MANAGER (Veri yükleme & feature engineering)
# ============================================================================

class DataManager:
    """
    CSV'lerden veri yükler, feature'lar hesaplar.
    2003-2024 tam veri desteği, özel yıl aralıkları.
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data = {}  # {pair: DataFrame}
    
    def load_data(self, pairs: List[str], start_year: int, end_year: int) -> Dict[str, pd.DataFrame]:
        """Belirtilen yıl aralığında veriyi yükle - Geliştirilmiş format desteği"""
        self.logger.info(f"📂 Veri yükleniyor: {start_year}-{end_year}")
        
        for pair in pairs:
            data_path = self.config.PAIR_DATA_PATHS[pair]
            
            self.logger.info(f"🔍 {pair}: Klasör kontrol ediliyor: {data_path}")
            
            if not data_path.exists():
                self.logger.error(f"❌ {pair} data klasörü bulunamadı: {data_path}")
                continue
            
            # Tüm CSV dosyalarını bul
            csv_files = sorted(data_path.glob("*Candlestick*.csv"))
            
            self.logger.info(f"📁 {pair}: {len(csv_files)} CSV dosyası bulundu")
            
            if not csv_files:
                self.logger.warning(f"⚠️ {pair}: Hiç CSV dosyası bulunamadı!")
                continue
            
            dfs = []
            for csv_file in csv_files:
                self.logger.info(f"  🔎 İnceleniyor: {csv_file.name}")
                try:
                    # HER DOSYAYI YÜKLE, SONRA TARİH FİLTRELE (daha basit ve güvenilir)
                    df = pd.read_csv(csv_file)
                    
                    # Sütun adlarını normalize et (küçük harf, boşluksuz)
                    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
                    
                    # Time sütunu yoksa local_time'ı time'a çevir
                    if 'time' not in df.columns and 'local_time' in df.columns:
                        df.rename(columns={'local_time': 'time'}, inplace=True)
                    
                    # Time sütununu datetime'a çevir
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                    
                    # NaT satırları temizle
                    df = df.dropna(subset=['time'])
                    
                    # Yıl filtreleme (veriyi yıl bazında filtrele)
                    df = df[(df['time'].dt.year >= start_year) & (df['time'].dt.year <= end_year)]
                    
                    if len(df) > 0:
                        dfs.append(df)
                        self.logger.info(f"  ✅ {csv_file.name}: {len(df)} bars ({df['time'].min().year}-{df['time'].max().year})")
                    else:
                        self.logger.debug(f"  ⊘ {csv_file.name}: İstenen yıl aralığında veri yok")
                
                except Exception as e:
                    import traceback
                    self.logger.error(f"  ❌ {csv_file.name}: {type(e).__name__}: {e}")
                    self.logger.debug(traceback.format_exc())
                    continue
            
            if dfs:
                # Tüm dataframe'leri birleştir
                combined = pd.concat(dfs, ignore_index=True)
                combined.sort_values('time', inplace=True)
                combined.drop_duplicates(subset='time', keep='first', inplace=True)
                combined.reset_index(drop=True, inplace=True)
                
                # 15M verisi varsa 1H'a resample et (opsiyonel, daha hızlı çalışır)
                # Timeframe kontrolü
                if len(combined) > 50000:  # Çok fazla bar varsa (15M olabilir)
                    self.logger.info(f"  ⚙️  {pair}: {len(combined)} bars, 1H'a resample ediliyor...")
                    combined.set_index('time', inplace=True)
                    combined_1h = combined.resample('1H').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    combined_1h.reset_index(inplace=True)
                    combined = combined_1h
                    self.logger.info(f"  ✅ Resample: {len(combined)} bars (1H)")
                
                self.data[pair] = combined
                self.logger.info(f"✅ {pair}: {len(combined)} bars ({start_year}-{end_year})")
            else:
                self.logger.warning(f"⚠️ {pair}: Veri bulunamadı!")
        
        return self.data
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Teknik göstergeler ve feature'lar ekle"""
        df = df.copy()
        
        # Simple Moving Averages
        for period in self.config.SMA_PERIODS:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in self.config.EMA_PERIODS:
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = df['close'].ewm(span=self.config.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.config.MACD_SLOW, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=self.config.MACD_SIGNAL, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        sma_bb = df['close'].rolling(window=self.config.BB_PERIOD).mean()
        std_bb = df['close'].rolling(window=self.config.BB_PERIOD).std()
        df['BB_upper'] = sma_bb + (std_bb * self.config.BB_STD)
        df['BB_lower'] = sma_bb - (std_bb * self.config.BB_STD)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=self.config.ATR_PERIOD).mean()
        
        # ADX (Average Directional Index)
        df['ADX'] = self._calculate_adx(df, self.config.ADX_PERIOD)
        
        # Volatility (rolling std)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Time-based features
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        
        # Price change
        df['returns'] = df['close'].pct_change()
        
        # Drop NaN
        df.dropna(inplace=True)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """ADX hesapla"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx


# ============================================================================
# 11. RISK MANAGER (VaR, CVaR, Kelly Criterion, Position Sizing)
# ============================================================================

class RiskManager:
    """
    Gelişmiş risk yönetimi:
    - VaR / CVaR
    - Kelly Criterion
    - ATR bazlı SL/TP
    - Dinamik position sizing
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.returns_history = deque(maxlen=500)
    
    def calculate_var_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """VaR ve CVaR hesapla"""
        if len(returns) < 10:
            return 0.0, 0.0
        
        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()
        
        return var, cvar
    
    def kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly Criterion ile lot hesapla"""
        if avg_loss == 0 or win_rate == 0:
            return self.config.DEFAULT_LOT_SIZE
        
        win_loss_ratio = abs(avg_win / avg_loss)
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Kelly'nin 1/4'ünü kullan (güvenli)
        kelly_fraction = kelly * self.config.KELLY_FRACTION
        
        # Lot size'a çevir
        lot_size = kelly_fraction * self.config.MAX_LOT_SIZE
        lot_size = np.clip(lot_size, self.config.MIN_LOT_SIZE, self.config.MAX_LOT_SIZE)
        
        return round(lot_size, 2)
    
    def calculate_sl_tp(self, entry_price: float, direction: str, atr: float) -> Tuple[float, float]:
        """ATR bazlı SL ve TP hesapla"""
        sl_distance = atr * self.config.SL_ATR_MULTIPLIER
        tp_distance = atr * self.config.TP_ATR_MULTIPLIER
        
        if direction == "LONG":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # SHORT
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        return round(sl, 5), round(tp, 5)
    
    def update_returns(self, trade_return: float):
        """Trade return'unu kaydet"""
        self.returns_history.append(trade_return)
    
    def get_risk_metrics(self) -> Dict:
        """Risk metriklerini al"""
        if len(self.returns_history) < 10:
            return {}
        
        returns = np.array(self.returns_history)
        var, cvar = self.calculate_var_cvar(returns)
        
        return {
            'var_95': var,
            'cvar_95': cvar,
            'avg_return': returns.mean(),
            'std_return': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
        }


# ============================================================================
# 12. SEQUENTIAL LOCK (Art arda kayıp/kar kilidi)
# ============================================================================

class SequentialLock:
    """
    - 3 art arda kayıp → trading durdur
    - Günlük profit hedefinin %20'sine ulaşıldığında art arda 2 kar → dur
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.recent_trades = deque(maxlen=5)
        self.daily_profit_target = config.INITIAL_CAPITAL * config.DAILY_RISK_LIMIT * 2  # Hedef: risk'in 2 katı
        self.current_daily_profit = 0.0
    
    def add_trade_result(self, profit: float):
        """Trade sonucunu kaydet"""
        self.recent_trades.append(profit)
        self.current_daily_profit += profit
    
    def is_locked(self) -> bool:
        """Trading kilitli mi?"""
        if len(self.recent_trades) < 3:
            return False
        
        # 3 art arda kayıp kontrolü
        last_3 = list(self.recent_trades)[-3:]
        if all(p < 0 for p in last_3):
            self.logger.warning("🔒 SEQUENTIAL LOCK: 3 art arda kayıp! Trading durduruldu.")
            return True
        
        # Profit lock: Günlük hedefin %20'sine ulaşıldıysa ve son 2 trade kar
        profit_threshold = self.daily_profit_target * self.config.SEQUENTIAL_WIN_PROFIT_THRESHOLD
        if self.current_daily_profit >= profit_threshold:
            last_2 = list(self.recent_trades)[-2:]
            if len(last_2) == 2 and all(p > 0 for p in last_2):
                self.logger.warning("🔒 PROFIT LOCK: Günlük hedef %20'ye ulaşıldı ve 2 art arda kar! Dur.")
                return True
        
        return False
    
    def reset_daily(self):
        """Günlük kilit sıfırla"""
        self.recent_trades.clear()
        self.current_daily_profit = 0.0


# ============================================================================
# 13. TRADING ENVIRONMENT (Gym-like environment for RL)
# ============================================================================

class TradingEnvironment:
    """
    Reinforcement Learning için trading environment.
    State, action, reward.
    """
    
    def __init__(self, df: pd.DataFrame, config: BotConfig, logger: logging.Logger, 
                 pair: str = "UNKNOWN", email_notifier=None, trade_logger=None):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.logger = logger
        self.pair = pair
        self.email_notifier = email_notifier
        self.trade_logger = trade_logger
        
        self.current_step = 0
        self.max_steps = len(df) - 1
        
        # State space (features sayısı)
        self.state_size = self._get_state_size()
        
        # Action space: 0=HOLD, 1=LONG, 2=SHORT
        self.action_size = 3
        
        # Portfolio
        self.balance = config.INITIAL_CAPITAL
        self.position = None  # {'type': 'LONG'/'SHORT', 'entry': price, 'sl': price, 'tp': price, 'lot': size}
        
        self.total_profit = 0.0
        self.trade_history = []
    
    def _get_state_size(self) -> int:
        """State boyutunu hesapla"""
        # Numerik feature'ları say
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # 'time' hariç
        state_cols = [c for c in numeric_cols if c not in ['time']]
        return len(state_cols)
    
    def reset(self):
        """Environment'i sıfırla"""
        self.current_step = self.config.RL_SEQUENCE_LENGTH  # İlk N bar'ı skip et
        self.balance = self.config.INITIAL_CAPITAL
        self.position = None
        self.total_profit = 0.0
        self.trade_history = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Mevcut state'i al"""
        # Son RL_SEQUENCE_LENGTH bar'ın feature'larını al
        start = max(0, self.current_step - self.config.RL_SEQUENCE_LENGTH)
        end = self.current_step + 1
        
        window = self.df.iloc[start:end]
        
        # Numerik feature'ları seç
        numeric_cols = window.select_dtypes(include=[np.number]).columns
        state_cols = [c for c in numeric_cols if c not in ['time']]
        
        state = window[state_cols].values
        
        # Padding (eğer sequence yeterince uzun değilse)
        if len(state) < self.config.RL_SEQUENCE_LENGTH:
            padding = np.zeros((self.config.RL_SEQUENCE_LENGTH - len(state), state.shape[1]))
            state = np.vstack([padding, state])
        
        # Normalize (basit z-score)
        state = (state - state.mean(axis=0)) / (state.std(axis=0) + 1e-8)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Bir adım at.
        Returns: (next_state, reward, done, info)
        """
        current_bar = self.df.iloc[self.current_step]
        current_price = current_bar['close']
        atr = current_bar.get('ATR', 0.0001)
        
        reward = 0.0
        done = False
        
        # Mevcut pozisyon var mı, SL/TP kontrolü
        if self.position is not None:
            reward = self._check_position(current_bar)
        
        # Yeni action
        if action == 1 and self.position is None:
            # LONG aç
            self._open_position('LONG', current_price, atr)
        elif action == 2 and self.position is None:
            # SHORT aç
            self._open_position('SHORT', current_price, atr)
        # action == 0: HOLD (hiçbir şey yapma)
        
        # Sonraki step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        next_state = self._get_state()
        
        info = {
            'balance': self.balance,
            'total_profit': self.total_profit,
            'position': self.position
        }
        
        return next_state, reward, done, info
    
    def _open_position(self, pos_type: str, entry_price: float, atr: float):
        """Pozisyon aç"""
        sl_distance = atr * self.config.SL_ATR_MULTIPLIER
        tp_distance = atr * self.config.TP_ATR_MULTIPLIER
        
        if pos_type == 'LONG':
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        lot_size = self.config.DEFAULT_LOT_SIZE
        
        self.position = {
            'type': pos_type,
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'lot': lot_size,
            'open_step': self.current_step
        }
        
        # Enhanced logging ve notifications
        if self.trade_logger:
            # Mevcut bar'ı al
            current_bar = self.df.iloc[self.current_step] if self.current_step < len(self.df) else None
            
            # Indikatör değerlerini topla
            indicators = {}
            if current_bar is not None:
                indicator_cols = ['RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 
                                'ATR', 'ADX', 'SMA_20', 'SMA_50', 'SMA_200']
                for col in indicator_cols:
                    if col in current_bar:
                        indicators[col] = current_bar[col]
            
            # Trade'i logla
            trade_id = self.trade_logger.log_trade_open(
                pair=self.pair,
                direction=pos_type,
                bar=current_bar,
                lot_size=lot_size,
                entry_price=entry_price,
                sl=sl,
                tp=tp,
                indicators=indicators
            )
            
            # Trade ID'yi position'a ekle
            self.position['trade_id'] = trade_id
        
        # Email notification
        if self.email_notifier:
            self.email_notifier.send_trade_alert(
                pair=self.pair,
                direction=pos_type,
                lot_size=lot_size,
                entry_price=entry_price,
                sl=sl,
                tp=tp
            )
        
        # Console log
        self.logger.info(f"📊 TRADE AÇILDI - {pos_type} {self.pair} @ {entry_price:.5f}")
    
    def _check_position(self, bar: pd.Series) -> float:
        """Pozisyonu kontrol et, SL/TP tetiklenirse kapat"""
        if self.position is None:
            return 0.0
        
        high = bar['high']
        low = bar['low']
        close = bar['close']
        
        pos_type = self.position['type']
        entry = self.position['entry']
        sl = self.position['sl']
        tp = self.position['tp']
        
        profit = 0.0
        close_reason = None
        
        if pos_type == 'LONG':
            if low <= sl:
                # SL hit
                profit = (sl - entry) * 100000 * self.position['lot']  # Pip cinsinden
                close_reason = 'SL'
            elif high >= tp:
                # TP hit
                profit = (tp - entry) * 100000 * self.position['lot']
                close_reason = 'TP'
        else:  # SHORT
            if high >= sl:
                # SL hit
                profit = (entry - sl) * 100000 * self.position['lot']
                close_reason = 'SL'
            elif low <= tp:
                # TP hit
                profit = (entry - tp) * 100000 * self.position['lot']
                close_reason = 'TP'
        
        if close_reason:
            self.balance += profit
            self.total_profit += profit
            
            # Calculate pips and duration
            pips = profit / self.position['lot']
            duration_bars = self.current_step - self.position['open_step']
            duration_str = f"{duration_bars} bars"
            
            # Exit price
            exit_price = sl if close_reason == 'SL' else tp
            
            # Enhanced logging
            if self.trade_logger and 'trade_id' in self.position:
                self.trade_logger.log_trade_close(
                    trade_id=self.position['trade_id'],
                    exit_price=exit_price,
                    profit=profit,
                    pips=pips,
                    close_reason=close_reason,
                    duration=duration_str
                )
            
            # Email notification
            if self.email_notifier:
                self.email_notifier.send_trade_closed(
                    pair=self.pair,
                    direction=pos_type,
                    profit=profit,
                    pips=pips,
                    duration=duration_str
                )
            
            self.trade_history.append({
                'entry_step': self.position['open_step'],
                'exit_step': self.current_step,
                'type': pos_type,
                'profit': profit,
                'reason': close_reason
            })
            self.position = None
            return profit * 0.01  # Reward normalizasyonu
        
        return 0.0


# ============================================================================
# 14. RL AGENT (Rainbow DQN + LSTM)
# ============================================================================

class RainbowDQNAgent(nn.Module):
    """
    Rainbow DQN + LSTM Hybrid Agent.
    - Dueling architecture
    - Noisy layers
    - LSTM for sequence processing
    """
    
    def __init__(self, state_size: int, action_size: int, config: BotConfig):
        super(RainbowDQNAgent, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=config.RL_LSTM_HIDDEN_SIZE,
            num_layers=config.RL_LSTM_LAYERS,
            batch_first=True,
            dropout=0.2 if config.RL_LSTM_LAYERS > 1 else 0
        )
        
        # Dueling DQN
        self.value_stream = nn.Sequential(
            nn.Linear(config.RL_LSTM_HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.RL_LSTM_HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    
    def forward(self, x):
        """
        Forward pass.
        x shape: (batch, sequence_length, state_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Son time step'i al
        
        # Dueling
        value = self.value_stream(lstm_out)
        advantage = self.advantage_stream(lstm_out)
        
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    """DQN Training & Inference"""
    
    def __init__(self, state_size: int, action_size: int, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"🤖 DQN Device: {self.device}")
        
        # Networks
        self.policy_net = RainbowDQNAgent(state_size, action_size, config).to(self.device)
        self.target_net = RainbowDQNAgent(state_size, action_size, config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.RL_LEARNING_RATE)
        
        # Replay buffer
        self.memory = ReplayBuffer(config.RL_MEMORY_SIZE)
        
        # Epsilon (exploration)
        self.epsilon = config.RL_EPSILON_START
        self.epsilon_end = config.RL_EPSILON_END
        self.epsilon_decay = config.RL_EPSILON_DECAY
        
        self.steps_done = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.policy_net.action_size)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()
    
    def train_step(self):
        """Bir training step"""
        if len(self.memory) < self.config.RL_BATCH_SIZE:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.RL_BATCH_SIZE)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values (target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.config.RL_GAMMA * next_q
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.config.RL_TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path: Path):
        """Model kaydet"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
        self.logger.info(f"💾 Model kaydedildi: {path}")
    
    def load_model(self, path: Path):
        """Model yükle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.logger.info(f"📂 Model yüklendi: {path}")


# ============================================================================
# 15. ULTIMATE TRADING SYSTEM (Tüm parçaları bir araya getir)
# ============================================================================

class UltimateTradingSystem:
    """
    V7.0 Professional Trading System.
    Tüm 12 strateji noktasını entegre eder.
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Bileşenler
        self.rights_manager = RightsManager(config, logger)
        self.range_learner = WeeklyRangeLearner(config, logger)
        self.news_blackout = NewsBlackout(config, logger)
        self.volatility_guards = VolatilityGuards(config, self.range_learner, logger)
        self.trend_filter = TrendFilter(config, logger)
        self.correlation_control = CorrelationControl(config, logger)
        self.hourly_allocator = HourlyAllocator(config, logger)
        self.thompson_bandit = ThompsonBandit(config, logger)
        self.telegram = TelegramReporter(config, logger)
        self.risk_manager = RiskManager(config, logger)
        self.sequential_lock = SequentialLock(config, logger)
        self.data_manager = DataManager(config, logger)
        
        # Enhanced modules
        self.news_manager = NewsManager(config.NEWS_CALENDAR_FILE)
        self.weekly_reporter = WeeklyReporter()
        self.blackout_config = create_blackout_config(
            critical_before=config.NEWS_BLACKOUT_CRITICAL_BEFORE,
            critical_after=config.NEWS_BLACKOUT_CRITICAL_AFTER,
            high_before=config.NEWS_BLACKOUT_HIGH_BEFORE,
            high_after=config.NEWS_BLACKOUT_HIGH_AFTER,
            medium_before=config.NEWS_BLACKOUT_MEDIUM_BEFORE,
            medium_after=config.NEWS_BLACKOUT_MEDIUM_AFTER
        )
        
        # Email ve detaylı logging modülleri
        self.email_notifier = EmailNotifier(config, logger)
        self.trade_logger = EnhancedTradeLogger(config, logger)
        
        # Veri & agent
        self.data = {}
        self.agents = {}  # {pair: DQNTrainer}
        self.environments = {}  # {pair: TradingEnvironment}
        
        # Trading state
        self.open_positions = {}  # {pair: position_info}
        self.closed_trades = []
        self.equity_curve = []
        
        self.current_balance = config.INITIAL_CAPITAL
    
    def load_data_and_initialize(self, start_year: int, end_year: int):
        """Veri yükle ve sistemleri başlat"""
        self.logger.info("🚀 Sistem başlatılıyor...")
        
        # Veriyi yükle
        self.data = self.data_manager.load_data(self.config.PAIRS, start_year, end_year)
        
        # Her pair için feature ekle ve environment oluştur
        for pair, df in self.data.items():
            self.logger.info(f"⚙️  {pair} için feature'lar hesaplanıyor...")
            df = self.data_manager.add_features(df)
            self.data[pair] = df
            
            # Environment
            env = TradingEnvironment(df, self.config, self.logger, pair=pair,
                                   email_notifier=self.email_notifier,
                                   trade_logger=self.trade_logger)
            self.environments[pair] = env
            
            # Agent
            agent = DQNTrainer(env.state_size, env.action_size, self.config, self.logger)
            self.agents[pair] = agent
            
            self.logger.info(f"✅ {pair} hazır: {len(df)} bars, state_size={env.state_size}")
    
    def backtest(self, start_year: int, end_year: int):
        """Backtest modu"""
        self.logger.info(f"📊 BACKTEST: {start_year}-{end_year}")
        
        self.load_data_and_initialize(start_year, end_year)
        
        # Her pair için sırayla backtest (basitleştirilmiş)
        for pair, env in self.environments.items():
            self.logger.info(f"\n{'='*60}\n🔍 {pair} Backtest Başlıyor...\n{'='*60}")
            
            state = env.reset()
            done = False
            step_count = 0
            
            while not done:
                # Agent action seç
                action = self.agents[pair].select_action(state)
                
                # Environment step
                next_state, reward, done, info = env.step(action)
                
                # Memory'ye ekle
                self.agents[pair].memory.push(state, action, reward, next_state, done)
                
                # Train
                if step_count % 4 == 0:
                    self.agents[pair].train_step()
                
                state = next_state
                step_count += 1
                
                if step_count % 1000 == 0:
                    self.logger.info(f"  Step {step_count}/{env.max_steps}, Balance: ${info['balance']:.2f}")
            
            # Sonuçlar
            final_balance = env.balance
            total_profit = env.total_profit
            num_trades = len(env.trade_history)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"✅ {pair} Backtest Tamamlandı!")
            self.logger.info(f"   💰 Final Balance: ${final_balance:.2f}")
            self.logger.info(f"   📈 Total Profit: ${total_profit:.2f}")
            self.logger.info(f"   🔢 Total Trades: {num_trades}")
            
            if num_trades > 0:
                wins = sum(1 for t in env.trade_history if t['profit'] > 0)
                win_rate = wins / num_trades * 100
                self.logger.info(f"   🎯 Win Rate: {win_rate:.1f}%")
            
            self.logger.info(f"{'='*60}\n")
            
            # Weekly reporter'a trade'leri ekle
            for trade in env.trade_history:
                # Convert to reporter format
                trade_data = {
                    'pair': pair,
                    'entry_time': env.df.iloc[trade.get('entry_step', 0)]['datetime'] if 'entry_step' in trade else datetime.now(),
                    'exit_time': env.df.iloc[trade.get('exit_step', 0)]['datetime'] if 'exit_step' in trade else datetime.now(),
                    'direction': trade.get('type', 'UNKNOWN'),
                    'lot_size': trade.get('lot', 0.0),
                    'entry_price': trade.get('entry_price', 0.0),
                    'exit_price': trade.get('exit_price', 0.0),
                    'pnl': trade.get('profit', 0.0),
                    'result': 'WIN' if trade.get('profit', 0) > 0 else 'LOSS',
                    'strategy_type': 'RL',
                    'nearby_news': []  # Will be filled later
                }
                self.weekly_reporter.add_trade(trade_data)
        
        # Genel özet
        self._send_backtest_summary()
        
        # Haftalık rapor oluştur
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 Haftalık Rapor Oluşturuluyor...")
        self.logger.info("="*60)
        
        weekly_report = self.weekly_reporter.generate_weekly_report()
        if weekly_report:
            report_text = self.weekly_reporter.format_report_text(weekly_report)
            self.logger.info("\n" + report_text)
            
            # Telegram'a gönder
            if self.config.TELEGRAM_ENABLED:
                try:
                    asyncio.run(self.telegram._send_message(report_text))
                    self.logger.info("✅ Haftalık rapor Telegram'a gönderildi")
                except Exception as e:
                    self.logger.error(f"❌ Haftalık rapor gönderilemedi: {e}")
    
    def train(self, start_year: int, end_year: int, episodes: int = 10):
        """Training modu"""
        self.logger.info(f"🤖 TRAINING: {start_year}-{end_year}, {episodes} episodes")
        
        self.load_data_and_initialize(start_year, end_year)
        
        for pair, env in self.environments.items():
            self.logger.info(f"\n{'='*60}\n🏋️  {pair} Training Başlıyor...\n{'='*60}")
            
            for episode in range(episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = self.agents[pair].select_action(state)
                    next_state, reward, done, info = env.step(action)
                    
                    self.agents[pair].memory.push(state, action, reward, next_state, done)
                    self.agents[pair].train_step()
                    
                    episode_reward += reward
                    state = next_state
                
                self.logger.info(
                    f"  Episode {episode+1}/{episodes}: Reward={episode_reward:.2f}, "
                    f"Balance=${env.balance:.2f}, Epsilon={self.agents[pair].epsilon:.3f}"
                )
            
            # Model kaydet
            model_path = self.config.MODELS_DIR / f"{pair}_rainbow_dqn_v7.pth"
            self.agents[pair].save_model(model_path)
            
            self.logger.info(f"✅ {pair} training tamamlandı!\n{'='*60}\n")
    
    def paper_trade(self):
        """Paper trading modu (MT5 gerekir)"""
        if not self.config.MT5_ENABLED:
            self.logger.error("❌ Paper trading için MT5 etkinleştirilmeli!")
            return
        
        self.logger.info("📡 PAPER TRADING modu - MT5 bağlantısı bekleniyor...")
        
        # MT5 entegrasyonu burada yapılır (opsiyonel)
        # Bu örnekte sadece placeholder
        
        self.logger.warning("⚠️ Paper trading henüz tam implemente edilmedi. Backtest kullanın.")
    
    def _send_backtest_summary(self):
        """Backtest özeti gönder"""
        summary = {
            'net_profit': sum(env.total_profit for env in self.environments.values()),
            'total_trades': sum(len(env.trade_history) for env in self.environments.values()),
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        all_trades = []
        for env in self.environments.values():
            all_trades.extend(env.trade_history)
        
        if all_trades:
            summary['winning_trades'] = sum(1 for t in all_trades if t['profit'] > 0)
            summary['losing_trades'] = len(all_trades) - summary['winning_trades']
            summary['win_rate'] = summary['winning_trades'] / len(all_trades) * 100
        
        self.telegram.send_daily_summary(summary)


# ============================================================================
# 16. MAIN ENTRY POINT
# ============================================================================

def parse_arguments():
    """Komut satırı argümanları"""
    parser = argparse.ArgumentParser(
        description="Ultimate FTMO Trading Bot V7.0 Professional",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  # Backtest (2020-2024)
  python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2020 --end-year 2024
  
  # Training (2003-2019)
  python3 ultimate_bot_v7_professional.py --mode train --start-year 2003 --end-year 2019 --episodes 20
  
  # Paper Trading
  python3 ultimate_bot_v7_professional.py --mode paper
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'train', 'paper'],
        required=True,
        help='Çalışma modu: backtest, train, paper'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2020,
        help='Başlangıç yılı (varsayılan: 2020)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        default=2024,
        help='Bitiş yılı (varsayılan: 2024)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Training episode sayısı (varsayılan: 10)'
    )
    
    return parser.parse_args()


def main():
    """Ana program"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ULTIMATE FTMO TRADING BOT V7.0 PROFESSIONAL                   ║
║   "Clockwork Reliability, Maximum Transparency"                  ║
║                                                                  ║
║   🎯 12-Point Strategy | 🛡️ Advanced Risk Management            ║
║   📊 Full Data 2003-2024 | 🤖 Rainbow DQN + LSTM               ║
║   📱 Turkish Telegram Reports                                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Argümanları parse et
    args = parse_arguments()
    
    # Konfigürasyonu doğrula
    try:
        BotConfig.validate()
    except FileNotFoundError as e:
        print(f"\n❌ HATA: {e}\n")
        print("Lütfen KULLANIM_KILAVUZU.md dosyasındaki talimatları takip edin.\n")
        sys.exit(1)
    
    # Logging setup
    logger = setup_logging(BotConfig.LOGS_DIR, BotConfig.LOG_LEVEL)
    logger.info("="*70)
    logger.info("🚀 ULTIMATE FTMO TRADING BOT V7.0 BAŞLATILIYOR...")
    logger.info(f"📋 Mod: {args.mode.upper()}")
    logger.info(f"📅 Yıl Aralığı: {args.start_year}-{args.end_year}")
    logger.info("="*70)
    
    # Trading system oluştur
    system = UltimateTradingSystem(BotConfig, logger)
    
    # Moda göre çalıştır
    try:
        if args.mode == 'backtest':
            system.backtest(args.start_year, args.end_year)
        
        elif args.mode == 'train':
            system.train(args.start_year, args.end_year, args.episodes)
        
        elif args.mode == 'paper':
            system.paper_trade()
    
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Kullanıcı tarafından durduruldu (Ctrl+C)")
    
    except Exception as e:
        logger.error(f"\n\n❌ HATA: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("="*70)
        logger.info("🏁 BOT DURDURULDU")
        logger.info("="*70)
        print("\n✅ Program sonlandı. Log dosyasını kontrol edin: ~/Desktop/JTTWS/logs/\n")


if __name__ == "__main__":
    main()
