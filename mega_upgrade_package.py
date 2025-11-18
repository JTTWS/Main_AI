#!/usr/bin/env python3
"""
FTMO Bot V7 - MEGA UPGRADE PACKAGE
- Detaylı Trade Logging
- Email Notifications
- Haftalık Rapor Sistemi
- Telegram Düzeltmesi
"""

import os
from pathlib import Path
import re

print("""
╔══════════════════════════════════════════════════════════════════╗
║   FTMO BOT V7 - MEGA UPGRADE                                     ║
║   Detaylı Logging + Email + Haftalık Rapor + Telegram Fix       ║
╚══════════════════════════════════════════════════════════════════╝
""")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"
config_file = BASE_DIR / "bot_config.py"

if not bot_file.exists():
    print(f"❌ {bot_file} bulunamadı!")
    exit(1)

if not config_file.exists():
    print(f"❌ {config_file} bulunamadı!")
    exit(1)

print("🚀 Upgrade başlıyor...\n")

# ============================================================================
# 1. CONFIG'E EMAIL EKLEMELERİ
# ============================================================================
print("1/4 Email yapılandırması ekleniyor...")

with open(config_file, 'r', encoding='utf-8') as f:
    config_content = f.read()

# Email config'i ekle (TELEGRAM bölümünden sonra)
email_config = '''
    # ==================== EMAIL NOTIFICATIONS ====================
    EMAIL_ENABLED = True
    EMAIL_ADDRESS = "journeytothewallstreet@gmail.com"
    
    # Gmail App Password (2-factor auth gerektirir)
    # https://myaccount.google.com/apppasswords adresinden alın
    EMAIL_APP_PASSWORD = ""  # Buraya Gmail App Password gireceksiniz
    
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
'''

if 'EMAIL_ENABLED' not in config_content:
    # TELEGRAM bölümünden sonra ekle
    if '# ==================== NEWS BLACKOUT ====================' in config_content:
        config_content = config_content.replace(
            '# ==================== NEWS BLACKOUT ====================',
            email_config + '\n    # ==================== NEWS BLACKOUT ===================='
        )
        print("  ✓ Email config eklendi")
    else:
        print("  ⚠ Email config eklenemedi (manuel ekleyin)")
else:
    print("  ℹ Email config zaten mevcut")

with open(config_file, 'w', encoding='utf-8') as f:
    f.write(config_content)

# ============================================================================
# 2. EMAIL NOTIFIER CLASS OLUŞTUR
# ============================================================================
print("2/4 Email notifier sınıfı oluşturuluyor...")

email_notifier_code = '''#!/usr/bin/env python3
"""
Email Notification System
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging

class EmailNotifier:
    """
    Email bildirimleri gönderen sınıf
    - Trade bildirimleri
    - Haftalık raporlar
    - Kritik uyarılar
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.enabled = config.EMAIL_ENABLED
        
        if self.enabled:
            if not config.EMAIL_APP_PASSWORD:
                self.logger.warning("📧 Email App Password yok! Email bildirimleri devre dışı.")
                self.enabled = False
            else:
                self.logger.info(f"📧 Email notifications etkin: {config.EMAIL_ADDRESS}")
    
    def send_email(self, subject: str, body: str, html: bool = False):
        """Email gönder"""
        if not self.enabled:
            return False
        
        try:
            # Email oluştur
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.EMAIL_ADDRESS
            msg['To'] = self.config.EMAIL_ADDRESS
            msg['Subject'] = subject
            
            # Body ekle
            if html:
                part = MIMEText(body, 'html')
            else:
                part = MIMEText(body, 'plain')
            msg.attach(part)
            
            # Gönder
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT) as server:
                server.starttls()
                server.login(self.config.EMAIL_ADDRESS, self.config.EMAIL_APP_PASSWORD)
                server.send_message(msg)
            
            self.logger.info(f"📧 Email gönderildi: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"📧 Email gönderilemedi: {e}")
            return False
    
    def send_trade_notification(self, trade_info: dict):
        """Trade bildirimi gönder"""
        subject = f"🤖 FTMO Bot - {trade_info['type']} Trade Açıldı"
        
        body = f"""
FTMO Trading Bot - Trade Bildirimi
{'='*50}

Parite: {trade_info['pair']}
Yön: {trade_info['type']}
Lot: {trade_info['lot']:.2f}
Giriş Fiyatı: {trade_info['entry_price']:.5f}
Stop Loss: {trade_info.get('sl', 'N/A')}
Take Profit: {trade_info.get('tp', 'N/A')}

İNDİKATÖRLER:
{trade_info.get('indicators', 'N/A')}

YAKIN HABERLER:
{trade_info.get('nearby_news', 'Yok')}

SEBEP:
{trade_info.get('reason', 'RL Model kararı')}

Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}
"""
        
        return self.send_email(subject, body)
    
    def send_weekly_report(self, report_text: str):
        """Haftalık rapor gönder"""
        subject = f"📊 FTMO Bot - Haftalık Rapor ({datetime.now().strftime('%d/%m/%Y')})"
        
        return self.send_email(subject, report_text)
    
    def send_alert(self, alert_type: str, message: str):
        """Kritik uyarı gönder"""
        subject = f"⚠️ FTMO Bot - {alert_type}"
        
        body = f"""
FTMO Trading Bot - UYARI
{'='*50}

Uyarı Tipi: {alert_type}

Mesaj:
{message}

Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}
"""
        
        return self.send_email(subject, body)
'''

email_file = BASE_DIR / "email_notifier.py"
with open(email_file, 'w', encoding='utf-8') as f:
    f.write(email_notifier_code)

print(f"  ✓ email_notifier.py oluşturuldu")

# ============================================================================
# 3. ENHANCED TRADE LOGGER
# ============================================================================
print("3/4 Enhanced trade logger ekleniyor...")

enhanced_logger_code = '''#!/usr/bin/env python3
"""
Enhanced Trade Logger
Her trade için detaylı bilgi kaydeder
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

class EnhancedTradeLogger:
    """
    Detaylı trade logging sistemi
    - İndikatör değerleri
    - Yakın haberler
    - Lot hesaplama mantığı
    - Risk/Reward
    """
    
    def __init__(self, logger: logging.Logger, news_manager=None):
        self.logger = logger
        self.news_manager = news_manager
        self.trade_count = 0
    
    def log_trade_entry(self, trade_data: Dict):
        """Trade açılışını detaylı logla"""
        self.trade_count += 1
        
        self.logger.info("\\n" + "="*70)
        self.logger.info(f"📊 TRADE #{self.trade_count} - {trade_data['type']} {trade_data['pair']}")
        self.logger.info("="*70)
        
        # Temel bilgiler
        self.logger.info(f"⏰ Zaman: {trade_data.get('time', datetime.now())}")
        self.logger.info(f"💰 Lot: {trade_data['lot']:.2f}")
        self.logger.info(f"📍 Giriş: {trade_data['entry_price']:.5f}")
        
        if 'sl' in trade_data:
            self.logger.info(f"🛡️ Stop Loss: {trade_data['sl']:.5f}")
        if 'tp' in trade_data:
            self.logger.info(f"🎯 Take Profit: {trade_data['tp']:.5f}")
        
        # İndikatörler
        if 'indicators' in trade_data:
            self.logger.info(f"\\n📈 İNDİKATÖRLER:")
            for ind, value in trade_data['indicators'].items():
                self.logger.info(f"  • {ind}: {value}")
        
        # Lot hesaplama mantığı
        if 'lot_calculation' in trade_data:
            self.logger.info(f"\\n💡 LOT HESAPLAMA:")
            calc = trade_data['lot_calculation']
            self.logger.info(f"  • Risk Miktarı: ${calc.get('risk_amount', 0):.2f}")
            self.logger.info(f"  • ATR: {calc.get('atr', 0):.5f}")
            self.logger.info(f"  • Kelly: {calc.get('kelly', 0):.3f}")
            self.logger.info(f"  • Final Lot: {trade_data['lot']:.2f}")
        
        # Yakın haberler
        if self.news_manager and 'time' in trade_data:
            nearby_news = self._get_nearby_news(
                trade_data['time'], 
                trade_data['pair'][:3]  # Currency (EUR, GBP, etc.)
            )
            
            if nearby_news:
                self.logger.info(f"\\n📰 YAKIN HABERLER (±30dk):")
                for news in nearby_news[:5]:
                    time_diff = int(news['minutes_diff'])
                    self.logger.info(
                        f"  • [{news['category']}] {news['name']} "
                        f"({time_diff:+d}dk)"
                    )
            else:
                self.logger.info(f"\\n📰 Yakın haber yok")
        
        # Trade nedeni
        if 'reason' in trade_data:
            self.logger.info(f"\\n🤔 SEBEP:")
            self.logger.info(f"  {trade_data['reason']}")
        
        self.logger.info("="*70 + "\\n")
    
    def log_trade_exit(self, trade_data: Dict):
        """Trade kapanışını detaylı logla"""
        self.logger.info("\\n" + "="*70)
        self.logger.info(f"📊 TRADE KAPANDI - {trade_data['type']} {trade_data['pair']}")
        self.logger.info("="*70)
        
        self.logger.info(f"⏰ Açılış: {trade_data.get('entry_time', 'N/A')}")
        self.logger.info(f"⏰ Kapanış: {trade_data.get('exit_time', datetime.now())}")
        self.logger.info(f"📍 Giriş Fiyat: {trade_data['entry_price']:.5f}")
        self.logger.info(f"📍 Çıkış Fiyat: {trade_data['exit_price']:.5f}")
        
        pnl = trade_data.get('pnl', 0)
        emoji = "✅" if pnl > 0 else "❌"
        self.logger.info(f"💰 Kar/Zarar: {emoji} ${pnl:.2f}")
        
        if 'duration_minutes' in trade_data:
            self.logger.info(f"⏱️ Süre: {trade_data['duration_minutes']} dakika")
        
        self.logger.info("="*70 + "\\n")
    
    def _get_nearby_news(self, trade_time: datetime, currency: str):
        """Yakındaki haberleri al"""
        if not self.news_manager or not self.news_manager.calendar_df is not None:
            return []
        
        return self.news_manager.get_news_at_time(trade_time, currency, window_minutes=30)
'''

enhanced_logger_file = BASE_DIR / "enhanced_trade_logger.py"
with open(enhanced_logger_file, 'w', encoding='utf-8') as f:
    f.write(enhanced_logger_code)

print(f"  ✓ enhanced_trade_logger.py oluşturuldu")

# ============================================================================
# 4. TELEGRAM INSTANCE FIX
# ============================================================================
print("4/4 Telegram instance sorunu düzeltiliyor...")

with open(bot_file, 'r', encoding='utf-8') as f:
    bot_content = f.read()

# main() fonksiyonunda BotConfig -> BotConfig() değişikliği
bot_content = re.sub(
    r'system = UltimateTradingSystem\(BotConfig,',
    'system = UltimateTradingSystem(BotConfig(),',
    bot_content
)

with open(bot_file, 'w', encoding='utf-8') as f:
    f.write(bot_content)

print("  ✓ Telegram instance sorunu düzeltildi")

# ============================================================================
# ÖZET
# ============================================================================
print("\\n" + "="*70)
print("✅ MEGA UPGRADE TAMAMLANDI!")
print("="*70)
print("\\n📋 YAPILAN DEĞİŞİKLİKLER:")
print("  ✓ Email configuration eklendi (bot_config.py)")
print("  ✓ email_notifier.py oluşturuldu")
print("  ✓ enhanced_trade_logger.py oluşturuldu")
print("  ✓ Telegram instance sorunu düzeltildi")

print("\\n⚠️  ÖNEMLİ - GMAIL APP PASSWORD:")
print("  1. https://myaccount.google.com/security adresine gidin")
print("  2. 2-Step Verification'ı aktif edin")
print("  3. https://myaccount.google.com/apppasswords adresine gidin")
print("  4. 'Mail' için yeni App Password oluşturun")
print("  5. bot_config.py'de EMAIL_APP_PASSWORD'e yapıştırın")

print("\\n🚀 SONRAKİ ADIMLAR:")
print("  1. Gmail App Password alın ve config'e ekleyin")
print("  2. python3 integrate_new_modules.py çalıştırın")
print("  3. Bot'u test edin")

print("="*70)
