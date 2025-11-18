#!/usr/bin/env python3
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
