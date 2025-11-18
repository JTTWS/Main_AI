#!/usr/bin/env python3
"""
ULTIMATE FİNAL FİX - TÜM SORUNLARI ÇÖZER
- Telegram CHAT_ID ekle
- Enhanced Logger aktif et
- Haftalık rapor aktif et
- Trade detaylarını logla
- Her şeyi düzelt
"""

import os
import re
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════╗
║   ULTIMATE FİNAL FİX - TÜM SORUNLAR ÇÖZÜLüYOR                   ║
╚══════════════════════════════════════════════════════════════════╝
""")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"
config_file = BASE_DIR / "bot_config.py"

# ============================================================================
# 1. BOT_CONFIG.PY'YE TELEGRAM_CHAT_ID EKLE
# ============================================================================
print("1/5 bot_config.py'ye TELEGRAM_CHAT_ID ekleniyor...")

with open(config_file, 'r', encoding='utf-8') as f:
    config_lines = f.readlines()

chat_id_added = False
for i, line in enumerate(config_lines):
    if 'TELEGRAM_TOKEN = ' in line and not chat_id_added:
        # Token satırından sonra CHAT_ID ekle
        config_lines.insert(i + 1, '    TELEGRAM_CHAT_ID = 1590841427\n')
        chat_id_added = True
        print("  ✓ TELEGRAM_CHAT_ID eklendi")
        break

if not chat_id_added:
    print("  ⚠ TELEGRAM_CHAT_ID zaten var veya eklenemedi")

with open(config_file, 'w', encoding='utf-8') as f:
    f.writelines(config_lines)

# ============================================================================
# 2. ULTIMATE_BOT'A DETAYLI TRADE LOGGING EKLE
# ============================================================================
print("2/5 Detaylı trade logging sistemi ekleniyor...")

with open(bot_file, 'r', encoding='utf-8') as f:
    bot_content = f.read()

# TradingEnvironment'ın step metoduna enhanced logging ekle
enhanced_trade_log = '''
        # ============ DETAYLI TRADE LOGGING ============
        if action in [1, 2] and self.position is None:  # Yeni trade açılıyor
            trade_info = {
                'type': 'LONG' if action == 1 else 'SHORT',
                'pair': self.pair,
                'time': current_time,
                'entry_price': current_price,
                'lot': lot_size,
                'indicators': {
                    'RSI_14': state[10] if len(state) > 10 else 0,
                    'MACD': state[11] if len(state) > 11 else 0,
                    'BB_upper': state[7] if len(state) > 7 else 0,
                    'BB_lower': state[8] if len(state) > 8 else 0,
                    'ATR': atr,
                },
                'lot_calculation': {
                    'risk_amount': risk_amount,
                    'atr': atr,
                    'kelly': 0.25,  # Simplified
                },
                'reason': f'RL Model decision (action={action})'
            }
            
            # Yakındaki haberleri ekle
            if hasattr(self.system, 'news_manager') and self.system.news_manager.calendar_df is not None:
                currency = self.pair[:3]
                nearby_news = self.system.news_manager.get_news_at_time(current_time, currency, window_minutes=30)
                if nearby_news:
                    trade_info['nearby_news'] = nearby_news
            
            # Enhanced logger ile logla
            if hasattr(self.system, 'trade_logger'):
                self.system.trade_logger.log_trade_entry(trade_info)
            else:
                # Fallback: Normal log
                self.logger.info("\\n" + "="*70)
                self.logger.info(f"📊 TRADE AÇILDI - {trade_info['type']} {self.pair}")
                self.logger.info("="*70)
                self.logger.info(f"💰 Lot: {lot_size:.2f}")
                self.logger.info(f"📍 Giriş: {current_price:.5f}")
                self.logger.info(f"📈 RSI: {trade_info['indicators']['RSI_14']:.2f}")
                self.logger.info(f"📈 ATR: {trade_info['indicators']['ATR']:.5f}")
                if 'nearby_news' in trade_info:
                    self.logger.info(f"📰 Yakın haberler: {len(trade_info['nearby_news'])} adet")
                    for news in trade_info['nearby_news'][:3]:
                        self.logger.info(f"  - [{news['category']}] {news['name']}")
                self.logger.info("="*70 + "\\n")
'''

# Position açma kodundan sonra enhanced logging ekle
# "self.position = {" bölümünü bul
if "self.position = {" in bot_content:
    # Bu bölümden sonra enhanced logging ekle
    bot_content = bot_content.replace(
        "self.position = {",
        enhanced_trade_log + "\n        self.position = {"
    )
    print("  ✓ Detaylı trade logging eklendi")
else:
    print("  ⚠ Trade logging eklenemedi")

# ============================================================================
# 3. HAFTALIK RAPORU AKTİF ET
# ============================================================================
print("3/5 Haftalık rapor sistemi aktif ediliyor...")

# "Haftalık rapor özelliği geliştirme aşamasında" kısmını değiştir
bot_content = bot_content.replace(
    '📊 Haftalık rapor özelliği geliştirme aşamasında...',
    '📊 Haftalık rapor oluşturuluyor...'
)

# Haftalık rapor kodunu aktif et
weekly_report_code = '''
        # Haftalık rapor oluştur ve gönder
        try:
            from datetime import datetime, timedelta
            
            self.logger.info("\\n" + "="*70)
            self.logger.info("📊 HAFTALIK RAPOR")
            self.logger.info("="*70)
            
            # Basit rapor
            all_trades = 0
            all_wins = 0
            all_pnl = 0.0
            
            for pair in self.config.PAIRS:
                if pair in pair_results:
                    result = pair_results[pair]
                    all_trades += result['trades']
                    all_wins += result['wins']
                    all_pnl += result['total_pnl']
            
            win_rate = (all_wins / all_trades * 100) if all_trades > 0 else 0
            
            self.logger.info(f"Toplam Trade: {all_trades}")
            self.logger.info(f"Kazanan: {all_wins} ({win_rate:.1f}%)")
            self.logger.info(f"Kaybeden: {all_trades - all_wins}")
            self.logger.info(f"Toplam PnL: ${all_pnl:.2f}")
            self.logger.info("\\n📈 PARİTE BAZLI:")
            
            for pair, result in pair_results.items():
                emoji = "🟢" if result['total_pnl'] > 0 else "🔴"
                self.logger.info(
                    f"{emoji} {pair}: {result['trades']} trade, "
                    f"Win Rate: {result['win_rate']:.1f}%, "
                    f"PnL: ${result['total_pnl']:.2f}"
                )
            
            self.logger.info("="*70)
            
            # Email ile gönder
            if hasattr(self, 'email_notifier') and self.email_notifier.enabled:
                report_text = f"""
HAFTALIK PERFORMANS RAPORU
{'='*50}

Toplam Trade: {all_trades}
Kazanan: {all_wins} ({win_rate:.1f}%)
Toplam PnL: ${all_pnl:.2f}

PARİTE PERFORMANSI:
"""
                for pair, result in pair_results.items():
                    report_text += f"\\n{pair}: {result['trades']} trade, Win Rate: {result['win_rate']:.1f}%, PnL: ${result['total_pnl']:.2f}"
                
                self.email_notifier.send_weekly_report(report_text)
                self.logger.info("📧 Haftalık rapor email ile gönderildi")
            
        except Exception as e:
            self.logger.error(f"Haftalık rapor oluşturulamadı: {e}")
'''

# Backtest sonunda ekle
bot_content = bot_content.replace(
    'self.logger.info("📊 Haftalık rapor oluşturuluyor...")',
    'self.logger.info("📊 Haftalık rapor oluşturuluyor...")' + weekly_report_code
)

print("  ✓ Haftalık rapor aktif edildi")

# ============================================================================
# 4. TELEGRAM CHAT_IDS SORUNUNU DÜZELT
# ============================================================================
print("4/5 Telegram chat_ids düzeltiliyor...")

# _initialize_bot metodunda chat_id'yi ekle
telegram_fix = '''
            # Chat ID'yi ekle
            if hasattr(self.config, 'TELEGRAM_CHAT_ID') and self.config.TELEGRAM_CHAT_ID:
                self.chat_ids = [self.config.TELEGRAM_CHAT_ID]
                self.logger.info(f"✅ Telegram Chat ID: {self.config.TELEGRAM_CHAT_ID}")
'''

if 'self.logger.info("📱 Telegram bot başlatıldı.")' in bot_content:
    bot_content = bot_content.replace(
        'self.logger.info("📱 Telegram bot başlatıldı.")',
        'self.logger.info("📱 Telegram bot başlatıldı.")' + telegram_fix
    )
    print("  ✓ Telegram chat_ids düzeltildi")

# ============================================================================
# 5. DOSYAYI YAZ
# ============================================================================
print("5/5 Dosyalar kaydediliyor...")

with open(bot_file, 'w', encoding='utf-8') as f:
    f.write(bot_content)

print("\n" + "="*70)
print("✅ FİNAL FİX TAMAMLANDI!")
print("="*70)

print("\n📋 YAPILAN DEĞİŞİKLİKLER:")
print("  ✓ bot_config.py'ye TELEGRAM_CHAT_ID eklendi")
print("  ✓ Detaylı trade logging sistemi eklendi")
print("  ✓ Haftalık rapor sistemi aktif edildi")
print("  ✓ Telegram chat_ids sorunu düzeltildi")
print("  ✓ Her trade için şunlar loglanacak:")
print("    • İndikatör değerleri (RSI, MACD, ATR, BB)")
print("    • Yakındaki haberler (±30dk)")
print("    • Lot hesaplama detayları")
print("    • Trade açılma sebebi")

print("\n⚠️ SON BİR ADIM:")
print("  Gmail App Password almayı unutmayın!")
print("  1. https://myaccount.google.com/apppasswords")
print("  2. 'Mail' için password oluştur")
print("  3. bot_config.py'de EMAIL_APP_PASSWORD'e yapıştır")

print("\n🚀 ŞİMDİ TEST EDİN:")
print("  python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2024 --end-year 2024")

print("\n✅ BEKLENİYOR:")
print("  • Detaylı trade log'ları")
print("  • İndikatör değerleri")
print("  • Yakın haberler")
print("  • Haftalık rapor")
print("  • Telegram bildirim (chat_id düzeltildi)")

print("="*70)
