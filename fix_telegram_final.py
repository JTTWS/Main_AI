#!/usr/bin/env python3
"""
Telegram Chat ID Sorunu - Son Düzeltme
"""

import os
from pathlib import Path

print("🔧 Telegram Chat ID sorunu düzeltiliyor...\n")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

if not bot_file.exists():
    print(f"❌ {bot_file} bulunamadı!")
    exit(1)

# Dosyayı oku
with open(bot_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Eski Telegram chat_id kontrolü
old_check = """        # Chat ID kontrolü
        if not self.config.TELEGRAM_CHAT_ID:
            self.logger.warning("📱 Telegram chat_id yok. /start gönderin.")
            self.enabled = False"""

# Yeni Telegram chat_id kontrolü
new_check = """        # Chat ID kontrolü
        if self.config.TELEGRAM_CHAT_ID is None or self.config.TELEGRAM_CHAT_ID == 0:
            self.logger.warning("📱 Telegram chat_id yok. Bot config'te ayarlayın.")
            self.enabled = False
        else:
            self.logger.info(f"✅ Telegram etkin - Chat ID: {self.config.TELEGRAM_CHAT_ID}")"""

# Değiştir
if old_check in content:
    content = content.replace(old_check, new_check)
    print("✅ Telegram chat_id kontrolü düzeltildi!")
    
    # Dosyayı yaz
    with open(bot_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n" + "="*70)
    print("✅ TELEGRAM DÜZELTMESİ TAMAMLANDI!")
    print("="*70)
    print("\n🚀 ŞİMDİ BOT'U ÇALIŞTIRIN:")
    print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2023 --end-year 2024")
    print("\n✅ BEKLENEN ÇIKTI:")
    print("   ✅ News calendar yüklendi: 83,522 events")
    print("   ✅ Telegram etkin - Chat ID: 1590841427")
    print("="*70)
else:
    print("⚠️  Eski kod bloğu bulunamadı.")
    print("   Muhtemelen zaten güncel veya farklı bir formatta.")
    print("\n📋 Alternatif: TelegramReporter'ı manuel kontrol edin")
    print("   Dosya: ~/Desktop/JTTWS/ultimate_bot_v7_professional.py")
    print("   Arama: 'class TelegramReporter'")
