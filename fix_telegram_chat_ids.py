#!/usr/bin/env python3
"""
Telegram chat_ids listesine config'teki chat_id'yi ekle
"""

from pathlib import Path

print("🔧 Telegram chat_ids sorunu düzeltiliyor...\n")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

if not bot_file.exists():
    print(f"❌ {bot_file} bulunamadı!")
    exit(1)

# Dosyayı oku
with open(bot_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# _initialize_bot metodunu bul ve chat_id'yi ekle
modified = False

for i, line in enumerate(lines):
    if 'def _initialize_bot(self):' in line:
        # Bu metodun içine chat_id ekleme kodunu ekle
        # "Telegram bot başlatıldı" log'undan sonra ekleyelim
        
        for j in range(i, min(i+20, len(lines))):
            if '"📱 Telegram bot başlatıldı."' in lines[j]:
                # Bu satırdan sonra chat_id ekle
                indent = '            '
                
                # Yeni satırlar ekle
                new_lines = [
                    f'{indent}\n',
                    f'{indent}# Chat ID\'yi listeye ekle\n',
                    f'{indent}if self.config.TELEGRAM_CHAT_ID:\n',
                    f'{indent}    self.chat_ids = [self.config.TELEGRAM_CHAT_ID]\n',
                    f'{indent}    self.logger.info(f"✅ Telegram Chat ID eklendi: {{self.config.TELEGRAM_CHAT_ID}}")\n',
                    f'{indent}else:\n',
                    f'{indent}    self.logger.warning("⚠️  Telegram Chat ID config\'te bulunamadı")\n',
                ]
                
                # Satırı ekle
                lines[j] = lines[j].rstrip() + '\n'
                for new_line in reversed(new_lines):
                    lines.insert(j + 1, new_line)
                
                modified = True
                print("✅ _initialize_bot metoduna chat_id ekleme kodu eklendi!")
                break
        
        if modified:
            break

if modified:
    # Dosyayı yaz
    with open(bot_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("\n" + "="*70)
    print("✅ TELEGRAM CHAT_IDS DÜZELTMESİ TAMAMLANDI!")
    print("="*70)
    print("\n📋 YAPILAN DEĞİŞİKLİK:")
    print("  ✓ _initialize_bot metodunda chat_id listeye ekleniyor")
    print("  ✓ Config'teki TELEGRAM_CHAT_ID (1590841427) kullanılacak")
    print("\n🚀 ŞİMDİ BOT'U ÇALIŞTIRIN:")
    print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2023 --end-year 2024")
    print("\n✅ BEKLENEN ÇIKTI:")
    print("   ✅ News calendar yüklendi: 83,522 events")
    print("   ✅ Telegram bot başlatıldı.")
    print("   ✅ Telegram Chat ID eklendi: 1590841427")
    print("="*70)
else:
    print("❌ _initialize_bot metodunda 'Telegram bot başlatıldı' satırı bulunamadı!")
    print("   Dosya zaten güncellenmiş olabilir veya format farklı.")
