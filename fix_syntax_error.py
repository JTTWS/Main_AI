#!/usr/bin/env python3
"""
Syntax hatası düzeltmesi
\n karakterlerini temizle
"""

from pathlib import Path

print("🔧 Syntax hatası düzeltiliyor...\n")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

if not bot_file.exists():
    print(f"❌ {bot_file} bulunamadı!")
    exit(1)

# Dosyayı oku
with open(bot_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Yanlış eklenen \n karakterlerini düzelt
content = content.replace('\\n        # Email notifications\\n        ', '\n        # Email notifications\n        ')
content = content.replace('\\n        \\n        # Enhanced trade logger\\n        ', '\n        \n        # Enhanced trade logger\n        ')
content = content.replace('\\n        ', '\n        ')

# Dosyayı yaz
with open(bot_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Syntax hatası düzeltildi!")
print("\n🚀 Şimdi bot'u test edin:")
print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2024 --end-year 2024")
