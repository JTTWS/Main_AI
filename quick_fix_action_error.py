#!/usr/bin/env python3
"""
Action error hızlı düzeltme
"""

from pathlib import Path

print("🔧 Action hatası düzeltiliyor...\n")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

with open(bot_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Hatalı kodu bul ve kaldır
old_code = '''
        # ============ DETAYLI TRADE LOGGING ============
        if action in [1, 2] and self.position is None:  # Yeni trade açılıyor'''

# Basit logging ile değiştir
new_code = '''
        # ============ DETAYLI TRADE LOGGING ============
        # Trade açılıyor - detaylar loglanıyor'''

content = content.replace(old_code, new_code)

# action kontrolünü kaldır, sadece logging yap
content = content.replace(
    "if action in [1, 2] and self.position is None:  # Yeni trade açılıyor",
    "if self.position is None:  # Yeni trade açılıyor (detaylı log)"
)

# Dosyayı yaz
with open(bot_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Action hatası düzeltildi!")
print("\n🚀 Şimdi tekrar test edin:")
print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2024 --end-year 2024")
