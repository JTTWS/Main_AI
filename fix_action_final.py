#!/usr/bin/env python3
"""
Action referanslarını tamamen temizle
"""

from pathlib import Path

print("🔧 Tüm action referansları temizleniyor...\n")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

with open(bot_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 'LONG' if action == 1 else 'SHORT' → direction kullan
content = content.replace(
    "'type': 'LONG' if action == 1 else 'SHORT',",
    "'type': direction,"
)

# action referanslarını kaldır
content = content.replace(
    "'reason': f'RL Model decision (action={action})'",
    "'reason': f'RL Model decision ({direction})'"
)

# Dosyayı yaz
with open(bot_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Tüm action referansları temizlendi!")
print("✅ Artık 'direction' parametresi kullanılıyor")
print("\n🚀 Test edin:")
print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2024 --end-year 2024")
