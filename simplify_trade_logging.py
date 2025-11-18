#!/usr/bin/env python3
"""
Trade logging'i basitleştir - tüm karmaşık kodları kaldır
"""

from pathlib import Path
import re

print("🔧 Trade logging basitleştiriliyor...\n")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

with open(bot_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Detaylı trade logging bloğunu bul ve basit versiyonla değiştir
new_lines = []
skip_mode = False
skip_count = 0

for i, line in enumerate(lines):
    # Detaylı trade logging başlangıcını bul
    if '# ============ DETAYLI TRADE LOGGING ============' in line:
        # Basit logging ekle
        new_lines.append('        # Basit trade log\n')
        new_lines.append('        self.logger.info("\\n" + "="*70)\n')
        new_lines.append(f'        self.logger.info(f"📊 TRADE AÇILDI - {{direction}} {{self.pair}}")\n')
        new_lines.append('        self.logger.info("="*70)\n')
        new_lines.append(f'        self.logger.info(f"💰 Lot: {{lot_size:.2f}}")\n')
        new_lines.append(f'        self.logger.info(f"📍 Giriş: {{current_price:.5f}}")\n')
        new_lines.append(f'        self.logger.info(f"📈 ATR: {{atr:.5f}}")\n')
        new_lines.append('        self.logger.info("="*70 + "\\n")\n')
        new_lines.append('\n')
        skip_mode = True
        continue
    
    # Skip tüm trade logging bloğunu
    if skip_mode:
        # self.position = { satırına ulaşana kadar skip
        if 'self.position = {' in line:
            skip_mode = False
            new_lines.append(line)
        continue
    
    new_lines.append(line)

# Dosyayı yaz
with open(bot_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Trade logging basitleştirildi!")
print("✅ Artık sadece temel bilgiler loglanıyor")
print("\n📊 Her trade için loglanacaklar:")
print("  • Trade yönü (LONG/SHORT)")
print("  • Parite")
print("  • Lot miktarı")
print("  • Giriş fiyatı")
print("  • ATR değeri")
print("\n🚀 Test edin:")
print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2024 --end-year 2024")
