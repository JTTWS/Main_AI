#!/usr/bin/env python3
"""
"Telegram chat_id yok" uyarısını burada bulalım
"""

from pathlib import Path

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

if not bot_file.exists():
    print(f"❌ {bot_file} bulunamadı!")
    exit(1)

# Dosyayı oku
with open(bot_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("🔍 'Telegram chat_id yok' uyarısını arıyorum...\n")

found_lines = []

for i, line in enumerate(lines):
    if 'chat_id yok' in line.lower() or 'telegram chat_id' in line.lower():
        found_lines.append((i, line))

if found_lines:
    print(f"✓ {len(found_lines)} satırda bulundu:\n")
    
    for line_num, line_content in found_lines:
        print(f"Satır {line_num+1}: {line_content.strip()}")
        
        # Çevresindeki 10 satırı da göster
        print("\n" + "="*70)
        print(f"ÇEVRE (Satır {max(0, line_num-5)+1} - {min(len(lines), line_num+6)})")
        print("="*70)
        for j in range(max(0, line_num-5), min(len(lines), line_num+6)):
            marker = ">>> " if j == line_num else "    "
            print(f"{marker}{j+1:4d} | {lines[j]}", end='')
        print("="*70 + "\n")
else:
    print("❌ Bulunamadı!")
    print("\nAlternatif arama: 'TELEGRAM_CHAT_ID' içeren satırlar:")
    
    for i, line in enumerate(lines):
        if 'TELEGRAM_CHAT_ID' in line:
            print(f"  Satır {i+1}: {line.strip()}")
