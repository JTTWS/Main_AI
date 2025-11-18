#!/usr/bin/env python3
"""
TelegramReporter kodunu kontrol et
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

print("🔍 TelegramReporter sınıfını arıyorum...\n")

# TelegramReporter'ı bul
in_telegram = False
telegram_start = -1
init_start = -1
init_end = -1

for i, line in enumerate(lines):
    if 'class TelegramReporter:' in line:
        in_telegram = True
        telegram_start = i
        print(f"✓ TelegramReporter bulundu (satır {i+1})")
        continue
    
    if in_telegram and 'def __init__' in line:
        init_start = i
        print(f"✓ __init__ metodu bulundu (satır {i+1})")
        
        # __init__'in sonunu bul (ilk 50 satır içinde)
        for j in range(i+1, min(i+50, len(lines))):
            if lines[j].strip().startswith('def ') or lines[j].strip().startswith('async def '):
                init_end = j
                break
        
        if init_end == -1:
            init_end = min(i+50, len(lines))
        
        print(f"✓ __init__ metodu sonu (satır {init_end+1})")
        print("\n" + "="*70)
        print("__INIT__ METODUNUN İÇERİĞİ:")
        print("="*70)
        for k in range(init_start, init_end):
            print(f"{k+1:4d} | {lines[k]}", end='')
        print("="*70)
        break
    
    # TelegramReporter'ın dışına çıktıysak dur
    if in_telegram and line.startswith('class ') and 'TelegramReporter' not in line:
        break

if init_start == -1:
    print("❌ __init__ metodu bulunamadı!")
