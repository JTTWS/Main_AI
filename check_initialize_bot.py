#!/usr/bin/env python3
"""
TelegramReporter _initialize_bot metodunu kontrol et
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

print("🔍 _initialize_bot metodunu arıyorum...\n")

# _initialize_bot'u bul
found = False
method_start = -1
method_end = -1

for i, line in enumerate(lines):
    if 'def _initialize_bot' in line:
        method_start = i
        found = True
        print(f"✓ _initialize_bot metodu bulundu (satır {i+1})")
        
        # Metodun sonunu bul (bir sonraki metod başlangıcına kadar)
        for j in range(i+1, min(i+100, len(lines))):
            if (lines[j].strip().startswith('def ') or 
                lines[j].strip().startswith('async def ')) and '_initialize_bot' not in lines[j]:
                method_end = j
                break
        
        if method_end == -1:
            method_end = min(i+100, len(lines))
        
        print(f"✓ _initialize_bot metodu sonu (satır {method_end+1})")
        print("\n" + "="*70)
        print("_INITIALIZE_BOT METODUNUN İÇERİĞİ:")
        print("="*70)
        for k in range(method_start, method_end):
            print(f"{k+1:4d} | {lines[k]}", end='')
        print("="*70)
        break

if not found:
    print("❌ _initialize_bot metodu bulunamadı!")
    print("\n🔍 TelegramReporter içindeki tüm metodları listeleyeyim:")
    
    in_telegram = False
    for i, line in enumerate(lines):
        if 'class TelegramReporter:' in line:
            in_telegram = True
            continue
        
        if in_telegram:
            if line.startswith('class ') and 'TelegramReporter' not in line:
                break
            
            if 'def ' in line:
                print(f"  Satır {i+1}: {line.strip()}")
