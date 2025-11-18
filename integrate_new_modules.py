#!/usr/bin/env python3
"""
Yeni modülleri ana bot'a entegre et
"""

import os
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════╗
║   YENİ MODÜLLER ENTEGRE EDİLİYOR                                 ║
║   Email + Enhanced Logger + Weekly Reporter                      ║
╚══════════════════════════════════════════════════════════════════╝
""")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

if not bot_file.exists():
    print(f"❌ {bot_file} bulunamadı!")
    exit(1)

with open(bot_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("🔧 Import'lar ekleniyor...")

# Import bölümünü bul ve yeni import'ları ekle
new_imports = """# Enhanced modules
from email_notifier import EmailNotifier
from enhanced_trade_logger import EnhancedTradeLogger

"""

import_added = False
for i, line in enumerate(lines):
    if 'from weekly_reporter import WeeklyReporter' in line:
        # Bu satırdan sonra ekle
        lines.insert(i + 1, new_imports)
        import_added = True
        print("  ✓ Import'lar eklendi")
        break

if not import_added:
    print("  ⚠ Import eklenemedi (manuel ekleyin)")

# UltimateTradingSystem __init__'e email ve enhanced logger ekle
print("\\n🔧 UltimateTradingSystem'e yeni modüller ekleniyor...")

in_init = False
init_modified = False

for i, line in enumerate(lines):
    if 'def __init__(self, config:' in line and not init_modified:
        in_init = True
    
    if in_init and 'self.weekly_reporter = WeeklyReporter()' in line:
        # Bu satırdan sonra email ve enhanced logger ekle
        indent = '        '
        new_lines = [
            f'{indent}\\n',
            f'{indent}# Email notifications\\n',
            f'{indent}self.email_notifier = EmailNotifier(config, logger)\\n',
            f'{indent}\\n',
            f'{indent}# Enhanced trade logger\\n',
            f'{indent}self.trade_logger = EnhancedTradeLogger(logger, self.news_manager)\\n',
        ]
        
        for j, new_line in enumerate(new_lines):
            lines.insert(i + 1 + j, new_line)
        
        init_modified = True
        print("  ✓ __init__'e email ve enhanced logger eklendi")
        break

if not init_modified:
    print("  ⚠ __init__ modifikasyonu yapılamadı")

# Dosyayı yaz
with open(bot_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\\n" + "="*70)
print("✅ ENTEGRASYON TAMAMLANDI!")
print("="*70)

print("\\n📋 YAPILAN DEĞİŞİKLİKLER:")
print("  ✓ email_notifier ve enhanced_trade_logger import edildi")
print("  ✓ UltimateTradingSystem'e email_notifier eklendi")
print("  ✓ UltimateTradingSystem'e trade_logger eklendi")

print("\\n🚀 ŞİMDİ BOT'U TEST EDİN:")
print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2024 --end-year 2024")

print("\\n📧 EMAIL AYARLARI:")
print("   Gmail App Password almayı unutmayın!")
print("   bot_config.py → EMAIL_APP_PASSWORD")

print("="*70)
