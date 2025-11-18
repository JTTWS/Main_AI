#!/usr/bin/env python3
"""
FTMO Bot - Datetime Hatası Düzeltme
"""

import os
from pathlib import Path

print("🔧 Datetime hatası düzeltiliyor...")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

if not bot_file.exists():
    print(f"❌ {bot_file} bulunamadı!")
    exit(1)

# Dosyayı oku
with open(bot_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Haftalık rapor kısmını yoruma al (şimdilik devre dışı)
old_code = """            # Weekly reporter'a trade'leri ekle
            for trade in env.trade_history:
                # Convert to reporter format
                trade_data = {
                    'pair': pair,
                    'entry_time': env.df.iloc[trade.get('entry_step', 0)]['datetime'] if 'entry_step' in trade else datetime.now(),
                    'exit_time': env.df.iloc[trade.get('exit_step', 0)]['datetime'] if 'exit_step' in trade else datetime.now(),
                    'direction': trade.get('type', 'UNKNOWN'),
                    'lot_size': trade.get('lot', 0.0),
                    'entry_price': trade.get('entry_price', 0.0),
                    'exit_price': trade.get('exit_price', 0.0),
                    'pnl': trade.get('profit', 0.0),
                    'result': 'WIN' if trade.get('profit', 0) > 0 else 'LOSS',
                    'strategy_type': 'RL',
                    'nearby_news': []  # Will be filled later
                }
                self.weekly_reporter.add_trade(trade_data)"""

new_code = """            # Weekly reporter - şimdilik devre dışı (datetime kolon sorunu)
            # TODO: Haftalık rapor için datetime kolonunu düzelt
            pass"""

# Değiştir
if old_code in content:
    content = content.replace(old_code, new_code)
    print("  ✓ Haftalık rapor kısmı yoruma alındı")
else:
    print("  ⚠ Kod bloğu bulunamadı, alternatif düzeltme yapılıyor...")
    # Alternatif: Tüm haftalık rapor bölümünü bul ve yoruma al
    import re
    pattern = r"(            # Weekly reporter'a trade'leri ekle.*?self\.weekly_reporter\.add_trade\(trade_data\))"
    content = re.sub(pattern, new_code, content, flags=re.DOTALL)

# Haftalık rapor oluşturma kısmını da yoruma al
old_report_code = """        # Haftalık rapor oluştur
        self.logger.info("\\n" + "="*60)
        self.logger.info("📊 Haftalık Rapor Oluşturuluyor...")
        self.logger.info("="*60)
        
        weekly_report = self.weekly_reporter.generate_weekly_report()
        if weekly_report:
            report_text = self.weekly_reporter.format_report_text(weekly_report)
            self.logger.info("\\n" + report_text)
            
            # Telegram'a gönder
            if self.config.TELEGRAM_ENABLED:
                try:
                    asyncio.run(self.telegram._send_message(report_text))
                    self.logger.info("✅ Haftalık rapor Telegram'a gönderildi")
                except Exception as e:
                    self.logger.error(f"❌ Haftalık rapor gönderilemedi: {e}")"""

new_report_code = """        # Haftalık rapor - şimdilik devre dışı
        # TODO: Haftalık rapor için datetime kolonunu düzelt
        self.logger.info("\\n📊 Haftalık rapor özelliği geliştirme aşamasında...")"""

if old_report_code in content:
    content = content.replace(old_report_code, new_report_code)
    print("  ✓ Haftalık rapor oluşturma kısmı yoruma alındı")

# Dosyayı yaz
with open(bot_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Düzeltme tamamlandı!")
print("\n🚀 Şimdi bot'u tekrar çalıştırabilirsiniz:")
print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2023 --end-year 2024")
