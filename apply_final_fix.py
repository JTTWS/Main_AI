#!/usr/bin/env python3
"""
Ultimate Bot V7 - FİNAL DÜZELTMELİ
NewsBlackout ve TelegramReporter'ı tamamen yeniden yazıyor
"""

import os
import re
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════╗
║   ULTIMATE BOT V7 - FINAL FIX                                    ║
║   NewsBlackout + Telegram tamamen düzeltiliyor...               ║
╚══════════════════════════════════════════════════════════════════╝
""")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

if not bot_file.exists():
    print(f"❌ {bot_file} bulunamadı!")
    exit(1)

# Dosyayı oku
with open(bot_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("🔍 NewsBlackout sınıfı bulunuyor...")

# NewsBlackout sınıfını tamamen değiştir
new_newsblackout_class = '''class NewsBlackout:
    """
    Gelişmiş haber blackout sistemi.
    - Kategori bazlı blackout süreleri (CRITICAL/HIGH/MEDIUM)
    - combined_economic_calendar.csv entegrasyonu
    """
    
    def __init__(self, config: BotConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.calendar_df = None
        self.load_calendar()
    
    def load_calendar(self):
        """Haber takvimini yükle"""
        if not self.config.NEWS_CALENDAR_FILE.exists():
            self.logger.warning(f"📰 News calendar dosyası yok: {self.config.NEWS_CALENDAR_FILE}")
            return
        
        try:
            self.calendar_df = pd.read_csv(self.config.NEWS_CALENDAR_FILE)
            
            # datetime kolonu zaten var (combined_economic_calendar.csv'de)
            if 'datetime' in self.calendar_df.columns:
                self.calendar_df['datetime'] = pd.to_datetime(self.calendar_df['datetime'])
                
                # İstatistikler
                total = len(self.calendar_df)
                critical = len(self.calendar_df[self.calendar_df['Category'] == 'CRITICAL'])
                high = len(self.calendar_df[self.calendar_df['Category'] == 'HIGH'])
                medium = len(self.calendar_df[self.calendar_df['Category'] == 'MEDIUM'])
                
                self.logger.info(f"✅ News calendar yüklendi: {total:,} events")
                self.logger.info(f"   CRITICAL: {critical:,} | HIGH: {high:,} | MEDIUM: {medium:,}")
            else:
                self.logger.error("❌ Calendar dosyasında 'datetime' kolonu yok!")
                self.calendar_df = None
            
        except Exception as e:
            self.logger.error(f"❌ News calendar yüklenemedi: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.calendar_df = None
    
    def is_blackout(self, dt: datetime, currency: str) -> Tuple[bool, Optional[str]]:
        """
        Belirli bir zamanda haber blackout'u var mı kontrol et.
        Returns: (is_blackout, reason)
        """
        if self.calendar_df is None:
            return False, None
        
        try:
            # İlgili currency için haberleri filtrele
            df = self.calendar_df[self.calendar_df['Currency'] == currency].copy()
            
            if df.empty:
                return False, None
            
            # Her kategori için farklı blackout süreleri
            for category in ['CRITICAL', 'HIGH', 'MEDIUM']:
                cat_events = df[df['Category'] == category]
                
                if cat_events.empty:
                    continue
                
                # Blackout sürelerini belirle
                if category == 'CRITICAL':
                    before_min = self.config.NEWS_BLACKOUT_CRITICAL_BEFORE
                    after_min = self.config.NEWS_BLACKOUT_CRITICAL_AFTER
                elif category == 'HIGH':
                    before_min = self.config.NEWS_BLACKOUT_HIGH_BEFORE
                    after_min = self.config.NEWS_BLACKOUT_HIGH_AFTER
                else:  # MEDIUM
                    before_min = self.config.NEWS_BLACKOUT_MEDIUM_BEFORE
                    after_min = self.config.NEWS_BLACKOUT_MEDIUM_AFTER
                
                # Zaman aralığı kontrolü
                for _, event in cat_events.iterrows():
                    event_time = event['datetime']
                    before_time = event_time - timedelta(minutes=before_min)
                    after_time = event_time + timedelta(minutes=after_min)
                    
                    # Şu an blackout penceresinde mi?
                    if before_time <= dt <= after_time:
                        reason = f"{category}: {event['Name']} at {event_time.strftime('%H:%M')}"
                        self.logger.debug(f"🚫 Blackout active: {reason}")
                        return True, reason
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"NewsBlackout kontrolünde hata: {e}")
            return False, None

'''

# NewsBlackout sınıfını bul ve değiştir
in_newsblackout = False
start_line = -1
end_line = -1

for i, line in enumerate(lines):
    if 'class NewsBlackout:' in line:
        in_newsblackout = True
        start_line = i
        continue
    
    if in_newsblackout and line.startswith('class ') and 'NewsBlackout' not in line:
        end_line = i
        break
    
    if in_newsblackout and line.startswith('# ====') and 'VOLATILITY' in line:
        end_line = i
        break

if start_line != -1 and end_line != -1:
    print(f"  ✓ NewsBlackout bulundu (satır {start_line+1} - {end_line+1})")
    # NewsBlackout sınıfını değiştir
    lines[start_line:end_line] = [new_newsblackout_class + '\n\n']
    print("  ✓ NewsBlackout tamamen yeniden yazıldı")
else:
    print("  ⚠ NewsBlackout sınıfı bulunamadı")

# TelegramReporter __init__ metodunu düzelt
print("\n🔍 TelegramReporter düzeltiliyor...")

in_telegram_init = False
init_start = -1
init_end = -1

for i, line in enumerate(lines):
    if 'class TelegramReporter:' in line:
        # __init__ metodunu bul
        for j in range(i, min(i+100, len(lines))):
            if 'def __init__' in lines[j]:
                init_start = j
                # __init__'in sonunu bul (bir sonraki metod başlangıcına kadar)
                for k in range(j+1, min(j+100, len(lines))):
                    if lines[k].strip().startswith('def ') and '__init__' not in lines[k]:
                        init_end = k
                        break
                    if lines[k].strip().startswith('async def '):
                        init_end = k
                        break
                break
        break

if init_start != -1 and init_end != -1:
    print(f"  ✓ TelegramReporter __init__ bulundu (satır {init_start+1} - {init_end+1})")
    
    # __init__ içindeki chat_id kontrolünü değiştir
    for i in range(init_start, init_end):
        if 'if not self.config.TELEGRAM_CHAT_ID:' in lines[i]:
            # Bu satırı ve sonraki 2 satırı değiştir
            indent = '        '
            lines[i] = f'{indent}# Chat ID kontrolü\n'
            lines[i+1] = f'{indent}if self.config.TELEGRAM_CHAT_ID is None or self.config.TELEGRAM_CHAT_ID == 0:\n'
            lines[i+2] = f'{indent}    self.logger.warning("📱 Telegram chat_id yok. Bot config\'te ayarlayın.")\n'
            lines.insert(i+3, f'{indent}    self.enabled = False\n')
            lines.insert(i+4, f'{indent}else:\n')
            lines.insert(i+5, f'{indent}    self.logger.info(f"✅ Telegram etkin - Chat ID: {{self.config.TELEGRAM_CHAT_ID}}")\n')
            print("  ✓ Telegram chat_id kontrolü düzeltildi")
            break

# Dosyayı yaz
with open(bot_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\n" + "="*70)
print("✅ FİNAL DÜZELTMELERİ TAMAMLANDI!")
print("="*70)
print("\n📋 YAPILAN DEĞİŞİKLİKLER:")
print("  ✓ NewsBlackout sınıfı tamamen yeniden yazıldı")
print("  ✓ - combined_economic_calendar.csv formatını okuyor")
print("  ✓ - datetime kolonunu kullanıyor")
print("  ✓ - Kategori bazlı blackout (CRITICAL/HIGH/MEDIUM)")
print("  ✓ TelegramReporter chat_id kontrolü düzeltildi")
print("\n🚀 ŞİMDİ BOT'U ÇALIŞTIRIN:")
print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2023 --end-year 2024")
print("\n✅ BEKLENEN ÇIKTI:")
print("   ✅ News calendar yüklendi: 83,522 events")
print("   ✅ Telegram etkin - Chat ID: 1590841427")
print("="*70)
