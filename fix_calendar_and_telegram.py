#!/usr/bin/env python3
"""
Calendar ve Telegram sorunlarını düzelt
"""

import os
from pathlib import Path

print("🔧 Calendar ve Telegram düzeltiliyor...\n")

BASE_DIR = Path.home() / "Desktop" / "JTTWS"
bot_file = BASE_DIR / "ultimate_bot_v7_professional.py"

if not bot_file.exists():
    print(f"❌ {bot_file} bulunamadı!")
    exit(1)

# Dosyayı oku
with open(bot_file, 'r', encoding='utf-8') as f:
    content = f.read()

# ============================================================================
# FIX 1: NewsBlackout sınıfını güncelle
# ============================================================================
print("1/2 NewsBlackout sınıfı güncelleniyor (yeni calendar formatı)...")

# Eski NewsBlackout is_blackout metodunu bul
old_newsblackout = """    def is_blackout(self, dt: datetime, currency: str) -> Tuple[bool, Optional[str]]:
        \"\"\"
        Belirli bir zamanda haber blackout'u var mı kontrol et.
        Returns: (is_blackout, reason)
        \"\"\"
        if self.calendar_df is None:
            return False, None
        
        try:
            # İlgili currency için haberleri filtrele
            df = self.calendar_df[self.calendar_df['Currency'] == currency].copy()
            
            # Zaman aralığı oluştur
            before_time = dt - timedelta(minutes=self.config.NEWS_BLACKOUT_BEFORE)
            after_time = dt + timedelta(minutes=self.config.NEWS_BLACKOUT_AFTER)
            
            # Zaman içinde haber var mı?
            mask = (df['time'] >= before_time) & (df['time'] <= after_time)
            
            if mask.any():
                event = df[mask].iloc[0]
                reason = f"News: {event['Event']} at {event['time'].strftime('%H:%M')}"
                return True, reason
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"NewsBlackout kontrolünde hata: {e}")
            return False, None"""

new_newsblackout = """    def is_blackout(self, dt: datetime, currency: str) -> Tuple[bool, Optional[str]]:
        \"\"\"
        Belirli bir zamanda haber blackout'u var mı kontrol et.
        Returns: (is_blackout, reason)
        \"\"\"
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
                
                # Zaman aralığı oluştur
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
            return False, None"""

if old_newsblackout in content:
    content = content.replace(old_newsblackout, new_newsblackout)
    print("  ✓ NewsBlackout is_blackout metodu güncellendi")
else:
    print("  ⚠ NewsBlackout metodu bulunamadı (zaten güncel olabilir)")

# NewsBlackout load_calendar metodunu da güncelle
old_load = """    def load_calendar(self):
        \"\"\"Haber takvimini yükle\"\"\"
        if not self.config.NEWS_CALENDAR_FILE.exists():
            self.logger.warning(f"📰 News calendar dosyası yok, manuel filtre kullanılacak.")
            return
        
        try:
            self.calendar_df = pd.read_csv(self.config.NEWS_CALENDAR_FILE)
            
            # Tarih parse
            self.calendar_df['time'] = pd.to_datetime(self.calendar_df['Time'])
            
            self.logger.info(f"✅ News calendar yüklendi: {len(self.calendar_df)} events")
            
        except Exception as e:
            self.logger.error(f"❌ News calendar yüklenemedi: {e}")
            self.calendar_df = None"""

new_load = """    def load_calendar(self):
        \"\"\"Haber takvimini yükle\"\"\"
        if not self.config.NEWS_CALENDAR_FILE.exists():
            self.logger.warning(f"📰 News calendar dosyası yok: {self.config.NEWS_CALENDAR_FILE}")
            return
        
        try:
            self.calendar_df = pd.read_csv(self.config.NEWS_CALENDAR_FILE)
            
            # datetime kolonu zaten var (combined_economic_calendar.csv'de)
            if 'datetime' in self.calendar_df.columns:
                self.calendar_df['datetime'] = pd.to_datetime(self.calendar_df['datetime'])
            else:
                self.logger.error("❌ Calendar dosyasında 'datetime' kolonu yok!")
                self.calendar_df = None
                return
            
            # İstatistikler
            total = len(self.calendar_df)
            critical = len(self.calendar_df[self.calendar_df['Category'] == 'CRITICAL'])
            high = len(self.calendar_df[self.calendar_df['Category'] == 'HIGH'])
            
            self.logger.info(f"✅ News calendar yüklendi: {total:,} events")
            self.logger.info(f"   CRITICAL: {critical:,} | HIGH: {high:,}")
            
        except Exception as e:
            self.logger.error(f"❌ News calendar yüklenemedi: {e}")
            self.calendar_df = None"""

if old_load in content:
    content = content.replace(old_load, new_load)
    print("  ✓ NewsBlackout load_calendar metodu güncellendi")

# ============================================================================
# FIX 2: Telegram chat_id kontrolünü düzelt
# ============================================================================
print("2/2 Telegram chat_id kontrolü düzeltiliyor...")

# TelegramReporter __init__ metodunda chat_id kontrolünü bul
old_telegram_init = """        # Chat ID kontrolü
        if not self.config.TELEGRAM_CHAT_ID:
            self.logger.warning("📱 Telegram chat_id yok. /start gönderin.")
            self.enabled = False"""

new_telegram_init = """        # Chat ID kontrolü (int veya None olabilir)
        if self.config.TELEGRAM_CHAT_ID is None or self.config.TELEGRAM_CHAT_ID == 0:
            self.logger.warning("📱 Telegram chat_id yok. Bot config'te ayarlayın.")
            self.enabled = False
        else:
            self.logger.info(f"📱 Telegram chat_id: {self.config.TELEGRAM_CHAT_ID}")"""

if old_telegram_init in content:
    content = content.replace(old_telegram_init, new_telegram_init)
    print("  ✓ Telegram chat_id kontrolü düzeltildi")
else:
    print("  ⚠ Telegram init metodu bulunamadı (farklı olabilir)")

# Dosyayı yaz
with open(bot_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✅ Düzeltmeler tamamlandı!")
print("\n📋 YENİLİKLER:")
print("  ✓ Calendar artık yeni formatı (datetime kolonu) okuyor")
print("  ✓ CRITICAL/HIGH/MEDIUM kategorilere göre farklı blackout süreleri")
print("  ✓ Telegram chat_id doğru kontrol ediliyor")
print("\n🚀 Bot'u tekrar çalıştırın:")
print("   python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2023 --end-year 2024")
