# 🔧 BOT GÜNCELLEMESİ - ADIM ADIM KILAVUZ

## ⚠️ SORUN
Bot çalışırken `AttributeError: NEWS_BLACKOUT_CRITICAL_BEFORE` hatası alıyorsunuz çünkü:
- Emergent ortamındaki dosyalar güncellendi ✅
- Ama sizin MacBook'unuzdaki dosyalar eski haliyle ❌

## ✅ ÇÖZÜM: 5 ADIMDA GÜNCELLEME

### ADIM 1: Yeni Dosyaları İndir
Emergent arayüzünden şu dosyaları indirin:
1. `combine_calendars.py`
2. `news_manager.py`
3. `weekly_reporter.py`
4. `bot_config.py` (YENİ HALİ)
5. `ultimate_bot_v7_professional.py` (YENİ HALİ)

### ADIM 2: Dosyaları Kopyala
```bash
# MacBook terminalinde:
cd ~/Desktop/JTTWS

# İndirdiğiniz dosyaları buraya kopyalayın:
# - combine_calendars.py
# - news_manager.py
# - weekly_reporter.py
# - bot_config.py (eskisinin yerine)
# - ultimate_bot_v7_professional.py (eskisinin yerine)
```

### ADIM 3: Calendar'ı Birleştir
```bash
cd ~/Desktop/JTTWS
python3 combine_calendars.py
```

Bu komut:
- 18 CSV'yi birleştirecek
- `data/combined_economic_calendar.csv` oluşturacak

### ADIM 4: Test Et
```bash
# Backtest dene
python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2023 --end-year 2024
```

### ADIM 5: Eğitim Başlat
```bash
# Eğitime başla
python3 ultimate_bot_v7_professional.py --mode train --start-year 2003 --end-year 2022 --episodes 50
```

---

## 🚨 HIZLI ÇÖZÜM (Sadece bot_config.py)

Eğer sadece hatayı düzeltmek istiyorsanız, bot_config.py dosyanızda şu satırları bulun:

```python
# ESKI (satır ~93-99):
    # ==================== NEWS BLACKOUT ====================
    # Yüksek etkili haberlerden önce/sonra işlem yapma (dakika)
    NEWS_BLACKOUT_BEFORE = 30  # Haber öncesi 30 dk
    NEWS_BLACKOUT_AFTER = 30   # Haber sonrası 30 dk
    
    # Haber takvimi dosyası (opsiyonel)
    NEWS_CALENDAR_FILE = DATA_DIR / "news_calendar.csv"
```

Şununla değiştirin:

```python
# YENİ:
    # ==================== NEWS BLACKOUT ====================
    # Birleştirilmiş ekonomik takvim dosyası
    NEWS_CALENDAR_FILE = DATA_DIR / "combined_economic_calendar.csv"
    
    # Haber kategorilerine göre blackout süreleri (dakika)
    NEWS_BLACKOUT_CRITICAL_BEFORE = 60  # CRITICAL haberler öncesi 60 dk
    NEWS_BLACKOUT_CRITICAL_AFTER = 60   # CRITICAL haberler sonrası 60 dk
    
    NEWS_BLACKOUT_HIGH_BEFORE = 30      # HIGH haberler öncesi 30 dk
    NEWS_BLACKOUT_HIGH_AFTER = 30       # HIGH haberler sonrası 30 dk
    
    NEWS_BLACKOUT_MEDIUM_BEFORE = 15    # MEDIUM haberler öncesi 15 dk
    NEWS_BLACKOUT_MEDIUM_AFTER = 15     # MEDIUM haberler sonrası 15 dk
    
    # LOW impact haberler için blackout YOK
```

Kaydedin ve tekrar deneyin!

---

## 📦 DOSYA LİSTESİ

İndirmeniz gereken dosyalar:
- [ ] `combine_calendars.py` - Calendar birleştirici
- [ ] `news_manager.py` - Haber yönetim sistemi
- [ ] `weekly_reporter.py` - Haftalık rapor sistemi
- [ ] `bot_config.py` - Güncellenmiş config
- [ ] `ultimate_bot_v7_professional.py` - Güncellenmiş ana bot

---

## 🆘 SORUN MU VAR?

Hala hata alıyorsanız:
1. `bot_config.py` içinde `NEWS_BLACKOUT_CRITICAL_BEFORE` arayın
2. Yoksa yukarıdaki kod bloğunu ekleyin
3. `NEWS_CALENDAR_FILE` satırını `combined_economic_calendar.csv` olarak değiştirin

İşe yaramazsa, tüm dosyaları yeniden indirin ve kopyalayın.
