# 🚀 ULTIMATE FTMO TRADING BOT V7.0 PROFESSIONAL
## Türkçe Kullanım Kılavuzu - MacBook M1 Kurulum ve Çalıştırma

---

## 📋 İçindekiler
1. [Sistem Gereksinimleri](#sistem-gereksinimleri)
2. [Kurulum Adımları](#kurulum-adımları)
3. [İlk Yapılandırma](#ilk-yapılandırma)
4. [Bot'u Çalıştırma](#botu-çalıştırma)
5. [Telegram Entegrasyonu](#telegram-entegrasyonu)
6. [Özellikler ve Stratejiler](#özellikler-ve-stratejiler)
7. [Sorun Giderme](#sorun-giderme)
8. [SSS](#sss)

---

## 🖥️ Sistem Gereksinimleri

### Donanım
- **İşlemci:** Apple M1 veya üzeri
- **RAM:** Minimum 8GB (16GB önerilir)
- **Depolama:** En az 5GB boş alan

### Yazılım
- **İşletim Sistemi:** macOS 11 (Big Sur) veya üzeri
- **Python:** 3.9 veya 3.10 (3.11 de çalışır)
- **Homebrew:** Python ve bağımlılıkları yüklemek için

---

## 📦 Kurulum Adımları

### Adım 1: Python Kurulumu (Eğer yoksa)

Terminal'i açın ve şunları çalıştırın:

```bash
# Homebrew ile Python yükleyin
brew install python@3.10

# Python versiyonunu kontrol edin
python3 --version
```

### Adım 2: Proje Klasörünü Hazırlayın

Masaüstünüzde `JTTWS` klasörü olmalı. Eğer yoksa:

```bash
cd ~/Desktop
mkdir -p JTTWS/data
mkdir -p JTTWS/logs
mkdir -p JTTWS/models
mkdir -p JTTWS/outputs
```

### Adım 3: Data Dosyalarını Kontrol Edin

`~/Desktop/JTTWS/data/` klasöründe şunlar olmalı:

```
JTTWS/
├── data/
│   ├── EURUSD2003-2024/          # EURUSD candlestick verileri
│   ├── GBPUSD2003-2024/          # GBPUSD candlestick verileri
│   ├── USDJPY2003-2024/          # USDJPY candlestick verileri
│   ├── EURUSD_weekly_ranges.csv  # EURUSD haftalık range'ler
│   ├── GBPUSD_weekly_ranges.csv  # GBPUSD haftalık range'ler
│   └── USDJPY_weekly_ranges.csv  # USDJPY haftalık range'ler
```

**Kontrol için:**

```bash
ls -lh ~/Desktop/JTTWS/data/
```

### Adım 4: Bot Dosyalarını Yerleştirin

İndirdiğiniz bot dosyalarını `~/Desktop/JTTWS/` klasörüne kopyalayın:

- `ultimate_bot_v7_professional.py`
- `bot_config.py`
- `requirements.txt`
- `KULLANIM_KILAVUZU.md` (bu dosya)

```bash
cd ~/Desktop/JTTWS
ls -l *.py
# Şunları görmelisiniz:
# ultimate_bot_v7_professional.py
# bot_config.py
```

### Adım 5: Python Sanal Ortamı Oluşturun

```bash
cd ~/Desktop/JTTWS

# Sanal ortam oluşturun
python3 -m venv venv

# Sanal ortamı aktifleştirin
source venv/bin/activate

# Pip'i güncelleyin
pip install --upgrade pip
```

### Adım 6: Bağımlılıkları Yükleyin

**ÖNEMLİ:** TA-Lib sistem kütüphanesi gerektirir. Önce onu yükleyin:

```bash
# Homebrew ile TA-Lib yükleyin
brew install ta-lib

# Şimdi Python paketlerini yükleyin
pip install -r requirements.txt
```

**Not:** PyTorch M1 için optimize edilmiş versiyonu otomatik yüklenecektir.

**Yükleme sırasında sorun çıkarsa:**

```bash
# TA-Lib yerine pandas-ta kullanın (alternatif)
pip install pandas-ta
```

---

## ⚙️ İlk Yapılandırma

### 1. Telegram Bot Token'ınızı Ayarlayın

`bot_config.py` dosyasını açın ve token'ınızı kontrol edin:

```python
TELEGRAM_TOKEN = "8008545474:AAHansC5Xag1b9N96bMAGE0YLTfykXoOPyY"
TELEGRAM_ENABLED = True
```

**Token zaten ayarlanmış! ✅** Değiştirmenize gerek yok.

### 2. Diğer Ayarları İnceleyin (Opsiyonel)

`bot_config.py` içinde şunları özelleştirebilirsiniz:

- **Risk Ayarları:**
  - `DAILY_RISK_LIMIT` - Günlük toplam risk (%5 varsayılan)
  - `MAX_RISK_PER_PAIR` - Pair başına risk (%33 varsayılan)
  
- **Trading Saatleri:**
  - `TRADING_END_HOUR = 22` (22:30'dan sonra yeni giriş yok)
  - `FORCE_CLOSE_HOUR = 23` (23:00'da tüm pozisyonlar kapanır)

- **Volatilite Korumaları:**
  - `RANGE_GUARD_PERCENTILE = 95` (p95)
  - `GAP_GUARD_ATR_MULTIPLIER = 1.5`
  - `SHALLOW_HOUR_ATR_MULTIPLIER = 0.5`

---

## 🚀 Bot'u Çalıştırma

### Mod 1: Backtest (Geçmiş verilerde test)

2020-2024 arası verilerde backtest:

```bash
cd ~/Desktop/JTTWS
source venv/bin/activate

python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2020 --end-year 2024
```

**Çıktı:**
- Ekranda adım adım log'lar
- Telegram'a özet bildirimler
- `outputs/` klasöründe detaylı rapor ve grafikler

### Mod 2: Train (Model eğitimi)

2003-2019 arası verilerle RL modelini eğitin:

```bash
python3 ultimate_bot_v7_professional.py --mode train --start-year 2003 --end-year 2019
```

**Not:** Eğitim uzun sürebilir (saatler). İlerleme çubuğu gösterilir.

### Mod 3: Paper Trading (Canlı simülasyon)

**ÖNEMLİ:** Paper trading için MT5 bağlantısı gerekir. Önce `bot_config.py`'de ayarlayın:

```python
MT5_ENABLED = True
MT5_LOGIN = "sizin_login"
MT5_PASSWORD = "sizin_password"
MT5_SERVER = "sizin_server"
```

Sonra çalıştırın:

```bash
python3 ultimate_bot_v7_professional.py --mode paper
```

**Not:** M1 Mac'te MT5 native çalışmayabilir. Backtest modunu tercih edin.

---

## 📱 Telegram Entegrasyonu

### Bot'unuzu Telegram'da Bulun

1. Telegram'da `@jttws_egitim_bot` kullanıcı adını arayın
2. Bota `/start` komutunu gönderin
3. Bot size hoş geldin mesajı gönderecek

### Telegram Komutları

Bot çalışırken şu komutları kullanabilirsiniz:

- `/start` - Bot'u başlat ve chat_id al
- `/status` - Güncel durum özeti
- `/positions` - Açık pozisyonlar
- `/performance` - Performans özeti
- `/stop` - Bot'u durdur (sadece bildirimler)

### Otomatik Bildirimler

Bot şunları otomatik gönderir:

✅ **Trade Bildirimleri:**
- Yeni pozisyon açıldığında
- Pozisyon kapatıldığında (kar/zarar)
- Stop-loss veya take-profit tetiklendiğinde

📊 **Günlük Raporlar:**
- Gün sonu özeti (23:00'dan sonra)
- Toplam kar/zarar
- Win rate, Sharpe ratio
- En iyi/en kötü trade

⚠️ **Uyarılar:**
- Günlük risk limitine yaklaşıldığında
- Art arda kayıp durumunda
- Volatilite koruması devreye girdiğinde

---

## 🎯 Özellikler ve Stratejiler

### V7.0'da Neler Var?

#### 1. **RightsManager (Hak Yönetimi)**
- Günlük bütçe takibi
- Pair başına adil risk dağılımı
- Saatlik işlem hakkı tahsisi

#### 2. **WeeklyRangeLearner**
- CSV'den haftalık range verileri okur
- Her pair için istatistikler hesaplar (avg, p95, max)
- RangeGuard için threshold belirler

#### 3. **NewsBlackout**
- Yüksek etkili haberler etrafında işlem yapma
- Haber öncesi 30 dk, sonrası 30 dk
- (Opsiyonel: `news_calendar.csv` ekleyebilirsiniz)

#### 4. **VolatilityGuards (3 Koruma)**

**a) RangeGuard:**
- Haftalık range > p95 ise giriş yapma
- "Bu hafta çok volatil, bekle!"

**b) GapGuard:**
- Açılış farkı > 1.5x ATR ise giriş yapma
- "Sabah büyük gap var, riskli!"

**c) ShallowHour:**
- Saatlik bar range < 0.5x ATR ise giriş yapma
- "Bu saat çok durgun, sinyal güvenilmez!"

#### 5. **TrendFilter & Distance**
- Trend yönü: SMA20 vs SMA50
- Trend gücü: ADX > 25
- Distance: Fiyat SMA'dan 2 ATR'den fazla uzaksa giriş yapma

#### 6. **CorrelationControl**
- Aynı yönde maksimum 2 pozisyon
- "EURUSD long + GBPUSD long = OK"
- "EURUSD long + GBPUSD long + USDJPY long = HAYIR!"

#### 7. **Sequential Loss/Profit Lock**
- 3 art arda kayıp → trading durdur
- Günlük profit hedefinin %20'sine ulaşıldığında art arda 2 kar → dur

#### 8. **HourlyAllocator**
- Her saate 3 işlem hakkı
- Haklar biterse o saatte yeni giriş yok
- Saat başında haklar yenilenir

#### 9. **ThompsonBandit**
- 4 sinyal tipi: TREND, MEAN_REVERSION, BREAKOUT, MOMENTUM
- Her sinyal başarı/başarısızlık kaydedilir
- En iyi performans gösteren sinyal otomatik seçilir

#### 10. **TelegramReporter**
- Zengin formatla Türkçe bildirimler
- Emoji'li, renkli mesajlar
- Grafikler ve performans özeti

#### 11. **23:00 Forced Close & 22:30 No New Entries**
- Saat kontrolü (UTC+3)
- Pozisyonlar gecede kalmaz!

#### 12. **Dynamic Position Sizing**
- Kelly Criterion (1/4 fraksiyonu)
- VaR/CVaR hesabı
- ATR bazlı SL/TP

---

## 🔧 Sorun Giderme

### Sorun 1: "Data klasörü bulunamadı"

**Çözüm:**
```bash
# Klasör yapısını kontrol edin
ls -la ~/Desktop/JTTWS/data/

# Eksik klasörleri oluşturun
mkdir -p ~/Desktop/JTTWS/data/EURUSD2003-2024
```

### Sorun 2: "TA-Lib yüklenemiyor"

**Çözüm 1:** Homebrew ile yükleyin
```bash
brew install ta-lib
pip install TA-Lib
```

**Çözüm 2:** Alternatif kullanın
```bash
pip uninstall TA-Lib
pip install pandas-ta
```

Sonra `ultimate_bot_v7_professional.py` içinde import'u değiştirin:
```python
# import talib as ta
import pandas_ta as ta
```

### Sorun 3: "PyTorch yüklenemiyor (M1 Mac)"

**Çözüm:**
```bash
# M1 için native PyTorch
pip3 install torch torchvision torchaudio
```

### Sorun 4: "Telegram bildirimleri gelmiyor"

**Kontroller:**
1. Bot'a `/start` gönderdiniz mi?
2. `bot_config.py`'de `TELEGRAM_ENABLED = True` mı?
3. Token doğru mu?

**Debug:**
```bash
python3 -c "from bot_config import BotConfig; print(BotConfig.TELEGRAM_TOKEN)"
```

### Sorun 5: "Bot çok yavaş çalışıyor"

**Çözüm:**
- Backtest yıl aralığını küçültün (örn. 2022-2024)
- `bot_config.py`'de `RL_BATCH_SIZE`'ı azaltın (32)
- Sadece 1 pair ile test edin

### Sorun 6: "No trades" - Bot hiç işlem yapmıyor

**Muhtemel Sebepler:**

1. **Volatilite korumaları çok sıkı:**
   - `bot_config.py`'de `RANGE_GUARD_PERCENTILE = 99` yapın
   - `GAP_GUARD_ATR_MULTIPLIER = 3.0` artırın

2. **Trend filtresi çok katı:**
   - `MIN_TREND_STRENGTH = 0.1` azaltın

3. **Korelasyon limiti çok düşük:**
   - `MAX_CORRELATED_POSITIONS = 3` artırın

**Test için tüm filtreleri geçici devre dışı bırakın:**

Bot içinde `_check_all_filters()` fonksiyonunu bulun ve return True yapın.

---

## ❓ SSS (Sıkça Sorulan Sorular)

### S: Bot gerçek parada çalışır mı?
**C:** Hayır, bu backtest ve paper trading botudur. Gerçek para için broker entegrasyonu gerekir.

### S: Kaç sermaye ile başlamalıyım?
**C:** `bot_config.py`'de `INITIAL_CAPITAL` değişkenini ayarlayın. Varsayılan $100,000.

### S: Bot 7/24 çalışır mı?
**C:** Backtest modunda hayır (bitince durur). Paper trading modunda evet (sürekli çalışabilir).

### S: Hangi zaman dilimini kullanıyor?
**C:** UTC+3 (İstanbul saati). Trading saatleri buna göre ayarlıdır.

### S: Veriler nereden geliyor?
**C:** Siz kendi verilerinizi sağlıyorsunuz (`data/` klasöründe). Bot bunları okur.

### S: Model ne kadar sürede eğitilir?
**C:** 2003-2019 arası 17 yıllık veri ile 4-6 saat (M1 Mac'te). Daha kısa periyot seçebilirsiniz.

### S: Telegram olmadan çalışır mı?
**C:** Evet! `bot_config.py`'de `TELEGRAM_ENABLED = False` yapın.

### S: Başka currency pair ekleyebilir miyim?
**C:** Evet. Verilerini ekleyin, `bot_config.py`'de `PAIRS` listesine ekleyin.

---

## 📞 Destek ve İletişim

**Sorun mu yaşıyorsunuz?**

1. Önce bu kılavuzu baştan okuyun
2. "Sorun Giderme" bölümüne bakın
3. Log dosyalarını inceleyin: `~/Desktop/JTTWS/logs/`
4. Telegram bot'a `/status` gönderin

**Hala çözemediyseniz:**

- Bot loglarını kaydedin
- Hata mesajını tam olarak not edin
- Terminal çıktısını screenshot alın

---

## 🎉 Başarılar!

Bot'unuz hazır! **Küçük test ile başlayın:**

```bash
cd ~/Desktop/JTTWS
source venv/bin/activate
python3 ultimate_bot_v7_professional.py --mode backtest --start-year 2023 --end-year 2024
```

**İlk backtest'iniz başarılı olduysa, tebrikler! 🚀**

Şimdi parametreleri optimize edin ve sistemi daha da geliştirin.

---

**Ultimate FTMO Trading Bot V7.0 Professional**  
*"Clockwork Reliability, Maximum Transparency"* ⚙️📊
