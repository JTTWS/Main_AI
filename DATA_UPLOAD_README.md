# 📦 JTTWS V8 VERİ YÜKLEME REHBERİ

## 🎯 GENEL BAKIŞ

Bu rehber, lokal bilgisayarınızdaki JTTWS forex verilerini Emergent konteynerine yükleme sürecini açıklar.

---

## 📊 VERİ YAPISI

### Kaynak (Lokal Bilgisayarınız):
```
~/Desktop/JTTWS/data/
├── EURUSD2003-2024/          (9 CSV, 43MB, 15M timeframe)
├── GBPUSD2003-2024/          (9 CSV, 42MB, 15M timeframe)
├── USDJPY2003-2024/          (9 CSV, 42MB, 15M timeframe)
├── EURUSD_weekly_ranges.csv  (138KB, 1118 hafta)
├── GBPUSD_weekly_ranges.csv  (138KB, 1118 hafta)
├── USDJPY_weekly_ranges.csv  (133KB, 1118 hafta)
├── combined_economic_calendar.csv (9.4MB)
└── ... (parçalı weekly range dosyaları)

TOPLAM: ~138MB (uncompressed), ~40-50MB (compressed)
```

### Hedef (Emergent Konteyner):
```
/app/data/
└── (aynı yapı)
```

---

## 🚀 ADIMLAR

### ADIM 1: VERİLERİ ZIP'LEYIN (Lokal Terminal)

Lokal Mac terminalinizde şu komutları çalıştırın:

```bash
cd ~/Desktop/JTTWS/
tar -czf jttws_data_complete.tar.gz data/
ls -lh jttws_data_complete.tar.gz
```

**Beklenen Çıktı:**
```
-rw-r--r--  1 serkanozturk  staff    42M  6 Kas 17:00 jttws_data_complete.tar.gz
```

---

### ADIM 2: ZIP DOSYASINI KONTEYNER'A KOPYALAYIN

Bu konteyner dosya sistemine erişiminiz var. İki seçenek:

#### Seçenek A: Emergent UI Üzerinden Upload
1. Emergent.sh arayüzüne gidin
2. File Manager açın
3. `jttws_data_complete.tar.gz` dosyasını `/app/` klasörüne upload edin

#### Seçenek B: Manuel Kopyalama (Eğer erişiminiz varsa)
```bash
# Lokal terminalden
cp ~/Desktop/JTTWS/jttws_data_complete.tar.gz /path/to/container/app/
```

---

### ADIM 3: EXTRACT EDİN (Emergent Terminal)

Emergent konteyner terminalinde:

```bash
cd /app
python upload_data.py
```

**Bu script:**
- ✅ Tar dosyasını bulur ve boyutunu kontrol eder
- ✅ `/app/data/` klasörüne extract eder
- ✅ Veri yapısını doğrular
- ✅ Özet rapor sunar

---

### ADIM 4: VERİ YÜKLEME TESTİ

DataManagerV8'i test edin:

```bash
python data_manager_v8.py
```

**Beklenen Çıktı:**
```
📂 DataManagerV8 initialized with data_dir: /app/data
📥 Loading EURUSD data (2003-01-01 to 2024-12-31)...
📂 Found 9 CSV files for EURUSD
   ✓ Loaded ... rows from EURUSD_Candlestick_15_M_BID_01.01.2006-01.01.2009.csv
   ...
✅ Loaded 500000+ rows for EURUSD from /app/data/EURUSD2003-2024

✅ EURUSD Data Shape: (500000, 6)
   timestamp                open     high      low    close  volume
0  2006-01-01 00:00:00  1.18460  1.18460  1.18460  1.18460       0
...
```

---

## 🧪 V8 TRAINING'İ BAŞLATIN

Veriler yüklendikten sonra:

### Test 1: Backtest Mode (Hızlı Test)
```bash
python ultimate_bot_v8_ppo.py --mode backtest --years 2020-2024 --use-ppo
```

### Test 2: Walk-Forward Training (Grok Önerileriyle)
```bash
python ultimate_bot_v8_ppo.py --mode train --optuna-trials 50 --years 2020-2024
```

**Parametreler (Grok Optimizasyonu):**
- 🔹 Window: 180 gün train / 60 gün test
- 🔹 Optuna Trials: 50 (ilk test), 100 (production)
- 🔹 Decay Threshold: 20% (ilk 3 periyot), 12% (sonrası)

---

## 📋 KONTROL LİSTESİ

- [ ] Lokal verileri ZIP'ledim (`tar -czf`)
- [ ] ZIP boyutunu kontrol ettim (~40-50MB)
- [ ] ZIP'i `/app/` klasörüne kopyaladım
- [ ] `upload_data.py` çalıştırdım
- [ ] Veri yapısı doğrulandı
- [ ] `data_manager_v8.py` test ettim
- [ ] EURUSD/GBPUSD/USDJPY yüklendi
- [ ] Weekly ranges yüklendi
- [ ] Economic calendar yüklendi
- [ ] V8 backtest başarılı
- [ ] Walk-forward training başladı

---

## ❓ SORUN GİDERME

### Sorun 1: "Tar file not found"
**Çözüm:** ZIP dosyasının `/app/jttws_data_complete.tar.gz` yolunda olduğundan emin olun.

```bash
ls -lh /app/jttws_data_complete.tar.gz
```

### Sorun 2: "No CSV files found for EURUSD"
**Çözüm:** Extract işlemi başarısız olmuş olabilir. Tekrar deneyin:

```bash
cd /app
tar -xzf jttws_data_complete.tar.gz
ls -lh data/
```

### Sorun 3: "Data directory not found after extraction"
**Çözüm:** Tar dosyası içindeki yapı hatalı olabilir. Kontrol edin:

```bash
tar -tzf jttws_data_complete.tar.gz | head -20
```

**Beklenen:** `data/EURUSD2003-2024/...` gibi yollar olmalı.

---

## 📊 GERÇEKÇİ BEKLENT İLER (Grok Analizi)

### OHLCV Verileri:
- **Timeframe:** 15 dakika
- **Satır Sayısı:** ~500,000 - 1,000,000 per parite
- **Dönem:** 2003-2024 (21 yıl)

### Walk-Forward Training Süreleri:
- **1 Optuna Trial:** ~5-10 saniye
- **50 Trial (1 period):** ~5-10 dakika
- **5 Period (180/60 window):** ~30-50 dakika

### Performans Metrikleri:
- **Sharpe Ratio:** 1.1 - 1.4 (hedef)
- **Win Rate:** 55% - 62%
- **Max Drawdown:** -8% to -12% (FTMO: <-5% ideal)
- **Profit Factor:** 1.4 - 1.8

---

## 🚀 SONRAKI ADIMLAR

1. ✅ Veriler yüklendi
2. ⏳ Walk-forward training (50 trial)
3. ⏳ V7 vs V8 karşılaştırma
4. ⏳ Paper trading hazırlığı
5. ⏳ FTMO challenge optimizasyonu

---

## 📞 DESTEK

Sorun yaşarsanız:
1. `upload_data.py` çıktısını paylaşın
2. `ls -lah /app/data/` çıktısını gönderin
3. Hata mesajlarını kopyalayın

**Hazır mısınız? Hadi başlayalım! 🚀**
