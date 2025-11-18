# ⚡ JTTWS V8 - Hızlı Başlangıç

## 🎯 5 Dakikada Çalıştır

### 1. Dosyaları İndir

Bu klasörü (`JTTWS_V8_COMPLETE`) bilgisayarınıza indirin:
```bash
~/Desktop/JTTWS/
```

### 2. Data Kontrol

```bash
cd ~/Desktop/JTTWS
python fix_local_paths.py
```

**Çıktı şöyle olmalı:**
```
✅ Data directory found!
   ✅ EURUSD2003-2024: 144 CSV files
   ✅ GBPUSD2003-2024: 144 CSV files
   ✅ USDJPY2003-2024: 144 CSV files

✅ All required data files found!
✅ All required packages installed!
```

### 3. Botu Çalıştır

```bash
python ultimate_bot_v8_ppo.py --mode train --years 2020-2024 --optuna-trials 10
```

---

## ❌ Hata: "No data found"

**Çözüm:**

Data klasörü eksik veya boş. İndir ve extract et:

```bash
cd ~/Desktop/JTTWS

# Google Drive'dan indir (324MB):
# https://drive.google.com/file/d/15q9AymGt2HzdZbmER8Oomfj7anyFGfBO/view

# Extract et:
tar -xzf jttws_data_complete.tar.gz

# Kontrol et:
ls -la data/
```

Görmelisin:
```
data/
├── EURUSD2003-2024/  (144 CSV dosyası)
├── GBPUSD2003-2024/  (144 CSV dosyası)
├── USDJPY2003-2024/  (144 CSV dosyası)
└── *.csv             (4 ek dosya)
```

---

## ❌ Hata: "ModuleNotFoundError"

**Çözüm:**

```bash
pip install -r requirements.txt
```

---

## 📊 Beklenen Çıktı

Bot çalıştığında göreceksin:

```
📂 Loading data: 2020-2024
✅ Loaded 520000+ rows for EURUSD from .../data/EURUSD2003-2024
✅ Loaded 520000+ rows for GBPUSD from .../data/GBPUSD2003-2024
✅ Loaded 520000+ rows for USDJPY from .../data/USDJPY2003-2024

📊 Period 1:
   Train Sharpe: 0.293 | Reward: 0.002819
   Test Sharpe:  0.341 | Reward: 0.003215
   Decay: 16.21%
   ✅ Decay within threshold
```

**ÖNEMLİ:** 
- ❌ "generating mock data" görürsen → Data path yanlış!
- ❌ Decay %100+ ise → Mock data kullanılıyor!
- ✅ "Loaded 500000+ rows" → Doğru! Gerçek data kullanılıyor

---

## 🚀 Sonraki Adımlar

1. **İlk test başarılı olduysa:**
```bash
python ultimate_bot_v8_ppo.py --mode train --years 2020-2024 --optuna-trials 20
```

2. **Sonuçları incele:**
```bash
cat outputs_v8/walk_forward_results_v8.csv
```

3. **Backtest yap:**
```bash
python ultimate_bot_v8_ppo.py --mode backtest --years 2020-2024
```

4. **Paper trading:**
```bash
python ultimate_bot_v8_ppo.py --mode paper
```

---

## 📁 Önemli Dosyalar

- `fix_local_paths.py` → Environment check
- `LOCAL_SETUP_TR.md` → Detaylı kurulum rehberi
- `CHANGES_SUMMARY.md` → Ne değişti?
- `ultimate_bot_v8_ppo.py` → Ana bot
- `requirements.txt` → Python paketleri

---

## ✅ Başarı Kriterleri

Bot düzgün çalışıyorsa:

- ✅ "Loaded XXXXX rows for EURUSD" mesajları
- ✅ Decay %12-50 arası
- ✅ Her period için Optuna optimization çalışıyor
- ✅ 27 period tamamlanıyor
- ✅ Model kaydediliyor: `models_v8/best_ppo_model.zip`
- ✅ Sonuçlar kaydediliyor: `outputs_v8/walk_forward_results_v8.csv`

---

**Başarılar! 🎉**

Sorun yaşarsan: `LOCAL_SETUP_TR.md` dosyasına bak veya `fix_local_paths.py` çalıştır.
