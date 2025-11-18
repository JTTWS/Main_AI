# 🚀 JTTWS V8 - Lokal Kurulum Rehberi

## 📋 Gereksinimler

- Python 3.8+
- 8GB+ RAM
- 2GB+ disk alanı (data için)

---

## ⚡ Hızlı Başlangıç (3 Adım)

### 1️⃣ Data Dosyalarını İndirin

Eğer data klasörünüz yoksa:

```bash
cd ~/Desktop/JTTWS

# Google Drive'dan data dosyasını indirin (324MB)
# Link: https://drive.google.com/file/d/15q9AymGt2HzdZbmER8Oomfj7anyFGfBO/view

# İndirdikten sonra extract edin
tar -xzf jttws_data_complete.tar.gz
```

### 2️⃣ Python Paketlerini Kurun

```bash
cd ~/Desktop/JTTWS
pip install -r requirements.txt
```

### 3️⃣ Environment Check

```bash
python fix_local_paths.py
```

Bu script:
- ✅ Data dosyalarını kontrol eder
- ✅ Python paketlerini kontrol eder
- ✅ Eksikleri gösterir

---

## 🎯 Botu Çalıştırma

### Train Mode (Recommended)

```bash
python ultimate_bot_v8_ppo.py --mode train --years 2020-2024 --optuna-trials 10
```

**Ne yapar?**
- Walk-forward training (180/60 gün windows)
- Optuna hyperparameter optimization
- 27 period üzerinde eğitim
- Model kaydı: `models_v8/`
- Sonuçlar: `outputs_v8/`

**Parametreler:**
- `--years`: Eğitim yılları (örn: `2020-2024`)
- `--optuna-trials`: Her period için Optuna trial sayısı (default: 10)

### Backtest Mode

```bash
python ultimate_bot_v8_ppo.py --mode backtest --years 2020-2024
```

### Paper Trading Mode

```bash
python ultimate_bot_v8_ppo.py --mode paper
```

---

## 🔧 Sorun Giderme

### Problem: "No data found" hatası

**Çözüm:**
```bash
# 1. Data klasörünü kontrol edin
ls -la ~/Desktop/JTTWS/data/

# 2. Klasör yapısı şöyle olmalı:
# data/
#   ├── EURUSD2003-2024/*.csv
#   ├── GBPUSD2003-2024/*.csv
#   ├── USDJPY2003-2024/*.csv
#   ├── combined_economic_calendar.csv
#   └── *_weekly_ranges.csv

# 3. Yoksa data'yı indirip extract edin
tar -xzf jttws_data_complete.tar.gz
```

### Problem: "ModuleNotFoundError: No module named 'gym'"

**Çözüm:**
```bash
pip install -r requirements.txt
```

### Problem: "Gym has been unmaintained" uyarısı

**Çözüm:**
Bu sadece bir warning, bot çalışır. Ama V8'de artık `gymnasium` kullanıyoruz:
```bash
pip install gymnasium
```

### Problem: Çok yüksek decay oranları

**Neden?**
- Mock data kullanılıyor olabilir
- Data path yanlış olabilir

**Kontrol:**
```bash
python fix_local_paths.py
```

Eğer "✅ All required data files found!" görmüyorsanız, data dosyalarını yeniden indirin.

---

## 📊 Beklenen Sonuçlar

### Gerçek Data ile:
- Avg Train Sharpe: 0.10 - 0.25
- Avg Test Sharpe: 0.10 - 0.40
- Decay: %12 - %15 (ilk 3 period %20'ye kadar normal)

### Mock Data ile:
- Avg Decay: %1000+ (Çok yüksek!)
- Güvenilmez sonuçlar

**Not:** Eğer decay %100+ görüyorsanız, muhtemelen mock data kullanılıyor!

---

## 📁 Klasör Yapısı

```
~/Desktop/JTTWS/
├── data/                          # Data dosyaları (2GB+)
│   ├── EURUSD2003-2024/
│   ├── GBPUSD2003-2024/
│   ├── USDJPY2003-2024/
│   └── *.csv
├── models_v8/                     # Trained models
├── outputs_v8/                    # Training results
├── ultimate_bot_v8_ppo.py         # Main V8 bot
├── data_manager_v8.py             # Data loading
├── data_aggregator_v8.py          # Data aggregation
├── ppo_agent.py                   # PPO agent
├── walk_forward_trainer.py        # Walk-forward training
├── optuna_optimizer.py            # Hyperparameter tuning
├── reward_shaper.py               # Reward function
├── requirements.txt               # Python packages
└── fix_local_paths.py             # Setup check script
```

---

## 🎯 V8 vs V7 Farkları

| Özellik | V7 | V8 |
|---------|----|----|
| RL Agent | Rainbow DQN | PPO |
| Training | Single-pass | Walk-Forward (180/60) |
| Optimization | Manual | Optuna (automatic) |
| Data Loading | Single file | Multi-file (chunked) |
| Path Handling | Hardcoded | Dynamic (relative) |
| Overfitting Control | ❌ | ✅ Decay monitoring |

---

## 💡 İpuçları

1. **İlk çalıştırma:** `--optuna-trials 5` ile başlayın (hızlı test)
2. **Production:** `--optuna-trials 20-50` kullanın (daha iyi sonuç)
3. **GPU varsa:** Otomatik kullanılır (PyTorch)
4. **RAM yetersizse:** `--years 2022-2024` ile küçük dataset kullanın
5. **Sonuçları izleyin:** `tail -f logs/ultimate_bot_v8.log`

---

## 📞 Destek

Sorun yaşıyorsanız:

1. `fix_local_paths.py` çalıştırın
2. Log dosyasını kontrol edin: `logs/ultimate_bot_v8.log`
3. Data klasörünü kontrol edin: `ls -la data/`

---

## ✅ Checklist

- [ ] Python 3.8+ kurulu
- [ ] Data dosyaları indirildi ve extract edildi
- [ ] `pip install -r requirements.txt` çalıştırıldı
- [ ] `python fix_local_paths.py` ✅ verdi
- [ ] Bot çalıştırıldı ve gerçek data kullanıyor ("No data found" yok)
- [ ] Decay oranları makul seviyede (%15-50 arası)

---

**Son güncelleme:** 7 Kasım 2025  
**Versiyon:** 8.0 PPO Hybrid  
**Status:** Production Ready ✅
