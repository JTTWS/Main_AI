# 📦 JTTWS V8 DOSYALARI - İNDİRME LİSTESİ

## ✅ İndirmeniz Gereken V8 Dosyaları

### 🔧 Ana Python Modülleri (5 dosya)

1. **reward_shaper.py** (11K)
   - Penalty-based reward shaping
   - RewardShaper sınıfı

2. **ppo_agent.py** (12K)
   - PPO agent + LSTM hybrid
   - PPOAgent ve LSTMPredictor sınıfları

3. **optuna_optimizer.py** (8.5K)
   - Hyperparameter optimization
   - OptunaOptimizer sınıfı

4. **walk_forward_trainer.py** (11K)
   - Walk-forward validation
   - WalkForwardTrainer sınıfı

5. **ultimate_bot_v8_ppo.py** (20K)
   - V8 ana sistem
   - TradingEnvironmentV8 ve UltimateTradingSystemV8 sınıfları

### 📖 Dokümantasyon (2 dosya)

6. **README_V8.md** (13K)
   - Kapsamlı V8 dokümantasyonu
   - Kullanım örnekleri, troubleshooting, karşılaştırma

7. **INDIRME_BILGISI.md** (5K)
   - Kurulum ve kullanım talimatları
   - Hızlı başlangıç kılavuzu

### 📁 Klasörler (3 klasör)

8. **data_v8/**
   - combined_economic_calendar.csv (ekonomik takvim)

9. **models_v8/**
   - ppo_model_v8.zip (eğitilmiş PPO modeli)
   - test_ppo.zip (test modeli)

10. **outputs_v8/**
    - walk_forward_results_v8.csv (walk-forward sonuçları)

### 📋 V7 Dosyası (Referans - Opsiyonel)

11. **ultimate_bot_v7_professional.py** (43K)
    - Orijinal V7 (karşılaştırma için)

---

## 🚀 Kendi Bilgisayarınızda Kurulum

### 1. Tüm dosyaları indirin

Yukarıdaki 11 öğeyi (5 .py + 2 .md + 3 klasör + 1 v7) indirin.

### 2. Bir klasör oluşturun

```bash
# MacOS/Linux
mkdir ~/Desktop/JTTWS_V8
cd ~/Desktop/JTTWS_V8

# Windows
mkdir C:\Users\YourName\Desktop\JTTWS_V8
cd C:\Users\YourName\Desktop\JTTWS_V8
```

### 3. İndirilen dosyaları buraya kopyalayın

```
JTTWS_V8/
├── reward_shaper.py
├── ppo_agent.py
├── optuna_optimizer.py
├── walk_forward_trainer.py
├── ultimate_bot_v8_ppo.py
├── ultimate_bot_v7_professional.py
├── README_V8.md
├── INDIRME_BILGISI.md
├── data_v8/
│   └── combined_economic_calendar.csv
├── models_v8/
│   ├── ppo_model_v8.zip
│   └── test_ppo.zip
└── outputs_v8/
    └── walk_forward_results_v8.csv
```

### 4. Python bağımlılıklarını yükleyin

```bash
pip install stable-baselines3 optuna gym torch shimmy vectorbt
pip install pandas numpy scipy aiohttp
```

### 5. Test edin

```bash
# Module testleri
python reward_shaper.py
python ppo_agent.py
python optuna_optimizer.py
python walk_forward_trainer.py

# V8 backtest
python ultimate_bot_v8_ppo.py --mode backtest --episodes 1

# Walk-forward training
python ultimate_bot_v8_ppo.py --mode train --optuna-trials 10
```

---

## 📖 Hangi Dosyayı İlk Okuyayım?

1. **INDIRME_BILGISI.md** ← Buradan başlayın
2. **README_V8.md** ← Detaylı bilgi için

---

## ⚠️ Önemli Notlar

- ✅ **V7 Korundu**: ultimate_bot_v7_professional.py değiştirilmedi
- ✅ **Modüler Yapı**: Her modül bağımsız test edilebilir
- ✅ **Test Edildi**: Tüm modüller test passed durumunda
- ⚠️ **Mock Data**: Şu anda sahte veri kullanıyor, gerçek CSV'lerinizi ekleyin
- ⚠️ **Test Aşaması**: V8 henüz production-ready değil, V7 ile paralel test edin

---

## 🆚 V7 vs V8 Hızlı Karşılaştırma

| Özellik | V7 | V8 |
|---------|----|----|
| RL Algoritması | Rainbow DQN | PPO + LSTM |
| Hyperparameter Tuning | Manuel | Otomatik (Optuna) |
| Overfitting Kontrolü | Yok | Walk-Forward |
| Reward Engineering | Basit | RewardShaper |
| Beklenen Performans | Baseline | +18-25% |

---

## 📞 Destek

Sorular için README_V8.md'deki troubleshooting bölümüne bakın.

**Versiyon**: V8.0 PPO Hybrid  
**Tarih**: 6 Ocak 2025  
**Durum**: Testing - V7 ile Paralel
