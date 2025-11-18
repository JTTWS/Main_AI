# 📦 JTTWS V8 İNDİRME KILAVUZU

## 🎯 İndirilen Dosyalar

Bu klasör, **Ultimate FTMO Trading Bot V8.0 PPO Hybrid** projesinin tamamını içerir.

### 📁 Dosya Listesi

#### 🔧 V8 Ana Dosyaları
- **ultimate_bot_v8_ppo.py** (20K) - V8 ana sistem
- **reward_shaper.py** (11K) - Penalty-based reward shaping
- **ppo_agent.py** (12K) - PPO agent + LSTM hybrid
- **optuna_optimizer.py** (8.5K) - Hyperparameter optimization
- **walk_forward_trainer.py** (11K) - Walk-forward validation

#### 📚 V7 Dosyaları (Korunan)
- **ultimate_bot_v7_professional.py** (43K) - V7 core (değiştirilmedi)

#### 📖 Dokümantasyon
- **README_V8.md** (13K) - Kapsamlı V8 dokümantasyonu
- **README_KULLANIM.md** (9.9K) - Türkçe kullanım kılavuzu (V7)
- **INDIRME_BILGISI.md** - Bu dosya

#### 📊 Klasörler
- **data/** - Ekonomik takvim ve piyasa verileri
- **models/** - Eğitilmiş PPO modelleri (ppo_model_v8.zip, test_ppo.zip)
- **outputs/** - Walk-forward sonuçları (walk_forward_results_v8.csv)
- **logs/** - Bot çalışma logları

#### ⚙️ Diğer
- **requirements.txt** - Python bağımlılıkları

## 🚀 Kurulum (Kendi Bilgisayarınızda)

### 1. Python Sanal Ortamı Oluşturun (Önerilen)

```bash
# MacOS/Linux
cd ~/Desktop
mkdir JTTWS_V8
cd JTTWS_V8
python3 -m venv trading_env
source trading_env/bin/activate

# Windows
cd C:\Users\YourName\Desktop
mkdir JTTWS_V8
cd JTTWS_V8
python -m venv trading_env
trading_env\Scripts\activate
```

### 2. İndirilen Dosyaları Kopyalayın

Tüm dosyaları `JTTWS_V8_COMPLETE` klasöründen yeni oluşturduğunuz `JTTWS_V8` klasörüne kopyalayın.

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt

# Eksik paketler için:
pip install stable-baselines3 optuna gym torch shimmy vectorbt
pip install pandas numpy scipy aiohttp
```

### 4. Test Edin

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

## 📊 Gerçek Veri ile Çalıştırma

Şu anda bot mock (sahte) veri kullanıyor. Gerçek verilerinizle çalıştırmak için:

1. **data/** klasörüne EURUSD/GBPUSD/USDJPY CSV dosyalarınızı ekleyin
2. Format: `EURUSD_Candlestick_15_M_BID_2020-2024.csv`
3. Kolonlar: time, open, high, low, close, volume

Alternatif olarak, V7'deki `DataManager` sınıfı otomatik olarak `data/` klasöründen CSV'leri yükler.

## 🎯 Hızlı Başlangıç Komutları

```bash
# V8 backtest (mock data ile)
python ultimate_bot_v8_ppo.py --mode backtest --years 2020-2024 --episodes 5

# Walk-forward training + Optuna (önerilen)
python ultimate_bot_v8_ppo.py --mode train --optuna-trials 50

# Paper trading
python ultimate_bot_v8_ppo.py --mode paper --use-ppo

# V7 ile karşılaştırma (V7 çalıştır)
python ultimate_bot_v7_professional.py --mode backtest --years 2020-2024
```

## 📖 Dokümantasyon

Detaylı kullanım için:
- **README_V8.md** - V8 özellikler, karşılaştırma, troubleshooting
- **README_KULLANIM.md** - V7 Türkçe kullanım kılavuzu

## 🆚 V7 vs V8 Seçimi

### V7'yi Kullanın Eğer:
- ✅ Production-ready, test edilmiş sistem istiyorsanız
- ✅ Basit, kolay debug edilebilir kod tercih ediyorsanız
- ✅ Türkçe dokümantasyon istiyorsanız

### V8'i Kullanın Eğer:
- ✅ %18-25 daha iyi performans hedefliyorsanız
- ✅ Otomatik hyperparameter tuning istiyorsanız
- ✅ Overfitting kontrolü (walk-forward) istiyorsanız
- ✅ Modern RL teknikleri (PPO, LSTM) denemek istiyorsanız

### Tavsiye: Her İkisini Paralel Çalıştırın!
1. V7 ile production trading yapın
2. V8'i backtest ve paper trading'de test edin
3. V8'in sonuçları V7'den %15+ iyi olunca geçiş yapın

## 🛠️ Gerekli Kütüphaneler

```
stable-baselines3==2.7.0
optuna==4.5.0
gym==0.26.2
shimmy>=2.0
torch>=2.0
vectorbt==0.28.1
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
aiohttp (Telegram bildirimleri için)
```

## ⚠️ Önemli Notlar

1. **V8 Test Aşamasında**: Gerçek para ile kullanmadan önce kapsamlı backtest yapın
2. **Mock Data**: Şu anda sahte veri kullanılıyor, gerçek CSV'lerinizi ekleyin
3. **V7 Korundu**: Orijinal V7 dosyası değiştirilmedi, her zaman geri dönebilirsiniz
4. **Walk-Forward Sonuçları**: `outputs/walk_forward_results_v8.csv` dosyasında

## 📞 Destek

Sorularınız için:
1. README_V8.md'deki troubleshooting bölümüne bakın
2. Test scriptlerini çalıştırın (her modül kendi testini içerir)
3. Walk-forward sonuçlarını inceleyin (CSV)

## 🎓 İleri Seviye

**Custom Reward Function**:
- `reward_shaper.py` içinde `compute_penalty()` metodunu düzenleyin

**Custom Environment**:
- `ultimate_bot_v8_ppo.py` içinde `TradingEnvironmentV8` sınıfını extend edin

**Ensemble Models**:
- PPO + DQN + SAC birlikte kullanın (V8.1'de gelecek)

---

**Son Güncelleme**: 6 Ocak 2025
**Versiyon**: V8.0 PPO Hybrid
**Durum**: Testing - V7 ile Paralel
