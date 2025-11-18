# 📧 EMAIL AYARLARI VE KULLANIMI

## Bot V7.0 - Email Bildirimleri Entegrasyonu

### ✅ Tamamlanan Özellikler

1. **EmailNotifier Modülü** (`email_notifier.py`)
   - Gmail SMTP ile otomatik email gönderimi
   - HTML formatında profesyonel bildirimler
   - Trade açılış/kapanış bildirimleri
   - Haftalık performans raporları
   - Hata bildirimleri

2. **EnhancedTradeLogger Modülü** (`enhanced_trade_logger.py`)
   - Her trade için detaylı loglama
   - Tüm teknik indikatör değerleri (RSI, MACD, ATR, SMA, vb.)
   - Risk/Reward oranı hesaplaması
   - Potential profit/loss analizi
   - CSV export özelliği
   - Trade istatistikleri (win rate, avg profit, vb.)

### 🔧 Yapılandırma

`bot_config.py` dosyasını düzenleyin:

```python
# ==================== EMAIL AYARLARI ====================
EMAIL_ENABLED = True  # False yaparsanız email bildirimleri kapalı olur
EMAIL_ADDRESS = "sizin_email@gmail.com"  # BURAYA KENDİ EMAİLİNİZİ YAZIN
EMAIL_APP_PASSWORD = "vorw noth yfey efuz"  # Gmail App Password (zaten ayarlandı)
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_TO_ADDRESS = "sizin_email@gmail.com"  # BİLDİRİMLERİN GÖNDERİLECEĞİ ADRES
```

### 📝 DEĞİŞTİRMENİZ GEREKEN SATIRLAR:

1. **Satır 32**: `EMAIL_ADDRESS = "sizin_email@gmail.com"`
   - Burada Gmail adresinizi yazın (email gönderen adres)

2. **Satır 36**: `EMAIL_TO_ADDRESS = "sizin_email@gmail.com"`
   - Bildirimlerin gönderileceği email adresini yazın
   - Genellikle gönderen ile aynı adres olur

3. **App Password zaten ayarlandı**: `vorw noth yfey efuz`
   - JTTWS uygulaması için oluşturulmuş Gmail App Password
   - Bu şifreyi DEĞİŞTİRMEYİN!

### 📬 Gönderilecek Email Bildirimleri

#### 1. Trade Açılış Bildirimi
```
Konu: 🚀 TRADE OPENED: 🟢 LONG EURUSD

İçerik:
- Direction (LONG/SHORT)
- Lot Size
- Entry Price
- Stop Loss
- Take Profit
- Timestamp
```

#### 2. Trade Kapanış Bildirimi
```
Konu: ✅ TRADE CLOSED: EURUSD (+125.50$)

İçerik:
- Direction
- Profit/Loss ($)
- Pips
- Trade Duration
- Timestamp
```

#### 3. Haftalık Performans Raporu
```
Konu: 📊 WEEKLY REPORT - Week 42

İçerik:
- Total Trades
- Win Rate (%)
- Total Profit
- Average Profit
- Max Drawdown
```

### 🔍 Detaylı Trade Logging

Bot artık her trade için şu bilgileri logluyor:

**Trade Açılışında:**
- Entry price, SL, TP
- Lot size
- Risk/Reward oranı (1:X.XX)
- Potential profit ve loss
- Tüm teknik indikatörler:
  - RSI, MACD, MACD_signal
  - Bollinger Bands (üst/alt)
  - ATR, ADX
  - SMA_20, SMA_50, SMA_200

**Trade Kapanışında:**
- Exit price
- Actual profit/loss ($)
- Pips
- Duration (kaç bar sürdü)
- Close reason (SL/TP)

### 📊 Trade İstatistiklerini Görüntüleme

Bot çalıştırıldığında, tüm trade'lerin istatistiklerini görebilirsiniz:

```python
from enhanced_trade_logger import EnhancedTradeLogger

# Bot içinden
stats = system.trade_logger.get_trade_stats()
print(stats)

# Çıktı:
{
    'total_trades': 150,
    'closed_trades': 145,
    'open_trades': 5,
    'winning_trades': 98,
    'losing_trades': 47,
    'win_rate': 67.6,
    'total_profit': 12500.50,
    'avg_profit': 86.21,
    'avg_win': 200.30,
    'avg_loss': -85.60
}
```

### 💾 Trade Loglarını CSV'ye Aktarma

```python
# Bot çalıştıktan sonra
system.trade_logger.export_to_csv('/path/to/trades_export.csv')
```

### ⚠️ Önemli Notlar

1. **Gmail App Password**:
   - Bu şifre (`vorw noth yfey efuz`) JTTWS uygulaması için özel oluşturulmuştur
   - Normal Gmail şifreniz DEĞİLDİR
   - Bu şifreyi kimseyle paylaşmayın

2. **Email Devre Dışı Bırakma**:
   - Email istemiyorsanız: `EMAIL_ENABLED = False` yapın
   - Bot normal çalışmaya devam eder, sadece email göndermez

3. **SMTP Güvenliği**:
   - Gmail, "Less secure app access" özelliğini kapattığı için App Password gereklidir
   - App Password ile 2-Factor Authentication güvenliği sağlanır

### 🎯 Test Etme

Bot'u test etmek için:

```bash
cd ~/Desktop/JTTWS
python3 ultimate_bot_v7_professional.py --mode backtest
```

İlk trade açıldığında email almalısınız!

### 🐛 Sorun Giderme

**Email gitmiyor:**
1. `bot_config.py`'de email adreslerini kontrol edin
2. Internet bağlantınızı kontrol edin
3. Gmail hesabınızın 2FA aktif olduğundan emin olun
4. App Password'un doğru olduğunu kontrol edin

**"Email App Password yok!" uyarısı alıyorsanız:**
- `bot_config.py`'de `EMAIL_APP_PASSWORD` satırını kontrol edin
- Şifrenin boş olmadığından emin olun

### 📞 Destek

Sorun yaşarsanız:
1. Konsol loglarını kontrol edin
2. Email ayarlarını tekrar gözden geçirin
3. Test email göndererek SMTP bağlantısını test edin

---

**Bot V7.0 Professional - Enhanced Transparency & Notifications** 🚀
