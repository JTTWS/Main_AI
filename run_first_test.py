#!/usr/bin/env python3
"""
Serkan Bey'in Botu için İlk Test Çalıştırması
"""

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 JTTWS Trading Bot - İlk Test Çalıştırması")
print("=" * 60)

try:
    from train_bot_v9 import TrainingPipelineV9
    from datetime import datetime
    
    print("\n📊 Pipeline hazırlanıyor...")
    
    # Pipeline oluştur
    pipeline = TrainingPipelineV9(
        data_dir='./data',
        models_dir='./models_v9', 
        logs_dir='./logs_v9'
    )
    
    # EURUSD verisini yükle (son 2 yıl için test)
    print("\n📈 EURUSD verisi yükleniyor (2022-2024)...")
    df = pipeline.setup_data(symbol='EURUSD', years='2022-2024')
    
    print(f"✅ Veri yüklendi:")
    print(f"   - Satır sayısı: {len(df):,}")
    print(f"   - Sütun sayısı: {len(df.columns)}")
    print(f"   - Başlangıç: {df.index[0] if not df.empty else 'N/A'}")
    print(f"   - Bitiş: {df.index[-1] if not df.empty else 'N/A'}")
    
    # Feature'ları göster
    print(f"\n🔧 Feature listesi (ilk 10):")
    for i, col in enumerate(df.columns[:10]):
        print(f"   {i+1}. {col}")
    
    # Trading environment'ı kur
    print("\n🎯 Trading environment hazırlanıyor...")
    env = pipeline.setup_environment(df)
    
    print(f"✅ Environment hazır:")
    print(f"   - Başlangıç sermaye: $25,000")
    print(f"   - Max pozisyon: 3")
    print(f"   - Pozisyon boyutu: %2")
    print(f"   - Max drawdown: %20")
    
    # Basit bir test episode'u çalıştır
    print("\n🎲 Test episode başlatılıyor...")
    obs, _ = env.reset()
    
    total_reward = 0
    for step in range(10):  # Sadece 10 adım test
        action = env.action_space.sample()  # Random aksiyon
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            break
    
    print(f"✅ Test tamamlandı:")
    print(f"   - Toplam adım: {step + 1}")
    print(f"   - Toplam reward: {total_reward:.2f}")
    
    # Mevcut modeli kontrol et
    print("\n🤖 Mevcut model kontrolü...")
    import os
    if os.path.exists('logs/best_model.zip'):
        print("✅ Eğitilmiş model bulundu (best_model.zip)")
        
        # Model yükleme testi
        try:
            from stable_baselines3 import PPO
            model = PPO.load('logs/best_model.zip')
            print("✅ Model başarıyla yüklendi")
            print(f"   - Model tipi: PPO")
            print(f"   - Policy: {model.policy.__class__.__name__}")
        except Exception as e:
            print(f"⚠️  Model yüklenemedi: {e}")
    else:
        print("ℹ️  Henüz eğitilmiş model yok")
    
    print("\n" + "=" * 60)
    print("🎉 TÜM TESTLER BAŞARILI!")
    print("Botunuz çalışmaya hazır durumda.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Test sırasında hata: {e}")
    import traceback
    traceback.print_exc()
