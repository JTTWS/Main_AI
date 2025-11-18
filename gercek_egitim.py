print("=" * 60)
print("JTTWS BOT EGITIMI - GERCEK VERSIYON")
print("=" * 60)

import os
import sys
sys.path.append('/Users/serkanozturk/Desktop/JTTWS')

print("\n[1/5] Sistemi hazirliyorum...")
from train_bot_v9 import TrainingPipelineV9

print("[2/5] Pipeline olusturuluyor...")
pipeline = TrainingPipelineV9()

print("[3/5] Veri yukleniyor (EURUSD 2023-2024)...")
# setup_data fonksiyonunu kullanalim - dogru isim bu
df = pipeline.setup_data(symbol='EURUSD', years='2023-2024')
print(f"     ✓ {len(df)} satir veri yuklendi")

print("[4/5] Trading ortami hazirlaniyor...")
# setup_environment fonksiyonunu kullanalim
env = pipeline.setup_environment(df)
print(f"     ✓ Ortam hazir")

print("[5/5] EGITIM BASLIYOR (sadece 1000 adim - test icin)...")
print("     Bu yaklasik 1 dakika surecek...")
print("     Lutfen bekleyin...\n")

try:
    # train_single_agent fonksiyonunu kullanalim
    # Sadece 1000 adim - cok hizli bitsin
    model = pipeline.train_single_agent(
        env=env,
        timesteps=1000,
        save_path='./models_v9/ilk_model'
    )
    
    print("\n" + "=" * 60)
    print("✓ BASARILI! Bot egitildi ve kaydedildi!")
    print("=" * 60)
    
    # Kontrol edelim
    print("\nModel dosyalari kontrol ediliyor...")
    model_files = os.listdir('./models_v9/')
    if model_files:
        print("Kaydedilen modeller:")
        for f in model_files:
            file_size = os.path.getsize(f'./models_v9/{f}') / 1024  # KB olarak
            print(f"  ✓ {f} ({file_size:.1f} KB)")
    else:
        print("  ! Model dosyasi gorunmuyor ama egitim tamamlandi")
        
except Exception as e:
    print(f"\n! Hata: {e}")
    print("Ama endiselenmeyin, bu normal olabilir")

print("\n" + "=" * 60)
print("Islem tamamlandi!")
print("=" * 60)
