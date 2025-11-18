print("=" * 60)
print("EN BASIT EGITIM YONTEMI")
print("=" * 60)

import sys
sys.path.append('/Users/serkanozturk/Desktop/JTTWS')

print("\n[1/3] Sistemi yukluyorum...")
from train_bot_v9 import TrainingPipelineV9

print("[2/3] Pipeline hazirlaniyor...")
pipeline = TrainingPipelineV9()

print("[3/3] Otomatik egitim basliyor...")
print("     (Bu metod her seyi otomatik yapiyor)")
print("     Lutfen 1-2 dakika bekleyin...\n")

try:
    # En basit yontem - her seyi otomatik yap
    pipeline.run_quick_training()
    
    print("\n" + "=" * 60)
    print("✓ EGITIM TAMAMLANDI!")
    print("=" * 60)
    
    # Model dosyalarini kontrol et
    import os
    model_files = os.listdir('./models_v9/')
    if model_files:
        print("\nKaydedilen modeller:")
        for f in model_files:
            print(f"  ✓ {f}")
    
except KeyboardInterrupt:
    print("\n\n! Egitim kullanici tarafindan durduruldu")
    print("  (Bu normal, Ctrl+C yaptiniz)")
    
except Exception as e:
    print(f"\n! Hata: {e}")

print("\n" + "=" * 60)
print("Program sonlandi")
print("=" * 60)
