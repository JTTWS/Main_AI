print("Basit model testi")
print("=" * 50)

import os
import sys
sys.path.append('/Users/serkanozturk/Desktop/JTTWS')

# Oncelikle mevcut modelleri kontrol edelim
print("1. Mevcut model dosyalari:")
models_path = './models_v9'
if not os.path.exists(models_path):
    print("   models_v9 klasoru yok, olusturuyorum...")
    os.makedirs(models_path, exist_ok=True)

files = os.listdir(models_path)
if files:
    for f in files:
        print(f"   - {f}")
else:
    print("   Hic model yok")

print("\n2. Train_bot_v9 modulunu kontrol ediyorum:")
try:
    # Sadece import edelim, calistirmayalim
    import train_bot_v9
    print("   Modul basariyla yuklendi")
    
    # Pipeline'i olusturalim
    pipeline = train_bot_v9.TrainingPipelineV9()
    print("   Pipeline olusturuldu")
    
    # Hangi metodlar var bakalim
    print("\n3. Kullanilabilir metodlar:")
    methods = [m for m in dir(pipeline) if not m.startswith('_')]
    for m in methods[:10]:  # Ilk 10 metodu goster
        print(f"   - {m}")
    
except Exception as e:
    print(f"   Hata: {e}")

print("=" * 50)
print("Test tamamlandi")
