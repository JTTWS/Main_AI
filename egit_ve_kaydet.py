print("Bot egitimi baslatiliyor...")
print("Bu islem yaklasik 2 dakika surecek")
print("=" * 50)

import os
import sys
sys.path.append('/Users/serkanozturk/Desktop/JTTWS')

print("1. Adim: Gerekli modulleri yukluyorum...")
from train_bot_v9 import TrainingPipelineV9

print("2. Adim: Egitim ayarlarini hazirliyorum...")
# Cok kisa bir egitim yapalim - sadece test icin
pipeline = TrainingPipelineV9()

print("3. Adim: Egitimi baslat (1000 adim - cok kisa)...")
try:
    # Sadece 1000 adimlik hizli egitim
    pipeline.train_single_agent(
        env=pipeline.setup_environment(pipeline.load_and_prepare_data()),
        timesteps=1000,  # Cok kisa - sadece test icin
        save_path='./models_v9/test_model'
    )
    print("4. Adim: Model kaydedildi!")
    print("Kontrol edelim:")
    if os.path.exists('./models_v9/test_model.zip'):
        print("BASARILI! Model dosyasi olusturuldu.")
    else:
        print("Model dosyasi bulunamadi ama egitim tamamlandi")
except Exception as e:
    print(f"Hata olustu: {e}")

print("=" * 50)
print("Egitim tamamlandi!")
