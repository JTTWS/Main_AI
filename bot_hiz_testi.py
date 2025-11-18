print("Bot hiz testi basliyor...")
print("=" * 50)

import time
from datetime import datetime

print("Simdi botun ne kadar hizli calistigini olcelim")
print("10 saniye bekleyin...")

baslangic = datetime.now()
time.sleep(10)
bitis = datetime.now()

print(f"Baslangic: {baslangic}")
print(f"Bitis: {bitis}")
print(f"Gecen sure: {bitis - baslangic}")

print("=" * 50)
print("Simdi bot ayarlarini kontrol edelim")

try:
    with open('train_bot_v9.py', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'timesteps' in line.lower() or 'capital' in line.lower():
                print(f"Satir {i+1}: {line.strip()}")
                if i > 200:  # Ilk 200 satirdan sonra dur
                    break
except:
    print("Dosya okunamadi")

print("=" * 50)
print("Test bitti")
