#!/usr/bin/env python3
"""Simple Test - Veri yükleme testi"""

import os
import glob
import pandas as pd
from datetime import datetime

# Test: EURUSD verisini yükle
data_path = os.path.expanduser("~/Desktop/JTTWS/data")
symbol = "EURUSD"
subfolder = f"{symbol}2003-2024"
pattern = os.path.join(data_path, subfolder, f"{symbol}_Candlestick*.csv")

print(f"🔍 Aranan pattern: {pattern}")
files = glob.glob(pattern)
print(f"📁 Bulunan dosyalar: {len(files)}")

if files:
    for f in files[:3]:  # İlk 3 dosyayı göster
        print(f"   ✅ {os.path.basename(f)}")
    
    # İlk dosyayı yükle
    df = pd.read_csv(files[0])
    print(f"\n📊 İlk dosya içeriği:")
    print(df.head())
    print(f"\n✅ Toplam satır: {len(df)}")
else:
    print("❌ Hiç dosya bulunamadı!")
    print(f"🔍 Kontrol et: {os.path.join(data_path, subfolder)}")
