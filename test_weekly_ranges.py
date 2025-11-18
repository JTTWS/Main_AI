#!/usr/bin/env python3
"""Test Weekly Ranges - Haftalık Range Verilerini Test Et"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

data_path = os.path.expanduser("~/Desktop/JTTWS/data")

print("="*70)
print("HAFTALIK RANGE VERİLERİ TEST")
print("="*70)

for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
    print(f"\n{'='*70}")
    print(f"📊 {symbol} WEEKLY RANGES")
    print(f"{'='*70}")
    
    filename = os.path.join(data_path, f"{symbol}_weekly_ranges.csv")
    
    if not os.path.exists(filename):
        print(f"❌ Dosya bulunamadı: {filename}")
        continue
    
    # Veriyi yükle
    df = pd.read_csv(filename)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    print(f"\n✅ Dosya yüklendi: {len(df)} hafta")
    print(f"📅 Tarih aralığı: {df['time'].min()} - {df['time'].max()}")
    print(f"\n📈 Örnek 3 hafta:")
    print(df.head(3).to_string())
    
    # İstatistikler
    print(f"\n📊 Range İstatistikleri:")
    print(f"   Ortalama range: {df['range'].mean():.5f} ({df['range_pips'].mean():.1f} pips)")
    print(f"   Min range: {df['range'].min():.5f} ({df['range_pips'].min():.1f} pips)")
    print(f"   Max range: {df['range'].max():.5f} ({df['range_pips'].max():.1f} pips)")
    print(f"   p95 range: {df['range_pips'].quantile(0.95):.1f} pips")
    print(f"   p99 range: {df['range_pips'].quantile(0.99):.1f} pips")
    
    # En volatil 5 hafta
    print(f"\n🔥 En Volatil 5 Hafta:")
    top5 = df.nlargest(5, 'range_pips')[['time', 'range_pips', 'high', 'low']]
    for idx, row in top5.iterrows():
        print(f"   {row['time'].strftime('%Y-%m-%d')}: {row['range_pips']:.0f} pips ({row['low']:.5f} - {row['high']:.5f})")
    
    # 2020 örneği
    df_2020 = df[df['time'].dt.year == 2020]
    if len(df_2020) > 0:
        print(f"\n📅 2020 Yılı:")
        print(f"   {len(df_2020)} hafta")
        print(f"   Ort range: {df_2020['range_pips'].mean():.1f} pips")
        print(f"   Max range: {df_2020['range_pips'].max():.1f} pips")

print(f"\n{'='*70}")
print("TEST TAMAMLANDI")
print(f"{'='*70}\n")
