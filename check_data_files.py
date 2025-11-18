#!/usr/bin/env python3
import os
from pathlib import Path

print("=" * 70)
print("DATA KLASÖRÜ ANALİZİ")
print("=" * 70)

base_dir = Path.home() / "Desktop" / "JTTWS" / "data"
print(f"\nBase Directory: {base_dir}")
print(f"Exists: {base_dir.exists()}")

if base_dir.exists():
    pairs = ["EURUSD2003-2024", "GBPUSD2003-2024", "USDJPY2003-2024"]
    
    for pair in pairs:
        pair_path = base_dir / pair
        print(f"\n{'='*70}")
        print(f"📁 {pair}")
        print(f"Path: {pair_path}")
        print(f"Exists: {pair_path.exists()}")
        
        if pair_path.exists():
            # Tüm CSV dosyalarını listele
            csv_files = sorted(pair_path.glob("*Candlestick*.csv"))
            print(f"Toplam CSV dosyası: {len(csv_files)}")
            
            if csv_files:
                print("\nDosya listesi:")
                for i, f in enumerate(csv_files, 1):
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  {i}. {f.name} ({size_mb:.2f} MB)")
                
                # İlk dosyayı detaylı incele
                first_file = csv_files[0]
                print(f"\n🔍 İlk dosya detayı: {first_file.name}")
                
                try:
                    with open(first_file, 'r') as f:
                        lines = f.readlines()[:5]
                    print("  İlk 5 satır:")
                    for line in lines:
                        print(f"    {line.strip()}")
                except Exception as e:
                    print(f"  ❌ Okuma hatası: {e}")
            else:
                print("  ⚠️ Hiç Candlestick CSV dosyası bulunamadı!")
        else:
            print("  ❌ Klasör bulunamadı!")

print("\n" + "=" * 70)
print("ANALİZ TAMAMLANDI")
print("=" * 70)
