#!/usr/bin/env python3
"""
CSV Dosya Analiz Aracı
Serkan Bey'in forex verilerini kontrol eder
Her dosyanın içeriğini, kalitesini ve doğruluğunu test eder
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("FOREX VERİ ANALİZ ARACI")
print("=" * 70)

# Veri klasörünü tara
data_path = Path.home() / "Desktop" / "JTTWS" / "data"
print(f"\nTaranacak klasör: {data_path}")
print("-" * 70)

# Tüm CSV dosyalarını bul (trading_env_backup hariç)
csv_files = []
for csv_file in data_path.glob("**/*.csv"):
    # trading_env_backup klasörünü atla
    if "trading_env_backup" not in str(csv_file):
        csv_files.append(csv_file)

print(f"\nToplam {len(csv_files)} CSV dosyası bulundu\n")

# Her dosyayı analiz et
results = []
for i, csv_file in enumerate(csv_files, 1):
    print(f"\n[{i}/{len(csv_files)}] Analiz ediliyor: {csv_file.name}")
    print("-" * 50)
    
    try:
        # Dosya bilgileri
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"Dosya boyutu: {file_size_mb:.2f} MB")
        
        # İlk birkaç satırı oku
        df_sample = pd.read_csv(csv_file, nrows=10)
        print(f"Kolon sayısı: {len(df_sample.columns)}")
        print(f"Kolonlar: {', '.join(df_sample.columns[:10])}")  # İlk 10 kolon
        
        # Tam veriyi yükle (sadece satır sayısı için)
        total_rows = sum(1 for line in open(csv_file)) - 1  # Başlık satırını çıkar
        print(f"Toplam satır: {total_rows:,}")
        
        # Tarih aralığını bulmaya çalış
        df_full = pd.read_csv(csv_file, nrows=1000)  # İlk 1000 satır
        
        # Olası tarih kolonlarını ara
        date_columns = ['Date', 'date', 'Time', 'time', 'Timestamp', 'timestamp', 
                       'Gmt time', 'Gmt Time', 'GMT time', 'GMT Time']
        
        date_col = None
        for col in date_columns:
            if col in df_full.columns:
                date_col = col
                break
        
        if date_col:
            try:
                # İlk ve son tarihi bul
                df_dates = pd.read_csv(csv_file, usecols=[date_col])
                df_dates[date_col] = pd.to_datetime(df_dates[date_col], errors='coerce')
                
                first_date = df_dates[date_col].min()
                last_date = df_dates[date_col].max()
                
                print(f"Tarih aralığı: {first_date} - {last_date}")
                
                # Hafta sonu verisi var mı kontrol et
                if not df_dates[date_col].isna().all():
                    weekend_data = df_dates[df_dates[date_col].dt.dayofweek >= 5]
                    if not weekend_data.empty:
                        print(f"⚠️  UYARI: {len(weekend_data)} hafta sonu verisi bulundu!")
                    else:
                        print("✅ Hafta sonu verisi yok (iyi)")
                        
            except Exception as e:
                print(f"Tarih analizi başarısız: {str(e)[:50]}")
        else:
            print("Tarih kolonu bulunamadı")
        
        # Fiyat kolonlarını kontrol et
        price_columns = ['Close', 'close', 'High', 'high', 'Low', 'low', 'Open', 'open']
        found_price_cols = [col for col in price_columns if col in df_full.columns]
        
        if found_price_cols:
            print(f"✅ Fiyat kolonları bulundu: {', '.join(found_price_cols)}")
            
            # Fiyat mantık kontrolü
            for price_col in found_price_cols[:1]:  # Sadece ilk fiyat kolonunu kontrol et
                if price_col in df_full.columns:
                    prices = pd.to_numeric(df_full[price_col], errors='coerce')
                    if not prices.isna().all():
                        min_price = prices.min()
                        max_price = prices.max()
                        mean_price = prices.mean()
                        
                        print(f"Fiyat aralığı: {min_price:.5f} - {max_price:.5f}")
                        print(f"Ortalama fiyat: {mean_price:.5f}")
                        
                        # Mantık kontrolü
                        if min_price <= 0:
                            print("❌ HATA: Negatif veya sıfır fiyat bulundu!")
                        elif max_price / min_price > 100:
                            print("⚠️  UYARI: Çok geniş fiyat aralığı (muhtemelen hatalı)")
                        else:
                            print("✅ Fiyat aralığı mantıklı")
        else:
            print("❌ Fiyat kolonu bulunamadı")
        
        # Sonucu kaydet
        results.append({
            'file': csv_file.name,
            'size_mb': file_size_mb,
            'rows': total_rows,
            'status': 'OK'
        })
        
    except Exception as e:
        print(f"❌ HATA: {str(e)[:100]}")
        results.append({
            'file': csv_file.name,
            'size_mb': 0,
            'rows': 0,
            'status': 'ERROR'
        })

# Özet rapor
print("\n" + "=" * 70)
print("ÖZET RAPOR")
print("=" * 70)

df_results = pd.DataFrame(results)
print(f"\nToplam dosya: {len(df_results)}")
print(f"Başarılı: {len(df_results[df_results['status'] == 'OK'])}")
print(f"Hatalı: {len(df_results[df_results['status'] == 'ERROR'])}")
print(f"Toplam veri boyutu: {df_results['size_mb'].sum():.2f} MB")
print(f"Toplam satır sayısı: {df_results['rows'].sum():,}")

print("\n" + "=" * 70)
print("Analiz tamamlandı!")
