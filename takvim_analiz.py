# takvim_analiz.py
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TAKVİM VERİSİ DETAYLI ANALİZİ")
print("=" * 70)

# Ana takvim dosyasını yükle
takvim_dosyasi = 'data/combined_economic_calendar.csv'

try:
    # Dosyayı oku
    df = pd.read_csv(takvim_dosyasi)
    print(f"\n✓ Dosya başarıyla yüklendi")
    print(f"Toplam kayıt sayısı: {len(df):,}")
    
    # Kolon isimlerini göster
    print(f"\nKolonlar: {', '.join(df.columns)}")
    
    # İlk 5 satırı göster
    print("\nİlk 5 Kayıt:")
    print("-" * 50)
    print(df.head())
    
    # Tarih aralığını bul
    if 'Start' in df.columns:
        df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
        print(f"\n📅 TARİH ARALIĞI:")
        print(f"En eski haber: {df['Start'].min()}")
        print(f"En yeni haber: {df['Start'].max()}")
        
        # Yıllara göre dağılım
        df['Year'] = df['Start'].dt.year
        yillik_dagilim = df['Year'].value_counts().sort_index()
        print(f"\n📊 YILLARA GÖRE HABER SAYILARI:")
        for yil, sayi in yillik_dagilim.items():
            if pd.notna(yil):
                print(f"  {int(yil)}: {sayi:,} haber")
    
    # Para birimlerine göre dağılım
    if 'Currency' in df.columns:
        print(f"\n💱 PARA BİRİMLERİNE GÖRE DAĞILIM:")
        para_dagilim = df['Currency'].value_counts()
        for para, sayi in para_dagilim.head(10).items():
            print(f"  {para}: {sayi:,} haber")
        
        # Major çiftler için özel analiz
        major_paralar = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        print(f"\n🌟 MAJOR PARA BİRİMLERİ:")
        for para in major_paralar:
            if para in para_dagilim.index:
                print(f"  {para}: {para_dagilim[para]:,} haber")
    
    # Haber önem derecelerine göre dağılım
    if 'Impact' in df.columns:
        print(f"\n⚡ ÖNEM DERECELERİNE GÖRE DAĞILIM:")
        onem_dagilim = df['Impact'].value_counts()
        for onem, sayi in onem_dagilim.items():
            yuzde = (sayi / len(df)) * 100
            print(f"  {onem}: {sayi:,} haber (%{yuzde:.1f})")
    
    # En sık görülen haberler
    if 'Name' in df.columns:
        print(f"\n📰 EN SIK GÖRÜLEN HABER TÜRLERİ (Top 20):")
        haber_sikligi = df['Name'].value_counts()
        for haber, sayi in haber_sikligi.head(20).items():
            print(f"  {sayi:4} kez: {haber[:60]}...")
    
    # Kritik haberler (Non-Farm, FOMC, ECB vs.)
    print(f"\n🔴 KRİTİK HABERLER:")
    kritik_kelimeler = ['Non-Farm', 'FOMC', 'ECB', 'BoE', 'BoJ', 'NFP', 
                        'Interest Rate', 'GDP', 'CPI', 'Employment', 
                        'Inflation', 'Payrolls']
    
    for kelime in kritik_kelimeler:
        if 'Name' in df.columns:
            ilgili_haberler = df[df['Name'].str.contains(kelime, case=False, na=False)]
            if len(ilgili_haberler) > 0:
                print(f"  {kelime}: {len(ilgili_haberler):,} kayıt")
    
    # Haftalık ve günlük dağılım
    if 'Start' in df.columns and df['Start'].notna().any():
        df['DayOfWeek'] = df['Start'].dt.day_name()
        df['Hour'] = df['Start'].dt.hour
        
        print(f"\n📅 GÜNLERE GÖRE DAĞILIM:")
        gun_dagilim = df['DayOfWeek'].value_counts()
        gun_sirasi = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for gun in gun_sirasi:
            if gun in gun_dagilim.index:
                print(f"  {gun}: {gun_dagilim[gun]:,} haber")
        
        print(f"\n⏰ SAATLERE GÖRE DAĞILIM (Top 10):")
        saat_dagilim = df['Hour'].value_counts().sort_index()
        for saat, sayi in saat_dagilim.head(10).items():
            if pd.notna(saat):
                print(f"  {int(saat):02d}:00 - {int(saat):02d}:59: {sayi:,} haber")
    
    # Kategori analizi (eğer varsa)
    if 'Category' in df.columns:
        print(f"\n📂 KATEGORİLERE GÖRE DAĞILIM:")
        kategori_dagilim = df['Category'].value_counts()
        for kategori, sayi in kategori_dagilim.head(10).items():
            print(f"  {kategori}: {sayi:,} haber")
    
    # Veri kalitesi kontrolü
    print(f"\n✅ VERİ KALİTE KONTROLÜ:")
    print(f"  Toplam satır: {len(df):,}")
    print(f"  Boş hücreler:")
    for col in df.columns:
        bos_sayi = df[col].isna().sum()
        if bos_sayi > 0:
            bos_yuzde = (bos_sayi / len(df)) * 100
            print(f"    {col}: {bos_sayi:,} boş (%{bos_yuzde:.1f})")
    
    # Özet istatistikler
    print(f"\n📊 ÖZET:")
    if 'Start' in df.columns and df['Start'].notna().any():
        yil_sayisi = df['Year'].nunique()
        print(f"  Kapsanan yıl sayısı: {yil_sayisi}")
        
    if 'Currency' in df.columns:
        para_sayisi = df['Currency'].nunique()
        print(f"  Farklı para birimi sayısı: {para_sayisi}")
        
    if 'Name' in df.columns:
        haber_turu = df['Name'].nunique()
        print(f"  Farklı haber türü sayısı: {haber_turu}")
    
    if 'Impact' in df.columns and 'High' in df['Impact'].values:
        yuksek_etki = len(df[df['Impact'] == 'High'])
        print(f"  Yüksek etkili haber sayısı: {yuksek_etki:,}")

except Exception as e:
    print(f"❌ HATA: {str(e)}")
    print("\nAlternatif olarak, tek tek calendar dosyalarına bakalım...")
    
    # Calendar dosyalarını bul ve analiz et
    import glob
    calendar_files = glob.glob('data/calendar-event-list-*.csv')
    
    print(f"\nBulunan calendar dosyaları: {len(calendar_files)}")
    
    toplam_kayit = 0
    for dosya in sorted(calendar_files):
        try:
            temp_df = pd.read_csv(dosya)
            print(f"  {dosya}: {len(temp_df):,} kayıt")
            toplam_kayit += len(temp_df)
        except:
            print(f"  {dosya}: OKUNAMADI")
    
    print(f"\nToplam kayıt sayısı: {toplam_kayit:,}")

print("\n" + "=" * 70)
print("Analiz tamamlandı!")
