#!/usr/bin/env python3
"""
JTTWS QUICK START & TEST SCRIPT V2
===================================
Date column sorunu düzeltilmiş versiyonu test eder
"""

import subprocess
import sys
import os

def check_requirements():
    """Gerekli paketleri kontrol et"""
    print("📦 PAKET KONTROLÜ")
    print("-" * 40)
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} yüklü")
        except ImportError:
            print(f"❌ {package_name} eksik")
            missing.append(package_name)
    
    if missing:
        print("\n⚠️ Eksik paketleri yüklemek için:")
        print(f"pip install {' '.join(missing)} --break-system-packages")
        return False
    
    return True

def check_data():
    """Veri dosyalarını kontrol et"""
    print("\n📊 VERİ KONTROLÜ")
    print("-" * 40)
    
    data_path = os.path.expanduser("~/Desktop/JTTWS/data")
    
    if not os.path.exists(data_path):
        print(f"❌ Veri klasörü bulunamadı: {data_path}")
        print("   Lütfen veri dosyalarınızın doğru yerde olduğundan emin olun")
        return False
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    found_any = False
    
    for symbol in symbols:
        symbol_dir = os.path.join(data_path, f"{symbol}2003-2024")
        if os.path.exists(symbol_dir):
            files = [f for f in os.listdir(symbol_dir) if f.endswith('.csv')]
            if files:
                print(f"✅ {symbol}: {len(files)} dosya bulundu")
                found_any = True
            else:
                print(f"⚠️ {symbol}: CSV dosyası bulunamadı")
        else:
            print(f"⚠️ {symbol}: Klasör bulunamadı")
    
    return found_any

def run_test(version="v2"):
    """Ana testi çalıştır"""
    print(f"\n🚀 TEST BAŞLATILIYOR (Version: {version})")
    print("-" * 40)
    
    # Hangi dosyayı çalıştıracağımızı belirle
    if version == "v2":
        script_name = "JTTWS_Training_v2.py"
    else:
        script_name = "JTTWS_Training.py"
    
    # Dosyanın varlığını kontrol et
    if not os.path.exists(script_name):
        print(f"❌ {script_name} dosyası bulunamadı!")
        return False
    
    try:
        # Önce veri kontrolü yapalım
        print("\n📋 Veri Yapısı Kontrolü:")
        print("-" * 40)
        result = subprocess.run(
            [sys.executable, script_name, "check_data"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(result.stdout)
        
        # Sonra paper trading testi
        print("\n💹 Paper Trading Testi:")
        print("-" * 40)
        result = subprocess.run(
            [sys.executable, script_name, "test_paper"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ Uyarılar/Hatalar:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ TEST BAŞARILI!")
            return True
        else:
            print("\n❌ Test sırasında hata oluştu")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Test 60 saniyeden uzun sürdü")
        return False
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

def main():
    """Ana fonksiyon"""
    print("=" * 50)
    print("JTTWS TRAINING SYSTEM V2 - QUICK TEST")
    print("=" * 50)
    
    # 1. Paket kontrolü
    if not check_requirements():
        print("\n⚠️ Önce eksik paketleri yükleyin!")
        return
    
    # 2. Veri kontrolü
    if not check_data():
        print("\n⚠️ Veri dosyaları eksik veya yanlış yerde!")
        print("Beklenen konum: ~/Desktop/JTTWS/data/")
        return
    
    # 3. Test çalıştır
    print("\n" + "=" * 50)
    if run_test("v2"):
        print("\n🎉 SİSTEM ÇALIŞIYOR!")
        print("\n✅ DÜZELTMELER:")
        print("  • Date kolonu sorunu çözüldü")
        print("  • Position sizing düzeltildi")
        print("  • Virtual cost hesaplaması düzeltildi")
        print("  • Balance yönetimi iyileştirildi")
        
        print("\n📝 KULLANIM:")
        print("  python JTTWS_Training_v2.py test_paper   # Paper trading testi")
        print("  python JTTWS_Training_v2.py check_data    # Veri kontrolü")
    else:
        print("\n⚠️ Sistem çalışmıyor. Lütfen hataları kontrol edin.")
        print("\nDeneyebilecekleriniz:")
        print("1. CSV dosyalarınızın kolonlarını kontrol edin")
        print("2. python JTTWS_Training_v2.py check_data komutunu çalıştırın")
        print("3. Log dosyasını inceleyin: ~/Desktop/JTTWS/logs/")

if __name__ == "__main__":
    main()
