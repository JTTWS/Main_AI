#!/usr/bin/env python3
"""
Serkan Bey'in Bot Sistemi için Özel Test Script
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

print("🔍 Bot Sistem Testi Başlıyor...")

# Modülleri kontrol et
modules_ok = True
required_modules = [
    'train_bot_v9',
    'ppo_agent', 
    'feature_engineer_v9',
    'data_manager_v8',
    'trading_environment_pro'
]

for module in required_modules:
    try:
        __import__(module)
        print(f"✅ {module} yüklendi")
    except ImportError as e:
        print(f"❌ {module} yüklenemedi: {e}")
        modules_ok = False

if not modules_ok:
    print("\n⚠️  Bazı modüller eksik. Lütfen gerekli kütüphaneleri yükleyin.")
    sys.exit(1)

print("\n✅ Tüm modüller başarıyla yüklendi!")
print("📊 Basit sistem kontrolü yapılıyor...")

try:
    # Feature engineer'ı test edelim
    from feature_engineer_v9 import FeatureEngineerV9
    fe = FeatureEngineerV9()
    print("✅ Feature Engineer hazır")
    
    # Data manager'ı test edelim
    from data_manager_v8 import DataManagerV8
    dm = DataManagerV8(data_dir='./data')
    print("✅ Data Manager hazır")
    
    print("\n🎉 Sistem testi başarılı! Bot çalışmaya hazır.")
    
except Exception as e:
    print(f"❌ Test sırasında hata: {e}")
    import traceback
    traceback.print_exc()
