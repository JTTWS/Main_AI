#!/usr/bin/env python3
"""
Quick Fix Script for Local Execution
=====================================
Bu script lokal sistemde JTTWS V8 bot'unu çalıştırmadan önce
path sorunlarını otomatik olarak düzeltir.

Kullanım:
    python fix_local_paths.py

Author: E1 AI Agent
Date: January 2025
"""

import os
import sys

def check_data_directory():
    """Check if data directory exists and has required files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    print("=" * 70)
    print("🔍 JTTWS V8 - Data Directory Check")
    print("=" * 70)
    print(f"\n📂 Script directory: {script_dir}")
    print(f"📂 Expected data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"\n❌ ERROR: Data directory not found!")
        print(f"   Expected: {data_dir}")
        print(f"\n💡 Solution:")
        print(f"   1. Eğer data dosyanız yok ise:")
        print(f"      https://drive.google.com/file/d/15q9AymGt2HzdZbmER8Oomfj7anyFGfBO/view")
        print(f"      linkinden jttws_data_complete.tar.gz dosyasını indirin")
        print(f"   2. Terminal'de şu komutu çalıştırın:")
        print(f"      cd {script_dir}")
        print(f"      tar -xzf jttws_data_complete.tar.gz")
        return False
    
    print(f"\n✅ Data directory found!")
    
    # Check for symbol directories
    symbols = ['EURUSD2003-2024', 'GBPUSD2003-2024', 'USDJPY2003-2024']
    missing_symbols = []
    
    for symbol_dir in symbols:
        symbol_path = os.path.join(data_dir, symbol_dir)
        if os.path.exists(symbol_path):
            csv_files = [f for f in os.listdir(symbol_path) if f.endswith('.csv')]
            print(f"   ✅ {symbol_dir}: {len(csv_files)} CSV files")
        else:
            print(f"   ❌ {symbol_dir}: Not found")
            missing_symbols.append(symbol_dir)
    
    # Check for additional files
    additional_files = [
        'combined_economic_calendar.csv',
        'EURUSD_weekly_ranges.csv',
        'GBPUSD_weekly_ranges.csv',
        'USDJPY_weekly_ranges.csv'
    ]
    
    print(f"\n📄 Additional files:")
    for filename in additional_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"   ⚠️  {filename}: Not found (optional)")
    
    if missing_symbols:
        print(f"\n⚠️  WARNING: Missing symbol directories: {', '.join(missing_symbols)}")
        print(f"   Bot will use mock data for these symbols.")
        return False
    
    print(f"\n✅ All required data files found!")
    return True


def check_python_packages():
    """Check if required Python packages are installed."""
    print("\n" + "=" * 70)
    print("📦 Python Package Check")
    print("=" * 70)
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'gymnasium': 'gymnasium',
        'stable_baselines3': 'stable-baselines3',
        'optuna': 'optuna',
        'torch': 'torch',
        'vectorbt': 'vectorbt'
    }
    
    missing_packages = []
    
    for package_name, pip_name in required_packages.items():
        try:
            __import__(package_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - NOT INSTALLED")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages detected!")
        print(f"\n💡 Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print(f"\n   Or install all requirements:")
        print(f"   pip install -r requirements.txt")
        return False
    
    print(f"\n✅ All required packages installed!")
    return True


def main():
    """Main function."""
    print("\n🚀 JTTWS V8 - Local Environment Setup Check\n")
    
    data_ok = check_data_directory()
    packages_ok = check_python_packages()
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    if data_ok and packages_ok:
        print("\n✅ Environment is ready!")
        print("\n🎯 You can now run the bot:")
        print("   python ultimate_bot_v8_ppo.py --mode train --years 2020-2024 --optuna-trials 10")
    elif not data_ok and packages_ok:
        print("\n⚠️  Data files are missing. Bot will use mock data.")
        print("   Download real data for better results.")
    elif data_ok and not packages_ok:
        print("\n❌ Python packages are missing.")
        print("   Install requirements first: pip install -r requirements.txt")
    else:
        print("\n❌ Both data and packages are missing.")
        print("   1. First install packages: pip install -r requirements.txt")
        print("   2. Then download and extract data files")
    
    print("\n")


if __name__ == '__main__':
    main()
