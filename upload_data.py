#!/usr/bin/env python3
"""
================================================================================
DATA UPLOAD SCRIPT FOR JTTWS V8
================================================================================

Handles uploading and extracting compressed data files.

Usage:
    1. Compress data locally:
       cd ~/Desktop/JTTWS/
       tar -czf jttws_data_complete.tar.gz data/
    
    2. Copy the tar.gz file to /app/
    
    3. Run this script:
       python upload_data.py

================================================================================
"""

import os
import sys
import tarfile
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataUploader')


def extract_data(tar_path: str = '/app/jttws_data_complete.tar.gz', extract_to: str = '/app'):
    """
    Extract compressed data file to target directory.
    
    Args:
        tar_path: Path to compressed tar.gz file
        extract_to: Target directory for extraction
    """
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║           JTTWS V8 DATA UPLOAD & EXTRACTION                ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")
    
    # Check if tar file exists
    if not os.path.exists(tar_path):
        logger.error(f"❌ Tar file not found: {tar_path}")
        logger.info("\n📝 INSTRUCTIONS:")
        logger.info("   1. Lokal terminalinizde çalıştırın:")
        logger.info("      cd ~/Desktop/JTTWS/")
        logger.info("      tar -czf jttws_data_complete.tar.gz data/")
        logger.info("")
        logger.info("   2. Dosyayı /app/ klasörüne kopyalayın")
        logger.info("   3. Bu scripti tekrar çalıştırın")
        return False
    
    # Get file size
    file_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    logger.info(f"📦 Found tar file: {tar_path} ({file_size_mb:.2f} MB)")
    
    # Extract
    try:
        logger.info(f"📂 Extracting to: {extract_to}")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            # List contents
            members = tar.getmembers()
            logger.info(f"📄 Archive contains {len(members)} files/directories")
            
            # Extract all
            tar.extractall(path=extract_to)
            logger.info(f"✅ Extraction complete!")
        
        # Verify extraction
        data_dir = os.path.join(extract_to, 'data')
        if os.path.exists(data_dir):
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
            
            logger.info(f"\n📊 Verification:")
            logger.info(f"   Data directory: {data_dir}")
            logger.info(f"   Subdirectories: {len(subdirs)}")
            logger.info(f"   Files: {len(files)}")
            
            if subdirs:
                logger.info(f"\n📁 Subdirectories found:")
                for d in subdirs:
                    subdir_path = os.path.join(data_dir, d)
                    file_count = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
                    logger.info(f"      {d}: {file_count} files")
            
            logger.info("\n✅ Data uploaded successfully!")
            logger.info("\n🚀 Next steps:")
            logger.info("   1. Test data loading:")
            logger.info("      python data_manager_v8.py")
            logger.info("")
            logger.info("   2. Run V8 training:")
            logger.info("      python ultimate_bot_v8_ppo.py --mode train --optuna-trials 50")
            
            return True
        else:
            logger.error(f"❌ Data directory not found after extraction: {data_dir}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return False


def check_data_structure():
    """Check if data directory structure is correct."""
    logger.info("\n╔════════════════════════════════════════════════════════════╗")
    logger.info("║              DATA STRUCTURE VERIFICATION                   ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")
    
    data_dir = '/app/data'
    
    if not os.path.exists(data_dir):
        logger.warning(f"⚠️  Data directory not found: {data_dir}")
        return False
    
    # Expected structure
    expected = {
        'directories': ['EURUSD2003-2024', 'GBPUSD2003-2024', 'USDJPY2003-2024'],
        'files': [
            'EURUSD_weekly_ranges.csv',
            'GBPUSD_weekly_ranges.csv',
            'USDJPY_weekly_ranges.csv',
            'combined_economic_calendar.csv'
        ]
    }
    
    logger.info(f"\n📂 Checking: {data_dir}")
    
    all_good = True
    
    # Check directories
    logger.info("\n📁 Expected directories:")
    for dirname in expected['directories']:
        dirpath = os.path.join(data_dir, dirname)
        if os.path.exists(dirpath):
            file_count = len([f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))])
            logger.info(f"   ✅ {dirname}: {file_count} files")
        else:
            logger.warning(f"   ❌ {dirname}: NOT FOUND")
            all_good = False
    
    # Check files
    logger.info("\n📄 Expected files:")
    for filename in expected['files']:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            logger.info(f"   ✅ {filename}: {size_kb:.1f} KB")
        else:
            logger.warning(f"   ❌ {filename}: NOT FOUND")
            all_good = False
    
    if all_good:
        logger.info("\n✅ Data structure is correct!")
    else:
        logger.warning("\n⚠️  Some files/directories are missing")
    
    return all_good


if __name__ == '__main__':
    print("\n")
    
    # Check if data already exists
    if os.path.exists('/app/data'):
        logger.info("📂 Data directory already exists, checking structure...")
        check_data_structure()
        
        response = input("\n🔄 Do you want to re-extract? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            logger.info("✅ Skipping extraction")
            sys.exit(0)
    
    # Extract data
    success = extract_data()
    
    if success:
        # Verify structure
        check_data_structure()
    
    print("\n")
