#!/usr/bin/env python3
"""
Economic Calendar Combiner & Categorizer
"""

import pandas as pd
import requests
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CALENDAR_URLS = [
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/xklmovt8_calendar-event-list-2.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/a86rism0_calendar-event-list-3.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/ohxovss0_calendar-event-list-4.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/ts9aja5f_calendar-event-list-5.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/yhra1dck_calendar-event-list-6.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/vntb64jw_calendar-event-list-7.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/zpjwlssq_calendar-event-list-8.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/bxrpxyqh_calendar-event-list-9.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/97xy7m8b_calendar-event-list-10.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/yb023b57_calendar-event-list-11.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/rdxcc0mr_calendar-event-list-12.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/8ltlqrhh_calendar-event-list-13.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/gf9mtt8l_calendar-event-list-14.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/jecvkle1_calendar-event-list-15.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/n3pwfrz6_calendar-event-list-16.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/gwt5lu1n_calendar-event-list-17.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/6oljo9d4_calendar-event-list-18.csv",
    "https://customer-assets.emergentagent.com/job_ftmo-algo-trader/artifacts/ej2w5t8w_calendar-event-list-19.csv",
]

CRITICAL_NEWS = [
    'Non-Farm', 'NFP', 'Nonfarm', 'Employment Change',
    'FOMC', 'Fed Interest Rate', 'Federal Funds Rate',
    'ECB Interest Rate', 'ECB Press Conference',
    'BoE Interest Rate', 'Bank of England',
    'BoJ Interest Rate', 'Bank of Japan',
    'SNB Interest Rate', 'Swiss National Bank',
]

HIGH_IMPACT_NEWS = [
    'Consumer Price Index', 'CPI',
    'Gross Domestic Product', 'GDP',
    'Unemployment Rate',
    'Retail Sales',
    'Trade Balance',
    'Manufacturing PMI',
    'Services PMI',
    'Industrial Production',
    'Consumer Confidence',
    'Producer Price Index', 'PPI',
]

MEDIUM_IMPACT_NEWS = [
    'Building Permits',
    'Housing Starts',
    'Existing Home Sales',
    'New Home Sales',
    'Durable Goods',
    'Factory Orders',
    'Business Confidence',
    'ZEW Economic Sentiment',
]

def categorize_news(name, original_impact):
    name_upper = name.upper()
    for keyword in CRITICAL_NEWS:
        if keyword.upper() in name_upper:
            return 'CRITICAL'
    for keyword in HIGH_IMPACT_NEWS:
        if keyword.upper() in name_upper:
            return 'HIGH'
    for keyword in MEDIUM_IMPACT_NEWS:
        if keyword.upper() in name_upper:
            return 'MEDIUM'
    if original_impact == 'HIGH':
        return 'HIGH'
    elif original_impact == 'MEDIUM':
        return 'MEDIUM'
    else:
        return 'LOW'

def download_and_combine_calendars(output_file='~/Desktop/JTTWS/data/combined_economic_calendar.csv'):
    all_dfs = []
    logger.info(f"Starting to download {len(CALENDAR_URLS)} files...")
    
    for i, url in enumerate(CALENDAR_URLS, 1):
        try:
            logger.info(f"Downloading file {i}/{len(CALENDAR_URLS)}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            temp_file = f'/tmp/calendar_{i}.csv'
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            df = pd.read_csv(temp_file)
            logger.info(f"  ✓ Loaded {len(df)} events")
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            continue
    
    if not all_dfs:
        logger.error("No files downloaded!")
        return None
    
    logger.info("Combining data...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    logger.info("Parsing dates...")
    combined_df['datetime'] = pd.to_datetime(combined_df['Start'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
    failed_mask = combined_df['datetime'].isna()
    if failed_mask.any():
        combined_df.loc[failed_mask, 'datetime'] = pd.to_datetime(
            combined_df.loc[failed_mask, 'Start'], 
            format='%d/%m/%Y %H:%M:%S', 
            errors='coerce'
        )
    
    combined_df = combined_df.dropna(subset=['datetime'])
    combined_df = combined_df.drop_duplicates(subset=['Start', 'Name', 'Currency'])
    
    logger.info("Categorizing news...")
    combined_df['Category'] = combined_df.apply(
        lambda row: categorize_news(row['Name'], row['Impact']), 
        axis=1
    )
    
    combined_df = combined_df.sort_values('datetime')
    
    logger.info("="*60)
    logger.info(f"Total events: {len(combined_df)}")
    logger.info(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
    for category in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = len(combined_df[combined_df['Category'] == category])
        pct = (count / len(combined_df) * 100)
        logger.info(f"  {category:10s}: {count:6d} events ({pct:.1f}%)")
    
    output_file = os.path.expanduser(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved to: {output_file}")
    
    return combined_df

if __name__ == '__main__':
    df = download_and_combine_calendars()
    if df is not None:
        logger.info("✓ Complete!")
    else:
        logger.error("✗ Failed!")
