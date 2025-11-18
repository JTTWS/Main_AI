#!/usr/bin/env python3
"""
News Manager Module - Enhanced News Blackout System
Manages economic calendar events and trading restrictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class NewsManager:
    """
    Gelişmiş haber yönetim sistemi
    - Haber kategorilerine göre farklı blackout süreleri
    - Haber bazlı volatilite profili
    - Detaylı loglama
    """
    
    def __init__(self, calendar_file: Path):
        """
        Args:
            calendar_file: Combined economic calendar CSV file
        """
        self.calendar_file = calendar_file
        self.calendar_df = None
        self.news_stats = {}
        self.load_calendar()
    
    def load_calendar(self):
        """Load and prepare economic calendar"""
        try:
            if not self.calendar_file.exists():
                logger.warning(f"Calendar file not found: {self.calendar_file}")
                logger.warning("NewsBlackout will be DISABLED")
                return
            
            self.calendar_df = pd.read_csv(self.calendar_file)
            
            # Parse datetime column
            if 'datetime' not in self.calendar_df.columns:
                logger.error("Calendar file missing 'datetime' column")
                return
            
            self.calendar_df['datetime'] = pd.to_datetime(self.calendar_df['datetime'])
            
            # Validate required columns
            required_cols = ['datetime', 'Name', 'Impact', 'Currency', 'Category']
            missing_cols = [col for col in required_cols if col not in self.calendar_df.columns]
            if missing_cols:
                logger.error(f"Calendar missing columns: {missing_cols}")
                return
            
            # Statistics
            total_events = len(self.calendar_df)
            categories = self.calendar_df['Category'].value_counts().to_dict()
            
            logger.info("=" * 60)
            logger.info("NEWS MANAGER INITIALIZED")
            logger.info("=" * 60)
            logger.info(f"Total events: {total_events:,}")
            logger.info(f"Date range: {self.calendar_df['datetime'].min()} to {self.calendar_df['datetime'].max()}")
            logger.info(f"Categories:")
            for cat, count in sorted(categories.items()):
                pct = (count / total_events * 100)
                logger.info(f"  {cat:10s}: {count:6,d} events ({pct:5.1f}%)")
            
            # Build news statistics (volatility profiles will be calculated during training)
            self._build_news_stats()
            
            logger.info("✓ News Manager ready!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error loading calendar: {e}")
            self.calendar_df = None
    
    def _build_news_stats(self):
        """Build statistics for each news type"""
        if self.calendar_df is None:
            return
        
        # Group by news name and category
        for name in self.calendar_df['Name'].unique():
            news_events = self.calendar_df[self.calendar_df['Name'] == name]
            category = news_events['Category'].iloc[0]
            currencies = news_events['Currency'].unique().tolist()
            
            self.news_stats[name] = {
                'category': category,
                'currencies': currencies,
                'count': len(news_events),
                'avg_volatility': None,  # Will be calculated during training
                'win_rate_after': None,   # Will be calculated during training
            }
    
    def is_blackout_period(
        self, 
        current_time: datetime, 
        currency: str,
        blackout_config: Dict[str, int]
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if current time is in a news blackout period
        
        Args:
            current_time: Current datetime
            currency: Currency to check (USD, EUR, GBP, JPY)
            blackout_config: Dictionary with blackout minutes for each category
                Example: {
                    'CRITICAL_BEFORE': 60,
                    'CRITICAL_AFTER': 60,
                    'HIGH_BEFORE': 30,
                    'HIGH_AFTER': 30,
                    'MEDIUM_BEFORE': 15,
                    'MEDIUM_AFTER': 15
                }
        
        Returns:
            (is_blackout, event_info)
            - is_blackout: True if in blackout period
            - event_info: Dict with event details if in blackout, else None
        """
        if self.calendar_df is None:
            return False, None
        
        # Filter events for this currency
        currency_events = self.calendar_df[self.calendar_df['Currency'] == currency].copy()
        
        if currency_events.empty:
            return False, None
        
        # Check each category
        for category in ['CRITICAL', 'HIGH', 'MEDIUM']:
            before_key = f'{category}_BEFORE'
            after_key = f'{category}_AFTER'
            
            if before_key not in blackout_config or after_key not in blackout_config:
                continue
            
            before_minutes = blackout_config[before_key]
            after_minutes = blackout_config[after_key]
            
            # Filter events of this category
            cat_events = currency_events[currency_events['Category'] == category]
            
            for _, event in cat_events.iterrows():
                event_time = event['datetime']
                
                # Check if we're in blackout window
                start_blackout = event_time - timedelta(minutes=before_minutes)
                end_blackout = event_time + timedelta(minutes=after_minutes)
                
                if start_blackout <= current_time <= end_blackout:
                    time_to_event = (event_time - current_time).total_seconds() / 60
                    
                    return True, {
                        'category': category,
                        'name': event['Name'],
                        'event_time': event_time,
                        'time_to_event_minutes': time_to_event,
                        'currency': currency,
                        'before_minutes': before_minutes,
                        'after_minutes': after_minutes
                    }
        
        return False, None
    
    def get_upcoming_news(
        self, 
        current_time: datetime, 
        currency: str, 
        lookahead_hours: int = 24
    ) -> List[Dict]:
        """
        Get upcoming news events for a currency
        
        Args:
            current_time: Current datetime
            currency: Currency code
            lookahead_hours: How many hours ahead to look
        
        Returns:
            List of upcoming news events
        """
        if self.calendar_df is None:
            return []
        
        end_time = current_time + timedelta(hours=lookahead_hours)
        
        upcoming = self.calendar_df[
            (self.calendar_df['Currency'] == currency) &
            (self.calendar_df['datetime'] >= current_time) &
            (self.calendar_df['datetime'] <= end_time)
        ].sort_values('datetime')
        
        events = []
        for _, event in upcoming.iterrows():
            events.append({
                'name': event['Name'],
                'datetime': event['datetime'],
                'category': event['Category'],
                'impact': event['Impact'],
                'hours_until': (event['datetime'] - current_time).total_seconds() / 3600
            })
        
        return events
    
    def get_news_at_time(self, target_time: datetime, currency: str, window_minutes: int = 60) -> List[Dict]:
        """
        Get news events around a specific time
        
        Args:
            target_time: Time to check
            currency: Currency code
            window_minutes: Window size (before and after)
        
        Returns:
            List of news events in the window
        """
        if self.calendar_df is None:
            return []
        
        start_time = target_time - timedelta(minutes=window_minutes)
        end_time = target_time + timedelta(minutes=window_minutes)
        
        events = self.calendar_df[
            (self.calendar_df['Currency'] == currency) &
            (self.calendar_df['datetime'] >= start_time) &
            (self.calendar_df['datetime'] <= end_time)
        ]
        
        result = []
        for _, event in events.iterrows():
            result.append({
                'name': event['Name'],
                'datetime': event['datetime'],
                'category': event['Category'],
                'impact': event['Impact'],
                'minutes_diff': (event['datetime'] - target_time).total_seconds() / 60
            })
        
        return result
    
    def log_news_impact(self, trade_time: datetime, currency: str, result: str, pnl: float):
        """
        Log the impact of news on a trade (for learning)
        
        Args:
            trade_time: When the trade was opened/closed
            currency: Currency pair
            result: 'win' or 'loss'
            pnl: Profit/loss amount
        """
        # Get news around this time
        nearby_news = self.get_news_at_time(trade_time, currency, window_minutes=120)
        
        if nearby_news:
            logger.debug(f"Trade at {trade_time} | {currency} | {result} | PnL: ${pnl:.2f}")
            logger.debug(f"  Nearby news events:")
            for news in nearby_news:
                logger.debug(f"    - {news['name']} ({news['category']}) at {news['datetime']} ({news['minutes_diff']:.0f}m)")
    
    def get_statistics_summary(self) -> Dict:
        """Get summary statistics about the calendar"""
        if self.calendar_df is None:
            return {}
        
        return {
            'total_events': len(self.calendar_df),
            'categories': self.calendar_df['Category'].value_counts().to_dict(),
            'currencies': self.calendar_df['Currency'].value_counts().to_dict(),
            'date_range': (
                self.calendar_df['datetime'].min(),
                self.calendar_df['datetime'].max()
            ),
            'unique_news_types': len(self.news_stats)
        }


# Convenience function for creating blackout config
def create_blackout_config(critical_before=60, critical_after=60,
                          high_before=30, high_after=30,
                          medium_before=15, medium_after=15):
    """Helper function to create blackout configuration"""
    return {
        'CRITICAL_BEFORE': critical_before,
        'CRITICAL_AFTER': critical_after,
        'HIGH_BEFORE': high_before,
        'HIGH_AFTER': high_after,
        'MEDIUM_BEFORE': medium_before,
        'MEDIUM_AFTER': medium_after,
    }


if __name__ == '__main__':
    # Test the news manager
    from bot_config import BotConfig
    
    logging.basicConfig(level=logging.INFO)
    
    news_mgr = NewsManager(BotConfig.NEWS_CALENDAR_FILE)
    
    # Test blackout check
    test_time = datetime(2023, 3, 10, 13, 15)  # Example: around NFP time
    blackout_cfg = create_blackout_config()
    
    is_blackout, event = news_mgr.is_blackout_period(test_time, 'USD', blackout_cfg)
    
    if is_blackout:
        print(f"\n⚠️ BLACKOUT ACTIVE!")
        print(f"Event: {event['name']}")
        print(f"Category: {event['category']}")
        print(f"Event time: {event['event_time']}")
        print(f"Time to event: {event['time_to_event_minutes']:.0f} minutes")
    else:
        print(f"\n✓ No blackout at {test_time}")
    
    # Test upcoming news
    upcoming = news_mgr.get_upcoming_news(test_time, 'USD', lookahead_hours=48)
    print(f"\nUpcoming USD news (next 48h): {len(upcoming)} events")
    for news in upcoming[:5]:
        print(f"  - {news['name']} ({news['category']}) in {news['hours_until']:.1f}h")
