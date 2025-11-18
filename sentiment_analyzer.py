#!/usr/bin/env python3
"""
================================================================================
SENTIMENT ANALYZER V9 - Economic Calendar & News Blackout
================================================================================

Provides:
- Economic calendar integration (investpy/pandas)
- High-impact news detection
- Trading blackout periods
- Sentiment scoring (basic)

Author: E1 AI Agent (Emergent.sh)
Date: January 2025
Version: 9.0 FREE PRO
================================================================================
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('SentimentAnalyzerV9')


class SentimentAnalyzerV9:
    """
    Economic Calendar and News Sentiment Analysis.
    
    Features:
    - Load economic calendar from CSV
    - Detect high-impact events
    - Generate trading blackout periods
    - Basic sentiment scoring
    """
    
    def __init__(
        self,
        calendar_path: str = None,
        blackout_before_minutes: int = 30,
        blackout_after_minutes: int = 15,
        high_impact_keywords: List[str] = None
    ):
        """
        Initialize Sentiment Analyzer.
        
        Args:
            calendar_path: Path to economic calendar CSV
            blackout_before_minutes: Minutes before high-impact event to stop trading
            blackout_after_minutes: Minutes after high-impact event to resume
            high_impact_keywords: Keywords to identify high-impact events
        """
        self.calendar_path = calendar_path
        self.blackout_before = timedelta(minutes=blackout_before_minutes)
        self.blackout_after = timedelta(minutes=blackout_after_minutes)
        
        # High-impact keywords (default)
        self.high_impact_keywords = high_impact_keywords or [
            'NFP', 'Non-Farm', 'Interest Rate', 'GDP', 'CPI', 'Inflation',
            'FOMC', 'ECB', 'Fed', 'Central Bank', 'Employment', 'Unemployment'
        ]
        
        self.calendar_df = None
        self.blackout_periods = []
        
        # Load calendar if provided
        if calendar_path and os.path.exists(calendar_path):
            self.load_calendar(calendar_path)
        
        logger.info(f"📰 SentimentAnalyzerV9 initialized")
        logger.info(f"   Blackout: -{blackout_before_minutes}m to +{blackout_after_minutes}m")
        logger.info(f"   High-impact keywords: {len(self.high_impact_keywords)}")
    
    def load_calendar(self, path: str) -> pd.DataFrame:
        """
        Load economic calendar from CSV.
        
        Expected columns: datetime, Name, Impact, Currency, Category
        
        Args:
            path: Path to CSV file
        
        Returns:
            DataFrame with calendar data
        """
        logger.info(f"📥 Loading economic calendar from {path}...")
        
        try:
            df = pd.read_csv(path)
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Parse datetime
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            elif 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            else:
                logger.warning("⚠️  No datetime column found, using index")
                df['datetime'] = pd.to_datetime(df.index)
            
            # Drop rows with invalid datetime
            df = df.dropna(subset=['datetime'])
            
            # Ensure required columns exist
            if 'name' not in df.columns:
                df['name'] = 'Unknown Event'
            if 'impact' not in df.columns:
                df['impact'] = 'Medium'
            if 'currency' not in df.columns:
                df['currency'] = 'USD'
            
            # Standardize impact values
            df['impact'] = df['impact'].str.lower()
            
            self.calendar_df = df
            logger.info(f"✅ Loaded {len(df)} economic events")
            
            # Generate blackout periods
            self._generate_blackout_periods()
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Failed to load calendar: {e}")
            return pd.DataFrame()
    
    def _generate_blackout_periods(self):
        """
        Generate blackout periods from high-impact events.
        """
        if self.calendar_df is None or self.calendar_df.empty:
            logger.warning("⚠️  No calendar data, no blackout periods generated")
            return
        
        self.blackout_periods = []
        
        # Filter high-impact events
        high_impact = self.calendar_df[
            (self.calendar_df['impact'] == 'high') |
            self.calendar_df['name'].str.contains('|'.join(self.high_impact_keywords), case=False, na=False)
        ]
        
        # Create blackout periods
        for _, event in high_impact.iterrows():
            start = event['datetime'] - self.blackout_before
            end = event['datetime'] + self.blackout_after
            self.blackout_periods.append({
                'start': start,
                'end': end,
                'event': event['name'],
                'currency': event.get('currency', 'USD')
            })
        
        logger.info(f"🚫 Generated {len(self.blackout_periods)} blackout periods")
    
    def is_blackout(self, timestamp: pd.Timestamp, currency: str = None) -> bool:
        """
        Check if timestamp is in a blackout period.
        
        Args:
            timestamp: Time to check
            currency: Filter by currency (e.g., 'USD', 'EUR')
        
        Returns:
            True if in blackout period
        """
        if not self.blackout_periods:
            return False
        
        for period in self.blackout_periods:
            # Currency filter
            if currency and period.get('currency') != currency:
                continue
            
            # Check if in period
            if period['start'] <= timestamp <= period['end']:
                return True
        
        return False
    
    def get_upcoming_events(self, timestamp: pd.Timestamp, hours_ahead: int = 24) -> pd.DataFrame:
        """
        Get upcoming economic events.
        
        Args:
            timestamp: Current time
            hours_ahead: Look-ahead window in hours
        
        Returns:
            DataFrame with upcoming events
        """
        if self.calendar_df is None or self.calendar_df.empty:
            return pd.DataFrame()
        
        future_time = timestamp + timedelta(hours=hours_ahead)
        upcoming = self.calendar_df[
            (self.calendar_df['datetime'] >= timestamp) &
            (self.calendar_df['datetime'] <= future_time)
        ]
        
        return upcoming.sort_values('datetime')
    
    def compute_sentiment_score(self, timestamp: pd.Timestamp, symbol: str = 'EURUSD') -> float:
        """
        Compute basic sentiment score.
        
        Args:
            timestamp: Current time
            symbol: Trading symbol
        
        Returns:
            Sentiment score (-1 to 1, 0 is neutral)
        """
        # Simple heuristic: negative sentiment during blackout, neutral otherwise
        if self.is_blackout(timestamp):
            return -0.5
        
        # Check upcoming high-impact events (within 2 hours)
        upcoming = self.get_upcoming_events(timestamp, hours_ahead=2)
        high_impact_upcoming = upcoming[upcoming['impact'] == 'high']
        
        if len(high_impact_upcoming) > 0:
            return -0.3  # Slightly negative before high-impact events
        
        return 0.0  # Neutral
    
    def get_stats(self) -> Dict:
        """
        Get sentiment analyzer statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            'calendar_events': len(self.calendar_df) if self.calendar_df is not None else 0,
            'blackout_periods': len(self.blackout_periods),
            'high_impact_keywords': len(self.high_impact_keywords),
            'blackout_window': f"-{self.blackout_before.seconds//60}m to +{self.blackout_after.seconds//60}m"
        }


# =============================================================================
# Test Functions
# =============================================================================

def test_sentiment_analyzer():
    """Test SentimentAnalyzerV9 with sample data."""
    print("🧪 Testing SentimentAnalyzerV9...")
    
    # Create sample calendar data
    sample_data = {
        'datetime': [
            '2024-01-15 14:30:00',
            '2024-01-16 08:30:00',
            '2024-01-17 12:00:00'
        ],
        'name': ['NFP Report', 'GDP Release', 'Minor Event'],
        'impact': ['high', 'high', 'low'],
        'currency': ['USD', 'EUR', 'GBP']
    }
    
    # Save to temp CSV
    df = pd.DataFrame(sample_data)
    temp_path = '/tmp/test_calendar.csv'
    df.to_csv(temp_path, index=False)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzerV9(
        calendar_path=temp_path,
        blackout_before_minutes=30,
        blackout_after_minutes=15
    )
    
    # Test 1: Blackout detection
    test_time = pd.Timestamp('2024-01-15 14:25:00')  # 5 min before NFP
    is_blackout = analyzer.is_blackout(test_time)
    print(f"✓ Test 1: Blackout at {test_time}: {is_blackout}")
    assert is_blackout, "Should be in blackout period"
    
    # Test 2: No blackout
    safe_time = pd.Timestamp('2024-01-15 10:00:00')
    is_safe = analyzer.is_blackout(safe_time)
    print(f"✓ Test 2: Blackout at {safe_time}: {is_safe}")
    assert not is_safe, "Should not be in blackout period"
    
    # Test 3: Upcoming events
    current = pd.Timestamp('2024-01-15 12:00:00')
    upcoming = analyzer.get_upcoming_events(current, hours_ahead=6)
    print(f"✓ Test 3: Upcoming events (6h): {len(upcoming)}")
    assert len(upcoming) > 0, "Should find upcoming events"
    
    # Test 4: Sentiment score
    sentiment = analyzer.compute_sentiment_score(test_time)
    print(f"✓ Test 4: Sentiment during blackout: {sentiment}")
    assert sentiment < 0, "Sentiment should be negative during blackout"
    
    # Test 5: Stats
    stats = analyzer.get_stats()
    print(f"\n📊 Stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n✅ SentimentAnalyzerV9 tests passed!\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_sentiment_analyzer()
