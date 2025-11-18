#!/usr/bin/env python3
"""
Enhanced Trade Logger
Her trade için detaylı bilgi kaydeder
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

class EnhancedTradeLogger:
    """
    Detaylı trade logging sistemi
    - İndikatör değerleri
    - Yakın haberler
    - Lot hesaplama mantığı
    - Risk/Reward
    """
    
    def __init__(self, logger: logging.Logger, news_manager=None):
        self.logger = logger
        self.news_manager = news_manager
        self.trade_count = 0
    
    def log_trade_entry(self, trade_data: Dict):
        """Trade açılışını detaylı logla"""
        self.trade_count += 1
        
        self.logger.info("\n" + "="*70)
        self.logger.info(f"📊 TRADE #{self.trade_count} - {trade_data['type']} {trade_data['pair']}")
        self.logger.info("="*70)
        
        # Temel bilgiler
        self.logger.info(f"⏰ Zaman: {trade_data.get('time', datetime.now())}")
        self.logger.info(f"💰 Lot: {trade_data['lot']:.2f}")
        self.logger.info(f"📍 Giriş: {trade_data['entry_price']:.5f}")
        
        if 'sl' in trade_data:
            self.logger.info(f"🛡️ Stop Loss: {trade_data['sl']:.5f}")
        if 'tp' in trade_data:
            self.logger.info(f"🎯 Take Profit: {trade_data['tp']:.5f}")
        
        # İndikatörler
        if 'indicators' in trade_data:
            self.logger.info(f"\n📈 İNDİKATÖRLER:")
            for ind, value in trade_data['indicators'].items():
                self.logger.info(f"  • {ind}: {value}")
        
        # Lot hesaplama mantığı
        if 'lot_calculation' in trade_data:
            self.logger.info(f"\n💡 LOT HESAPLAMA:")
            calc = trade_data['lot_calculation']
            self.logger.info(f"  • Risk Miktarı: ${calc.get('risk_amount', 0):.2f}")
            self.logger.info(f"  • ATR: {calc.get('atr', 0):.5f}")
            self.logger.info(f"  • Kelly: {calc.get('kelly', 0):.3f}")
            self.logger.info(f"  • Final Lot: {trade_data['lot']:.2f}")
        
        # Yakın haberler
        if self.news_manager and 'time' in trade_data:
            nearby_news = self._get_nearby_news(
                trade_data['time'], 
                trade_data['pair'][:3]  # Currency (EUR, GBP, etc.)
            )
            
            if nearby_news:
                self.logger.info(f"\n📰 YAKIN HABERLER (±30dk):")
                for news in nearby_news[:5]:
                    time_diff = int(news['minutes_diff'])
                    self.logger.info(
                        f"  • [{news['category']}] {news['name']} "
                        f"({time_diff:+d}dk)"
                    )
            else:
                self.logger.info(f"\n📰 Yakın haber yok")
        
        # Trade nedeni
        if 'reason' in trade_data:
            self.logger.info(f"\n🤔 SEBEP:")
            self.logger.info(f"  {trade_data['reason']}")
        
        self.logger.info("="*70 + "\n")
    
    def log_trade_exit(self, trade_data: Dict):
        """Trade kapanışını detaylı logla"""
        self.logger.info("\n" + "="*70)
        self.logger.info(f"📊 TRADE KAPANDI - {trade_data['type']} {trade_data['pair']}")
        self.logger.info("="*70)
        
        self.logger.info(f"⏰ Açılış: {trade_data.get('entry_time', 'N/A')}")
        self.logger.info(f"⏰ Kapanış: {trade_data.get('exit_time', datetime.now())}")
        self.logger.info(f"📍 Giriş Fiyat: {trade_data['entry_price']:.5f}")
        self.logger.info(f"📍 Çıkış Fiyat: {trade_data['exit_price']:.5f}")
        
        pnl = trade_data.get('pnl', 0)
        emoji = "✅" if pnl > 0 else "❌"
        self.logger.info(f"💰 Kar/Zarar: {emoji} ${pnl:.2f}")
        
        if 'duration_minutes' in trade_data:
            self.logger.info(f"⏱️ Süre: {trade_data['duration_minutes']} dakika")
        
        self.logger.info("="*70 + "\n")
    
    def _get_nearby_news(self, trade_time: datetime, currency: str):
        """Yakındaki haberleri al"""
        if not self.news_manager or not self.news_manager.calendar_df is not None:
            return []
        
        return self.news_manager.get_news_at_time(trade_time, currency, window_minutes=30)
