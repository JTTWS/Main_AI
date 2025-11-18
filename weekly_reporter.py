#!/usr/bin/env python3
"""
Weekly Reporter Module
Generates detailed weekly performance reports with:
- Per-pair profitability
- News reaction analysis
- Lot size analytics
- Win/loss patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class WeeklyReporter:
    """
    Haftalık performans raporu oluşturur
    - Parite bazlı kar/zarar
    - Haber bazlı reaksiyon analizi
    - Lot analizi
    - Win/loss pattern'leri
    """
    
    def __init__(self):
        self.trade_history = []
        self.news_impacts = defaultdict(list)
        self.current_week_start = None
    
    def add_trade(self, trade_data: Dict):
        """
        Add a completed trade to history
        
        Args:
            trade_data: Dictionary containing:
                - pair: str
                - entry_time: datetime
                - exit_time: datetime
                - direction: 'LONG' or 'SHORT'
                - lot_size: float
                - entry_price: float
                - exit_price: float
                - pnl: float
                - result: 'WIN' or 'LOSS'
                - strategy_type: str (TREND, BREAKOUT, etc.)
                - nearby_news: List[Dict] (optional)
        """
        self.trade_history.append(trade_data)
        
        # Track news impacts
        if 'nearby_news' in trade_data and trade_data['nearby_news']:
            for news in trade_data['nearby_news']:
                key = f"{news['name']}_{news['category']}"
                self.news_impacts[key].append({
                    'pair': trade_data['pair'],
                    'result': trade_data['result'],
                    'pnl': trade_data['pnl'],
                    'time_to_news': news['minutes_diff']
                })
    
    def generate_weekly_report(self, week_start: datetime = None) -> Dict:
        """
        Generate comprehensive weekly report
        
        Args:
            week_start: Start of the week (if None, uses last 7 days)
        
        Returns:
            Dictionary with report data
        """
        if week_start is None:
            week_start = datetime.now() - timedelta(days=7)
        
        week_end = week_start + timedelta(days=7)
        
        # Filter trades for this week
        week_trades = [
            t for t in self.trade_history
            if week_start <= t['exit_time'] < week_end
        ]
        
        if not week_trades:
            logger.warning(f"No trades found for week starting {week_start.date()}")
            return {}
        
        report = {
            'week_start': week_start,
            'week_end': week_end,
            'total_trades': len(week_trades),
            'pairs': self._analyze_pairs(week_trades),
            'news_reactions': self._analyze_news_reactions(week_trades),
            'lot_analytics': self._analyze_lots(week_trades),
            'time_analytics': self._analyze_time_patterns(week_trades),
            'strategy_performance': self._analyze_strategies(week_trades),
            'overall_metrics': self._calculate_overall_metrics(week_trades)
        }
        
        return report
    
    def _analyze_pairs(self, trades: List[Dict]) -> Dict:
        """Analyze performance by currency pair"""
        pair_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'total_lots': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.0
        })
        
        for trade in trades:
            pair = trade['pair']
            stats = pair_stats[pair]
            
            stats['trades'] += 1
            stats['total_pnl'] += trade['pnl']
            stats['total_lots'] += trade['lot_size']
            
            if trade['result'] == 'WIN':
                stats['wins'] += 1
            else:
                stats['losses'] += 1
            
            if trade['pnl'] > stats['best_trade']:
                stats['best_trade'] = trade['pnl']
            if trade['pnl'] < stats['worst_trade']:
                stats['worst_trade'] = trade['pnl']
        
        # Calculate averages and win rates
        for pair, stats in pair_stats.items():
            stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
            stats['win_rate'] = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        
        # Sort by total PnL
        sorted_pairs = dict(sorted(pair_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True))
        
        return sorted_pairs
    
    def _analyze_news_reactions(self, trades: List[Dict]) -> Dict:
        """Analyze how news events affected trades"""
        news_stats = defaultdict(lambda: {
            'trades_affected': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'category': 'UNKNOWN'
        })
        
        for trade in trades:
            if 'nearby_news' not in trade or not trade['nearby_news']:
                continue
            
            for news in trade['nearby_news']:
                key = news['name']
                stats = news_stats[key]
                
                stats['trades_affected'] += 1
                stats['category'] = news['category']
                stats['total_pnl'] += trade['pnl']
                
                if trade['result'] == 'WIN':
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1
        
        # Calculate metrics
        for news_name, stats in news_stats.items():
            if stats['trades_affected'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['trades_affected'] * 100)
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades_affected']
        
        # Sort by trades affected
        sorted_news = dict(sorted(news_stats.items(), key=lambda x: x[1]['trades_affected'], reverse=True))
        
        return sorted_news
    
    def _analyze_lots(self, trades: List[Dict]) -> Dict:
        """Analyze lot sizing patterns"""
        lot_sizes = [t['lot_size'] for t in trades]
        pnls = [t['pnl'] for t in trades]
        
        # Correlation between lot size and PnL
        correlation = np.corrcoef(lot_sizes, pnls)[0, 1] if len(lot_sizes) > 1 else 0
        
        # Group by lot size ranges
        lot_ranges = {
            '0.01-0.05': [],
            '0.05-0.10': [],
            '0.10-0.20': [],
            '0.20-0.50': [],
            '0.50+': []
        }
        
        for trade in trades:
            lot = trade['lot_size']
            if lot < 0.05:
                lot_ranges['0.01-0.05'].append(trade)
            elif lot < 0.10:
                lot_ranges['0.05-0.10'].append(trade)
            elif lot < 0.20:
                lot_ranges['0.10-0.20'].append(trade)
            elif lot < 0.50:
                lot_ranges['0.20-0.50'].append(trade)
            else:
                lot_ranges['0.50+'].append(trade)
        
        lot_range_stats = {}
        for range_name, range_trades in lot_ranges.items():
            if range_trades:
                wins = sum(1 for t in range_trades if t['result'] == 'WIN')
                total_pnl = sum(t['pnl'] for t in range_trades)
                lot_range_stats[range_name] = {
                    'trades': len(range_trades),
                    'win_rate': (wins / len(range_trades) * 100),
                    'total_pnl': total_pnl,
                    'avg_pnl': total_pnl / len(range_trades)
                }
        
        return {
            'min_lot': min(lot_sizes),
            'max_lot': max(lot_sizes),
            'avg_lot': np.mean(lot_sizes),
            'median_lot': np.median(lot_sizes),
            'lot_pnl_correlation': correlation,
            'lot_ranges': lot_range_stats
        }
    
    def _analyze_time_patterns(self, trades: List[Dict]) -> Dict:
        """Analyze time-based patterns"""
        hour_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0.0})
        day_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0.0})
        
        for trade in trades:
            hour = trade['entry_time'].hour
            day = trade['entry_time'].strftime('%A')
            
            hour_stats[hour]['trades'] += 1
            hour_stats[hour]['total_pnl'] += trade['pnl']
            if trade['result'] == 'WIN':
                hour_stats[hour]['wins'] += 1
            
            day_stats[day]['trades'] += 1
            day_stats[day]['total_pnl'] += trade['pnl']
            if trade['result'] == 'WIN':
                day_stats[day]['wins'] += 1
        
        # Calculate win rates
        for hour, stats in hour_stats.items():
            stats['win_rate'] = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        
        for day, stats in day_stats.items():
            stats['win_rate'] = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        
        # Find best and worst hours
        best_hour = max(hour_stats.items(), key=lambda x: x[1]['total_pnl'])
        worst_hour = min(hour_stats.items(), key=lambda x: x[1]['total_pnl'])
        
        return {
            'hourly': dict(hour_stats),
            'daily': dict(day_stats),
            'best_hour': {'hour': best_hour[0], **best_hour[1]},
            'worst_hour': {'hour': worst_hour[0], **worst_hour[1]}
        }
    
    def _analyze_strategies(self, trades: List[Dict]) -> Dict:
        """Analyze performance by strategy type"""
        strategy_stats = defaultdict(lambda: {
            'trades': 0,
            'wins': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_pnl': 0.0
        })
        
        for trade in trades:
            strategy = trade.get('strategy_type', 'UNKNOWN')
            stats = strategy_stats[strategy]
            
            stats['trades'] += 1
            stats['total_pnl'] += trade['pnl']
            if trade['result'] == 'WIN':
                stats['wins'] += 1
        
        # Calculate metrics
        for strategy, stats in strategy_stats.items():
            if stats['trades'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['trades'] * 100)
                stats['avg_pnl'] = stats['total_pnl'] / stats['trades']
        
        return dict(strategy_stats)
    
    def _calculate_overall_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate overall performance metrics"""
        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        losses = len(trades) - wins
        
        win_pnls = [t['pnl'] for t in trades if t['result'] == 'WIN']
        loss_pnls = [t['pnl'] for t in trades if t['result'] == 'LOSS']
        
        return {
            'total_trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / len(trades) * 100) if trades else 0,
            'total_pnl': total_pnl,
            'avg_win': np.mean(win_pnls) if win_pnls else 0,
            'avg_loss': np.mean(loss_pnls) if loss_pnls else 0,
            'profit_factor': abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else 0,
            'largest_win': max(win_pnls) if win_pnls else 0,
            'largest_loss': min(loss_pnls) if loss_pnls else 0
        }
    
    def format_report_text(self, report: Dict) -> str:
        """Format report as readable text for Telegram"""
        if not report:
            return "❌ Rapor oluşturulamadı - veri yok"
        
        text = f"""
📊 HAFTALIK PERFORMANS RAPORU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Tarih: {report['week_start'].strftime('%d/%m/%Y')} - {report['week_end'].strftime('%d/%m/%Y')}

💰 GENEL PERFORMANS
  • Toplam Trade: {report['overall_metrics']['total_trades']}
  • Kazanan: {report['overall_metrics']['wins']} ({report['overall_metrics']['win_rate']:.1f}%)
  • Kaybeden: {report['overall_metrics']['losses']}
  • Toplam PnL: ${report['overall_metrics']['total_pnl']:.2f}
  • Profit Factor: {report['overall_metrics']['profit_factor']:.2f}
  • Ortalama Kazanç: ${report['overall_metrics']['avg_win']:.2f}
  • Ortalama Kayıp: ${report['overall_metrics']['avg_loss']:.2f}

📈 PARİTE BAZLI PERFORMANS
"""
        
        for pair, stats in report['pairs'].items():
            emoji = "🟢" if stats['total_pnl'] > 0 else "🔴"
            text += f"""{emoji} {pair}
  • Trade: {stats['trades']} | Win Rate: {stats['win_rate']:.1f}%
  • PnL: ${stats['total_pnl']:.2f} | Avg: ${stats['avg_pnl']:.2f}
  • En İyi: ${stats['best_trade']:.2f} | En Kötü: ${stats['worst_trade']:.2f}
  • Toplam Lot: {stats['total_lots']:.2f}

"""
        
        # Top news reactions
        if report['news_reactions']:
            text += "📰 EN ÇOK ETKİLEYEN HABERLER (Top 5)\n"
            top_news = list(report['news_reactions'].items())[:5]
            for news_name, stats in top_news:
                emoji = "⚠️" if stats['category'] == 'CRITICAL' else "📌"
                text += f"""{emoji} {news_name} ({stats['category']})
  • Etkilenen Trade: {stats['trades_affected']}
  • Win Rate: {stats['win_rate']:.1f}%
  • Avg PnL: ${stats['avg_pnl']:.2f}

"""
        
        # Lot analytics
        lot_stats = report['lot_analytics']
        text += f"""📊 LOT ANALİZİ
  • Min: {lot_stats['min_lot']:.2f} | Max: {lot_stats['max_lot']:.2f}
  • Ortalama: {lot_stats['avg_lot']:.2f} | Medyan: {lot_stats['median_lot']:.2f}
  • Lot-PnL Korelasyon: {lot_stats['lot_pnl_correlation']:.2f}

"""
        
        # Best/worst trading hours
        time_stats = report['time_analytics']
        text += f"""⏰ ZAMAN ANALİZİ
  • En İyi Saat: {time_stats['best_hour']['hour']}:00 
    ({time_stats['best_hour']['trades']} trade, ${time_stats['best_hour']['total_pnl']:.2f})
  • En Kötü Saat: {time_stats['worst_hour']['hour']}:00
    ({time_stats['worst_hour']['trades']} trade, ${time_stats['worst_hour']['total_pnl']:.2f})

"""
        
        text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n✅ Rapor oluşturma: " + datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        
        return text


if __name__ == '__main__':
    # Test the reporter
    reporter = WeeklyReporter()
    
    # Add sample trades
    sample_trades = [
        {
            'pair': 'EURUSD',
            'entry_time': datetime.now() - timedelta(days=3, hours=5),
            'exit_time': datetime.now() - timedelta(days=3, hours=3),
            'direction': 'LONG',
            'lot_size': 0.10,
            'entry_price': 1.0850,
            'exit_price': 1.0870,
            'pnl': 200.0,
            'result': 'WIN',
            'strategy_type': 'TREND',
            'nearby_news': [
                {'name': 'NFP', 'category': 'CRITICAL', 'minutes_diff': -120}
            ]
        },
        {
            'pair': 'GBPUSD',
            'entry_time': datetime.now() - timedelta(days=2, hours=8),
            'exit_time': datetime.now() - timedelta(days=2, hours=6),
            'direction': 'SHORT',
            'lot_size': 0.15,
            'entry_price': 1.2650,
            'exit_price': 1.2620,
            'pnl': -150.0,
            'result': 'LOSS',
            'strategy_type': 'BREAKOUT',
            'nearby_news': []
        }
    ]
    
    for trade in sample_trades:
        reporter.add_trade(trade)
    
    # Generate report
    report = reporter.generate_weekly_report()
    
    # Print formatted report
    print(reporter.format_report_text(report))
