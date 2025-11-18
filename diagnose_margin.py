#!/usr/bin/env python3
"""
diagnose_margin.py - Margin Calculation & Trade Execution Smoke Test

Validates:
1. FX margin calculation accuracy (EURUSD, USDJPY, cross pairs)
2. Pip value calculation per lot
3. Trade execution with proper margin checking
4. Balance updates after trades

Expected Results:
- EURUSD 0.01 lots: ~$10-15 margin (leverage 100)
- USDJPY 0.01 lots: ~$10-15 margin (leverage 100)
- Trades execute successfully
- Balances update correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from JTTWS_v5_COMPLETE_FIXED import (
    PaperTradingSystem, 
    fx_required_margin, 
    fx_pip_value_per_lot,
    Config
)

def test_margin_calculations():
    """Test margin calculation for different currency pairs"""
    print("=" * 80)
    print("MARGIN CALCULATION TESTS")
    print("=" * 80)
    
    test_cases = [
        # (symbol, price, lots, expected_margin_range)
        ('EURUSD', 1.10, 0.01, (10, 15)),
        ('EURUSD', 1.10, 0.10, (100, 150)),
        ('GBPUSD', 1.27, 0.01, (10, 20)),
        ('USDJPY', 150.0, 0.01, (9, 12)),
        ('USDJPY', 155.0, 0.01, (9, 12)),
        ('AUDUSD', 0.66, 0.01, (6, 10)),
    ]
    
    passed = 0
    failed = 0
    
    for symbol, price, lots, (min_expected, max_expected) in test_cases:
        margin = fx_required_margin(symbol, price, lots)
        
        status = "✓ PASS" if min_expected <= margin <= max_expected else "✗ FAIL"
        
        if min_expected <= margin <= max_expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | {symbol} {lots:.3f} lots @ {price:.5f} => "
              f"Margin: ${margin:.2f} (expected: ${min_expected}-${max_expected})")
    
    print(f"\nMargin Tests: {passed} passed, {failed} failed")
    return failed == 0


def test_pip_values():
    """Test pip value calculation"""
    print("\n" + "=" * 80)
    print("PIP VALUE TESTS")
    print("=" * 80)
    
    test_cases = [
        # (symbol, price, expected_pip_value_range)
        ('EURUSD', 1.10, (9, 11)),
        ('GBPUSD', 1.27, (9, 11)),
        ('USDJPY', 150.0, (6, 8)),
        ('AUDUSD', 0.66, (9, 11)),
    ]
    
    passed = 0
    failed = 0
    
    for symbol, price, (min_expected, max_expected) in test_cases:
        pip_value = fx_pip_value_per_lot(symbol, price)
        
        status = "✓ PASS" if min_expected <= pip_value <= max_expected else "✗ FAIL"
        
        if min_expected <= pip_value <= max_expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | {symbol} @ {price:.5f} => "
              f"Pip Value: ${pip_value:.2f}/lot (expected: ${min_expected}-${max_expected})")
    
    print(f"\nPip Value Tests: {passed} passed, {failed} failed")
    return failed == 0


def test_trade_execution():
    """Test actual trade execution with margin checking"""
    print("\n" + "=" * 80)
    print("TRADE EXECUTION TESTS")
    print("=" * 80)
    
    # Initialize paper trading system
    initial_balance = 25_000.0
    system = PaperTradingSystem(initial_balance)
    
    print(f"\nInitial Balance: ${system.balance:,.2f}")
    print(f"Initial Available: ${system.available_balance:,.2f}\n")
    
    # Test Case 1: Normal trade execution
    print("Test 1: Execute EURUSD BUY 0.01 lots")
    signal1 = {
        'symbol': 'EURUSD',
        'action': 'BUY',
        'quantity': 0.01,
        'entry_price': 1.10
    }
    
    result1 = system.execute_trade(signal1)
    
    print(f"  Result: {result1['status']}")
    if result1['status'] == 'EXECUTED':
        print(f"  Position ID: {result1['position_id']}")
        print(f"  Margin Used: ${result1['margin_used']:.2f}")
        print(f"  Available After: ${system.available_balance:,.2f}")
        test1_pass = (
            result1['status'] == 'EXECUTED' and 
            24_980 <= system.available_balance <= 24_995
        )
        print(f"  ✓ PASS" if test1_pass else f"  ✗ FAIL")
    else:
        print(f"  Reason: {result1['reason']}")
        print(f"  ✗ FAIL - Trade should have executed")
        test1_pass = False
    
    # Test Case 2: Close position and check PnL
    print("\nTest 2: Close EURUSD position with profit")
    if result1['status'] == 'EXECUTED':
        close_result = system.close_position(
            result1['position_id'], 
            exit_price=1.1050,  # 50 pip profit
            timestamp=None
        )
        
        print(f"  Result: {close_result['status']}")
        print(f"  PnL: ${close_result['pnl']:+.2f}")
        print(f"  Pips: {close_result['pips']:+.1f}")
        print(f"  Balance After: ${system.balance:,.2f}")
        
        test2_pass = (
            close_result['status'] == 'CLOSED' and
            close_result['pnl'] > 0 and
            40 <= close_result['pips'] <= 60
        )
        print(f"  ✓ PASS" if test2_pass else f"  ✗ FAIL")
    else:
        test2_pass = False
    
    # Test Case 3: USDJPY trade
    print("\nTest 3: Execute USDJPY SELL 0.01 lots")
    signal3 = {
        'symbol': 'USDJPY',
        'action': 'SELL',
        'quantity': 0.01,
        'entry_price': 150.0
    }
    
    result3 = system.execute_trade(signal3)
    print(f"  Result: {result3['status']}")
    if result3['status'] == 'EXECUTED':
        print(f"  Margin Used: ${result3['margin_used']:.2f}")
        test3_pass = result3['status'] == 'EXECUTED'
        print(f"  ✓ PASS")
    else:
        print(f"  Reason: {result3['reason']}")
        test3_pass = False
        print(f"  ✗ FAIL")
    
    # Test Case 4: Insufficient margin scenario
    print("\nTest 4: Attempt trade with insufficient margin")
    # Drain balance first
    system.available_balance = 5.0
    
    signal4 = {
        'symbol': 'EURUSD',
        'action': 'BUY',
        'quantity': 0.01,
        'entry_price': 1.10
    }
    
    result4 = system.execute_trade(signal4)
    print(f"  Result: {result4['status']}")
    if result4['status'] == 'REJECTED':
        print(f"  Reason: {result4['reason']}")
        test4_pass = 'margin' in result4['reason'].lower()
        print(f"  ✓ PASS - Correctly rejected")
    else:
        test4_pass = False
        print(f"  ✗ FAIL - Should have been rejected")
    
    # Summary
    all_pass = test1_pass and test2_pass and test3_pass and test4_pass
    print("\n" + "=" * 80)
    print(f"EXECUTION TESTS: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
    print("=" * 80)
    
    return all_pass


def test_position_sizing():
    """Test position size calculation with risk management"""
    print("\n" + "=" * 80)
    print("POSITION SIZING TESTS")
    print("=" * 80)
    
    from JTTWS_v5_COMPLETE_FIXED import UltimateTradingSystem
    
    system = UltimateTradingSystem()
    
    test_cases = [
        ('EURUSD', 1.10, 1.0),
        ('USDJPY', 150.0, 1.0),
        ('GBPUSD', 1.27, 0.5),
    ]
    
    passed = 0
    
    for symbol, price, signal_strength in test_cases:
        lots = system.calculate_position_size(symbol, price, signal_strength)
        
        # Position should be reasonable (0.001 to 0.1 lots)
        is_valid = 0.001 <= lots <= 0.1
        status = "✓ PASS" if is_valid else "✗ FAIL"
        
        if is_valid:
            passed += 1
        
        print(f"{status} | {symbol} @ {price:.5f} strength={signal_strength:.1f} => "
              f"{lots:.4f} lots")
    
    print(f"\nPosition Sizing: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def main():
    """Run all diagnostic tests"""
    print("\n" + "█" * 80)
    print(" " * 20 + "JTTWS v5 MARGIN DIAGNOSTIC SUITE")
    print("█" * 80 + "\n")
    
    results = {}
    
    # Run test suites
    results['margin'] = test_margin_calculations()
    results['pip_value'] = test_pip_values()
    results['execution'] = test_trade_execution()
    results['position_sizing'] = test_position_sizing()
    
    # Final summary
    print("\n" + "█" * 80)
    print("FINAL SUMMARY")
    print("█" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | {test_name.upper().replace('_', ' ')}")
    
    all_passed = all(results.values())
    
    print("\n" + ("=" * 80))
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED - REVIEW REQUIRED ✗✗✗")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
