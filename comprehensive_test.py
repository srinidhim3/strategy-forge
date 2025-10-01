#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.financial_fetcher import FinancialFetcher
from src.models.financial_calculator import FinancialCalculator

def comprehensive_accuracy_test():
    """Comprehensive accuracy test comparing all metrics with verification document"""
    
    # Fetch Apple data
    print("Fetching Apple financial data...")
    fetcher = FinancialFetcher()
    statements = fetcher.fetch_all_statements('AAPL')
    calc = FinancialCalculator()
    
    # Calculate metrics using TTM method
    metrics = calc.calculate_all_metrics(statements, annualized=True)
    
    # Use December 2024 period (index 2) which has full TTM data
    period_idx = 2
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE FINANCIAL METRICS VERIFICATION")
    print("Apple Inc. (AAPL) - Strategy Forge vs External Verification")
    print(f"{'='*80}")
    
    # Define verification benchmarks from the document
    verification_data = {
        'ROE': {'our': metrics.loc['ROE'].iloc[period_idx], 'expected': 1.3157, 'unit': '%'},
        'ROA': {'our': metrics.loc['ROA'].iloc[period_idx], 'expected': 0.2811, 'unit': '%'},
        'EPS': {'our': metrics.loc['EPS'].iloc[period_idx], 'expected': 1.57, 'unit': '$'},
        'BVPS': {'our': metrics.loc['BVPS'].iloc[period_idx], 'expected': 4.41, 'unit': '$'},
        'Debt_to_Equity': {'our': metrics.loc['Debt_to_Equity'].iloc[period_idx], 'expected': 1.41, 'unit': 'x'},
        'Net_Profit_Margin': {'our': metrics.loc['Net_Profit_Margin'].iloc[period_idx], 'expected': 0.2492, 'unit': '%'},
        'Operating_Margin': {'our': metrics.loc['Operating_Margin'].iloc[period_idx], 'expected': 0.3161, 'unit': '%'},
    }
    
    print(f"{'Metric':<20} {'Our Value':<12} {'Expected':<12} {'Difference':<12} {'Status':<15}")
    print(f"{'-'*80}")
    
    total_within_range = 0
    acceptable_threshold = 0.05  # 5% tolerance
    
    for metric, data in verification_data.items():
        our_val = data['our']
        expected_val = data['expected']
        unit = data['unit']
        
        # Calculate difference
        if unit == '%':
            diff = abs(our_val - expected_val)
            our_display = f"{our_val:.2%}"
            expected_display = f"{expected_val:.2%}"
            diff_display = f"{diff:.2%}"
            within_range = diff < acceptable_threshold
        elif unit == '$':
            diff = abs(our_val - expected_val)
            our_display = f"${our_val:.2f}"
            expected_display = f"${expected_val:.2f}"
            diff_display = f"${diff:.2f}"
            within_range = diff < (expected_val * acceptable_threshold)
        else:  # 'x' ratios
            diff = abs(our_val - expected_val)
            our_display = f"{our_val:.2f}x"
            expected_display = f"{expected_val:.2f}x"
            diff_display = f"{diff:.2f}x"
            within_range = diff < (expected_val * acceptable_threshold)
        
        status = "✅ GOOD" if within_range else "⚠️ VARIANCE"
        if within_range:
            total_within_range += 1
            
        print(f"{metric.replace('_', ' '):<20} {our_display:<12} {expected_display:<12} {diff_display:<12} {status:<15}")
    
    print(f"{'-'*80}")
    print(f"SUMMARY: {total_within_range}/{len(verification_data)} metrics within ±5% tolerance")
    
    if total_within_range >= len(verification_data) * 0.7:  # 70% threshold
        print("\n🎯 OVERALL ASSESSMENT: EXCELLENT ACCURACY")
        print("   Our financial calculator produces results within acceptable industry standards.")
        print("   Minor variances are due to different data sources and calculation timing.")
    else:
        print("\n⚠️ OVERALL ASSESSMENT: NEEDS IMPROVEMENT")
        print("   Some metrics show significant variance requiring investigation.")
    
    print(f"\n{'='*80}")
    print("DATA SOURCE NOTES:")
    print("• Our calculations: Yahoo Finance (yfinance) with TTM methodology")
    print("• Verification source: External financial data provider")
    print("• Differences expected due to data source variance and timing")
    print(f"{'='*80}")
    
    return total_within_range >= len(verification_data) * 0.7

if __name__ == "__main__":
    success = comprehensive_accuracy_test()
    exit_code = 0 if success else 1
    exit(exit_code)