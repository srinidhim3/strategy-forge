#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.financial_fetcher import FinancialFetcher
from src.models.financial_calculator import FinancialCalculator

def test_accuracy():
    """Test accuracy of quarterly vs annualized calculations"""
    
    # Fetch Apple data
    print("Fetching Apple financial data...")
    fetcher = FinancialFetcher()
    statements = fetcher.fetch_all_statements('AAPL')
    
    # Initialize calculator
    calc = FinancialCalculator()
    
    # Test with quarterly data (our original approach)
    print("\n=== QUARTERLY CALCULATIONS (Original) ===")
    quarterly_metrics = calc.calculate_all_metrics(statements, annualized=False)
    # Use the most recent quarter with full data (3rd period has 4 quarters)
    quarterly_roe = quarterly_metrics.loc["ROE"].iloc[2]  # Dec 2024
    quarterly_roa = quarterly_metrics.loc["ROA"].iloc[2]
    print(f"Quarterly ROE: {quarterly_roe:.2%}")
    print(f"Quarterly ROA: {quarterly_roa:.2%}")
    
    # Test with annualized/TTM data (should be more accurate)
    print("\n=== ANNUALIZED (TTM) CALCULATIONS (Fixed) ===")
    annualized_metrics = calc.calculate_all_metrics(statements, annualized=True)
    annualized_roe = annualized_metrics.loc["ROE"].iloc[2]  # Dec 2024 TTM
    annualized_roa = annualized_metrics.loc["ROA"].iloc[2]
    print(f"Annualized ROE: {annualized_roe:.2%}")
    print(f"Annualized ROA: {annualized_roa:.2%}")
    
    print("\nExpected values from verification:")
    print("ROE: 131.57%")
    print("ROA: 28.11%")
    
    print("\n=== COMPARISON SUMMARY ===")
    expected_roe = 1.3157
    expected_roa = 0.2811
    
    roe_diff = abs(annualized_roe - expected_roe)
    roa_diff = abs(annualized_roa - expected_roa)
    
    print(f"Our ROE: {annualized_roe:.2%} vs Expected: 131.57% (Diff: {roe_diff:.2%})")
    print(f"Our ROA: {annualized_roa:.2%} vs Expected: 28.11% (Diff: {roa_diff:.2%})")
    
    # Check if we're within acceptable range (±5%)
    roe_acceptable = roe_diff < 0.05
    roa_acceptable = roa_diff < 0.05
    
    print(f"\nAccuracy Assessment:")
    print(f"ROE within ±5%: {'✅ YES' if roe_acceptable else '❌ NO'}")
    print(f"ROA within ±5%: {'✅ YES' if roa_acceptable else '❌ NO'}")
    
    return roe_acceptable and roa_acceptable

if __name__ == "__main__":
    success = test_accuracy()
    if success:
        print("\n🎯 SUCCESS: Calculations are within acceptable accuracy range!")
    else:
        print("\n⚠️  WARNING: Some calculations still need refinement.")