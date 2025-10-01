#!/usr/bin/env python3
"""
Test script for RatioCalculator integration with DataAligner.

This script demonstrates the complete pipeline:
1. Fetch price and financial data
2. Align data with point-in-time accuracy
3. Calculate financial ratios
4. Validate results
"""

import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.price_fetcher import PriceFetcher
from data.financial_fetcher import FinancialFetcher
from models.financial_calculator import FinancialCalculator
from data.data_aligner import DataAligner
from models.ratio_calculator import RatioCalculator
import pandas as pd
from datetime import datetime, timedelta


def test_ratio_calculator_integration():
    """Test the complete data pipeline with ratio calculations."""
    
    print("🚀 Testing Strategy Forge Data Pipeline with Ratio Calculator")
    print("=" * 60)
    
    # Test parameters
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)  # 2 years of data
    
    try:
        # Step 1: Fetch price data
        print(f"\n📈 Step 1: Fetching price data for {symbol}")
        price_fetcher = PriceFetcher()
        price_data = price_fetcher.fetch_price_data(
            symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        )
        print(f"✅ Price data: {len(price_data)} rows from {price_data.index.min().date()} to {price_data.index.max().date()}")
        
        # Step 2: Fetch financial data
        print(f"\n📊 Step 2: Fetching financial statements for {symbol}")
        financial_fetcher = FinancialFetcher()
        statements = financial_fetcher.fetch_all_statements(symbol, "quarterly")
        print(f"✅ Financial statements fetched:")
        for stmt_type, data in statements.items():
            print(f"   - {stmt_type}: {data.shape}")
        
        # Step 3: Calculate financial metrics
        print(f"\n🧮 Step 3: Calculating financial metrics")
        calculator = FinancialCalculator()
        financial_metrics = calculator.calculate_all_metrics(statements)
        print(f"✅ Financial metrics: {financial_metrics.shape}")
        print(f"   Available metrics: {list(financial_metrics.index)}")
        
        # Step 4: Align data with point-in-time accuracy
        print(f"\n🔗 Step 4: Aligning data with 45-day reporting lag")
        aligner = DataAligner(reporting_lag_days=45)
        aligned_data = aligner.align(price_data, financial_metrics)
        print(f"✅ Aligned data: {aligned_data.shape}")
        
        # Show alignment summary
        alignment_summary = aligner.get_alignment_summary(aligned_data)
        print(f"   Coverage: {alignment_summary.get('coverage_percentage', 0):.1f}%")
        
        # Debug: Show available columns
        print(f"   Available columns: {list(aligned_data.columns)}")
        print(f"   Sample financial data:")
        financial_cols = [col for col in aligned_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
        recent_sample = aligned_data[financial_cols].dropna().tail(1)
        if len(recent_sample) > 0:
            for col in financial_cols[:8]:  # Show first 8 financial columns
                if col in recent_sample.columns:
                    print(f"      {col}: {recent_sample[col].iloc[0]:.3f}")
        
        
        # Step 5: Calculate ratios
        print(f"\n📏 Step 5: Calculating financial ratios")
        ratio_calculator = RatioCalculator(enable_validation=True)
        ratios_data = ratio_calculator.calculate_all_ratios(aligned_data)
        print(f"✅ Ratios calculated: {ratios_data.shape}")
        
        # Step 6: Analyze results
        print(f"\n📋 Step 6: Analyzing ratio calculation results")
        ratio_columns = ['pe_ratio', 'pb_ratio', 'peg_ratio', 'ps_ratio', 'ev_ebitda']
        
        for ratio in ratio_columns:
            if ratio in ratios_data.columns:
                valid_count = ratios_data[ratio].notna().sum()
                total_count = len(ratios_data)
                coverage_pct = (valid_count / total_count) * 100
                
                if valid_count > 0:
                    ratio_values = ratios_data[ratio].dropna()
                    print(f"   📊 {ratio.upper()}: {valid_count}/{total_count} values ({coverage_pct:.1f}% coverage)")
                    print(f"      Range: {ratio_values.min():.2f} - {ratio_values.max():.2f}")
                    print(f"      Median: {ratio_values.median():.2f}, Mean: {ratio_values.mean():.2f}")
                else:
                    print(f"   ❌ {ratio.upper()}: No valid values calculated")
        
        # Step 7: Show sample of recent data
        print(f"\n📈 Step 7: Sample of recent data with ratios")
        recent_data = ratios_data.tail(10)
        display_columns = ['Close', 'eps', 'book_value_per_share', 'pe_ratio', 'pb_ratio']
        available_display = [col for col in display_columns if col in recent_data.columns]
        
        if available_display:
            print(recent_data[available_display].round(3))
        else:
            print("No ratio data available for display")
        
        # Step 8: Validate specific ratios
        print(f"\n✅ Step 8: Ratio validation and benchmarks")
        
        for ratio in ['pe_ratio', 'pb_ratio']:
            if ratio in ratios_data.columns:
                benchmarks = ratio_calculator.get_ratio_benchmarks(ratios_data, ratio)
                if 'error' not in benchmarks:
                    print(f"\n   {ratio.upper()} Benchmarks:")
                    print(f"   - Count: {benchmarks['count']} valid values")
                    print(f"   - Median: {benchmarks['median']:.2f}")
                    print(f"   - P10-P90 range: {benchmarks['percentiles']['p10']:.2f} - {benchmarks['percentiles']['p90']:.2f}")
                    
                    if 'typical_range' in benchmarks:
                        typ_range = benchmarks['typical_range']
                        print(f"   - Typical range: {typ_range['min']} - {typ_range['max']}")
        
        print(f"\n🎉 Integration test completed successfully!")
        print(f"Strategy Forge pipeline is ready for Task 14: Data Processing Pipeline")
        
        return ratios_data
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_result = test_ratio_calculator_integration()
    
    if test_result is not None:
        print(f"\n✨ Test completed successfully with {len(test_result)} rows of ratio data")
    else:
        print(f"\n💥 Test failed - check error messages above")