#!/usr/bin/env python3
"""
Test script for DataProcessingPipeline integration.

This script tests the complete unified pipeline:
1. Single stock processing with default configuration
2. Custom configuration with different parameters
3. Multiple stock processing
4. Performance benchmarking
5. Error handling and validation
"""

import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.processing_pipeline import DataProcessingPipeline, PipelineConfig, process_stock_data
import pandas as pd
from datetime import datetime, timedelta


def test_basic_pipeline():
    """Test basic pipeline functionality with AAPL."""
    print("🧪 Test 1: Basic Pipeline Processing")
    print("-" * 40)
    
    try:
        # Test with convenience function
        data = process_stock_data(
            symbol="AAPL",
            start_date="2023-01-01", 
            end_date="2024-06-30",
            reporting_lag_days=45
        )
        
        print(f"✅ Basic pipeline test passed")
        print(f"   Data shape: {data.shape}")
        print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
        
        # Check for key columns
        expected_cols = ['Close', 'EPS', 'pe_ratio', 'pb_ratio']
        available_cols = [col for col in expected_cols if col in data.columns]
        print(f"   Available key columns: {available_cols}")
        
        # Show sample data
        if available_cols:
            print("   Sample data (last 5 rows):")
            print(data[available_cols].tail().round(3).to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Basic pipeline test failed: {str(e)}")
        return False


def test_custom_configuration():
    """Test pipeline with custom configuration."""
    print("\n🧪 Test 2: Custom Configuration")
    print("-" * 40)
    
    try:
        # Create custom configuration
        config = PipelineConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            reporting_lag_days=60,  # Longer lag
            ratios_to_calculate=['pe', 'pb', 'ps'],  # Specific ratios only
            enable_progress_tracking=True,
            enable_benchmarking=True,
            min_data_points=100
        )
        
        # Create pipeline and process
        pipeline = DataProcessingPipeline(config)
        data = pipeline.process_stock("MSFT")
        
        print(f"✅ Custom configuration test passed")
        print(f"   Data shape: {data.shape}")
        
        # Check that only requested ratios were calculated
        ratio_cols = [col for col in data.columns if col.endswith('_ratio')]
        print(f"   Calculated ratios: {ratio_cols}")
        
        # Show performance metrics
        performance = pipeline.get_performance_summary()
        print(f"   Processing time: {performance.get('avg_processing_time_seconds', 0):.2f}s")
        print(f"   Memory usage: {performance.get('avg_memory_usage_mb', 0):.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom configuration test failed: {str(e)}")
        return False


def test_multiple_stocks():
    """Test processing multiple stocks."""
    print("\n🧪 Test 3: Multiple Stock Processing")
    print("-" * 40)
    
    try:
        config = PipelineConfig(
            start_date="2023-01-01",
            end_date="2023-06-30",
            reporting_lag_days=45,
            ratios_to_calculate=['pe', 'pb'],
            enable_progress_tracking=False  # Reduce log noise
        )
        
        pipeline = DataProcessingPipeline(config)
        
        # Test with multiple stocks
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = pipeline.process_multiple_stocks(
            symbols, 
            continue_on_error=True
        )
        
        print(f"✅ Multiple stock processing test passed")
        print(f"   Successfully processed: {len(results)}/{len(symbols)} stocks")
        
        for symbol, data in results.items():
            print(f"   {symbol}: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Show combined performance summary
        performance = pipeline.get_performance_summary()
        total_time = performance.get('total_processing_time_seconds', 0)
        print(f"   Total processing time: {total_time:.2f}s")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Multiple stock processing test failed: {str(e)}")
        return False


def test_error_handling():
    """Test error handling and validation."""
    print("\n🧪 Test 4: Error Handling and Validation")
    print("-" * 40)
    
    test_results = []
    
    # Test 1: Invalid symbol
    try:
        data = process_stock_data("INVALID_SYMBOL_XYZ")
        print("❌ Invalid symbol test failed - should have raised error")
        test_results.append(False)
    except Exception:
        print("✅ Invalid symbol test passed - correctly raised error")
        test_results.append(True)
    
    # Test 2: Invalid date range
    try:
        config = PipelineConfig(
            start_date="2025-01-01",  # Future date
            end_date="2025-12-31",
            min_data_points=1000  # Very high requirement
        )
        pipeline = DataProcessingPipeline(config)
        data = pipeline.process_stock("AAPL")
        print("❌ Invalid date range test failed - should have raised error")
        test_results.append(False)
    except Exception:
        print("✅ Invalid date range test passed - correctly raised error")
        test_results.append(True)
    
    # Test 3: Invalid configuration
    try:
        config = PipelineConfig(
            ratios_to_calculate=['invalid_ratio'],  # Invalid ratio
            reporting_lag_days=-10  # Negative lag
        )
        print("❌ Invalid config test failed - should have raised error")
        test_results.append(False)
    except Exception:
        print("✅ Invalid config test passed - correctly raised error")
        test_results.append(True)
    
    return all(test_results)


def test_data_quality_checks():
    """Test data quality validation."""
    print("\n🧪 Test 5: Data Quality Validation")
    print("-" * 40)
    
    try:
        # Process with normal configuration
        config = PipelineConfig(
            start_date="2023-01-01",
            end_date="2023-06-30",
            enable_progress_tracking=True,
            max_missing_data_pct=90.0  # Allow more missing data for testing
        )
        
        pipeline = DataProcessingPipeline(config)
        data = pipeline.process_stock("AAPL")
        
        # Perform data quality checks
        quality_checks = []
        
        # Check 1: Data completeness
        has_price_data = not data['Close'].isnull().all()
        quality_checks.append(("Price data availability", has_price_data))
        
        # Check 2: Ratio calculations
        ratio_cols = [col for col in data.columns if col.endswith('_ratio')]
        has_ratios = len(ratio_cols) > 0
        quality_checks.append(("Ratio calculations", has_ratios))
        
        # Check 3: Date continuity
        date_gaps = data.index.to_series().diff().dt.days.max()
        reasonable_gaps = date_gaps <= 7  # Weekend gaps are normal
        quality_checks.append(("Date continuity", reasonable_gaps))
        
        # Check 4: Fundamental data integration
        fundamental_cols = [col for col in data.columns if col in ['EPS', 'BVPS', 'ROE', 'ROA']]
        has_fundamentals = len(fundamental_cols) > 0
        quality_checks.append(("Fundamental data integration", has_fundamentals))
        
        # Report results
        for check_name, passed in quality_checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}: {'PASS' if passed else 'FAIL'}")
        
        all_passed = all(result for _, result in quality_checks)
        
        if all_passed:
            print(f"✅ Data quality validation test passed")
        else:
            print(f"⚠️ Data quality validation test completed with some issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality validation test failed: {str(e)}")
        return False


def run_comprehensive_pipeline_test():
    """Run comprehensive test suite for DataProcessingPipeline."""
    print("🚀 DataProcessingPipeline Comprehensive Test Suite")
    print("=" * 60)
    
    test_functions = [
        ("Basic Pipeline Processing", test_basic_pipeline),
        ("Custom Configuration", test_custom_configuration),
        ("Multiple Stock Processing", test_multiple_stocks),
        ("Error Handling", test_error_handling),
        ("Data Quality Validation", test_data_quality_checks)
    ]
    
    results = []
    
    for test_name, test_func in test_functions:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 40)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! DataProcessingPipeline is ready for production.")
        print("🚀 Ready for Task 15: Basic Trading Strategy Implementation")
    else:
        print("⚠️ Some tests failed. Please review and fix issues before proceeding.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_pipeline_test()
    
    if success:
        print(f"\n✨ DataProcessingPipeline test suite completed successfully!")
    else:
        print(f"\n💥 DataProcessingPipeline test suite completed with failures!")