#!/usr/bin/env python3
"""
Test script to verify enhanced models are working correctly.
"""

def test_enhanced_imports():
    """Test if enhanced models can be imported successfully."""
    print("Testing Enhanced Models Import...")
    
    try:
        from indi_ml.enhanced_pipeline import run_enhanced_pipeline, compare_pipelines
        print("[OK] Enhanced pipeline imported successfully!")
        
        from indi_ml.models.enhanced_ensemble import train_enhanced_ensemble
        print("[OK] Enhanced ensemble imported successfully!")
        
        from indi_ml.models.enhanced_arima import train_enhanced_arima
        print("[OK] Enhanced ARIMA imported successfully!")
        
        from indi_ml.models.enhanced_lstm import train_enhanced_lstm
        print("[OK] Enhanced LSTM imported successfully!")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    print("\nTesting Dependencies...")
    
    dependencies = [
        'xgboost',
        'lightgbm', 
        'sklearn',
        'tensorflow',
        'statsmodels',
        'numpy',
        'pandas'
    ]
    
    missing = []
    
    for dep in dependencies:
        try:
            if dep == 'sklearn':
                import sklearn
            else:
                __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} missing")
            missing.append(dep)
    
    return len(missing) == 0

def test_enhanced_pipeline_basic():
    """Test basic functionality of enhanced pipeline."""
    print("\n🚀 Testing Enhanced Pipeline Basic Functionality...")
    
    try:
        from indi_ml.enhanced_pipeline import run_enhanced_pipeline
        
        # Test with a simple symbol and short period
        print("Running enhanced pipeline test...")
        results = run_enhanced_pipeline("RELIANCE", period="1mo")
        
        if results and 'best_model' in results:
            print(f"✅ Enhanced pipeline test successful!")
            print(f"   Best model: {results['best_model']['model']}")
            print(f"   Execution time: {results.get('execution_time', 0):.2f}s")
            return True
        else:
            print("❌ Enhanced pipeline returned unexpected results")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Enhanced Models Test Suite")
    print("=" * 50)
    
    # Test 1: Imports
    imports_ok = test_enhanced_imports()
    
    # Test 2: Dependencies
    deps_ok = test_dependencies()
    
    # Test 3: Basic functionality (only if imports work)
    if imports_ok and deps_ok:
        pipeline_ok = test_enhanced_pipeline_basic()
    else:
        pipeline_ok = False
        print("\n⏭️ Skipping pipeline test due to import/dependency issues")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"   Pipeline: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
    
    if imports_ok and deps_ok and pipeline_ok:
        print("\n🎉 All tests passed! Enhanced models are ready to use.")
        print("\n📱 You can now use enhanced models in the Streamlit app:")
        print("   1. Open the app: streamlit run app.py")
        print("   2. Select 'Enhanced Models' in the sidebar")
        print("   3. Run analysis to see improved results!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        
        if not imports_ok:
            print("\n🔧 To fix import issues:")
            print("   pip install xgboost lightgbm")
        
        if not deps_ok:
            print("\n🔧 To fix dependency issues:")
            print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
