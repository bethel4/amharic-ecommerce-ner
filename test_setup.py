#!/usr/bin/env python3
"""
Test script to verify the setup works correctly.
"""
import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import telethon
        print("✓ Telethon imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Telethon: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pandas: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import python-dotenv: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✓ tqdm imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import tqdm: {e}")
        return False
    
    return True

def test_env_file():
    """Test if .env file exists and has required variables."""
    print("\nTesting .env file...")
    
    env_file = Path('.env')
    if not env_file.exists():
        print("✗ .env file not found")
        return False
    
    print("✓ .env file exists")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['TELEGRAM_API_ID', 'TELEGRAM_API_HASH', 'TELEGRAM_PHONE']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value or value == f'your_{var.lower()}_here':
            missing_vars.append(var)
        else:
            print(f"✓ {var} is set")
    
    if missing_vars:
        print(f"✗ Missing or not configured: {', '.join(missing_vars)}")
        return False
    
    return True

def test_directory_structure():
    """Test if required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed', 
        'data/external',
        'notebooks',
        'scripts',
        'src',
        'models',
        'outputs',
        'tests'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path} exists")
        else:
            missing_dirs.append(dir_path)
            print(f"✗ {dir_path} missing")
    
    if missing_dirs:
        print(f"Missing directories: {', '.join(missing_dirs)}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🔍 Testing Amharic E-commerce NER Setup\n")
    
    tests = [
        test_imports,
        test_env_file,
        test_directory_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Update .env file with your Telegram API credentials")
        print("2. Run: python scripts/scraper.py")
        print("3. Run: python scripts/preprocess.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 