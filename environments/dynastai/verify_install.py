#!/usr/bin/env python3
"""
DynastAI Installation Verification Script

This script checks if all required packages are installed properly.
"""

import sys
import platform
import importlib.util

def check_module(module_name):
    """Check if a module is installed and report its version if available"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False, None
        
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "Unknown")
        return True, version
    except ImportError:
        return False, None

def main():
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print("\nChecking required modules:")
    
    required_modules = [
        "fastapi", "uvicorn", "pydantic", "requests", "httpx", 
        "python_multipart", "uuid", "aiohttp", "jinja2", 
        "tqdm", "numpy", "wandb", "datasets"
    ]
    
    all_found = True
    for module in required_modules:
        found, version = check_module(module)
        status = f"✓ Found (version: {version})" if found else "✗ Not found"
        print(f"- {module}: {status}")
        if not found:
            all_found = False
    
    # Special check for built-in uuid module which is critical
    if not check_module("uuid")[0]:
        print("\nWARNING: The UUID module is missing. This is a built-in Python module and should be available.")
        all_found = False
        
    if all_found:
        print("\nAll required packages are installed! You should be able to run DynastAI successfully.")
    else:
        print("\nSome packages are missing. Run 'python setup.py' to install them.")
    
if __name__ == "__main__":
    main()
