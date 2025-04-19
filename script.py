#!/usr/bin/env python
import os
import subprocess
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'flask', 'torch', 'torchvision', 'opencv-python', 'numpy', 
        'mediapipe', 'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.split('-')[0])  # Handle packages like 'opencv-python'
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """Install missing packages using pip"""
    print("Installing missing dependencies...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("All dependencies installed successfully!")

def main():
    # Check for Python version
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher is required.")
        sys.exit(1)
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        install_missing_packages(missing_packages)
    
    # Check if the server file exists
    server_file = "sign_language_web_server.py"
    if not os.path.exists(server_file):
        # Create the server file from the implementation
        print(f"Creating server file '{server_file}'...")
        with open(server_file, "w") as f:
            # Open the artifact file and copy its contents
            try:
                with open("sign-language-web-implementation.py", "r") as source:
                    f.write(source.read())
            except FileNotFoundError:
                print("Error: Could not find the implementation file.")
                print("Please make sure 'sign-language-web-implementation.py' exists in the current directory.")
                sys.exit(1)
    
    # Run the server
    print("Starting the Sign Language Recognition Web Server...")
    subprocess.call([sys.executable, server_file])

if __name__ == "__main__":
    main()