#!/bin/bash

echo "Setting up your Heart-Disease-Finding project..."

# Check if Python and pip are installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install Python3 and try again."
    exit
fi

if ! command -v pip3 &> /dev/null
then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit
fi

# Install virtualenv if not installed
if ! command -v virtualenv &> /dev/null
then
    echo "virtualenv is not installed. Installing it now..."
    pip3 install virtualenv
fi

# Create a virtual environment
echo "Creating a virtual environment..."
virtualenv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! To start working, activate the virtual environment using:"
echo "source venv/bin/activate"
