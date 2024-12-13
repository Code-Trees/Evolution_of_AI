#!/bin/bash

# Shell script to install Python dependencies from requirements.txt
# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo "Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command_exists pip3; then
    echo "pip is not installed. Installing pip..."
    python3 -m ensurepip --upgrade || { echo "Failed to install pip. Exiting."; exit 1; }
fi

# Upgrade pip
echo "Upgrading pip..."
pip3 install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip3 install -r requirements.txt || { echo "Failed to install some dependencies."; exit 1; }
    echo "Dependencies installed successfully."
else
    echo "requirements.txt not found. Exiting."
    exit 1
fi
