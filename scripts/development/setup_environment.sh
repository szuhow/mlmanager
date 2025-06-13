#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Updating system packages..."
apt-get update -y

echo "Installing required system dependencies..."
apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
    liblzma-dev python3-openssl

echo "Installing pyenv..."
if ! command -v pyenv &> /dev/null; then
    curl https://pyenv.run | bash
else
    echo "Pyenv is already installed."
fi

# Add pyenv to PATH and initialize
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo "Installing Python 3.11.9 using pyenv..."
pyenv install -s 3.11.9

echo "Setting Python 3.11.9 as global version..."
pyenv global 3.11.9

echo "Creating a virtual environment..."
python -m venv venv

echo "Activating the virtual environment..."
source venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup completed successfully!"
