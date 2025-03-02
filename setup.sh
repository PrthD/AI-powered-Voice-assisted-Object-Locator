#!/bin/bash

set -e
trap 'echo "❌ An error occurred during the setup. Please review the error messages above and refer to the Troubleshooting Guide in README.md." && exit 1' ERR

echo "🚀 Setting up AIVOL Environment..."

# ------------------ Step 1: Install System Dependencies ------------------

echo "🔧 Installing system dependencies..."
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
    liblzma-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# ------------------ Step 2: Install & Configure pyenv ------------------

echo "🐍 Checking for pyenv..."
if ! command -v pyenv &> /dev/null; then
    echo "📥 Installing pyenv..."
    curl https://pyenv.run | bash

    # Add pyenv to shell startup file
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

    source ~/.bashrc
else
    echo "✅ pyenv is already installed."
fi

# ------------------ Step 3: Install Python 3.11.4 ------------------

PYTHON_VERSION="3.11.4"

echo "🐍 Checking Python version..."
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "📥 Installing Python $PYTHON_VERSION..."
    pyenv install $PYTHON_VERSION
else
    echo "✅ Python $PYTHON_VERSION is already installed."
fi

echo "🔧 Setting Python version for this project..."
pyenv local $PYTHON_VERSION

# ------------------ Step 4: Create & Activate Virtual Environment ------------------

if [ ! -d "venv" ]; then
    echo "🛠️ Creating virtual environment..."
    python -m venv venv
fi

echo "✅ Activating virtual environment..."
source venv/bin/activate

# ------------------ Step 5: Upgrade pip & Install Dependencies ------------------

echo "🚀 Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "📥 Installing Python dependencies..."
pip install --upgrade -r requirements.txt

# ------------------ Step 6: Download YOLO Weights ------------------

YOLO_WEIGHTS_PATH="models/yolo/yolov8s.pt"
YOLO_WEIGHTS_URL="https://ultralytics.com/assets/yolov8s.pt"

echo "🛠️ Ensuring YOLO model directory exists..."
mkdir -p models/yolo

if [ ! -f "$YOLO_WEIGHTS_PATH" ]; then
    echo "📥 Downloading YOLO weights..."
    wget -O "$YOLO_WEIGHTS_PATH" "$YOLO_WEIGHTS_URL"
else
    echo "✅ YOLO weights already downloaded."
fi

# ------------------ Step 7: Final Checks & Summary ------------------

echo ""
echo "🎉 Setup Complete!"
echo "------------------------------------------------"
echo "🔹 Python Version: $(python --version)"
echo "🔹 Virtual Environment: $(which python)"
echo "🔹 Installed Packages:"
pip list
echo "🔹 YOLO Weights: $(ls -lh $YOLO_WEIGHTS_PATH)"
echo "------------------------------------------------"
echo "✅ Your system is now ready to run AIVOL!"
