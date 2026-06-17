set -eo pipefail  # Exit on error

echo "========================================="
echo "Starting custom provisioning..."
echo "========================================="

# Update and install git
echo "Installing git..."
sudo apt-get update && sudo apt-get install -y git

. /venv/main/bin/activate

echo "========================================="
echo "Installing Python 3.12"
echo "========================================="

conda install python=3.12 -y

echo "========================================="
echo "Installing packages"
echo "========================================="

pip install "mujoco>=3.2" "mujoco-warp>=3.7.0" "warp-lang==1.12.0" "mjlab==1.3.0" "scipy" "onnxruntime"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall


echo "Torch available: "
python -c "import torch; print(torch.cuda.is_available())"

pip install "viser==1.0.26" "mjviser==0.0.12"
pip install "gradio>=5.0" "gradio-rangeslider==0.0.8" "wandb>=0.21.0"
pip install "ml-collections>=0.1.1" "hydra-core" "omegaconf" "etils[epath]>=1.3.0"
pip install ipykernel nbformat

echo "Package installation complete"

# Clone your repository
echo "========================================="
echo "Cloning repository..."
echo "========================================="

cd /workspace
echo "Current directory after cd command: $(pwd)"

# Replace with your actual repo URL
if [ ! -d "myointeract_ui_course" ]; then
    echo "Before Starting cloning"
    git clone git clone https://github.com/fl0fischer/myointeract_ui_course.git
    echo "Cloning complete"
    cd myointeract_ui_course
    echo "Changed into folder myosuite"
else
    echo "Repository already exists, pulling latest..."
    cd myointeract_ui_course
    git pull
fi
# Install Python package in editable mode
echo "Installing Python package..."
pip install -e .

echo "========================================="
echo "Custom provisioning completed successfully!"
echo "========================================="

python -c "import myosuite"

