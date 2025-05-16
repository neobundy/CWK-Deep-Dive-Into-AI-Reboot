#!/bin/zsh
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate openwebui

# Upgrade openwebui
pip install -U open-webui

# Start openwebui
exec open-webui serve --host 0.0.0.0 --port 3000
