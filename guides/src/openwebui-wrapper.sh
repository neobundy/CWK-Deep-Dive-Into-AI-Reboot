sudo tee ~/start-openwebui.sh >/dev/null <<'EOF'
#!/bin/zsh
# load Conda functions
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate openwebui
# launch the server
exec open-webui serve --host 0.0.0.0 --port 3000
EOF

sudo chmod +x ~/start-openwebui.sh