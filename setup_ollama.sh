# set up ollama with sudo permission
# curl -fsSL https://ollama.com/install.sh | sh

# set up ollama without sudo permission
mkdir -p "$HOME/.local/ollama"
mkdir -p "$HOME/.local/bin"

wget https://ollama.com/download/ollama-linux-amd64.tar.zst # or the link corresponding to your machine

zstd -d ollama-linux-amd64.tar.zst -c |   tar -xf - -C "$HOME/.local/ollama"
ln -sf "$HOME/.local/ollama/bin/ollama"        "$HOME/.local/bin/ollama"
export PATH="$HOME/.local/bin:$PATH"

