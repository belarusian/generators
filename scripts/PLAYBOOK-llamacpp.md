# llama.cpp Server Playbook

Replacing Ollama with direct llama.cpp. One server per model, no middleman.

## Build llama.cpp

### macOS (Mac Studio / Apple Silicon)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

Metal backend uses unified memory -- the Mac Studio's full RAM is available
for model weights. No `-ngl` needed on Metal (all layers go to GPU by default).

### Linux (WSL2 with NVIDIA GPU)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Use the correct nvcc (not the old apt one)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --config Release -j$(nproc)
```

Requires: cmake, g++, CUDA toolkit 12.x (NOT the apt nvidia-cuda-toolkit
which is 11.5 and too old for RTX 40-series).

If WSL2 needs LAN access, set up port forwarding from Windows (admin PowerShell):

```powershell
$wslIp = (wsl hostname -I).Trim()
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslIp
netsh advfirewall firewall add rule name="llama.cpp" dir=in action=allow protocol=TCP localport=8080
```

## Get models (GGUF format)

Download from HuggingFace. Example:

```bash
mkdir -p ~/models

# Coding model (~20GB Q4, fits in 24GB VRAM or Mac unified memory)
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct-GGUF \
  qwen2.5-coder-32b-instruct-q4_k_m.gguf --local-dir ~/models

# OR reuse Ollama's cached blob (Windows -> WSL2 path):
# /mnt/c/Users/<user>/.ollama/models/blobs/sha256-<hash>
```

To find an Ollama blob hash:

```bash
cat /mnt/c/Users/<user>/.ollama/models/manifests/registry.ollama.ai/library/<model>/latest | python3 -m json.tool
# Look for the layer with mediaType "application/vnd.ollama.image.model"
```

## Start servers

Edit `scripts/llamacpp-boot.sh` to point MODEL paths at your actual files.

```bash
# Start all role servers
./scripts/llamacpp-boot.sh start

# Check status
./scripts/llamacpp-boot.sh status

# Set environment
source scripts/llamacpp-env.sh

# For remote Mac Studio:
LLAMA_HOST=10.106.1.184 source scripts/llamacpp-env.sh
```

## Port mapping

| Role    | Port | Purpose                        |
|---------|------|--------------------------------|
| coder   | 8080 | Worker -- coding, acting, etc. |
| oracle  | 8081 | Wisdom / deep reasoning        |
| vision  | 8082 | Image understanding            |
| embed   | 8083 | Embedding for RAG (--embedding)|

## Manual server launch (single model)

```bash
./build/bin/llama-server \
  -m ~/models/model.gguf \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

Key flags:
- `-ngl 99` : offload all layers to GPU (CUDA). Not needed for Metal.
- `--host 0.0.0.0` : listen on all interfaces (LAN access)
- `--embedding` : enable embedding endpoint (for embed role)
- `-c 8192` : context size (default varies by model)
- `-np 4` : number of parallel slots (concurrent requests)

## Verify

```bash
# Health check
curl http://localhost:8080/health

# What model is loaded?
curl http://localhost:8080/v1/models

# Quick completion test
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'
```

## Mac Studio notes

- Apple Silicon uses Metal backend (GGML_METAL=ON)
- Unified memory means the full 192GB (or whatever) is available
- Can run multiple large models simultaneously on different ports
- No driver/toolkit version headaches like CUDA
- Models that Ollama doesn't support can run here directly from GGUF
