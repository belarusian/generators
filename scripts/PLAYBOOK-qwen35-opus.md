# Running Qwen3.5-27B-Opus-Distilled on Windows (native)

Requires ~16.5GB VRAM. Fits on a 4090 (24GB) but **not alongside the vision model**.

## Prerequisites

- llama.cpp built at `C:\Users\kodep\llama.cpp\build\bin\llama-server.exe`
- `huggingface-cli` available (Python 3.11)

## 1. Stop the vision server

Kill whatever is on port 8082 (the Qwen3-VL model), or close the terminal running `run-vision.ps1`.

## 2. Download the model

```powershell
huggingface-cli download mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF --include "*Q4_K_M*" --local-dir C:\Users\kodep\models\Qwen3.5-27B-Opus-Distilled-Q4_K_M
```

## 3. Run the server

```powershell
& "C:\Users\kodep\llama.cpp\build\bin\llama-server.exe" `
    -m "C:\Users\kodep\models\Qwen3.5-27B-Opus-Distilled-Q4_K_M\Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.Q4_K_M.gguf" `
    -ngl 99 `
    --host 0.0.0.0 `
    --port 8080
```

The `-ngl 99` offloads all layers to the GPU.

## GPU budget (4090, 24GB)

| Model | VRAM | Port |
|-------|------|------|
| Qwen3-VL-30B vision | ~19GB | 8082 |
| Qwen3.5-27B-Opus | ~16.5GB | 8080 |
| Qwen3-Embedding-4B | ~2.4GB | 8083 |

Vision + Opus cannot run simultaneously. Pick one or the other.
Vision + Embed fits (~21.4GB). Opus + Embed fits (~19GB).
