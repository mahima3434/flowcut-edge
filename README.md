# FlowCut Edge — NVIDIA On-Device AI

Run NVIDIA AI models on the **ASUS Ascent GX10** (NVIDIA GB10 Blackwell) for FlowCut's AI features: chat, vision understanding, and video generation — all on-device.

## Models

| Model | Task | Size |
|-------|------|------|
| **LLaVA-NeXT 7B** | Image understanding, scene description | ~14 GB |
| **Cosmos Predict 2.5 (2B)** | Text-to-video, image-to-video, morph transitions | ~4 GB |

## Quick Start

### 1. Deploy base service

```bash
# On the GX10:
cd ~/flowcut-edge
bash deploy.sh
```

This installs dependencies and downloads the text/vision models.

### 2. Set up Cosmos video generation (optional)

```bash
bash setup_cosmos.sh
```

This clones [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5), sets up the Python environment with CUDA 13.0, and authenticates with HuggingFace. Model checkpoints (~4 GB for 2B) auto-download on first inference.

### 3. Start the server

```bash
bash start.sh
```

Server runs at `http://<device-ip>:8000`.

## API Endpoints

### Chat (OpenAI-compatible)

```
POST /v1/chat/completions
```

Automatically routes to vision or text model based on content.

### Video Generation

```
POST /v1/video/generate/text    — Text-to-video (Cosmos)
POST /v1/video/generate/image   — Image-to-video (Cosmos)
POST /v1/video/generate/morph   — Morph transition between two frames
GET  /v1/video/download/{file}  — Download generated video
```

### Health & Models

```
GET  /health      — GPU info, model status
GET  /v1/models   — List available models
```

## Architecture

```
FlowCut Desktop App
        │
        ▼
  HTTP (OpenAI API)
        │
        ▼
┌─────────────────────────────┐
│   FlowCut Edge (FastAPI)    │
│   http://<gx10-ip>:8000     │
├─────────────────────────────┤
│ LLaVA-NeXT 7B    (vision)    │
│ Runway           (video/cloud)│
└─────────────────────────────┘
      ASUS Ascent GX10
     NVIDIA GB10 Blackwell
       119 GB RAM, CUDA 13
```

## Device Info

- **Hardware**: ASUS Ascent GX10 (NVIDIA GB10 Blackwell)
- **RAM**: 119 GB unified memory
- **Storage**: 916 GB NVMe
- **OS**: Ubuntu 24.04 (aarch64)
- **CUDA**: 13.0 / Driver 580.123

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_MODEL` | `llava-hf/llava-v1.6-mistral-7b-hf` | HF vision model ID |
| `DEVICE` | auto-detect | `cuda` or `cpu` |

## FlowCut Integration

In FlowCut settings, set:
- **AI Model**: `llava-v1.6`
- **NVIDIA Edge API URL**: `http://<device-ip>:8000/v1`

The app will automatically use the edge device for all AI operations, falling back to cloud providers if the device is unreachable.
