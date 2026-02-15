# FlowCut Edge — NVIDIA On-Device AI

Run NVIDIA AI models on the **ASUS Ascent GX10** (NVIDIA GB10 Blackwell) for FlowCut's AI features: chat, vision understanding, and video generation — all on-device.

## Models

| Model | Task | Size |
|-------|------|------|
| **Nemotron-Mini 4B** | Text/reasoning, tool calling | ~8 GB |
| **Phi-3.5 Vision** | Image understanding, scene description | ~8 GB |
| **NVIDIA Cosmos 1.0 Diffusion 7B** | Text-to-video, image-to-video, morph transitions | ~14 GB |

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

This clones the [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) repo, authenticates with HuggingFace, and downloads the 7B diffusion checkpoints (~14 GB).

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
│ Nemotron-Mini 4B  (text)    │
│ Phi-3.5 Vision   (vision)   │
│ Cosmos 7B        (video)    │
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
| `TEXT_MODEL` | `nvidia/Nemotron-Mini-4B-Instruct` | HF text model ID |
| `VISION_MODEL` | `microsoft/Phi-3.5-vision-instruct` | HF vision model ID |
| `VIDEO_MODEL` | `Cosmos-1.0-Diffusion-7B-Text2World` | Cosmos checkpoint dir name |
| `COSMOS_DIR` | `~/Cosmos` | Path to cloned Cosmos repo |
| `DEVICE` | auto-detect | `cuda` or `cpu` |

## FlowCut Integration

In FlowCut settings, set:
- **AI Model**: `nvidia-edge/nemotron-mini-4b`
- **NVIDIA Edge API URL**: `http://<device-ip>:8000/v1`

The app will automatically use the edge device for all AI operations, falling back to cloud providers if the device is unreachable.
