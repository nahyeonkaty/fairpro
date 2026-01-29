# GenEval Evaluation Setup

This guide explains how to set up and run GenEval evaluation for generated images. GenEval is a benchmark for evaluating text-to-image generation models on compositional tasks.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Building the Docker Image](#building-the-docker-image)
- [Generating Images](#generating-images)
- [Running Evaluation](#running-evaluation)
- [Understanding Results](#understanding-results)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Docker** with NVIDIA Container Toolkit (nvidia-docker)
- **NVIDIA GPU** with CUDA support
- **Python 3.8+** (for image generation)
- Sufficient disk space for model weights (~10GB)

### Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu20.04 nvidia-smi
```

---

## Building the Docker Image

The GenEval Docker image contains all necessary dependencies including:
- PyTorch 2.1.2 + CUDA 12.1
- MMDetection 2.x with Mask2Former
- GenEval evaluation scripts

### Build the Image

```bash
cd geneval
docker build -t geneval:latest .
```

This process takes approximately 15-30 minutes as it:
1. Installs Python 3.8 and system dependencies
2. Installs PyTorch with CUDA support
3. Clones and installs MMDetection v2.26.0
4. Downloads Mask2Former model weights
5. Clones the GenEval repository

### Verify the Build

```bash
docker images | grep geneval
```

---

## Generating Images

### Using the Generation Script

Generate images for GenEval prompts using `generate_geneval.py`:

```bash
# Generate with QwenImage
python generate_geneval.py --model qwenimage --output-dir outputs/geneval

# Generate with QwenImage + FairPro
python generate_geneval.py --model qwenimage --fairpro --output-dir outputs/geneval

# Generate with SANA
python generate_geneval.py --model sana --output-dir outputs/geneval

# Generate with SANA + FairPro
python generate_geneval.py --model sana --fairpro --output-dir outputs/geneval

# Run all 4 configurations
python generate_geneval.py --all --output-dir outputs/geneval
```

### Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | - | Model to use: `qwenimage` or `sana` |
| `--fairpro` | False | Enable FairPro for fairness-aware generation |
| `--all` | False | Run all 4 model configurations |
| `--seeds` | 0 1 2 3 | Seeds for generation (4 images per prompt) |
| `--output-dir` | outputs/geneval | Output directory |
| `--height` | 1024 | Image height |
| `--width` | 1024 | Image width |
| `--steps` | 20 | Number of inference steps |
| `--guidance-scale` | 5.0 | Guidance scale |
| `--start-idx` | 0 | Start prompt index |
| `--end-idx` | None | End prompt index |

### Output Directory Structure

Generated images follow this structure:

```
outputs/geneval/
├── qwenimage/
│   ├── 00000/
│   │   ├── samples/
│   │   │   ├── 00000.png
│   │   │   ├── 00001.png
│   │   │   ├── 00002.png
│   │   │   └── 00003.png
│   │   ├── metadata.jsonl
│   │   └── grid.png
│   ├── 00001/
│   └── ...
├── qwenimage_fairpro/
├── sana/
└── sana_fairpro/
```

Each prompt folder contains:
- `samples/`: Individual generated images (one per seed)
- `metadata.jsonl`: Prompt metadata from GenEval
- `grid.png`: Grid visualization of all seeds

---

## Running Evaluation

### Quick Start

Use the provided wrapper script:

```bash
# Basic usage
./geneval.sh <image_dir> <output_file> [gpu_ids]

# Examples
./geneval.sh outputs/geneval/qwenimage outputs/geneval/qwenimage/results.jsonl
./geneval.sh outputs/geneval/qwenimage_fairpro outputs/geneval/qwenimage_fairpro/results.jsonl
```

### Manual Docker Execution

```bash
# Run evaluation with Docker directly
docker run --gpus all \
  --user "$(id -u):$(id -g)" \
  -v /path/to/images:/images \
  -v /path/to/output:/output \
  geneval:latest /images /output/results.jsonl
```

### GPU Selection

```bash
# Use specific GPU(s)
./geneval.sh outputs/geneval/qwenimage results.jsonl 0      # GPU 0 only
./geneval.sh outputs/geneval/qwenimage results.jsonl 0,1    # GPUs 0 and 1
./geneval.sh outputs/geneval/qwenimage results.jsonl all    # All GPUs (default)
```

### Image Directory Requirements

The evaluation script expects images organized as:
```
<image_dir>/
├── 00000/
│   └── samples/
│       ├── 00000.png
│       ├── 00001.png
│       └── ...
├── 00001/
└── ...
```

Each subfolder index corresponds to a prompt from `evaluation_metadata.jsonl`.

---

## Understanding Results

### Output Files

After evaluation, you'll have a `results.jsonl` file containing per-image evaluation results.

### Summary Scores

Run the summary script to get aggregated scores:

```bash
python geneval/summary_scores.py outputs/geneval/qwenimage/results.jsonl
```

### Sample Output

```
Summary
=======
Total images: 2200
Total prompts: 553
% correct images: 45.32%
% correct prompts: 62.18%

Task breakdown
==============
single_object    = 85.42% (123 / 144)
two_object       = 52.31% (136 / 260)
counting         = 28.75% (46 / 160)
colors           = 67.89% (180 / 265)
position         = 35.21% (89 / 253)
color_attribution= 42.15% (98 / 232)

Overall score (avg. over tasks): 0.51955
```

### Task Categories

GenEval evaluates compositional generation across these tasks:

| Task | Description |
|------|-------------|
| `single_object` | Generate a single specified object |
| `two_object` | Generate two different objects |
| `counting` | Generate a specific number of objects |
| `colors` | Generate objects with specific colors |
| `position` | Place objects in spatial relationships |
| `color_attribution` | Assign correct colors to multiple objects |

---

## Troubleshooting

### Common Issues

#### Permission Denied
```bash
# Run with current user to avoid permission issues
docker run --user "$(id -u):$(id -g)" ...
```

#### GPU Not Found
```bash
# Ensure nvidia-docker is installed
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

#### Out of Memory
- Reduce batch size or use fewer GPUs
- Ensure sufficient GPU VRAM (16GB+ recommended)

#### Model Download Issues
The Mask2Former weights are downloaded during Docker build. If the build fails:
```bash
# Manually download
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
```

### Rebuilding the Docker Image

```bash
# Force rebuild without cache
docker build --no-cache -t geneval:latest .
```

### Checking Container Logs

```bash
# Run interactively for debugging
docker run -it --gpus all geneval:latest /bin/bash
```

---

## References

- [GenEval Repository](https://github.com/djghosh13/geneval)
- [GenEval Paper](https://arxiv.org/abs/2310.11513)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
