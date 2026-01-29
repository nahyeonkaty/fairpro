# GenEval Dockerfile
# Based on proven working installation from: https://github.com/djghosh13/geneval/issues/12
# Using Python 3.8 + PyTorch 2.1.2 + CUDA 12.1 + mmdet 2.x

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set cache directories to /tmp (writable by any user)
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV XDG_CACHE_HOME=/tmp/cache
ENV TORCH_HOME=/tmp/torch
ENV HOME=/tmp

# Install system dependencies and Python 3.8
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    python -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.1.2 with CUDA 12.1 (proven working version)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install --no-cache-dir \
    networkx==2.8.8 \
    open-clip-torch==2.26.1 \
    clip-benchmark \
    openmim \
    einops \
    lightning \
    diffusers \
    transformers \
    tomli \
    platformdirs \
    numpy \
    pandas \
    scipy \
    tqdm \
    pyyaml \
    addict \
    yapf \
    pillow \
    opencv-python-headless

# Install mmengine and mmcv-full (proven working versions)
RUN mim install mmengine mmcv-full==1.7.2

# Clone and install mmdetection 2.x (required for GenEval)
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection && \
    cd /mmdetection && \
    git checkout v2.26.0 && \
    pip install -v -e .

# Clone the GenEval repository
RUN git clone https://github.com/djghosh13/geneval.git /geneval

# Create directories for models and outputs with world-writable permissions
RUN mkdir -p /geneval/models /geneval/outputs && \
    mkdir -p /tmp/matplotlib /tmp/huggingface /tmp/cache /tmp/torch /tmp/.cache && \
    chmod -R 777 /tmp

# Download Mask2Former model weights
RUN wget -q https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth \
    -O /geneval/models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth

# Set environment variables
ENV GENEVAL_MODEL_PATH=/geneval/models
ENV PYTHONPATH=/mmdetection

# Set working directory
WORKDIR /geneval

# Copy evaluation script
COPY run_eval.sh /geneval/run_eval.sh
RUN chmod +x /geneval/run_eval.sh

# Default: run evaluation script
ENTRYPOINT ["/geneval/run_eval.sh"]
CMD ["/data/images", "/data/results.jsonl"]
