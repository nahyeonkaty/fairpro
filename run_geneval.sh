#!/bin/bash

# GenEval Generation and Evaluation Script
# This script generates images using specified model and evaluates them with GenEval.
#
# Usage: ./run_geneval.sh --model <model> [options]
#
# Examples:
#   ./run_geneval.sh --model qwenimage                    # Generate with QwenImage
#   ./run_geneval.sh --model sana --fairpro               # Generate with SANA + FairPro
#   ./run_geneval.sh --model qwenimage --eval-only        # Evaluate existing images
#   ./run_geneval.sh --model sana --generate-only         # Generate only, skip evaluation

set -e

# Default values
MODEL=""
FAIRPRO=""
OUTPUT_DIR="outputs/geneval"
GPU_IDS="all"
SEEDS="0 1 2 3"
HEIGHT=1024
WIDTH=1024
STEPS=20
GUIDANCE_SCALE=5.0
GENERATE_ONLY=false
EVAL_ONLY=false
START_IDX=""
END_IDX=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --fairpro)
      FAIRPRO="--fairpro"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gpu)
      GPU_IDS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --guidance-scale)
      GUIDANCE_SCALE="$2"
      shift 2
      ;;
    --generate-only)
      GENERATE_ONLY=true
      shift
      ;;
    --eval-only)
      EVAL_ONLY=true
      shift
      ;;
    --start-idx)
      START_IDX="--start-idx $2"
      shift 2
      ;;
    --end-idx)
      END_IDX="--end-idx $2"
      shift 2
      ;;
    -h|--help)
      echo "GenEval Generation and Evaluation Script"
      echo ""
      echo "Usage: ./run_geneval.sh --model <model> [options]"
      echo ""
      echo "Required:"
      echo "  --model MODEL        Model to use: qwenimage or sana"
      echo ""
      echo "Options:"
      echo "  --fairpro            Enable FairPro for fairness-aware generation"
      echo "  --output-dir DIR     Output directory (default: outputs/geneval)"
      echo "  --gpu GPU_IDS        GPU IDs to use (default: all). E.g., '0' or '0,1'"
      echo "  --seeds SEEDS        Seeds for generation (default: '0 1 2 3')"
      echo "  --height HEIGHT      Image height (default: 1024)"
      echo "  --width WIDTH        Image width (default: 1024)"
      echo "  --steps STEPS        Inference steps (default: 20)"
      echo "  --guidance-scale S   Guidance scale (default: 5.0)"
      echo "  --generate-only      Only generate images, skip evaluation"
      echo "  --eval-only          Only evaluate existing images, skip generation"
      echo "  --start-idx IDX      Start index for prompts"
      echo "  --end-idx IDX        End index for prompts"
      echo "  -h, --help           Show this help message"
      echo ""
      echo "Examples:"
      echo "  ./run_geneval.sh --model qwenimage"
      echo "  ./run_geneval.sh --model sana --fairpro --gpu 0,1"
      echo "  ./run_geneval.sh --model qwenimage --eval-only"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required"
  echo "Use --help for usage information"
  exit 1
fi

if [[ "$MODEL" != "qwenimage" && "$MODEL" != "sana" ]]; then
  echo "Error: --model must be 'qwenimage' or 'sana'"
  exit 1
fi

# Determine output subdirectory name
if [[ -n "$FAIRPRO" ]]; then
  SUBDIR="${MODEL}_fairpro"
else
  SUBDIR="${MODEL}"
fi

IMAGE_DIR="${OUTPUT_DIR}/${SUBDIR}"
RESULTS_FILE="${IMAGE_DIR}/results.jsonl"

echo "============================================"
echo "GenEval Generation and Evaluation Pipeline"
echo "============================================"
echo "Model:        $MODEL"
echo "FairPro:      ${FAIRPRO:-disabled}"
echo "Output Dir:   $IMAGE_DIR"
echo "GPU(s):       $GPU_IDS"
echo "============================================"

# =============================================
# Step 1: Generate Images
# =============================================
if [[ "$EVAL_ONLY" == false ]]; then
  echo ""
  echo "[Step 1/2] Generating images..."
  echo "--------------------------------------------"

  # Set CUDA_VISIBLE_DEVICES if specific GPUs requested
  if [[ "$GPU_IDS" != "all" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "Using GPU(s): $GPU_IDS"
  fi

  python geneval/generate_geneval.py \
    --model "$MODEL" \
    $FAIRPRO \
    --output-dir "$OUTPUT_DIR" \
    --seeds $SEEDS \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --steps "$STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    $START_IDX \
    $END_IDX

  echo "Generation complete: $IMAGE_DIR"
else
  echo ""
  echo "[Step 1/2] Skipping generation (--eval-only)"
fi

# =============================================
# Step 2: Evaluate with GenEval
# =============================================
if [[ "$GENERATE_ONLY" == false ]]; then
  echo ""
  echo "[Step 2/2] Evaluating with GenEval..."
  echo "--------------------------------------------"

  # Convert to absolute paths (required by Docker)
  IMAGE_DIR_ABS="$(cd "$(dirname "$IMAGE_DIR")" && pwd)/$(basename "$IMAGE_DIR")"
  RESULTS_DIR_ABS="$(dirname "$RESULTS_FILE")"
  RESULTS_DIR_ABS="$(cd "$(dirname "$RESULTS_DIR_ABS")" && pwd)/$(basename "$RESULTS_DIR_ABS")"
  RESULTS_FILENAME="$(basename "$RESULTS_FILE")"

  # Create output directory if it doesn't exist
  mkdir -p "$RESULTS_DIR_ABS"

  # Determine GPU flag for docker
  if [[ "$GPU_IDS" == "all" ]]; then
    GPU_FLAG="all"
  else
    GPU_FLAG="\"device=${GPU_IDS}\""
  fi

  # Run GenEval evaluation in Docker
  docker run --gpus "$GPU_FLAG" \
    --user "$(id -u):$(id -g)" \
    -v "${IMAGE_DIR_ABS}":/images \
    -v "${RESULTS_DIR_ABS}":/output \
    geneval:latest /images "/output/${RESULTS_FILENAME}"

  echo "Evaluation complete: $RESULTS_FILE"

  # Summary scores if available
  if [[ -f "$RESULTS_FILE" ]]; then
    echo ""
    echo "--------------------------------------------"
    echo "Results Summary:"
    python geneval/summary_scores.py "$RESULTS_FILE" 2>/dev/null || echo "(Run summary_scores.py manually for detailed results)"
  fi
else
  echo ""
  echo "[Step 2/2] Skipping evaluation (--generate-only)"
fi

echo ""
echo "============================================"
echo "Pipeline Complete!"
echo "============================================"

# GPU options for Docker:
#   --gpus all                    # Use all GPUs
#   --gpus '"device=0"'           # Use GPU 0
#   --gpus '"device=0,1"'         # Use GPU 0 and 1
#
# User options:
#   --user "$(id -u):$(id -g)"    # Run as current user (avoids permission issues)
