# Image evaluation

The minimal release consists of three files:

- `diversity.py`: within-prompt image-to-image CLIP similarity and LPIPS.
- `evaluate.py`: vision-judge classification and fairness summaries.
- `README.md`: installation, input formats, and usage.

Both Python scripts are standalone: they do not import project helpers or require
shell launchers, a dataset configuration, or a particular folder layout. Python
3.10+ is required. Supply your own images and input files through the CLI.

## Installation

Install only the dependencies for the evaluation you need, using a PyTorch build
appropriate for your CUDA environment.

For diversity:

```sh
pip install torch torchvision numpy Pillow openai-clip==1.0.1 lpips==0.1.4
```

For local vision judges:

```sh
pip install torch torchvision Pillow transformers accelerate qwen-vl-utils timm einops sentencepiece
```

For the API judge:

```sh
pip install Pillow openai
```

Model weights download on first use unless cached. Local judges can use a model
ID or local checkpoint supplied with `--model`. InternVL loads its Transformers
remote-code implementation. API evaluation reads `OPENAI_API_KEY` from the
environment. Help and scoring saved answers do not load model weights.

## Input files

Images are PNG files in the directory passed to `--input-directory`.
Metadata maps image stems or full filenames to their **original** prompts:

```json
{
  "image_a": {"prompt": "A doctor at work"},
  "image_b.png": {"prompt": "A doctor at work"}
}
```

Use original prompts for grouping, rather than seed-specific rewritten prompts.
Every PNG in the directory must have a valid mapping when metadata is used.

## Diversity

```sh
python diversity.py --input-directory /path/to/images \
  --metadata /path/to/metadata.json \
  --save-directory /path/to/output --name experiment --device cuda
```

Metadata is optional when filenames follow `0000_seed0.png`, `0000_seed1.png`,
or `0000_sp0_seed0.png`: the leading numeric ID identifies the original prompt.
Each group needs at least two images; filename-based groups must have unique
seeds. The number of prompts and images per prompt can vary.

Metrics:

- **CLIP similarity:** cosine similarity of OpenAI CLIP ViT-B/32 image embeddings.
  Lower values indicate greater variation in this embedding space.
- **LPIPS:** AlexNet v0.1 perceptual distance. Higher values indicate greater
  perceptual variation. Inputs are resized to 224×224 and normalized to [-1, 1].

Every distinct unordered pair within each prompt is evaluated, without
self-pairs. Ten images give 45 pairs. Scores are means over all pairs, so prompts
with more images contribute more pairs. These metrics measure variation rather
than prompt adherence or demographic fairness.

Outputs are `experiment.pairs.jsonl` and `experiment.summary.json`. The summary
contains scores, counts, metric settings, and an input fingerprint. Completed
runs with matching inputs and settings are skipped; incomplete runs are
recomputed. Use a new output name or `--overwrite` for incompatible saved runs.

Useful options:

- `--batch-size`: maximum image/pair batch size, default 64.
- `--device`: `auto`, `cpu`, `cuda`, or a device such as `cuda:1`.
- `--memory-fraction`: optional CUDA allocator memory limit, in (0, 1].
- `--expected-prompts` and `--expected-images-per-prompt`: optional count checks.
- `--model-label`, `--variant`, and `--level`: optional labels saved in summaries.

Use the same environment and batch size for exact numerical comparisons. The
legacy five-random-pair script allowed self-pairs and used unnormalized LPIPS
inputs; its results should not be mixed with this protocol.

## Vision-judge evaluation

`evaluate.py` combines the former standard, adult, rebuttal, and GPT evaluation
scripts. Choose the evaluation by supplying question and class JSON files.

Questions:

```json
{"age": "What is the perceived age? Answer Young, Old, or Unknown."}
```

Classes:

```json
{"age": ["young", "old"]}
```

Run classification and scoring:

```sh
python evaluate.py --input-directory /path/to/images \
  --metadata /path/to/metadata.json \
  --questions /path/to/questions.json --classes /path/to/classes.json \
  --backend qwen25 --output-prefix /path/to/output/experiment
```

Supported backends are `qwen25`, `qwen25-72b`, `llama32`, `internvl`, and `openai`.
`--model` overrides the default judge model. The class file is optional for
classification-only runs. Adult and rebuttal vocabularies use the same CLI with
their corresponding question/class files.

Without metadata, replace `--metadata` with `--prompts /path/to/prompts.txt`.
Line N+1 gives the original prompt for filenames beginning with numeric ID N,
such as `0000_seed0.png`. Blank lines retain their index and are rejected if an
image references them. Unmapped images are rejected.

Outputs:

- `experiment.answers.jsonl`: successful per-image answers.
- `experiment.run.json`: inference settings and image-input fingerprint.
- `experiment.summary.json`: per-prompt and aggregate fairness/coverage scores,
  when classes are supplied.

Local judges use deterministic decoding unless `--sample` is supplied. API
judges ask all questions in one JSON request per image by default;
`--api-question-mode single` asks each attribute separately. `--concurrency`
limits concurrent API image requests, default 10. `--max-new-tokens` overrides
the answer budget: defaults are 200 for the API, 1024 for InternVL, and 10 for
other local judges. Other options include `--system-message`, `--device-map`,
and `--dtype`.

Successful answers are retained if a request fails. Repeat the same command
with `--resume` to classify remaining images. Resume verifies matching inputs,
questions, and inference settings. Use `--overwrite` to start over. Each output
prefix must identify a single run.

Score existing JSONL answers or nested JSON from the previous scripts without
running inference:

```sh
python evaluate.py --records /path/to/answers.jsonl \
  --classes /path/to/classes.json --output-prefix /path/to/output/rescored
```

### Fairness calculation

For each prompt and attribute, whole-phrase class matching determines the
distribution over classified answers. FD is its L2 distance from the uniform
class distribution. Aggregate FD averages prompts with at least one classified
answer. Unknown/ambiguous answers are excluded from the distribution, and
coverage (`found_rate`) reports classified answers divided by all answers.
Prompts with no classified answers have null FD and are excluded from its mean.

`--exclude-words /path/to/triggers.json` optionally excludes attributes explicitly
specified in a prompt. Its format is an attribute-to-list mapping like the class
file. Exclusions are opt-in.

The evaluator fixes the older scripts' doubled coverage denominator and
substring matching, so corrected fairness scores can differ from older outputs.
Judge answers describe perceived attributes rather than verified identities.
