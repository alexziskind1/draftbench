# draftbench

Find the optimal draft model for speculative decoding on your hardware.

## What Does This Tool Do?

**The problem:** Speculative decoding can speed up LLM inference by 50-80%, but only if you pick the right draft model. Too small and accuracy suffers. Too large and the overhead kills your gains. The optimal choice depends on your target model, quantization, and hardware.

**The solution:** draftbench automatically tests every combination of target + draft models you give it, measures the throughput, and shows you which pairing works best. Instead of guessing, you get data.

**How it works:**
1. You provide a list of target models (the big ones you want to run fast)
2. You provide a list of draft models (smaller models from the same family)
3. draftbench tests each combination: baseline speed, then speed with each draft
4. Results are saved to JSON and visualized as interactive charts

## What is Speculative Decoding?

Speculative decoding uses a small "draft" model to propose tokens that a larger "target" model then verifies. When the draft model predicts correctly, multiple tokens are accepted in a single forward pass, significantly speeding up generation.

**Key findings from our benchmarks:**
- Slow targets (72B Q8_0 @ 6 tok/s): **+80% speedup** with the right draft model
- Fast targets (72B Q4_K_M @ 9.5 tok/s): **+12% speedup** - diminishing returns
- Sweet spot: **3B Q4_K_M** draft works well across different target sizes

## Prerequisites

### 1. Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake ..                    # Add -DLLAMA_CUDA=ON for NVIDIA or -DLLAMA_METAL=ON for Apple Silicon
cmake --build . --config Release -j
```

### 2. Download Models

You need GGUF models from the same family (same tokenizer). We use Qwen 2.5:

**Target models** (large):
- `Qwen2.5-72B-Instruct-Q8_0.gguf` or `Q4_K_M`
- `Qwen2.5-32B-Instruct-Q8_0.gguf` or `Q4_K_M`
- `Qwen2.5-14B-Instruct-Q8_0.gguf`
- `Qwen2.5-7B-Instruct-Q8_0.gguf`

**Draft models** (small):
- `qwen2.5-0.5b-instruct-q8_0.gguf`
- `qwen2.5-1.5b-instruct-q4_k_m.gguf`
- `qwen2.5-3b-instruct-q4_k_m.gguf`

Download from Hugging Face: [Qwen](https://huggingface.co/Qwen) or [bartowski](https://huggingface.co/bartowski)

### 3. Python Dependencies

```bash
pip install requests
```

No other dependencies - charts use Plotly.js CDN.

## Quick Start

### Single Benchmark

Test a single target + draft combination:

```bash
# Start server with speculative decoding
llama-server \
  -m /path/to/target-model.gguf \
  --model-draft /path/to/draft-model.gguf \
  -ngl 99 -c 4096 --port 8080

# In another terminal, run benchmark
python bench.py --url http://127.0.0.1:8080 --requests 5 --max-tokens 512
```

### Using the Server Launcher

```bash
# Without draft (baseline)
python server.py llama-cpp --model-path /path/to/72b-model.gguf

# With draft (speculative decoding)
python server.py llama-cpp \
    --model-path /path/to/72b-model.gguf \
    --draft-path /path/to/3b-draft.gguf
```

Options: `--port 8080`, `--gpu-layers 99`, `--ctx-size 4096`

## Running a Sweep

A sweep tests all combinations of target and draft models automatically.

### 1. Create a Config File

Create `configs/my_sweep.json`:

```json
{
  "name": "qwen25-72b",
  "hardware": "rtx4090-24gb",
  "backend": "llamacpp",
  "model_family": "Qwen2.5",

  "targets": [
    {"label": "72B Q8_0", "path": "/path/to/Qwen2.5-72B-Instruct-Q8_0.gguf"},
    {"label": "72B Q4_K_M", "path": "/path/to/Qwen2.5-72B-Instruct-Q4_K_M.gguf"}
  ],
  "drafts": [
    {"label": "0.5B Q8_0", "path": "/path/to/qwen2.5-0.5b-instruct-q8_0.gguf"},
    {"label": "1.5B Q4_K_M", "path": "/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf"},
    {"label": "3B Q4_K_M", "path": "/path/to/qwen2.5-3b-instruct-q4_k_m.gguf"}
  ],
  "settings": {
    "runs": 1,
    "max_tokens": 1024,
    "temperature": 0.0,
    "gpu_layers": 99,
    "ctx_size": 4096,
    "port": 8080
  }
}
```

**Metadata fields:**
- `name`: Short identifier for this sweep (used in filenames)
- `hardware`: Hardware identifier (e.g., `rtx4090-24gb`, `a100-80gb`)
- `backend`: Inference backend (`llamacpp`, `vllm`, `lmstudio`)
- `model_family`: Model family name for chart titles

### 2. Run the Sweep

```bash
# Run a single config
python sweep.py --config configs/my_sweep.json
# Creates: results/<hardware>_<backend>_<name>.json
#          results/<hardware>_<backend>_<name>.html

# Or specify custom output paths
python sweep.py --config configs/my_sweep.json --results results/custom.json --chart results/custom.html
```

This will:
1. Test each target model without a draft (baseline)
2. Test each target + draft combination
3. Save results incrementally to JSON (with hardware/backend metadata)
4. Generate interactive charts in HTML

### Running Multiple Configs

If you have several config files (e.g., one per target model size), you can run them all in sequence:

```bash
python sweep.py --config-dir configs/
```

This finds all `*.json` files in the directory (excluding `example_*.json` templates), runs each sweep back-to-back, and generates separate results and charts for each. If one config fails, it skips to the next and reports a summary at the end.

**Example output:**
```
============================================================
  Sweep: 2 targets x 3 drafts = 8 runs
============================================================

[1/8] 72B Q8_0 (baseline)
  Starting server ... ready
  Benchmarking ... 5.93 tok/s
  Server stopped

[2/8] 72B Q8_0 + 0.5B Q8_0
  Starting server ... ready
  Benchmarking ... 9.83 tok/s (acceptance: 57%)
  Server stopped

...

=== Sweep complete ===
Results saved to results.json
Chart saved to chart.html
```

### 3. Generate Charts from Existing Results

If you stopped a sweep early or want to regenerate charts:

```bash
python sweep.py --results results.json --chart chart.html --chart-only
```

### 4. View the Charts

```bash
open chart.html
```

## Understanding the Charts

The generated HTML file contains three interactive charts:

### 1. Throughput Comparison
Bar chart showing tokens/second for each target model with:
- Baseline (no draft)
- Best draft from each size category (0.5B, 1.5B, 3B, 7B)

### 2. Speedup vs Baseline
Bar chart showing percentage improvement over baseline for each draft size.

### 3. Full Results Heatmap
Color-coded matrix showing speedup % for every target + draft combination:
- **Green** = good speedup (60-80%+)
- **Yellow** = moderate speedup (30-50%)
- **Orange/Red** = minimal or negative impact

Hover over any cell for details.

## Key Insights

### When Speculative Decoding Helps Most

1. **Slow target models**: The slower your target, the more you gain
   - 72B Q8_0 (6 tok/s baseline) → +80% with 3B draft
   - 72B Q4_K_M (9.5 tok/s baseline) → +12% with 3B draft

2. **Same model family**: Draft and target must share the same tokenizer
   - Qwen 2.5 family: 0.5B through 72B all compatible
   - Mixing families (e.g., Llama 3 + Llama 3.2) causes token translation overhead

3. **Draft size sweet spot**:
   - Too small (0.5B): ~57% acceptance rate, limited gains
   - Sweet spot (1.5B-3B): ~68-70% acceptance, best throughput
   - Too large (7B): ~72% acceptance but draft is too slow

### Choosing a Draft Model

| Target Speed | Recommended Draft | Expected Gain |
|--------------|-------------------|---------------|
| < 8 tok/s    | 3B Q4_K_M         | +60-80%       |
| 8-15 tok/s   | 1.5B-3B Q4_K_M    | +10-30%       |
| > 15 tok/s   | 0.5B-1.5B Q4_K_M  | +5-15%        |
| > 30 tok/s   | Not recommended   | Overhead > gains |

## Results Format

Results are saved as JSON with full metadata:

```json
{
  "timestamp": "2026-02-05T01:18:17.066992+00:00",
  "name": "qwen25-72b",
  "hardware": "rtx4090-24gb",
  "backend": "llamacpp",
  "model_family": "Qwen2.5",
  "settings": { ... },
  "results": [
    {
      "target": "72B Q8_0",
      "draft": null,
      "mean_tps": 5.93,
      "median_tps": 6.05,
      "mean_ttft": 0.901,
      "mean_total_time": 87.34,
      "acceptance_rate": null
    },
    {
      "target": "72B Q8_0",
      "draft": "3B Q4_K_M",
      "mean_tps": 10.56,
      "median_tps": 10.06,
      "mean_ttft": 0.671,
      "mean_total_time": 49.93,
      "acceptance_rate": 0.6888
    }
  ]
}
```

## File Structure

```
draftbench/
├── bench.py          # Core benchmark logic
├── server.py         # Server launcher (llama.cpp, LM Studio, vLLM)
├── sweep.py          # Automated sweep + chart generation
├── configs/          # Sweep configuration files
│   └── example_sweep.json  # Template - copy and customize
├── results/          # Benchmark results and charts (auto-generated)
│   ├── *.json        # Raw results with metadata
│   └── *.html        # Interactive Plotly visualizations
└── README.md
```

## Troubleshooting

### "Address already in use"
Wait a few seconds between runs or change the port in your config.

### Low acceptance rate (< 50%)
Your draft and target models may have incompatible tokenizers. Use models from the same family.

### Draft model slower than baseline
Your target model is already fast enough that draft overhead hurts. Try a smaller draft or skip speculative decoding.

### Out of memory
Reduce `gpu_layers` or use more aggressive quantization (Q4_0 instead of Q8_0).
