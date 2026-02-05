#!/usr/bin/env python3
"""
sweep.py - Automated speculative decoding benchmark sweep.

Runs all target+draft model combinations, collects results, and generates
interactive Plotly charts.

Usage:
    python sweep.py --config sweep_config.json
    python sweep.py --config sweep_config.json --results results.json --chart chart.html
    python sweep.py --config-dir configs/              # Run all configs in directory
    python sweep.py --results results.json --chart-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

from bench import run_bench, PROMPTS
from server import LlamaCppBackend


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def parse_acceptance_rate(log_path: str) -> float | None:
    """Extract the last draft acceptance rate from a llama.cpp server log."""
    if not log_path or not os.path.isfile(log_path):
        return None
    pattern = re.compile(r"draft acceptance rate = ([\d.]+)")
    last = None
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                last = float(m.group(1))
    return last


def run_single(
    target_path: str,
    draft_path: str | None,
    label: str,
    settings: dict,
) -> dict:
    """Start a server, benchmark it, stop it, and return result dict."""
    t_wall_start = time.monotonic()

    port = settings.get("port", 8080)
    log_file = os.path.join(tempfile.gettempdir(), f"draftbench_server_{port}.log")

    backend = LlamaCppBackend(
        model_path=target_path,
        draft_path=draft_path,
        host="127.0.0.1",
        port=port,
        gpu_layers=settings.get("gpu_layers", 99),
        ctx_size=settings.get("ctx_size", 4096),
        llama_bin=settings.get("llama_bin"),
        log_file=log_file,
    )

    backend.start()
    print(f"  Waiting for server ...")
    if not backend.wait_ready(timeout=180):
        print(f"  ERROR: Server failed to start", file=sys.stderr)
        backend.stop()
        return {"error": "server_failed"}

    print(f"  Server ready. Benchmarking ...")
    summary = run_bench(
        label=label,
        base_url=backend.base_url,
        model=os.path.basename(target_path),
        prompts=PROMPTS,
        runs=settings.get("runs", 1),
        max_tokens=settings.get("max_tokens", 512),
        temperature=settings.get("temperature", 0.0),
        api_key=None,
    )

    backend.stop()

    wall_time = round(time.monotonic() - t_wall_start, 2)

    # Parse acceptance rate from logs
    acceptance = parse_acceptance_rate(log_file) if draft_path else None

    tps_stat = summary.stat("tps")
    ttft_stat = summary.stat("ttft")
    total_stat = summary.stat("total_time")

    result = {
        "mean_tps": round(tps_stat.get("mean", 0), 2),
        "median_tps": round(tps_stat.get("median", 0), 2),
        "mean_ttft": round(ttft_stat.get("mean", 0), 3),
        "mean_total_time": round(total_stat.get("mean", 0), 2),
        "wall_time": wall_time,
        "acceptance_rate": round(acceptance, 4) if acceptance else None,
    }

    return result


def _load_existing_results(results_path: str) -> list[dict]:
    """Load existing results from a previous run, if any."""
    if not os.path.isfile(results_path):
        return []
    try:
        with open(results_path) as f:
            data = json.load(f)
        results = data.get("results", [])
        # Only keep successful results (no errors)
        return [r for r in results if "error" not in r]
    except (json.JSONDecodeError, KeyError):
        return []


def run_sweep(config: dict, results_path: str) -> list[dict]:
    """Run the full sweep and save results incrementally."""
    targets = config["targets"]
    drafts = config.get("drafts", [])
    settings = config.get("settings", {})

    total_runs = len(targets) * (1 + len(drafts))

    # Load existing results for resume support
    results = _load_existing_results(results_path)
    completed = {(r["target"], r.get("draft")) for r in results}
    skipped = len(completed)

    print(f"\n{'='*60}")
    print(f"  Sweep: {len(targets)} targets x {len(drafts)} drafts = {total_runs} runs")
    if skipped:
        print(f"  Resuming: {skipped} already completed, {total_runs - skipped} remaining")
    print(f"{'='*60}\n")

    run_idx = 0

    for target in targets:
        target_label = target["label"]
        target_path = target["path"]

        # --- baseline (no draft) ---
        run_idx += 1
        if (target_label, None) in completed:
            print(f"[{run_idx}/{total_runs}] {target_label} (baseline) -- already done, skipping")
        else:
            print(f"[{run_idx}/{total_runs}] {target_label} (baseline)")
            try:
                result = run_single(target_path, None, f"{target_label} baseline", settings)
            except Exception as e:
                print(f"  SKIPPED: {e}")
                # Skip all drafts for this target since we have no baseline
                run_idx += len(drafts)
                print(f"  Skipping {len(drafts)} draft runs for {target_label}\n")
                continue
            entry = {"target": target_label, "draft": None, **result}
            results.append(entry)
            completed.add((target_label, None))
            _save_results(results, config, results_path)
            _print_summary(entry)
            print()
            time.sleep(3)

        # --- with each draft ---
        for draft in drafts:
            run_idx += 1
            draft_label = draft["label"]
            draft_path = draft["path"]
            combo_label = f"{target_label} + {draft_label}"

            if (target_label, draft_label) in completed:
                print(f"[{run_idx}/{total_runs}] {combo_label} -- already done, skipping")
                continue

            print(f"[{run_idx}/{total_runs}] {combo_label}")
            try:
                result = run_single(target_path, draft_path, combo_label, settings)
            except Exception as e:
                print(f"  SKIPPED: {e}\n")
                continue
            entry = {"target": target_label, "draft": draft_label, **result}
            results.append(entry)
            completed.add((target_label, draft_label))
            _save_results(results, config, results_path)
            _print_summary(entry)
            print()

            time.sleep(3)

    return results


def _print_summary(entry: dict):
    """Print a one-line summary of a run."""
    if "error" in entry:
        print(f"  ERROR: {entry['error']}")
        return
    parts = [f"{entry['mean_tps']} tok/s"]
    if entry.get("acceptance_rate"):
        parts.append(f"acceptance: {entry['acceptance_rate']:.0%}")
    print(f"  Result: {', '.join(parts)}")


def _save_results(results: list[dict], config: dict, path: str):
    """Incrementally save results to JSON."""
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": config.get("name", "unnamed"),
        "hardware": config.get("hardware", "unknown"),
        "backend": config.get("backend", "unknown"),
        "model_family": config.get("model_family", "unknown"),
        "settings": config.get("settings", {}),
        "results": results,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(results: list[dict], output_path: str, metadata: dict = None):
    """Generate a standalone HTML file with Plotly charts."""
    metadata = metadata or {}
    hardware = metadata.get("hardware", "Unknown Hardware")
    backend = metadata.get("backend", "unknown")
    model_family = metadata.get("model_family", "")
    chart_title = f"Speculative Decoding Benchmark"
    chart_subtitle = f"{model_family} on {hardware} ({backend})".strip()

    # Group results by target
    targets_seen = []
    for r in results:
        if r["target"] not in targets_seen:
            targets_seen.append(r["target"])

    # Build baseline map
    baseline_map = {}
    for r in results:
        if r["draft"] is None and "error" not in r:
            baseline_map[r["target"]] = r["mean_tps"]

    # Categorize drafts by size
    def get_draft_size(draft_name: str) -> str:
        if draft_name.startswith("0.5B"):
            return "0.5B"
        elif draft_name.startswith("1.5B"):
            return "1.5B"
        elif draft_name.startswith("3B"):
            return "3B"
        elif draft_name.startswith("7B"):
            return "7B"
        elif draft_name.startswith("14B"):
            return "14B"
        return "Other"

    # Find best draft for each size category per target
    size_categories = ["0.5B", "1.5B", "3B", "7B", "14B"]
    best_by_size = {target: {} for target in targets_seen}

    for r in results:
        if r["draft"] is None or "error" in r:
            continue
        target = r["target"]
        size = get_draft_size(r["draft"])
        if size not in best_by_size[target] or r["mean_tps"] > best_by_size[target][size]["mean_tps"]:
            best_by_size[target][size] = r

    # Build summary bar chart traces (baseline + best per size)
    colors = {
        "Baseline": "#636EFA",
        "0.5B": "#EF553B",
        "1.5B": "#00CC96",
        "3B": "#AB63FA",
        "7B": "#FFA15A",
        "14B": "#19D3F3",
    }

    summary_traces = []

    # Baseline trace
    summary_traces.append({
        "x": targets_seen,
        "y": [baseline_map.get(t, 0) for t in targets_seen],
        "text": [f"{baseline_map.get(t, 0):.1f}" for t in targets_seen],
        "textposition": "outside",
        "name": "Baseline (no draft)",
        "type": "bar",
        "marker": {"color": colors["Baseline"]},
    })

    # Best draft per size category
    for size in size_categories:
        y_vals = []
        text_vals = []
        hover_vals = []
        has_data = False
        for target in targets_seen:
            if size in best_by_size[target]:
                r = best_by_size[target][size]
                y_vals.append(r["mean_tps"])
                acc = r.get("acceptance_rate", 0) or 0
                text_vals.append(f"{r['mean_tps']:.1f}")
                hover_vals.append(f"{r['draft']}<br>{r['mean_tps']:.1f} tok/s<br>{acc:.0%} acceptance")
                has_data = True
            else:
                y_vals.append(0)
                text_vals.append("")
                hover_vals.append("")

        if has_data:
            summary_traces.append({
                "x": targets_seen,
                "y": y_vals,
                "text": text_vals,
                "textposition": "outside",
                "hovertext": hover_vals,
                "hoverinfo": "text",
                "name": f"Best {size} Draft",
                "type": "bar",
                "marker": {"color": colors.get(size, "#888")},
            })

    # Build speedup bar chart (best per size)
    speedup_traces = []
    for size in size_categories:
        y_vals = []
        text_vals = []
        hover_vals = []
        has_data = False
        for target in targets_seen:
            base = baseline_map.get(target, 0)
            if size in best_by_size[target] and base > 0:
                r = best_by_size[target][size]
                speedup = (r["mean_tps"] - base) / base * 100
                y_vals.append(round(speedup, 1))
                acc = r.get("acceptance_rate", 0) or 0
                text_vals.append(f"+{speedup:.0f}%")
                hover_vals.append(f"{r['draft']}<br>+{speedup:.1f}% speedup<br>{acc:.0%} acceptance")
                has_data = True
            else:
                y_vals.append(0)
                text_vals.append("")
                hover_vals.append("")

        if has_data:
            speedup_traces.append({
                "x": targets_seen,
                "y": y_vals,
                "text": text_vals,
                "textposition": "outside",
                "hovertext": hover_vals,
                "hoverinfo": "text",
                "name": f"Best {size} Draft",
                "type": "bar",
                "marker": {"color": colors.get(size, "#888")},
            })

    # Build heatmap data (all drafts)
    drafts_seen = []
    for r in results:
        if r["draft"] and r["draft"] not in drafts_seen:
            drafts_seen.append(r["draft"])

    # Sort drafts by size then quant
    def draft_sort_key(d):
        size_order = {"0.5B": 0, "1.5B": 1, "3B": 2, "7B": 3, "14B": 4}
        size = get_draft_size(d)
        return (size_order.get(size, 99), d)

    drafts_seen.sort(key=draft_sort_key)

    heatmap_z = []
    heatmap_text = []
    for draft in drafts_seen:
        row = []
        text_row = []
        for target in targets_seen:
            match = [r for r in results if r["target"] == target and r["draft"] == draft and "error" not in r]
            base = baseline_map.get(target, 0)
            if match and base > 0:
                speedup = (match[0]["mean_tps"] - base) / base * 100
                row.append(round(speedup, 1))
                acc = match[0].get("acceptance_rate", 0) or 0
                text_row.append(f"+{speedup:.0f}%<br>{acc:.0%} acc")
            else:
                row.append(None)
                text_row.append("")
        heatmap_z.append(row)
        heatmap_text.append(text_row)

    summary_json = json.dumps(summary_traces)
    speedup_json = json.dumps(speedup_traces)
    heatmap_z_json = json.dumps(heatmap_z)
    heatmap_text_json = json.dumps(heatmap_text)
    targets_json = json.dumps(targets_seen)
    drafts_json = json.dumps(drafts_seen)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>draftbench - Speculative Decoding Sweep</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{
            text-align: center;
            color: #fff;
            margin-bottom: 10px;
        }}
        h2 {{
            text-align: center;
            color: #888;
            font-weight: normal;
            margin-top: 0;
        }}
        .chart {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin: 30px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .chart-title {{
            color: #fff;
            font-size: 1.2em;
            margin-bottom: 10px;
            padding-left: 10px;
        }}
    </style>
</head>
<body>
    <h1>{chart_title}</h1>
    <h2>{chart_subtitle}</h2>

    <div class="chart">
        <div class="chart-title">Throughput Comparison (tokens/sec)</div>
        <div id="summary-chart"></div>
    </div>

    <div class="chart">
        <div class="chart-title">Speedup vs Baseline (%)</div>
        <div id="speedup-chart"></div>
    </div>

    <div class="chart">
        <div class="chart-title">Full Results Heatmap (% Speedup)</div>
        <div id="heatmap-chart"></div>
    </div>

    <script>
        var darkLayout = {{
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#16213e',
            font: {{ color: '#eee' }},
            xaxis: {{ gridcolor: '#2a3a5e', title: 'Target Model' }},
            yaxis: {{ gridcolor: '#2a3a5e' }},
        }};

        var summaryTraces = {summary_json};
        Plotly.newPlot('summary-chart', summaryTraces, {{
            ...darkLayout,
            barmode: 'group',
            yaxis: {{ ...darkLayout.yaxis, title: 'Tokens per Second' }},
            legend: {{ orientation: 'h', y: -0.15, font: {{ color: '#eee' }} }},
            margin: {{ b: 80, t: 20 }},
            height: 400,
        }}, {{ responsive: true }});

        var speedupTraces = {speedup_json};
        Plotly.newPlot('speedup-chart', speedupTraces, {{
            ...darkLayout,
            barmode: 'group',
            yaxis: {{ ...darkLayout.yaxis, title: 'Speedup (%)' }},
            legend: {{ orientation: 'h', y: -0.15, font: {{ color: '#eee' }} }},
            margin: {{ b: 80, t: 20 }},
            height: 400,
            shapes: [{{
                type: 'line', x0: 0, x1: 1, xref: 'paper',
                y0: 0, y1: 0, yref: 'y',
                line: {{ color: '#666', width: 2, dash: 'dash' }}
            }}]
        }}, {{ responsive: true }});

        var heatmapTrace = [{{
            z: {heatmap_z_json},
            x: {targets_json},
            y: {drafts_json},
            hovertemplate: '<b>%{{y}}</b> → %{{x}}<br>Speedup: <b>%{{z:.1f}}%</b><extra></extra>',
            type: 'heatmap',
            colorscale: [
                [0, '#dc3545'],
                [0.4, '#fd7e14'],
                [0.5, '#ffc107'],
                [0.7, '#28a745'],
                [1, '#00ff88']
            ],
            zmin: 0,
            zmax: 85,
            colorbar: {{
                title: 'Speedup %',
                tickfont: {{ color: '#eee', size: 14 }},
                titlefont: {{ color: '#eee', size: 14 }},
                tickvals: [0, 20, 40, 60, 80],
                ticktext: ['0%', '+20%', '+40%', '+60%', '+80%'],
                len: 0.9
            }},
            hoverongaps: false,
            xgap: 2,
            ygap: 1,
        }}];

        var numDrafts = {drafts_json}.length;
        var heatmapHeight = Math.max(500, numDrafts * 22 + 100);

        Plotly.newPlot('heatmap-chart', heatmapTrace, {{
            ...darkLayout,
            xaxis: {{
                ...darkLayout.xaxis,
                side: 'top',
                tickfont: {{ size: 14, color: '#eee' }},
                tickangle: 0
            }},
            yaxis: {{
                ...darkLayout.yaxis,
                title: '',
                autorange: 'reversed',
                tickfont: {{ size: 12, color: '#eee' }},
                dtick: 1
            }},
            margin: {{ l: 100, t: 50, b: 20, r: 100 }},
            height: heatmapHeight,
        }}, {{ responsive: true }});
    </script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"  Chart saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _generate_output_paths(config: dict) -> tuple[str, str]:
    """Generate output paths from config metadata."""
    hardware = config.get("hardware", "unknown")
    backend = config.get("backend", "unknown")
    name = config.get("name", "sweep")

    base = f"results/{hardware}_{backend}_{name}"
    return f"{base}.json", f"{base}.html"


def _run_config_file(config_path: str, results_path: str | None = None, chart_path: str | None = None):
    """Run a single config file and generate charts."""
    print(f"\n{'#'*60}")
    print(f"  Loading config: {config_path}")
    print(f"{'#'*60}")

    with open(config_path) as f:
        config = json.load(f)

    # Auto-generate paths from config metadata if not specified
    auto_results, auto_chart = _generate_output_paths(config)
    results_path = results_path or auto_results
    chart_path = chart_path or auto_chart

    results = run_sweep(config, results_path)

    print(f"\n{'='*60}")
    print(f"  Sweep complete. Results saved to {results_path}")
    print(f"{'='*60}\n")

    generate_chart(results, chart_path, config)
    return results_path, chart_path


def main():
    parser = argparse.ArgumentParser(
        description="Run speculative decoding benchmark sweep and generate charts.",
    )
    parser.add_argument("--config", help="Path to sweep config JSON file")
    parser.add_argument("--config-dir", help="Path to directory containing config files (runs all *.json except example_*.json)")
    parser.add_argument("--results", help="Path to results JSON file (auto-generated from config if not specified)")
    parser.add_argument("--chart", help="Path to output HTML chart (auto-generated from config if not specified)")
    parser.add_argument("--chart-only", action="store_true", help="Skip benchmarking, just generate chart from existing results")

    args = parser.parse_args()

    if args.chart_only:
        if not args.results:
            parser.error("--results is required when using --chart-only")
        if not os.path.isfile(args.results):
            print(f"Error: results file not found: {args.results}", file=sys.stderr)
            sys.exit(1)
        with open(args.results) as f:
            data = json.load(f)
        chart_path = args.chart or args.results.replace(".json", ".html")
        generate_chart(data["results"], chart_path, data)
        return

    # Handle --config-dir: run all configs in directory
    if args.config_dir:
        if not os.path.isdir(args.config_dir):
            print(f"Error: directory not found: {args.config_dir}", file=sys.stderr)
            sys.exit(1)

        # Find all JSON files, excluding example_*.json templates
        config_files = sorted(glob.glob(os.path.join(args.config_dir, "*.json")))
        config_files = [f for f in config_files if not os.path.basename(f).startswith("example_")]

        if not config_files:
            print(f"Error: no config files found in {args.config_dir}", file=sys.stderr)
            print(f"  (files matching example_*.json are excluded)", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'#'*60}")
        print(f"  Running {len(config_files)} config(s) from {args.config_dir}")
        print(f"{'#'*60}")
        for i, cf in enumerate(config_files, 1):
            print(f"  [{i}] {os.path.basename(cf)}")

        completed = []
        for config_file in config_files:
            try:
                results_path, chart_path = _run_config_file(config_file)
                completed.append((config_file, results_path, chart_path))
            except Exception as e:
                print(f"\n  ERROR running {config_file}: {e}", file=sys.stderr)
                continue

        print(f"\n{'#'*60}")
        print(f"  All sweeps complete! {len(completed)}/{len(config_files)} succeeded")
        print(f"{'#'*60}")
        for cf, rp, cp in completed:
            print(f"  {os.path.basename(cf)}:")
            print(f"    Results: {rp}")
            print(f"    Chart:   {cp}")
        return

    # Handle single --config
    if not args.config:
        parser.error("--config or --config-dir is required (unless using --chart-only)")

    _run_config_file(args.config, args.results, args.chart)


if __name__ == "__main__":
    main()
