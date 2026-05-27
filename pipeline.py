# pipeline.py
# -*- coding: utf-8 -*-
"""
PE-Sleuth 端到端分析流水线。

输入单个 PE 二进制文件，输出完整的分析报告目录。

用法:
    python pipeline.py --input sample.exe --output ./reports
    python pipeline.py -i malware.exe -o ./reports --ida-path "D:\\IDA Pro"

输出目录结构:
    {output_dir}/{binary_name}/
    ├── summary.md                    # 综合分析报告
    ├── {binary_name}.c               # 反编译 C 源码
    ├── 1a_parsed_metadata/           # 静态特征 (字符串/API/可疑指标)
    ├── 1b_call_graphs/               # 函数调用图
    ├── 2_smart_chunks/               # 调用图感知分块
    ├── 3_visualizations/             # 交互式调用图 HTML
    ├── 4_chunk_summaries/            # 块级行为语义
    ├── 5_program_summaries/          # 程序级语义
    ├── 6_final_ir/                   # 融合中间表示
    ├── 7_classification_results/     # 分类预测 (含消融实验)
    └── 8_classification_rationale/   # 分类理由
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# =========================
# Paths
# =========================
REPO_ROOT = Path(__file__).resolve().parent
PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"

DECOMPILE_SCRIPT = REPO_ROOT / "batch_decompile_ida.py"
CLASSIFY_SCRIPT = REPO_ROOT / "classify_from_c_code.py"

DECOMPILE_INPUT_DIR = REPO_ROOT / "Decompile" / "Decompile_Input_Raw_PE"
DECOMPILE_OUTPUT_DIR = REPO_ROOT / "Decompile" / "Decompile_Output_C_Code"
CLASSIFY_INPUT_DIR = REPO_ROOT / "Classify" / "Classify_Input_C_Code"
CLASSIFY_OUTPUT_DIR = REPO_ROOT / "Classify" / "Classify_Output_All"

# Output sub-directories (must match classify_from_c_code.py ExperimentConfig)
STEP_DIRS = [
    "1a_parsed_metadata",
    "1b_call_graphs",
    "2_smart_chunks",
    "3_visualizations",
    "4_chunk_summaries",
    "5_program_summaries",
    "6_final_ir",
    "7_classification_results",
    "8_classification_rationale",
]

SETTINGS_PATH = REPO_ROOT / "settings.json"


# =========================
# Helpers
# =========================


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_settings() -> dict:
    """Load settings.json. Returns empty dict if file not found."""
    if SETTINGS_PATH.exists():
        return read_json(SETTINGS_PATH)
    return {}


def get_ida_path() -> Optional[str]:
    """Resolve IDA path from settings.json, then env var. Returns None if not configured."""
    settings = load_settings()
    ida = settings.get("ida_path", "")
    if ida:
        return ida
    return os.getenv("IDA_PATH")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def ensure_empty(path: Path) -> None:
    """Ensure directory exists and is empty. Create if needed."""
    if path.exists():
        for item in path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        path.mkdir(parents=True)


# =========================
# Workspace management
# =========================


def clean_workspace() -> None:
    """Clear all intermediate I/O directories for a fresh single-file run."""
    logging.info("Cleaning workspace...")

    # Clear decompile input (keep only .gitkeep if present)
    DECOMPILE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in DECOMPILE_INPUT_DIR.iterdir():
        if f.name != ".gitkeep":
            f.unlink()

    # Clear decompile output
    DECOMPILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in DECOMPILE_OUTPUT_DIR.iterdir():
        if f.name != ".gitkeep":
            f.unlink()

    # Clear classify input
    CLASSIFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in CLASSIFY_INPUT_DIR.iterdir():
        if f.name != ".gitkeep":
            f.unlink()

    # Clear classify output subdirectories
    CLASSIFY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in CLASSIFY_OUTPUT_DIR.iterdir():
        if subdir.is_dir():
            shutil.rmtree(subdir)

    # Clear failure log
    fail_log = REPO_ROOT / "Decompile" / "decompile_failure_log.txt"
    if fail_log.exists():
        fail_log.unlink()

    logging.info("Workspace cleaned.")


# =========================
# Subprocess runners
# =========================


def run_decompile(ida_path: str) -> bool:
    """Run batch_decompile_ida.py. Returns True on success."""
    env = os.environ.copy()
    env["IDA_PATH"] = ida_path
    env["PYTHONIOENCODING"] = "utf-8"

    logging.info(f"Running decompilation (IDA_PATH={ida_path})...")
    t0 = time.time()

    result = subprocess.run(
        [str(PYTHON), str(DECOMPILE_SCRIPT)],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        logging.info(f"Decompilation OK ({elapsed:.1f}s)")
        return True
    else:
        logging.error(f"Decompilation failed (exit {result.returncode}):")
        logging.error(result.stdout[-2000:] if result.stdout else "(no stdout)")
        logging.error(result.stderr[-2000:] if result.stderr else "(no stderr)")
        return False


def run_classify() -> bool:
    """Run classify_from_c_code.py. Returns True on success."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    logging.info("Running classification pipeline...")
    t0 = time.time()

    result = subprocess.run(
        [str(PYTHON), str(CLASSIFY_SCRIPT)],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=3600,
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        logging.info(f"Classification OK ({elapsed:.1f}s)")
        return True
    else:
        logging.error(f"Classification failed (exit {result.returncode}):")
        logging.error(result.stdout[-2000:] if result.stdout else "(no stdout)")
        logging.error(result.stderr[-2000:] if result.stderr else "(no stderr)")
        return False


# =========================
# Results collection
# =========================


def collect_results(binary_name: str, results_dir: Path) -> None:
    """Copy all pipeline outputs from Classify_Output_All/ into results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)

    for step_dir_name in STEP_DIRS:
        src_dir = CLASSIFY_OUTPUT_DIR / step_dir_name
        if not src_dir.exists():
            continue
        dst_dir = results_dir / step_dir_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for f in src_dir.iterdir():
            if f.is_file() and binary_name in f.name:
                shutil.copy2(f, dst_dir / f.name)

    # Also copy the decompiled C source
    c_file = DECOMPILE_OUTPUT_DIR / f"{binary_name}.c"
    if c_file.exists():
        shutil.copy2(c_file, results_dir / c_file.name)


# =========================
# Summary report
# =========================


def generate_summary(binary_name: str, results_dir: Path) -> None:
    """Generate summary.md consolidating key findings."""

    lines: list[str] = []
    lines.append(f"# PE-Sleuth Analysis Report: `{binary_name}`")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── Load settings ──
    try:
        settings = read_json(REPO_ROOT / "settings.json")
        model_name = settings.get("model_name", "unknown")
        lines.append(f"**Model:** {model_name}")
    except Exception:
        lines.append("**Model:** unknown")
    lines.append("")

    # ── 1. Classification ──
    clf_file = (
        results_dir
        / "7_classification_results"
        / f"{binary_name}_classification.json"
    )
    if clf_file.exists():
        clf = read_json(clf_file)
        variants = clf.get("variants", {})
        full = variants.get("full", {})

        lines.append("---")
        lines.append("## 1. Classification Result")
        lines.append("")
        prediction = full.get("prediction", "UNKNOWN")
        lines.append(f"**Verdict: `{prediction}`**")
        lines.append("")

        lines.append("| Variant | Prediction |")
        lines.append("|---------|------------|")
        for variant_name in ["full", "E-Meta", "E-Summary"]:
            v = variants.get(variant_name, {})
            label = v.get("prediction", "N/A")
            lines.append(f"| {variant_name} | `{label}` |")
        lines.append("")

        # Explanation of ablation variants
        lines.append(
            "> *Full* = static features + program semantics &nbsp;|&nbsp; "
            "*E-Meta* = program semantics only &nbsp;|&nbsp; "
            "*E-Summary* = static features only"
        )
        lines.append("")

    # ── 2. Rationale ──
    rationale_file = (
        results_dir
        / "8_classification_rationale"
        / f"{binary_name}_rationale.json"
    )
    if rationale_file.exists():
        rat = read_json(rationale_file)
        lines.append("---")
        lines.append("## 2. Analyst Rationale")
        lines.append("")
        lines.append(rat.get("rationale", "*No rationale generated.*"))
        lines.append("")

    # ── 3. Program-Level Semantics ──
    pgm_file = (
        results_dir
        / "5_program_summaries"
        / f"{binary_name}_program_summary.json"
    )
    if pgm_file.exists():
        pgm = read_json(pgm_file)
        lines.append("---")
        lines.append("## 3. Program-Level Semantics")
        lines.append("")
        lines.append("```text")
        lines.append(pgm.get("program_summary", ""))
        lines.append("```")
        lines.append("")

    # ── 4. Static Features ──
    meta_file = (
        results_dir / "1a_parsed_metadata" / f"{binary_name}_metadata.json"
    )
    if meta_file.exists():
        meta = read_json(meta_file)
        lines.append("---")
        lines.append("## 4. Static Features")
        lines.append("")

        # Suspicious indicators
        susp = meta.get("suspicious_indicators", {})
        if susp:
            triggered = [k for k, v in susp.items() if v]
            lines.append(f"**Suspicious Indicators:** {len(triggered)} triggered")
            if triggered:
                lines.append("")
                for ind in triggered:
                    lines.append(f"- [!!] `{ind}`")
            else:
                lines.append("")
                lines.append("*No suspicious indicators triggered.*")
            lines.append("")

        # API imports
        apis = meta.get("api_imports", [])
        lines.append(f"**API Imports:** {len(apis)} unique")
        if apis:
            lines.append("<details>")
            lines.append(f"<summary>Show {len(apis)} imports</summary>")
            lines.append("")
            lines.append("```")
            for api in apis[:80]:
                lines.append(api)
            if len(apis) > 80:
                lines.append(f"... and {len(apis) - 80} more")
            lines.append("```")
            lines.append("</details>")
        lines.append("")

        # Strings
        strings = meta.get("top_strings", [])
        lines.append(f"**Top Strings:** {len(strings)} extracted")
        if strings:
            lines.append("<details>")
            lines.append(f"<summary>Show {len(strings)} strings</summary>")
            lines.append("")
            lines.append("```")
            for s in strings[:60]:
                lines.append(s)
            if len(strings) > 60:
                lines.append(f"... and {len(strings) - 60} more")
            lines.append("```")
            lines.append("</details>")
        lines.append("")

        # Global declarations
        decls = meta.get("global_declarations", [])
        lines.append(f"**Global Declarations:** {len(decls)}")
        lines.append("")

    # ── 5. Call Graph ──
    cg_file = results_dir / "1b_call_graphs" / f"{binary_name}_callgraph.json"
    if cg_file.exists():
        cg = read_json(cg_file)
        lines.append("---")
        lines.append("## 5. Call Graph")
        lines.append("")
        lines.append(f"- **Nodes (functions):** {cg.get('node_count', 0)}")
        lines.append(f"- **Edges (calls):** {cg.get('edge_count', 0)}")
        lines.append("")

    # ── 6. Chunking ──
    chunk_file = results_dir / "2_smart_chunks" / f"{binary_name}_chunks.json"
    if chunk_file.exists():
        chunks = read_json(chunk_file)
        lines.append("---")
        lines.append("## 6. Smart Chunking")
        lines.append("")
        lines.append(f"- **Total blocks:** {chunks.get('total_blocks', 0)}")
        max_tok = chunks.get("max_block_tokens", 0)
        lines.append(f"- **Max block tokens:** {max_tok}")
        lines.append("")

    # ── 7. Artifacts ──
    lines.append("---")
    lines.append("## 7. Output Artifacts")
    lines.append("")
    for step_dir_name in STEP_DIRS:
        step_dir = results_dir / step_dir_name
        if step_dir.exists():
            files_in_step = list(step_dir.iterdir())
            if files_in_step:
                lines.append(f"- `{step_dir_name}/` ({len(files_in_step)} file(s))")
    lines.append(f"- `{binary_name}.c` (decompiled source)")
    lines.append("")

    # ── Write ──
    summary_path = results_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"Summary written to: {summary_path}")


# =========================
# Main pipeline
# =========================


def run(input_pe: str, output_dir: str, ida_path: str) -> bool:
    """
    Run the full PE-Sleuth pipeline on a single PE binary.

    Parameters:
        input_pe:   Path to the PE binary file.
        output_dir: Base directory for analysis reports.
        ida_path:   Path to IDA Pro installation directory.

    Returns:
        True if pipeline completed successfully.
    """
    pe_path = Path(input_pe).resolve()
    if not pe_path.exists():
        logging.error(f"PE file not found: {pe_path}")
        return False

    if not ida_path:
        logging.error(
            "IDA path not configured. Set 'ida_path' in settings.json, "
            "set $env:IDA_PATH, or pass --ida-path."
        )
        return False

    binary_name = pe_path.stem
    results_dir = Path(output_dir).resolve() / binary_name

    logging.info(f"{'='*60}")
    logging.info(f"PE-Sleuth Pipeline: {binary_name}")
    logging.info(f"  Input:  {pe_path}")
    logging.info(f"  Output: {results_dir}")
    logging.info(f"  IDA:    {ida_path}")
    logging.info(f"{'='*60}")

    # ── Step 0: Clean workspace ──
    clean_workspace()

    # ── Step 1: Copy PE to decompile input ──
    shutil.copy2(pe_path, DECOMPILE_INPUT_DIR / pe_path.name)
    logging.info(f"Copied {pe_path.name} -> decompile input.")

    # ── Step 2: Decompile ──
    if not run_decompile(ida_path):
        logging.error("Decompilation stage failed.")
        return False

    # ── Step 3: Check decompile output ──
    c_file = DECOMPILE_OUTPUT_DIR / f"{binary_name}.c"
    if not c_file.exists():
        logging.error(f"Decompiled C file not found: {c_file}")
        return False
    logging.info(f"Decompiled output: {c_file} ({c_file.stat().st_size:,} bytes)")

    # ── Step 4: Copy .c to classify input ──
    shutil.copy2(c_file, CLASSIFY_INPUT_DIR / c_file.name)
    logging.info(f"Copied {c_file.name} -> classify input.")

    # ── Step 5: Classify ──
    if not run_classify():
        logging.error("Classification stage failed.")
        return False

    # ── Step 6: Collect results ──
    collect_results(binary_name, results_dir)
    logging.info(f"Results collected to: {results_dir}")

    # ── Step 7: Generate summary report ──
    generate_summary(binary_name, results_dir)

    logging.info(f"{'='*60}")
    logging.info(f"Pipeline complete!")
    logging.info(f"  Report: {results_dir / 'summary.md'}")
    logging.info(f"  Visual: {results_dir / '3_visualizations' / f'{binary_name}_callgraph_chunks.html'}")
    logging.info(f"{'='*60}")
    return True


# =========================
# CLI entry
# =========================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(
        description="PE-Sleuth: single-file PE analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python pipeline.py -i sample.exe -o ./reports\n"
            '  python pipeline.py -i malware.exe -o ./reports --ida-path "D:\\IDA"'
        ),
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the PE binary file to analyze.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Base directory for analysis reports. A subfolder named after the binary will be created.",
    )
    parser.add_argument(
        "--ida-path",
        default=get_ida_path(),
        help="Path to IDA Pro installation directory (reads from settings.json 'ida_path' or $env:IDA_PATH).",
    )

    args = parser.parse_args()

    success = run(
        input_pe=args.input,
        output_dir=args.output,
        ida_path=args.ida_path,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
