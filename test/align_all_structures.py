"""
Align all protein structures in a folder against a reference using TM-align.

Usage example:
    python align_all_structures.py \
        --input-dir /path/to/pdbs \
        --output-dir /path/to/output \
        --tmalign-bin /home/wangzeli/Lab/FrameDiff/test/TMalign/TM-align \
        --extension .pdb

By default the first file (sorted alphabetically) is taken as reference.
Use --reference to specify a custom reference structure.
The script writes TM-align stdout to per-target log files and a CSV summary.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


def list_structures(input_dir: Path, extension: str) -> List[Path]:
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == extension.lower())
    return files


def parse_tmalign_output(stdout: str) -> Dict[str, Optional[float]]:
    """Extract key metrics from TM-align stdout (robust to spacing/wrapping)."""
    tm_score_ref = None
    tm_score_chain = None
    rmsd = None
    aligned_len = None

    for line in stdout.splitlines():
        # TM-score lines look like: "TM-score= 0.30004 (if normalized by length of Chain_1...)"
        if line.startswith("TM-score") and "Chain_2" not in line and tm_score_ref is None:
            try:
                tm_score_ref = float(line.split("=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        if line.startswith("TM-score") and "Chain_2" in line and tm_score_chain is None:
            try:
                tm_score_chain = float(line.split("=")[1].split()[0])
            except (IndexError, ValueError):
                pass

        # RMSD and aligned length often appear on one line:
        # "Aligned length= 120, RMSD=   6.04, Seq_ID= ..."
        if "Aligned length" in line and "RMSD" in line:
            try:
                aligned_len = int(line.split("Aligned length=")[1].split(",")[0].strip())
            except (IndexError, ValueError):
                pass
            try:
                rmsd = float(line.split("RMSD=")[1].split(",")[0].strip())
            except (IndexError, ValueError):
                pass

        # Fallback RMSD line: "RMSD of the common residues = 3.45"
        if rmsd is None and "RMSD of the common residues" in line:
            try:
                rmsd = float(line.split("=")[1].split()[0])
            except (IndexError, ValueError):
                pass

    return {
        "tm_score_ref": tm_score_ref,
        "tm_score_chain": tm_score_chain,
        "rmsd": rmsd,
        "aligned_len": aligned_len,
    }


def cleanup_outputs(output_prefix: Path, keep_all_outputs: bool) -> None:
    """Remove all TM-align outputs except PDB (and summary CSV)."""
    if keep_all_outputs:
        return
    base = output_prefix.name
    for path in output_prefix.parent.glob(base + "*"):
        # Only keep PDB files.
        if path.suffix == ".pdb":
            continue
        # Everything else (including base with no suffix and .log) is removed.
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def run_tmalign(
    *,
    tmalign_bin: Path,
    target: Path,
    reference: Path,
    output_dir: Path,
    keep_all_outputs: bool,
) -> Dict[str, Optional[float]]:
    output_prefix = output_dir / f"{target.stem}_to_{reference.stem}"
    cmd = [str(tmalign_bin), str(target), str(reference), "-o", str(output_prefix)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"TM-align failed on {target.name}: {result.stderr.strip()}")
    metrics = parse_tmalign_output(result.stdout)
    log_path = output_dir / f"{target.stem}_to_{reference.stem}.log"
    log_path.write_text(result.stdout)
    cleanup_outputs(output_prefix, keep_all_outputs=keep_all_outputs)
    return metrics


def write_summary(summary_path: Path, rows: List[Dict[str, Optional[float]]]) -> None:
    fieldnames = [
        "target",
        "reference",
        "tm_score_ref",
        "tm_score_chain",
        "rmsd",
        "aligned_len",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ensure_tmalign_path(tmalign_bin: Path) -> Optional[Path]:
    if tmalign_bin.is_absolute() and tmalign_bin.exists():
        return tmalign_bin
    if tmalign_bin.exists():
        return tmalign_bin.resolve()
    resolved = shutil.which(str(tmalign_bin))
    return Path(resolved) if resolved else None


def _worker(args) -> Tuple[str, Optional[Dict[str, Optional[float]]], Optional[str]]:
    target, reference, tmalign_bin, output_dir, keep_all_outputs = args
    try:
        metrics = run_tmalign(
            tmalign_bin=tmalign_bin,
            target=target,
            reference=reference,
            output_dir=output_dir,
            keep_all_outputs=keep_all_outputs,
        )
        return target.name, metrics, None
    except Exception as exc:  # noqa: BLE001
        return target.name, None, str(exc)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Align all proteins in a folder using TM-align")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing structures (e.g., PDB files)")
    parser.add_argument("--output-dir", type=Path, default=Path("alignment_output"), help="Directory to write outputs")
    parser.add_argument("--tmalign-bin", type=Path, default=Path("./TMalign/TM-align"), help="Path to TM-align binary")
    parser.add_argument("--reference", type=Path, default=None, help="Optional reference structure. If omitted, the first file is used.")
    parser.add_argument("--extension", type=str, default=".pdb", help="File extension to include (e.g., .pdb)")
    parser.add_argument(
        "--keep-all-outputs",
        action="store_true",
        help="Keep all TM-align generated files (default keeps only PDB + log)",
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1), help="Number of parallel TM-align processes")

    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    tmalign_bin: Path = args.tmalign_bin
    reference: Optional[Path] = args.reference
    extension: str = args.extension
    keep_all_outputs: bool = bool(args.keep_all_outputs)
    workers: int = max(1, args.workers)

    resolved_tmalign = _ensure_tmalign_path(tmalign_bin)
    if resolved_tmalign is None:
        print(f"TM-align binary not found (checked: {tmalign_bin} and PATH)", file=sys.stderr)
        return 1
    tmalign_bin = resolved_tmalign
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    structures = list_structures(input_dir, extension)
    if not structures:
        print(f"No structures with extension {extension} in {input_dir}", file=sys.stderr)
        return 1

    if reference is None:
        reference = structures[0]
        targets = structures[1:]
    else:
        if reference not in structures:
            structures.append(reference)
        structures = sorted(structures)
        targets = [p for p in structures if p != reference]

    if not targets:
        print("Only reference structure found; nothing to align.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Optional[float]]] = []
    print("=" * 60)
    print(f"Reference structure: {reference.name}")
    print(f"TM-align: {tmalign_bin}")
    print(f"Workers: {workers}")
    print(f"Keep all TM-align outputs: {keep_all_outputs}")
    print("=" * 60)

    tasks = [(t, reference, tmalign_bin, output_dir, keep_all_outputs) for t in targets]
    errors: List[Tuple[str, str]] = []
    with mp.Pool(processes=workers) as pool, tqdm(total=len(targets), desc="Aligning", dynamic_ncols=True) as pbar:
        for target_name, metrics, err in pool.imap_unordered(_worker, tasks):
            if metrics is None:
                errors.append((target_name, err or "unknown error"))
                row = {
                    "target": target_name,
                    "reference": reference.name,
                    "tm_score_ref": None,
                    "tm_score_chain": None,
                    "rmsd": None,
                    "aligned_len": None,
                }
            else:
                row = {
                    "target": target_name,
                    "reference": reference.name,
                    **metrics,
                }
            summary_rows.append(row)
            pbar.update(1)
            if err:
                pbar.write(f"[error] {target_name}: {err}")

    summary_path = output_dir / "alignment_summary.csv"
    write_summary(summary_path, summary_rows)
    print(f"Summary written to {summary_path}")
    if errors:
        print(f"Completed with {len(errors)} errors. See above messages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
