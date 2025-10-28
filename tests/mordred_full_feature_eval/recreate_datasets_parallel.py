#!/usr/bin/env python3
"""
High-throughput, parallel dataset recreation with full Mordred descriptors (2D and 2D+3D).

This alternative script focuses on throughput and memory efficiency on multi-core HPC nodes.
It recreates two directory trees mirroring datasets/molecular_function_features_fingerprints/:
  - datasets_2d_all/:   Full 2D Mordred descriptors (1613+ features)
  - datasets_2d3d_all/: Full 2D+3D Mordred descriptors (1826+ features)

Key improvements vs the baseline script (recreate_datasets.py):
  - Process-level parallelism (multiprocessing) with controlled per-process thread usage
  - Chunked CSV reading + batched descriptor computation to keep memory bounded
  - Single-pass SMILES parsing per chunk; selective left-join to avoid large merges
  - Stable schema across chunks (descriptor columns aligned) with early header discovery
  - Fingerprint filtering in streaming mode

Designed to run on 32-64 cores with 100-300 GB RAM. Use --workers to cap CPU usage.

Author: Alternative parallel implementation
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# Mordred
from mordred import Calculator, descriptors


# ---------------------------
# RDKit helpers
# ---------------------------
def smiles_to_rdkit_mol(s: str) -> Optional[Chem.Mol]:
    """Parse SMILES into a sanitized RDKit Mol (2D)."""
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def embed_3d(mol: Chem.Mol, seed: int = 42, max_attempts: int = 3) -> Optional[Chem.Mol]:
    """Generate a 3D conformer using ETKDG. Returns None on failure.

    Note: To prevent CPU oversubscription when using multiprocessing, we set
    params.numThreads = 1 and parallelize across processes instead.
    """
    try:
        m = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.numThreads = 1  # avoid oversubscription across processes

        for _ in range(max_attempts):
            code = AllChem.EmbedMolecule(m, params)
            if code == 0:
                # Optional: quick geometry optimization is often helpful
                try:
                    AllChem.UFFOptimizeMolecule(m, maxIters=200)
                except Exception:
                    pass
                return m
        return None
    except Exception:
        return None


# ---------------------------
# Mordred workers (multiprocessing)
# ---------------------------
_CALC_CACHE: dict[str, Calculator] = {}


def _get_calc(use_3d: bool) -> Calculator:
    key = "3d" if use_3d else "2d"
    calc = _CALC_CACHE.get(key)
    if calc is None:
        calc = Calculator(descriptors, ignore_3D=not use_3d)
        _CALC_CACHE[key] = calc
    return calc


def _worker_compute_batch(args: tuple[list[str], bool, int]) -> tuple[pd.DataFrame, list[str]]:
    """Worker: compute Mordred descriptors for a batch of SMILES.

    Returns (descriptor_df, failed_smiles). descriptor_df has a 'smiles' column first.
    """
    smiles_batch, use_3d, seed = args

    mols: list[Optional[Chem.Mol]] = []
    smiles_ok: list[str] = []
    failed: list[str] = []

    for s in smiles_batch:
        m = smiles_to_rdkit_mol(s)
        if m is None:
            failed.append(s)
            continue
        if use_3d:
            m3d = embed_3d(m, seed=seed)
            if m3d is None:
                failed.append(s)
                continue
            mols.append(m3d)
        else:
            mols.append(m)
        smiles_ok.append(s)

    if not mols:
        return pd.DataFrame(columns=["smiles"]), failed

    try:
        calc = _get_calc(use_3d)
        df = calc.pandas(mols)
    except Exception:
        # Fallback per-molecule to isolate problematic molecules
        rows = []
        new_failed = []
        calc = _get_calc(use_3d)
        for m, s in zip(mols, smiles_ok):
            try:
                r = calc(m)
                rows.append(r)
            except Exception:
                new_failed.append(s)
        failed.extend(new_failed)
        if rows:
            df = pd.DataFrame(rows)
        else:
            return pd.DataFrame(columns=["smiles"]), failed

    # to numeric; drop non-numeric columns
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.select_dtypes(include=[np.number])
    df.insert(0, "smiles", smiles_ok)
    # unique by smiles within this batch
    df = df.drop_duplicates(subset=["smiles"], keep="last")
    return df, failed


@dataclass
class Schema:
    descriptor_cols: list[str]


def discover_schema(smiles_sample: list[str], use_3d: bool, seed: int) -> Schema:
    """Compute descriptors for a tiny sample to lock the descriptor column set/order."""
    df, _ = _worker_compute_batch((smiles_sample, use_3d, seed))
    # First column is 'smiles'
    cols = [c for c in df.columns if c != "smiles"]
    return Schema(descriptor_cols=cols)


def compute_descriptors_parallel(
    smiles: Sequence[str],
    use_3d: bool,
    seed: int,
    workers: int,
    batch_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute Mordred for a list of SMILES using multiprocessing.

    Returns a concatenated DataFrame with 'smiles' + descriptor columns and a list of failed SMILES.
    """
    batches: list[list[str]] = [list(smiles[i : i + batch_size]) for i in range(0, len(smiles), batch_size)]
    failed_all: list[str] = []
    out_frames: list[pd.DataFrame] = []

    with mp.Pool(processes=workers, maxtasksperchild=100) as pool:
        for df, failed in tqdm(
            pool.imap_unordered(
                _worker_compute_batch,
                ((b, use_3d, seed) for b in batches),
                chunksize=1,
            ),
            total=len(batches),
            desc=f"Mordred {'3D' if use_3d else '2D'} (parallel)",
            leave=False,
        ):
            if not df.empty:
                out_frames.append(df)
            if failed:
                failed_all.extend(failed)

    if out_frames:
        all_df = pd.concat(out_frames, axis=0, ignore_index=True)
        all_df = all_df.drop_duplicates(subset=["smiles"], keep="last")
    else:
        all_df = pd.DataFrame(columns=["smiles"])  # empty
    return all_df, failed_all


# ---------------------------
# IO helpers
# ---------------------------
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def append_chunk(
    out_path: Path,
    df: pd.DataFrame,
    header: Optional[list[str]] = None,
    write_header_if_new: bool = True,
) -> None:
    """Append a dataframe to CSV, optionally enforcing a fixed column order (header).

    If header is provided, missing columns will be added and extra ones dropped.
    """
    if header is not None:
        for c in header:
            if c not in df.columns:
                df[c] = np.nan
        df = df[header]

    ensure_parent(out_path)
    write_header = write_header_if_new and (not out_path.exists())
    df.to_csv(out_path, mode="a", index=False, header=write_header)


def filter_fingerprints_by_features(
    fp_path: Path,
    feature_path: Path,
    output_path: Path,
    smiles_col: str = "SMILES",
) -> int:
    """Stream-filter a fingerprint CSV to only include rows whose SMILES appear in feature CSV."""
    if not fp_path.exists() or not feature_path.exists():
        return 0

    try:
        keep_smiles = set(
            pd.read_csv(feature_path, usecols=[smiles_col], low_memory=False)[smiles_col]
            .dropna()
            .astype(str)
            .tolist()
        )

        total_kept = 0
        first = True
        for chunk in pd.read_csv(fp_path, chunksize=50_000, low_memory=False):
            if smiles_col not in chunk.columns:
                continue
            kept = chunk[chunk[smiles_col].isin(keep_smiles)].copy()
            if not kept.empty:
                ensure_parent(output_path)
                kept.to_csv(output_path, mode="a", index=False, header=first)
                first = False
                total_kept += len(kept)
            del kept, chunk
            gc.collect()
        if total_kept == 0:
            # write empty header to keep structure
            header_df = pd.read_csv(fp_path, nrows=0)
            ensure_parent(output_path)
            header_df.to_csv(output_path, index=False)
        return total_kept
    except Exception:
        return 0


# ---------------------------
# Pipeline
# ---------------------------
def process_zinc(
    datasets_dir: Path,
    out_2d_dir: Path,
    out_2d3d_dir: Path,
    seed: int,
    workers: int,
    chunk_size: int,
    batch_2d: int,
    batch_3d: int,
    limit_zinc: Optional[int] = None,
) -> tuple[int, int, int]:
    """Process ZINC file: returns (initial_count, delivered_2d, delivered_2d3d)."""
    zinc_src = datasets_dir / "zinc" / "zinc_acquirable_extracted_features.csv"
    if not zinc_src.exists():
        print(f"  ⚠ ZINC not found: {zinc_src}")
        return 0, 0, 0

    meta_cols = ["ZINC_ID", "SMILES", "LABEL", "MANUFACTURER", "TRANCHE"]
    out_2d = out_2d_dir / "zinc" / "zinc_acquirable_extracted_features.csv"
    out_3d = out_2d3d_dir / "zinc" / "zinc_acquirable_extracted_features.csv"

    # initial lines (excluding header)
    initial_count = sum(1 for _ in open(zinc_src)) - 1
    if limit_zinc:
        initial_effective = min(limit_zinc, initial_count)
        print(f"  ZINC molecules: {initial_count:,} (limiting to {initial_effective:,})")
    else:
        initial_effective = initial_count
        print(f"  ZINC molecules: {initial_count:,}")

    delivered_2d = 0
    delivered_3d = 0
    seen_header_2d = False
    seen_header_3d = False
    schema_2d: Optional[Schema] = None
    schema_3d: Optional[Schema] = None

    processed = 0
    reader = pd.read_csv(zinc_src, chunksize=chunk_size, low_memory=False)
    for chunk_idx, df in enumerate(reader, 1):
        if limit_zinc and processed >= initial_effective:
            break
        if limit_zinc and processed + len(df) > initial_effective:
            df = df.iloc[: initial_effective - processed].copy()

        smiles = df["SMILES"].dropna().astype(str).tolist()
        processed += len(df)

        # Early header discovery on first non-empty batch per mode
        sample = smiles[: min(32, len(smiles))]
        if sample:
            if schema_2d is None:
                schema_2d = discover_schema(sample, use_3d=False, seed=seed)
            if schema_3d is None:
                schema_3d = discover_schema(sample, use_3d=True, seed=seed)

        # 2D descriptors
        desc2d, _ = compute_descriptors_parallel(
            smiles=smiles,
            use_3d=False,
            seed=seed,
            workers=workers,
            batch_size=batch_2d,
        )
        if not desc2d.empty:
            tmp = df[[c for c in meta_cols if c in df.columns]].merge(
                desc2d, left_on="SMILES", right_on="smiles", how="inner"
            )
            if "smiles" in tmp.columns:
                tmp.drop(columns=["smiles"], inplace=True)
            # Arrange columns: metadata + descriptors in fixed order
            if schema_2d:
                descriptor_cols = [c for c in schema_2d.descriptor_cols if c in tmp.columns]
                ordered = [c for c in meta_cols if c in tmp.columns] + descriptor_cols
                tmp = tmp[ordered]
            append_chunk(out_2d, tmp, header=list(tmp.columns), write_header_if_new=not seen_header_2d)
            seen_header_2d = True
            delivered_2d += len(tmp)
            del tmp
        del desc2d
        gc.collect()

        # 2D+3D descriptors
        desc3d, _ = compute_descriptors_parallel(
            smiles=smiles,
            use_3d=True,
            seed=seed,
            workers=workers,
            batch_size=batch_3d,
        )
        if not desc3d.empty:
            tmp = df[[c for c in meta_cols if c in df.columns]].merge(
                desc3d, left_on="SMILES", right_on="smiles", how="inner"
            )
            if "smiles" in tmp.columns:
                tmp.drop(columns=["smiles"], inplace=True)
            if schema_3d:
                descriptor_cols = [c for c in schema_3d.descriptor_cols if c in tmp.columns]
                ordered = [c for c in meta_cols if c in tmp.columns] + descriptor_cols
                tmp = tmp[ordered]
            append_chunk(out_3d, tmp, header=list(tmp.columns), write_header_if_new=not seen_header_3d)
            seen_header_3d = True
            delivered_3d += len(tmp)
            del tmp
        del desc3d
        gc.collect()

        print(f"    Chunk {chunk_idx}: 2D+={delivered_2d:,}  2D+3D+={delivered_3d:,}")
        del df
        gc.collect()

    # Filter fingerprints to match delivered features
    fp_src = datasets_dir / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"
    if fp_src.exists():
        out_fp_2d = out_2d_dir / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"
        out_fp_3d = out_2d3d_dir / "zinc" / "zinc_acquirable_extracted_fingerprints_ECFP4.csv"
        n2 = filter_fingerprints_by_features(fp_src, out_2d, out_fp_2d, smiles_col="SMILES")
        n3 = filter_fingerprints_by_features(fp_src, out_3d, out_fp_3d, smiles_col="SMILES")
        print(f"  ZINC fingerprints filtered: 2D={n2:,}, 2D+3D={n3:,}")

    return initial_count, delivered_2d, delivered_3d


def process_kw_files(
    datasets_dir: Path,
    out_2d_dir: Path,
    out_2d3d_dir: Path,
    seed: int,
    workers: int,
    chunk_size: int,
    batch_2d: int,
    batch_3d: int,
    limit_kw: Optional[int] = None,
) -> tuple[int, int, int]:
    """Process all KW-*_affinity_extracted_features.csv files.

    Returns (initial_total, delivered_2d_total, delivered_3d_total).
    """
    kw_files = sorted(
        [f for f in datasets_dir.glob("KW-*.csv") if "_affinity_extracted_features.csv" in f.name]
    )
    print(f"  KW files: {len(kw_files)}")

    meta_cols = [
        "Compound ChEMBL ID",
        "SMILES",
        "Target ChEMBL ID",
        "Target Name",
        "Activity Type",
        "Standard Value (nM)",
        "target_chembl_id",
        "accession",
    ]

    initial_total = 0
    delivered_2d_total = 0
    delivered_3d_total = 0

    for i, path in enumerate(kw_files, 1):
        print(f"\n  [{i}/{len(kw_files)}] {path.name}")
        try:
            # Read file once; optionally limit number of rows
            if limit_kw:
                df = pd.read_csv(path, nrows=int(limit_kw), low_memory=False)
            else:
                df = pd.read_csv(path, low_memory=False)

            smiles_col = "SMILES" if "SMILES" in df.columns else ("canonical_smiles" if "canonical_smiles" in df.columns else None)
            if smiles_col is None:
                print("    ✗ No SMILES column found; skipping")
                continue

            smiles = df[smiles_col].dropna().astype(str).tolist()
            initial_total += len(smiles)

            # Discover schema on a small sample
            sample = smiles[: min(32, len(smiles))]
            schema2 = discover_schema(sample, use_3d=False, seed=seed)
            schema3 = discover_schema(sample, use_3d=True, seed=seed)

            # 2D
            desc2d, _ = compute_descriptors_parallel(
                smiles=smiles, use_3d=False, seed=seed, workers=workers, batch_size=batch_2d
            )
            if not desc2d.empty:
                meta = df[[c for c in meta_cols if c in df.columns]].copy()
                out = meta.merge(desc2d, left_on=smiles_col, right_on="smiles", how="inner")
                if "smiles" in out.columns:
                    out.drop(columns=["smiles"], inplace=True)
                # order columns
                descriptor_cols = [c for c in schema2.descriptor_cols if c in out.columns]
                ordered = [c for c in meta_cols if c in out.columns] + descriptor_cols
                out = out[ordered]
                out_path = out_2d_dir / path.name
                out.to_csv(out_path, index=False)
                delivered_2d_total += len(out)
                del out, meta
            del desc2d
            gc.collect()

            # 2D+3D
            desc3d, _ = compute_descriptors_parallel(
                smiles=smiles, use_3d=True, seed=seed, workers=workers, batch_size=batch_3d
            )
            if not desc3d.empty:
                meta = df[[c for c in meta_cols if c in df.columns]].copy()
                out = meta.merge(desc3d, left_on=smiles_col, right_on="smiles", how="inner")
                if "smiles" in out.columns:
                    out.drop(columns=["smiles"], inplace=True)
                descriptor_cols = [c for c in schema3.descriptor_cols if c in out.columns]
                ordered = [c for c in meta_cols if c in out.columns] + descriptor_cols
                out = out[ordered]
                out_path = out_2d3d_dir / path.name
                out.to_csv(out_path, index=False)
                delivered_3d_total += len(out)
                del out, meta
            del desc3d
            gc.collect()

            # Fingerprints
            fp_name = path.name.replace("_extracted_features.csv", "_extracted_fingerprints_ECFP4.csv")
            fp_src = datasets_dir / fp_name
            if fp_src.exists():
                out_fp2 = out_2d_dir / fp_name
                out_fp3 = out_2d3d_dir / fp_name
                kept2 = filter_fingerprints_by_features(fp_src, out_2d_dir / path.name, out_fp2, smiles_col="SMILES")
                kept3 = filter_fingerprints_by_features(fp_src, out_2d3d_dir / path.name, out_fp3, smiles_col="SMILES")
                print(f"    ✓ Fingerprints filtered: 2D={kept2:,}, 2D+3D={kept3:,}")
            else:
                print("    ⚠ Fingerprint source not found; skipping")

        except Exception as e:
            print(f"    ✗ Error: {e}")
        finally:
            del df
            gc.collect()

    return initial_total, delivered_2d_total, delivered_3d_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alternative parallel dataset recreation with full Mordred descriptors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_dir", type=str, default=".", help="Base dir containing datasets/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 3D embedding")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2), help="Worker processes")
    parser.add_argument("--chunk-size", type=int, default=50_000, help="CSV chunk size for ZINC")
    parser.add_argument("--batch-2d", type=int, default=1000, help="SMILES per worker batch (2D)")
    parser.add_argument("--batch-3d", type=int, default=250, help="SMILES per worker batch (3D)")
    parser.add_argument("--limit-zinc", type=int, default=None, help="Limit ZINC molecules (testing)")
    parser.add_argument("--limit-kw", type=int, default=None, help="Limit per KW file (testing)")

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    datasets_dir = base_dir / "datasets" / "molecular_function_features_fingerprints"
    out_2d = out_root / "datasets_2d_all"
    out_3d = out_root / "datasets_2d3d_all"
    (out_2d / "zinc").mkdir(parents=True, exist_ok=True)
    (out_3d / "zinc").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("ALTERNATIVE PARALLEL DATASET RECREATION (MORDRED)")
    print("=" * 80)
    print(f"Base dir:     {base_dir}")
    print(f"Output dir:   {out_root}")
    print(f"Workers:      {args.workers}")
    print(f"Chunk size:   {args.chunk_size}")
    print(f"Batch sizes:  2D={args.batch_2d}, 3D={args.batch_3d}")

    # ZINC
    print("\n[1/2] Processing ZINC...")
    init_zinc, zinc_2d, zinc_3d = process_zinc(
        datasets_dir=datasets_dir,
        out_2d_dir=out_2d,
        out_2d3d_dir=out_3d,
        seed=args.seed,
        workers=args.workers,
        chunk_size=args.chunk_size,
        batch_2d=args.batch_2d,
        batch_3d=args.batch_3d,
        limit_zinc=args.limit_zinc,
    )

    # KW
    print("\n[2/2] Processing KW molecular function files...")
    init_kw, kw_2d, kw_3d = process_kw_files(
        datasets_dir=datasets_dir,
        out_2d_dir=out_2d,
        out_2d3d_dir=out_3d,
        seed=args.seed,
        workers=args.workers,
        chunk_size=args.chunk_size,
        batch_2d=args.batch_2d,
        batch_3d=args.batch_3d,
        limit_kw=args.limit_kw,
    )

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)
    print(f"ZINC: requested={init_zinc:,},  2D={zinc_2d:,},  2D+3D={zinc_3d:,}")
    print(f"KW:   requested={init_kw:,},  2D={kw_2d:,},  2D+3D={kw_3d:,}")
    total_req = init_zinc + init_kw
    total_2d = zinc_2d + kw_2d
    total_3d = zinc_3d + kw_3d
    print(f"Total delivered: 2D={total_2d:,} ({(total_2d / total_req * 100) if total_req else 0:.2f}%), "
          f"2D+3D={total_3d:,} ({(total_3d / total_req * 100) if total_req else 0:.2f}%)")
    print("Output directories:")
    print(f"  2D:    {out_2d}")
    print(f"  2D+3D: {out_3d}")


if __name__ == "__main__":
    # Prefer 'fork' on Linux for RDKit; Python defaults to 'fork' on POSIX.
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()
