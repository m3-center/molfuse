#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

# Default batch size mirrors project_and_analyze.py
CDIST_BATCH_SIZE_DEFAULT = 5000


def setup_logger(log_path: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode='a'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(message)s',
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark scipy.cdist vs exact NearestNeighbors (KDTree/BallTree) on real simspace data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--simspace_csv_path', required=True, help='Path to similarity space CSV (MF cloud + ZINC).')
    p.add_argument('--dr_short_name', required=True, help='DR short name, e.g., PCA or UMAP-... (prefix of coord columns).')
    p.add_argument('--simspace_dim', type=int, required=True, help='Number of coordinate dimensions to use.')
    p.add_argument('--actives_csv_path', default=None, help='Optional: path to projected actives CSV with same coord columns.')
    p.add_argument('--query_source', choices=['zinc', 'actives', 'both'], default='both', help='Which query set(s) to benchmark against MF cloud.')
    p.add_argument('--sample_mf', type=int, default=None, help='Optional: sample this many MF cloud points (random, without replacement).')
    p.add_argument('--sample_query', type=int, default=None, help='Optional: sample this many queries per source (zinc/actives).')
    p.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducible sampling.')
    p.add_argument('--batch_size', type=int, default=CDIST_BATCH_SIZE_DEFAULT, help='Batch size for cdist computation.')
    p.add_argument('--algorithm', choices=['auto', 'kd_tree', 'ball_tree'], default='auto', help='NN backend algorithm.')
    p.add_argument('--tolerance', type=float, default=1e-6, help='Absolute tolerance for distance equality check.')
    p.add_argument('--output_json', default='benchmark_cdist_vs_kdtree.json', help='Where to save JSON summary report.')
    p.add_argument('--save_sample_csv', default=None, help='Optional: save a small sample of paired distances for manual inspection.')
    p.add_argument('--log_path', default='analysis_scripts/benchmark_cdist_vs_kdtree.log', help='Log file path (appends).')
    return p.parse_args()


def get_coord_cols(dr_short_name: str, dim: int) -> list:
    return [f"{dr_short_name}-{i+1}" for i in range(dim)]


def split_mf_and_zinc(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if 'DataSource' in df.columns:
        mf_mask = df['DataSource'] == 'ChEMBL_MF'
        zinc_mask = df['DataSource'] == 'ZINC'
    elif 'MOLECULE ID' in df.columns:
        mf_mask = ~df['MOLECULE ID'].astype(str).str.startswith('ZINC', na=False)
        zinc_mask = df['MOLECULE ID'].astype(str).str.startswith('ZINC', na=False)
    else:
        logging.error("Neither 'DataSource' nor 'MOLECULE ID' present. Cannot split MF/ZINC.")
        return pd.DataFrame(), pd.DataFrame()
    return df[mf_mask].copy(), df[zinc_mask].copy()


def to_float32_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    sub = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
    return sub.values.astype(np.float32, copy=False)


def maybe_sample(arr: np.ndarray, n: Optional[int], seed: int) -> np.ndarray:
    if n is None or n <= 0 or arr.shape[0] <= n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[idx]


def min_cdist(query: np.ndarray, ref: np.ndarray, batch_size: int) -> np.ndarray:
    if query.size == 0 or ref.size == 0:
        return np.array([], dtype=np.float32)
    mins = np.empty(query.shape[0], dtype=np.float64)  # cdist returns float64
    w = 0
    for start in range(0, query.shape[0], batch_size):
        q_batch = query[start:start+batch_size]
        d = distance.cdist(q_batch, ref, metric='euclidean')
        mins[w:w+q_batch.shape[0]] = np.min(d, axis=1)
        w += q_batch.shape[0]
    return mins.astype(np.float32)


def min_nn(query: np.ndarray, ref: np.ndarray, algorithm: str) -> np.ndarray:
    if query.size == 0 or ref.size == 0:
        return np.array([], dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=1, algorithm=algorithm, metric='euclidean')
    nn.fit(ref)
    dists, _ = nn.kneighbors(query, return_distance=True)
    return dists.astype(np.float32).ravel()


def compare_distances(d1: np.ndarray, d2: np.ndarray, tol: float) -> Dict[str, float]:
    if d1.shape != d2.shape:
        return {
            'equal_length': False,
            'count': min(len(d1), len(d2)),
            'allclose': False,
            'max_abs_diff': float('inf'),
            'mean_abs_diff': float('inf'),
            'p99_abs_diff': float('inf'),
            'frac_exceeds_tol': 1.0,
        }
    diffs = np.abs(d1 - d2)
    if diffs.size == 0:
        return {
            'equal_length': True,
            'count': 0,
            'allclose': True,
            'max_abs_diff': 0.0,
            'mean_abs_diff': 0.0,
            'p99_abs_diff': 0.0,
            'frac_exceeds_tol': 0.0,
        }
    p99 = float(np.quantile(diffs, 0.99))
    return {
        'equal_length': True,
        'count': int(diffs.size),
        'allclose': bool(np.all(diffs <= tol)),
        'max_abs_diff': float(diffs.max()),
        'mean_abs_diff': float(diffs.mean()),
        'p99_abs_diff': p99,
        'frac_exceeds_tol': float((diffs > tol).mean()),
    }


def run_benchmark_for_queries(name: str,
                               queries: np.ndarray,
                               ref: np.ndarray,
                               batch_size: int,
                               algorithm: str,
                               tol: float,
                               save_sample_csv: Optional[str] = None,
                               sample_tag: Optional[str] = None) -> Dict[str, object]:
    result: Dict[str, object] = {
        'query_name': name,
        'num_queries': int(queries.shape[0]),
        'num_ref': int(ref.shape[0]),
        'cdist_time_sec': None,
        'nn_time_sec': None,
        'speedup_nn_over_cdist': None,
        'equality': None,
    }

    t0 = time.perf_counter()
    d_cdist = min_cdist(queries, ref, batch_size)
    t1 = time.perf_counter()
    result['cdist_time_sec'] = round(t1 - t0, 6)
    logging.info(f"[{name}] cdist computed for {len(d_cdist)} queries in {result['cdist_time_sec']} s")

    t2 = time.perf_counter()
    d_nn = min_nn(queries, ref, algorithm)
    t3 = time.perf_counter()
    result['nn_time_sec'] = round(t3 - t2, 6)
    logging.info(f"[{name}] NearestNeighbors ({algorithm}) computed for {len(d_nn)} queries in {result['nn_time_sec']} s")

    if result['cdist_time_sec'] and result['nn_time_sec'] and result['nn_time_sec'] > 0:
        result['speedup_nn_over_cdist'] = round(result['cdist_time_sec'] / result['nn_time_sec'], 4)

    eq = compare_distances(d_cdist, d_nn, tol)
    result['equality'] = eq
    logging.info(
        f"[{name}] equality: allclose={eq['allclose']}, max_abs_diff={eq['max_abs_diff']:.3e}, "
        f"mean_abs_diff={eq['mean_abs_diff']:.3e}, p99_abs_diff={eq['p99_abs_diff']:.3e}, "
        f"frac_exceeds_tol={eq['frac_exceeds_tol']:.3%}"
    )

    if save_sample_csv:
        try:
            sample_n = min(1000, len(d_cdist))
            if sample_n > 0:
                idx = np.linspace(0, len(d_cdist) - 1, sample_n, dtype=int)
                df_sample = pd.DataFrame({
                    'dist_cdist': d_cdist[idx],
                    'dist_nn': d_nn[idx],
                    'abs_diff': np.abs(d_cdist[idx] - d_nn[idx]),
                })
                base, ext = os.path.splitext(save_sample_csv)
                suffix = f"_{sample_tag}" if sample_tag else ""
                path = f"{base}{suffix}{ext or '.csv'}"
                df_sample.to_csv(path, index=False)
                logging.info(f"Saved sample distance comparison to {path} ({sample_n} rows)")
        except Exception as e:
            logging.warning(f"Failed to save sample CSV: {e}")

    return result


def main():
    args = parse_args()
    setup_logger(args.log_path)

    np.random.seed(args.random_seed)

    coord_cols = get_coord_cols(args.dr_short_name, args.simspace_dim)
    logging.info(f"Using coord cols: {coord_cols}")

    # Load simspace (MF + ZINC)
    if not os.path.exists(args.simspace_csv_path):
        logging.error(f"simspace_csv_path not found: {args.simspace_csv_path}")
        sys.exit(2)

    df_sim = pd.read_csv(args.simspace_csv_path, low_memory=False)
    missing = [c for c in coord_cols if c not in df_sim.columns]
    if missing:
        logging.error(f"Coord columns missing in simspace: {missing}")
        sys.exit(2)

    df_mf, df_zinc = split_mf_and_zinc(df_sim)
    logging.info(f"Loaded simspace: total={len(df_sim)}, MF={len(df_mf)}, ZINC={len(df_zinc)}")

    X_mf = to_float32_matrix(df_mf, coord_cols)
    if args.sample_mf:
        X_mf = maybe_sample(X_mf, args.sample_mf, args.random_seed)
    logging.info(f"MF cloud usable points (after dropna/sample): {X_mf.shape[0]}")

    # Prepare queries
    queries: Dict[str, np.ndarray] = {}

    if args.query_source in ('zinc', 'both'):
        X_zinc = to_float32_matrix(df_zinc, coord_cols)
        if args.sample_query:
            X_zinc = maybe_sample(X_zinc, args.sample_query, args.random_seed)
        logging.info(f"ZINC decoys usable points: {X_zinc.shape[0]}")
        queries['ZINC'] = X_zinc

    if args.query_source in ('actives', 'both') and args.actives_csv_path:
        if not os.path.exists(args.actives_csv_path):
            logging.error(f"actives_csv_path not found: {args.actives_csv_path}")
            sys.exit(2)
        df_act = pd.read_csv(args.actives_csv_path, low_memory=False)
        missing_act = [c for c in coord_cols if c not in df_act.columns]
        if missing_act:
            logging.error(f"Coord columns missing in actives CSV: {missing_act}")
            sys.exit(2)
        X_act = to_float32_matrix(df_act, coord_cols)
        if args.sample_query:
            X_act = maybe_sample(X_act, args.sample_query, args.random_seed)
        logging.info(f"Actives usable points: {X_act.shape[0]}")
        queries['ACTIVES'] = X_act
    elif args.query_source in ('actives', 'both') and not args.actives_csv_path:
        logging.warning("query_source requests 'actives' but --actives_csv_path not provided; skipping actives.")

    # Run benchmarks
    report = {
        'simspace_csv_path': args.simspace_csv_path,
        'actives_csv_path': args.actives_csv_path,
        'dr_short_name': args.dr_short_name,
        'simspace_dim': args.simspace_dim,
        'algorithm': args.algorithm,
        'batch_size': args.batch_size,
        'tolerance': args.tolerance,
        'mf_points': int(X_mf.shape[0]),
        'results': [],
    }

    if X_mf.shape[0] == 0:
        logging.error("MF cloud is empty after filtering. Nothing to benchmark.")
        sys.exit(2)

    for q_name, X_q in queries.items():
        res = run_benchmark_for_queries(
            name=q_name,
            queries=X_q,
            ref=X_mf,
            batch_size=args.batch_size,
            algorithm=args.algorithm,
            tol=args.tolerance,
            save_sample_csv=args.save_sample_csv,
            sample_tag=q_name.lower(),
        )
        report['results'].append(res)

    # Save JSON
    try:
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
        logging.info(f"Saved benchmark report to {args.output_json}")
    except Exception as e:
        logging.error(f"Failed to save JSON report {args.output_json}: {e}")

    # Print concise summary
    print("=== Benchmark Summary ===")
    print(f"MF points: {report['mf_points']} | dim={report['simspace_dim']} | alg={report['algorithm']} | batch={report['batch_size']}")
    for r in report['results']:
        eq = r['equality'] or {}
        print(
            f"{r['query_name']}: n={r['num_queries']} | cdist={r['cdist_time_sec']}s | nn={r['nn_time_sec']}s | "
            f"speedup={r['speedup_nn_over_cdist']} | allclose={eq.get('allclose')} | "
            f"maxΔ={eq.get('max_abs_diff'):.3e} | p99Δ={eq.get('p99_abs_diff'):.3e} | >tol={eq.get('frac_exceeds_tol'):.2%}"
        )


if __name__ == '__main__':
    main()
