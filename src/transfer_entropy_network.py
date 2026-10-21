#!/usr/bin/env python3
"""Transfer Entropy (TE) network analysis of MEA spike trains.

Input : CSV written by `detect_spikes.py` (columns `channel` and `time`, time in samples).
Steps
    1. Bin the spikes into a binary matrix (active channels x time bins).
    2. Compute directed TE between every pair of channels, with a circular-shift
       surrogate test for each pair.
    3. Keep significant edges (uncorrected and Benjamini-Hochberg FDR).
    4. Build a directed graph and compare its topology (shortest path, degree, clustering,
       modularity, eigenvector centrality) against degree-preserving random networks (z-scores).

Computing TE for all pairs is expensive. For a quick test use `--max-channels` and fewer surrogates.

Example
    python src/transfer_entropy_network.py --spikes outputs/Slc01_20min-4AP_240425_spikes.csv
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from networkx.algorithms.community import greedy_modularity_communities, modularity
from pyinform.transferentropy import transfer_entropy
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_spikes(path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["channel", "time"])


def build_binned_spike_matrix(spike_df, fs_original, fs_target, max_channels=None):
    """Binary spike matrix (active channels x time bins) at `fs_target`.

    Spike times are converted from samples (`fs_original`) to bins (`fs_target`); several spikes
    falling in the same bin count as one. Silent channels are dropped. If `max_channels` is set,
    only the most active channels are kept.

    Returns the matrix, the original channel id of each row and the number of spikes kept.
    """
    channels = spike_df["channel"].to_numpy(dtype=int)
    bins = np.floor(spike_df["time"].to_numpy() * fs_target / fs_original).astype(np.int64)
    n_bins = int(bins.max()) + 1

    ids, counts = np.unique(channels, return_counts=True)
    if max_channels is not None and max_channels < len(ids):
        keep = ids[np.argsort(counts)[::-1][:max_channels]]
        mask = np.isin(channels, keep)
        channels, bins = channels[mask], bins[mask]

    channel_ids, rows = np.unique(channels, return_inverse=True)
    matrix = np.zeros((len(channel_ids), n_bins), dtype=bool)
    matrix[rows, bins] = True
    return matrix, channel_ids, len(channels)


def plot_raster(matrix, channel_ids, fs_target, out_path):
    rows, bins = np.where(matrix)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bins / fs_target, channel_ids[rows], ".", markersize=1, color="black")
    ax.set(xlabel="Time (s)", ylabel="Channel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Transfer entropy
# --------------------------------------------------------------------------- #
def compute_pair_te(i, j, spike_matrix, k, n_surrogates, seed):
    """TE from channel i to channel j and its surrogate-based p-value.

    Surrogates are circular shifts of the source: they keep its statistics but destroy its
    temporal relation with the target.
    """
    source, target = spike_matrix[i], spike_matrix[j]
    real_te = transfer_entropy(source, target, k=k)

    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, len(source), size=n_surrogates)
    surrogate_te = np.array([transfer_entropy(np.roll(source, s), target, k=k) for s in shifts])

    p_value = (np.sum(surrogate_te >= real_te) + 1) / (n_surrogates + 1)
    return i, j, real_te, p_value


def compute_te_network(spike_matrix, k=1, n_surrogates=300, alpha=0.05, n_jobs=-1, seed=42):
    """Directed TE matrices; entry [i, j] is the flow from channel i to channel j.

    Returns (te_raw, te_naive, te_fdr, p_values):
        te_raw    all TE values
        te_naive  TE values with p <= alpha (no multiple-comparison correction)
        te_fdr    TE values that survive Benjamini-Hochberg FDR correction
        p_values  surrogate p-values (1 on the diagonal)
    """
    n = spike_matrix.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    jobs = (delayed(compute_pair_te)(i, j, spike_matrix, k, n_surrogates, seed + i * n + j)
            for i, j in pairs)
    results = list(tqdm(Parallel(n_jobs=n_jobs, return_as="generator")(jobs),
                        total=len(pairs), desc=f"Transfer entropy (k={k})"))

    te_raw = np.zeros((n, n))
    p_values = np.ones((n, n))
    for i, j, te, p in results:
        te_raw[i, j] = te
        p_values[i, j] = p

    te_naive = te_raw.copy()
    te_naive[p_values > alpha] = 0

    off_diagonal = ~np.eye(n, dtype=bool)
    significant, _ = fdrcorrection(p_values[off_diagonal], alpha=alpha, method="indep")
    fdr_mask = np.zeros((n, n), dtype=bool)
    fdr_mask[off_diagonal] = significant
    te_fdr = np.where(fdr_mask, te_raw, 0.0)

    return te_raw, te_naive, te_fdr, p_values


# --------------------------------------------------------------------------- #
# Graph analysis
# --------------------------------------------------------------------------- #
def graph_metrics(G: nx.DiGraph) -> dict:
    """Topology metrics of a directed graph."""
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(largest_scc)

    metrics = {
        "avg_shortest_path": (nx.average_shortest_path_length(G_scc) if len(G_scc) > 1 else np.nan),
        "avg_degree": np.mean([degree for _, degree in G.degree()]),
        "avg_clustering": np.mean(list(nx.clustering(G).values())),
        "modularity": modularity(G, greedy_modularity_communities(G.to_undirected())),
    }
    try:
        metrics["avg_eigenvector"] = np.mean(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
    except nx.PowerIterationFailedConvergence:
        metrics["avg_eigenvector"] = np.nan
    return metrics


def compare_with_null_models(G: nx.DiGraph, n_null=2000, seed=42) -> pd.DataFrame:
    """Compare graph metrics with degree-preserving random networks (directed configuration model)."""
    real = graph_metrics(G)
    in_degrees = [degree for _, degree in G.in_degree()]
    out_degrees = [degree for _, degree in G.out_degree()]

    null = {name: [] for name in real}
    for s in tqdm(range(n_null), desc="Null networks", leave=False):
        G_rand = nx.DiGraph(nx.directed_configuration_model(in_degrees, out_degrees, seed=seed + s))
        G_rand.remove_edges_from(list(nx.selfloop_edges(G_rand)))
        for name, value in graph_metrics(G_rand).items():
            null[name].append(value)

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, value in real.items():
            mean, std = np.nanmean(null[name]), np.nanstd(null[name])
            z_score = (value - mean) / std if std > 0 else np.nan
            rows.append({"metric": name, "real": value, "null_mean": mean,
                         "null_std": std, "z_score": z_score})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spikes", required=True, help="Spike CSV produced by detect_spikes.py")
    p.add_argument("--out-dir", default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--fs-original", type=float, default=7062, help="Sampling rate of the spike times, Hz")
    p.add_argument("--fs-target", type=float, default=1000, help="Sampling rate after binning, Hz")
    p.add_argument("--k", type=int, default=1, help="History length of the TE (default: 1)")
    p.add_argument("--n-surrogates", type=int, default=300, help="Surrogates per channel pair (default: 300)")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    p.add_argument("--n-null", type=int, default=2000, help="Random networks for z-scores (default: 2000)")
    p.add_argument("--max-channels", type=int, default=None,
                   help="Keep only the N most active channels (useful for quick tests)")
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1 = all cores)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.spikes).stem.replace("_spikes", "")

    # 1. Spike matrix -----------------------------------------------------------
    spike_df = load_spikes(args.spikes)
    matrix, channel_ids, n_spikes = build_binned_spike_matrix(
        spike_df, args.fs_original, args.fs_target, args.max_channels)
    n_channels, n_bins = matrix.shape
    duration_s = n_bins / args.fs_target
    n_binned = int(matrix.sum())

    print(f"Spikes in CSV: {len(spike_df)} | spikes used: {n_spikes} | after binning: {n_binned}")
    print(f"Active channels: {n_channels} | duration: {duration_s:.1f} s")
    print(f"Population firing rate: {n_binned / duration_s:.2f} Hz (sum over channels)")
    print(f"Mean firing rate per channel: {n_binned / duration_s / n_channels:.3f} Hz")
    plot_raster(matrix, channel_ids, args.fs_target, out_dir / f"{prefix}_raster_binned.png")

    # 2. Transfer entropy -------------------------------------------------------
    te_raw, te_naive, te_fdr, p_values = compute_te_network(
        matrix, k=args.k, n_surrogates=args.n_surrogates,
        alpha=args.alpha, n_jobs=args.n_jobs, seed=args.seed)

    np.savez_compressed(
        out_dir / f"{prefix}_te_network.npz",
        channel_ids=channel_ids,
        te_raw=te_raw.astype(np.float32),
        te_fdr=te_fdr.astype(np.float32),
        p_values=p_values.astype(np.float32))

    n_possible = n_channels * (n_channels - 1)
    n_edges_naive = np.count_nonzero(te_naive)
    n_edges_fdr = np.count_nonzero(te_fdr)
    print(f"Significant edges (uncorrected p <= {args.alpha}): {n_edges_naive}")
    print(f"Significant edges (FDR): {n_edges_fdr} | density: {n_edges_fdr / n_possible:.4f}")
    if n_edges_fdr == 0:
        print("No edge survived FDR correction: nothing to analyse. "
              "Try more surrogates (--n-surrogates) or a less strict --alpha.")
        return
    print(f"Mean TE of significant edges: {te_fdr[te_fdr > 0].mean():.6f}")

    # 3. Graph analysis -----------------------------------------------------------
    G = nx.from_numpy_array(te_fdr, create_using=nx.DiGraph)
    print(f"Graph | nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")

    results = compare_with_null_models(G, n_null=args.n_null, seed=args.seed)
    results.to_csv(out_dir / f"{prefix}_network_metrics.csv", index=False)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
