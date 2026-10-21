#!/usr/bin/env python3
"""Spike detection and localisation for high-density MEA recordings (3Brain BioCam, .brw).

Detection and localisation are done by HerdingSpikes (https://github.com/mhhennig/HS2), software
developed by other researchers (Muthmann et al., 2015; Hilgen et al., 2017). This script only calls it;
the algorithm is not implemented here.

Steps
    1. Load the recording and flag noisy channels (std above a threshold in a short segment).
    2. Run HerdingSpikes detection with the noisy channels masked.
    3. Export the spikes to CSV, draw a raster plot and run basic quality control
       (channels with very few events, example waveforms).

Example
    python src/detect_spikes.py --brw data/Slc01_20min-4AP_240425.brw --out-dir outputs
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are saved to disk, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BIOCAM_FS_HZ = 7062.0  # BioCam sampling rate
N_CHANNELS = 4096      # BioCam electrodes


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def find_noisy_channels(probe, t_start: int, t_end: int, std_threshold: float) -> list:
    """Channels whose signal std (in a short segment) exceeds `std_threshold`."""
    n_samples = t_end - t_start
    segment = np.asarray(probe.Read(t_start, t_end)).reshape(n_samples, probe.num_channels)
    return np.where(np.std(segment, axis=0) > std_threshold)[0].tolist()


def spikes_to_dataframe(spikes: pd.DataFrame, save_waveforms: bool = False) -> pd.DataFrame:
    """Convert HerdingSpikes' `spikes` table to a tidy table.

    Output columns: channel, time (samples), amplitude, x, y [, waveform].
    """
    def column(name):
        if name in spikes.columns:
            return spikes[name].to_numpy()
        return np.full(len(spikes), np.nan)

    df = pd.DataFrame({
        "channel": column("ch"),
        "time": column("t"),
        "amplitude": column("Amplitude"),
        "x": column("x"),
        "y": column("y"),
    })
    if save_waveforms and "Shape" in spikes.columns:
        df["waveform"] = [" ".join(f"{v:.1f}" for v in np.asarray(s)) for s in spikes["Shape"]]
    return df


def find_low_activity_channels(df: pd.DataFrame, min_events: int = 5) -> list:
    """Channels that detected at least one spike but fewer than `min_events`."""
    counts = df["channel"].value_counts()
    return sorted(int(c) for c in counts[counts < min_events].index)


def plot_raster(df, masked_channels, fs_hz, out_path):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(df["time"] / fs_hz, df["channel"], ".", color="black", markersize=1)
    ax.plot(np.zeros(len(masked_channels)), masked_channels, "o", color="red",
            markersize=4, label="masked channels")
    ax.set(xlabel="Time (s)", ylabel="Channel", ylim=(0, N_CHANNELS))
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_example_waveforms(spikes, n_channels, max_waveforms, out_path, seed=0):
    """Overlay waveforms of the `n_channels` most active channels (random subset per channel)."""
    if n_channels <= 0 or "Shape" not in spikes.columns:
        return
    rng = np.random.default_rng(seed)
    top_channels = spikes["ch"].value_counts().index[:n_channels]
    n_cols = 5
    n_rows = int(np.ceil(len(top_channels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), squeeze=False)

    for ax, channel in zip(axes.ravel(), top_channels):
        shapes = spikes.loc[spikes["ch"] == channel, "Shape"].to_list()
        if len(shapes) > max_waveforms:
            keep = rng.choice(len(shapes), max_waveforms, replace=False)
            shapes = [shapes[i] for i in keep]
        for shape in shapes:
            ax.plot(np.asarray(shape), alpha=0.4, lw=0.7)
        ax.set_title(f"channel {channel}", fontsize=9)
        ax.set(xlabel="Samples", ylabel="Amplitude")
    for ax in axes.ravel()[len(top_channels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brw", required=True, help="Path to the BioCam recording (.brw)")
    p.add_argument("--out-dir", default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--threshold", type=float, default=30, help="Spike detection threshold (default: 30)")
    p.add_argument("--no-localize", action="store_true",
                   help="Only detect spikes, do not localise them (localisation is needed for spike sorting)")
    p.add_argument("--masked-channels", type=int, nargs="*", default=None,
                   help="Channels to mask. If omitted, noisy channels are detected automatically")
    p.add_argument("--noise-threshold", type=float, default=15,
                   help="Std threshold used to detect noisy channels automatically (default: 15)")
    p.add_argument("--noise-window", type=int, nargs=2, default=[1_000_000, 1_100_000],
                   metavar=("START", "END"), help="Sample range used to look for noisy channels")
    p.add_argument("--left-cutout-ms", type=float, default=5.0, help="Waveform ms before the spike peak")
    p.add_argument("--right-cutout-ms", type=float, default=5.0, help="Waveform ms after the spike peak")
    p.add_argument("--amp-eval-ms", type=float, default=0.4, help="Window (ms) used to evaluate amplitude")
    p.add_argument("--spk-eval-ms", type=float, default=1.7, help="Window (ms) used to evaluate spikes")
    p.add_argument("--fs", type=float, default=BIOCAM_FS_HZ, help="Sampling rate in Hz (default: 7062)")
    p.add_argument("--save-waveforms", action="store_true",
                   help="Also store spike waveforms in the CSV (makes the file much larger)")
    p.add_argument("--waveform-channels", type=int, default=20,
                   help="Number of most active channels to plot waveforms for (0 = skip, default: 20)")
    return p.parse_args()


def main():
    args = parse_args()

    # HerdingSpikes is imported here so the helpers above work without it installed
    from herdingspikes.hs2 import HSDetection
    from herdingspikes.probe import BioCam

    brw_file = Path(args.brw)
    if not brw_file.exists():
        raise SystemExit(f"Recording not found: {brw_file}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = brw_file.stem

    # 1. Masked channels -----------------------------------------------------
    probe = BioCam(str(brw_file))
    print(f"Recording: {brw_file} | channels: {probe.num_channels}")

    if args.masked_channels is None:
        t_start, t_end = args.noise_window
        masked = find_noisy_channels(probe, t_start, t_end, args.noise_threshold)
        print(f"Noisy channels detected automatically (std > {args.noise_threshold}): {len(masked)}")
    else:
        masked = sorted(set(args.masked_channels))
        print(f"Using {len(masked)} user-provided masked channels")
    print("Masked channels:", ", ".join(map(str, masked)) or "none")

    probe = BioCam(str(brw_file), masked_channels=masked)

    # 2. Detection -------------------------------------------------------------
    detector = HSDetection(
        probe,
        num_com_centers=1,
        threshold=args.threshold,
        to_localize=not args.no_localize,
        maa=0,
        ahpthr=0,
        out_file_name=base,
        file_directory_name=str(out_dir),
        save_all=True,
        left_cutout_time=args.left_cutout_ms,
        right_cutout_time=args.right_cutout_ms,
        amp_evaluation_time=args.amp_eval_ms,
        spk_evaluation_time=args.spk_eval_ms,
        decay_filtering=True,
    )
    detector.DetectFromRaw()
    detector.LoadDetected()
    spikes = detector.spikes
    print(f"Spikes detected: {len(spikes)}")

    # 3. Export and quality control ------------------------------------------
    df = spikes_to_dataframe(spikes, save_waveforms=args.save_waveforms)
    csv_path = out_dir / f"{base}_spikes.csv"
    df.to_csv(csv_path, index=False)
    print(f"Spikes saved to: {csv_path}")

    raster_path = out_dir / f"{base}_raster_thr{args.threshold:g}.png"
    plot_raster(df, masked, args.fs, raster_path)
    print(f"Raster plot saved to: {raster_path}")

    low_activity = find_low_activity_channels(df, min_events=5)
    if low_activity:
        print(f"Channels with fewer than 5 events ({len(low_activity)}): {', '.join(map(str, low_activity))}")
        (out_dir / f"{base}_low_activity_channels.txt").write_text("\n".join(map(str, low_activity)))
    else:
        print("No low-activity channels.")

    plot_example_waveforms(spikes, args.waveform_channels, max_waveforms=100,
                           out_path=out_dir / f"{base}_example_waveforms.png")


if __name__ == "__main__":
    main()
