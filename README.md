# High-density MEA: spike detection and Transfer Entropy network analysis

Two-step pipeline for high-density multielectrode array (MEA) recordings:

1. **Spike detection and localisation** with [HerdingSpikes](https://github.com/mhhennig/HS2) on a 3Brain BioCam recording (`.brw`, 4096 electrodes).
2. **Directed functional connectivity** between channels with **Transfer Entropy (TE)**, surrogate-based significance testing, FDR correction, and graph analysis compared against degree-preserving random networks.

## Credits and scope

This repository contains the **analysis pipeline** (glue code, quality control, binning, per-pair surrogate testing, FDR correction, graph construction and null-model comparison). It builds on third-party software that was developed by other people and is **not** part of this repository:

- **Spike detection and localisation:** [HerdingSpikes](https://github.com/mhhennig/HS2), developed by its authors (Muthmann et al., 2015; Hilgen et al., 2017). The algorithm is not reimplemented here: `detect_spikes.py` only calls it through its Python API. If you use this pipeline, please cite the HerdingSpikes papers and check the HerdingSpikes license.
- **Transfer Entropy estimator:** [PyInform](https://github.com/ELIFE-ASU/PyInform) (Moore et al., 2018). The TE measure itself is from Schreiber (2000).
- **Graph algorithms and statistics:** [NetworkX](https://networkx.org/) and [statsmodels](https://www.statsmodels.org/) (FDR correction).

The MIT license of this repository applies only to the code in this repository, not to these dependencies.

## Repository structure

```
.
├── src/
│   ├── detect_spikes.py            # step 1: .brw -> spikes CSV + raster + QC
│   └── transfer_entropy_network.py # step 2: spikes CSV -> TE network + graph metrics
├── data/                           # put the recording here (not versioned)
├── requirements.txt                # step 2 dependencies
├── requirements-detection.txt      # step 1 dependency (HerdingSpikes)
└── LICENSE
```

## Pipeline

### Step 1: `detect_spikes.py`

1. Looks for noisy channels (standard deviation above a threshold in a short segment) or uses a list you provide.
2. Runs HerdingSpikes detection with those channels masked.
3. Saves the spikes to CSV (`channel`, `time` in samples, `amplitude`, `x`, `y`), a raster plot, the list of channels with fewer than 5 events, and example waveforms of the most active channels.

### Step 2: `transfer_entropy_network.py`

1. Bins spikes (7062 Hz to 1 kHz) into a binary matrix of active channels x time bins.
2. Computes directed TE for every channel pair, using `k` = 1 by default.
3. Tests each pair with circular-shift surrogates (300 by default): `p = (#surrogates with TE >= observed + 1) / (n_surrogates + 1)`.
4. Keeps significant edges, both uncorrected and Benjamini-Hochberg FDR-corrected.
5. Builds the directed graph from the FDR-corrected TE matrix and computes average shortest path (largest strongly connected component), average degree, clustering, modularity and eigenvector centrality.
6. Compares each metric with 2000 random networks that preserve in- and out-degrees (z-scores).

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/<your-user>/mea-transfer-entropy-network.git
cd mea-transfer-entropy-network

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Step 2 (and plotting)
pip install -r requirements.txt

# Step 1 (HerdingSpikes needs a C++ compiler)
pip install numpy cython
pip install herdingspikes==0.3.100 --no-build-isolation
```

## Data

The example recording `Slc01_20min-4AP_240425.brw` (**33.33 GB**) is not included in this repository. See [`data/README.md`](data/README.md) for the download link and where to place it.

## Usage

**Step 1: spike detection**

```bash
python src/detect_spikes.py --brw data/Slc01_20min-4AP_240425.brw --out-dir outputs
```

By default, noisy channels are detected automatically. To mask specific channels instead (for example, the ones masked in the original analysis):

```bash
python src/detect_spikes.py --brw data/Slc01_20min-4AP_240425.brw --out-dir outputs \
    --masked-channels 366 1078 1454 1471 1740 1860 2069 2277 2455 2456 2518 2519 2520 2521 2582 2583 2584 2647 2911 2956 3089 3673
```

Other useful options: `--threshold` (detection threshold, default 30), `--no-localize`, `--save-waveforms`, `--waveform-channels`. Run `python src/detect_spikes.py --help` for all of them.

**Step 2: Transfer Entropy network**

```bash
python src/transfer_entropy_network.py --spikes outputs/Slc01_20min-4AP_240425_spikes.csv
```

The full analysis computes TE for every pair of active channels, each with its surrogates, so it can take a very long time and use many CPU cores. For a quick test:

```bash
python src/transfer_entropy_network.py --spikes outputs/Slc01_20min-4AP_240425_spikes.csv \
    --max-channels 50 --n-surrogates 100 --n-null 100
```

Options: `--k`, `--n-surrogates`, `--alpha`, `--n-null`, `--max-channels`, `--n-jobs`, `--seed`, `--fs-original`, `--fs-target`.

## Outputs

Written to `outputs/` (not versioned):

| File | Description |
|------|-------------|
| `<name>_spikes.csv` | Detected spikes (step 1) |
| `<name>_raster_thr<T>.png` | Raster plot of all detected spikes |
| `<name>_low_activity_channels.txt` | Channels with fewer than 5 events |
| `<name>_example_waveforms.png` | Waveforms of the most active channels |
| `<name>_raster_binned.png` | Raster of the binned spike matrix (step 2) |
| `<name>_te_network.npz` | `channel_ids`, `te_raw`, `te_fdr`, `p_values` (entry `[i, j]` = flow from channel `i` to `j`) |
| `<name>_network_metrics.csv` | Real value, null-model mean and std, and z-score of each graph metric |

## Notes

- **Binning:** several spikes in the same 1 ms bin count as a single event.
- **Surrogate resolution:** with `n` surrogates the smallest possible p-value is `1 / (n + 1)` (about 0.0033 for 300). When thousands of pairs are tested, FDR correction can be very strict with few surrogates. Increase `--n-surrogates` if few or no edges survive.
- **Interpretation:** TE measures statistical directed dependence between spike trains, not necessarily anatomical or causal connections.

## References

- Hilgen, G. et al. (2017). Unsupervised spike sorting for large-scale, high-density multielectrode arrays. *Cell Reports*, 18(10), 2521-2532.
- Muthmann, J.-O. et al. (2015). Spike detection for large neural populations using high density multielectrode arrays. *Frontiers in Neuroinformatics*, 9, 28.
- Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461-464.
- Moore, D. G. et al. (2018). Inform: efficient information-theoretic analysis of collective behaviors. *Frontiers in Robotics and AI*, 5, 60. (PyInform)

## Author

Pedro Santos Rosa Muradas, Federal University of ABC (UFABC).

## License

MIT, see [LICENSE](LICENSE).
