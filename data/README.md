# Data

Recordings are **not** stored in this repository because of their size (GitHub rejects files above 100 MB).

## Using your own recording

The pipeline works with any BioCam `.brw` file. Place it in this folder and run:

```
python src/detect_spikes.py --brw data/<your_recording>.brw --out-dir outputs
```

