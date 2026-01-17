# dislocation-density

A reproducible batch processing program for estimating dislocation density from Transmission Electron Microscopy (TEM) images, implemented as an M1–M8 modular pipeline: IO & Calibration, Preprocessing, Line Enhancement, Segmentation, Skeletonization, Metrics, QA Visualization, CLI & Configuration.

## Installation

Recommended to use a virtual environment, then install from the root directory:

```bash
python -m pip install -U pip
python -m pip install -e .
```

## Quick Start

1. Copy config:

```bash
cp config.example.yaml config.yaml
```

2. Run batch processing:

```bash
dislo_density run --input ./ --out results/ --config config.yaml
```

## Web Interface (Flask)

Launch local web interface:

```bash
python -m pip install -e .
dislo_density_web --config config.yaml --host 127.0.0.1 --port 5000
```

Browser: `http://127.0.0.1:5000/`

## Output

Generates in `results/`:
- `summary.csv`: Summary results per image/ROI
- `summary_stats.csv`: Mean/Std statistics
- `results/<image_stem>/result.json`: Detailed JSON results
- `results/<image_stem>/roi_XX/overlay.png`: Skeleton overlay
- `results/<image_stem>/roi_XX/qc.png`: Quality control panel

## Calibration (nm/px)

Priority:
1. `m1.manual_nm_per_px`: Manual input (Recommended)
2. TIFF metadata
3. Scale bar detection + Manual value (OCR removed)
