# Time Series Project

This project implements a **complete pipeline for analyzing astronomical time-series data** (stellar light curves) with uneven sampling and observational gaps.  
It focuses on applying **Kurtz’s recursive DFT** (Discrete Fourier Transform) and **multi-sine least-squares fitting**, with extensions including **residual analysis**, **permutation-based significance testing (FAP)**, and **uncertainty quantification**.  
The project uses **Kepler mission light curves** as test data.

---

## Features (Modules)

- **Module 0–2**: Data download, cleaning, and segmentation (before/after gap)
- **Module 3**: Phase folding utilities
- **Module 4**: Kurtz recursive DFT scan + spectral window analysis
- **Module 5**: Multi-frequency sine fitting (joint LSQ fit)
- **Module 6**: View fitted model (overlay, residuals, summary statistics)
- **Module 7**: Residual periodogram (search for remaining signals)
- **Module 8**: Permutation-based significance test (global FAP thresholds)
- **Module 9**: Segment comparison (before vs after gaps)
- **Module 10**: Uncertainty quantification (frequency, amplitude, phase)

---

## Project Structure
```
.
├── data/ # Raw and cleaned Kepler light curve data
├── output/ # Results: plots, tables, fitted models
├── scripts/ # Executable scripts for each module (M0–M10)
│ ├── module0_download.py
│ ├── module2_prepare.py
│ ├── module3_phase_fold.py
│ ├── module4_kurtz_scan.py
│ ├── module5_multi_sine_fit.py
│ ├── module6_view_fit.py
│ ├── module7_resid_scan.py
│ ├── module8_significance.py
│ ├── module9_compare_segments.py
│ └── module10_uncertainty_report.py
├── src/ # Core Python utilities
│ ├── utils_data.py
│ ├── utils_freq.py
│ ├── utils_resid.py
│ ├── utils_fit.py
│ ├── utils_significance.py
│ ├── utils_phase.py
│ ├── utils_plot.py
│ ├── utils_compare.py
│ └── utils_errors.py
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone <your_repo_url>
cd time-series-project

python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt


---
### Dependencies

This project uses a minimal set of scientific Python packages:

- `numpy` — numerical arrays and linear algebra  
- `scipy` — numerical methods (optimization, signal processing)  
- `matplotlib` — plotting  
- `astropy` — astronomical time/units handling  
- `lightkurve` — Kepler light curve access & I/O  
- `tqdm` — progress bars

Install all dependencies with:

```bash
pip install -r requirements.txt


## Full pipeline (from data download)
```bash
# 0) Download and standardize raw data (will fetch Kepler light curves to ./data)
python -m scripts.module0_download

# 1) Prepare / clean and split by the largest gap
python -m scripts.module2_prepare

# 2) Kurtz recursive DFT scan + peaks
python -m scripts.module4_kurtz_scan

# 3) Multi-sine least squares fit
python -m scripts.module5_multi_sine_fit

# 4) View model overlay + residuals
python -m scripts.module6_view_fit

# 5) Residual periodogram
python -m scripts.module7_resid_scan

# 6) Permutation-based significance (global FAP)
python -m scripts.module8_significance

# 7) Uncertainty report
python -m scripts.module10_uncertainty_report

Note: No raw data or outputs are tracked in the repository. All inputs are downloaded by Module 0, and all figures/tables are generated into ./output/.

---

### Example Outputs
Periodograms: kurtz_periodogram_full.png, kurtz_periodogram_zoom.png
Peak tables: kurtz_top_peaks.txt, kurtz_resid_top_peaks_with_fap.txt
Fit results: multisine_fit_results.npz, fit_overlay_timeseries.png, fit_residual_timeseries.png
Significance plots: perm_cdf_maxpower.png, kurtz_resid_periodogram_with_thresholds.png
Final uncertainty table: final_table.csv, sigma_nu_vs_freq.png

Note: No raw data or outputs are tracked in the repository. All inputs are downloaded by Module 0, and all figures/tables are generated into ./output/.