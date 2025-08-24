# Time Series Project

This project implements a **complete pipeline for analyzing astronomical time-series data** (stellar light curves) with uneven sampling and observational gaps.  
It focuses on applying **Kurtz’s recursive DFT** (Discrete Fourier Transform) and **multi-sine least-squares fitting**, with extensions including **residual analysis**, **permutation-based significance testing (FAP)**, and **uncertainty quantification**.  
The project uses **Kepler mission light curves** as test data.

---

## Features (Modules)

- **Module 0–2**: Data download, cleaning, and segmentation (before/after gap)
- **Module 3**: Phase folding utilities (for visualisation only, not used in later computation) 
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

## Dependencies

This project uses a minimal set of scientific Python packages:
numpy — numerical arrays and linear algebra
scipy — numerical methods (optimization, signal processing)
matplotlib — plotting
astropy — astronomical time/units handling
lightkurve — Kepler light curve access & I/O
tqdm — progress bars

---

## Full Pipeline (from data download)
Each module can be run independently.
Here is the recommended end-to-end workflow:
```
# Module 0) Download and standardize raw data (saves to ./data)
python -m scripts.module0_download

# Module 1) Plot the light curve for the raw data
python -m scripts.module1_plot_raw

# Module 2) Prepare / clean and split by the largest gap
python -m scripts.module2_prepare

# Module 3) Phase folding (visualisation only, not required for analysis)
python -m scripts.module3_phase_fold

# Module 4) Kurtz recursive DFT scan + peaks
python -m scripts.module4_kurtz_scan

# Module 5) Multi-sine least squares fit
python -m scripts.module5_multi_sine_fit

# Module 6) View model overlay + residuals
python -m scripts.module6_view_fit

# Module 7) Residual periodogram
python -m scripts.module7_resid_scan

# Module 8) Permutation-based significance (global FAP)
python -m scripts.module8_significance

# Module 9) Segment comparison (before vs after)
python -m scripts.module9_compare_segments

# Module 10) Uncertainty report
python -m scripts.module10_uncertainty_report
```