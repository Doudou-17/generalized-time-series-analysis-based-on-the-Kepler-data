# ===Module 9: Compare segements (before vs after) ===
# This script compare the peaks fromtwo segements: before and after.
# Cross‑match before/after peaks for consistency
# This script uses the src.utils_compare, src.utils_resid.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt

from src.utils_compare import read_peaks_txt, direct_fr_fi, amp_phase_from_frfi, SUFFIX_MAP
from src.utils_resid import select_data_segment, summarize_timebase

# ------------------- Manual parameters -------------------
NPZ_PREP = "./output/cleaned_segments.npz"
OUT_DIR  = "./output"
K_BEFORE = 10    # Number of top peaks from "before" segment to match
K_AFTER  = 10    # Number of top peaks from "after" segment to match
ALPHA_FREQ = 1.0 # Frequency tolerance factor α (in multiples of Rayleigh, see below)
AMP_RATIO_MAX = 3.0  # Amplitude consistency threshold (A_before/A_after in [1/AMP_RATIO_MAX, AMP_RATIO_MAX])
CHECK_PHASE = True    # Whether to compute phase difference (degrees)

# Peak table paths (from Module 4)
PEAKS_BEFORE = os.path.join(OUT_DIR, "kurtz_top_peaks_before.txt")
PEAKS_AFTER  = os.path.join(OUT_DIR, "kurtz_top_peaks_after.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------- Load segments and peaks -------------------
# "before" segment
bundle_b = select_data_segment(NPZ_PREP, "before")
t_b, y_b = bundle_b.time, bundle_b.flux_raw
T_b, ray_b = summarize_timebase(t_b)
# "after" segment
bundle_a = select_data_segment(NPZ_PREP, "after")
t_a, y_a = bundle_a.time, bundle_a.flux_raw
T_a, ray_a = summarize_timebase(t_a)

# Frequency tolerance (μHz): ε_f = α * 1e6 * max(1/T_b, 1/T_a)
eps_uHz = ALPHA_FREQ * 1e6 * max(1.0 / max(T_b, 1e-12), 1.0 / max(T_a, 1e-12))
print(f"Rayleigh_before={ray_b:.3f} μHz, Rayleigh_after={ray_a:.3f} μHz, eps_uHz={eps_uHz:.3f} (α={ALPHA_FREQ})")

# Read peak tables (take top K)
freqs_b = read_peaks_txt(PEAKS_BEFORE)[:K_BEFORE]
freqs_a = read_peaks_txt(PEAKS_AFTER)[:K_AFTER]

# ------------------- Cross-match (nearest within ε) -------------------
matched = []  # (fb, fa, f_avg)
used_a = np.zeros(freqs_a.size, dtype=bool)
for fb in freqs_b:
    # Nearest after frequency
    diffs = np.abs(freqs_a - fb)
    j = int(np.argmin(diffs)) if diffs.size else -1
    if j >= 0 and diffs[j] <= eps_uHz and (not used_a[j]):
        fa = float(freqs_a[j])
        f_avg = 0.5 * (fb + fa)
        matched.append((float(fb), fa, f_avg, j))
        used_a[j] = True

only_b = [float(fb) for fb in freqs_b if not np.any(np.isclose(fb, [m[0] for m in matched], atol=eps_uHz))]
only_a = [float(fa) for k, fa in enumerate(freqs_a) if not used_a[k]]

print(f"Matched pairs: {len(matched)} | only_before: {len(only_b)} | only_after: {len(only_a)}")

# ------------------- Compute amplitude/phase consistency (evaluate at f_avg) -------------------
rows = []
for fb, fa, f_avg, _ in matched:
    FRb, FIb = direct_fr_fi(t_b, y_b, f_avg)
    FRa, FIa = direct_fr_fi(t_a, y_a, f_avg)
    Ab, phib = amp_phase_from_frfi(FRb, FIb, w_sum=len(y_b))
    Aa, phia = amp_phase_from_frfi(FRa, FIa, w_sum=len(y_a))
    amp_ratio = Ab / max(Aa, 1e-30)
    if CHECK_PHASE:
        # Phase difference (deg), folded to [-180,180]
        dphi = (phia - phib)
        while dphi >  np.pi: dphi -= 2*np.pi
        while dphi < -np.pi: dphi += 2*np.pi
        dphi_deg = np.degrees(dphi)
    else:
        dphi_deg = np.nan
    rows.append((fb, fa, f_avg, Ab, Aa, amp_ratio, dphi_deg))

# ------------------- Save result tables -------------------
M_MATCH = os.path.join(OUT_DIR, "matched_peaks.txt")
M_ONLYB = os.path.join(OUT_DIR, "only_before.txt")
M_ONLYA = os.path.join(OUT_DIR, "only_after.txt")

with open(M_MATCH, "w") as f:
    f.write("# f_before  f_after  f_avg   A_before   A_after   A_ratio  dphi_deg\n")
    for (fb, fa, favg, Ab, Aa, r, d) in rows:
        f.write(f"{fb:12.6f}  {fa:12.6f}  {favg:12.6f}  {Ab:10.6e}  {Aa:10.6e}  {r:8.3f}  {d:8.3f}\n")
with open(M_ONLYB, "w") as f:
    for fb in only_b:
        f.write(f"{fb:12.6f}\n")
with open(M_ONLYA, "w") as f:
    for fa in only_a:
        f.write(f"{fa:12.6f}\n")
print(f"Saved -> {M_MATCH}, {M_ONLYB}, {M_ONLYA}")

# ------------------- Visualization: amplitude comparison & phase difference -------------------
import matplotlib.ticker as mticker

PLOT_AMP = os.path.join(OUT_DIR, "amp_compare_before_after.png")
PLOT_PHI = os.path.join(OUT_DIR, "phase_diff_before_after.png")

if rows:
    Ab_arr = np.array([r[3] for r in rows])
    Aa_arr = np.array([r[4] for r in rows])
    favg_arr = np.array([r[2] for r in rows])
    dphi_arr = np.array([r[6] for r in rows])

    plt.figure(figsize=(5.2,5))
    plt.loglog(Ab_arr, Aa_arr, 'o', alpha=0.8)
    lo = max(np.min([Ab_arr, Aa_arr])*0.8, 1e-12)
    hi = np.max([Ab_arr, Aa_arr])*1.2
    plt.plot([lo, hi],[lo, hi], '--')
    plt.xlabel('A_before (norm)'); plt.ylabel('A_after (norm)')
    plt.title('Amplitude consistency (before vs after)')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOT_AMP, dpi=200); plt.close()
    print(f"Saved -> {PLOT_AMP}")

    if CHECK_PHASE:
        plt.figure(figsize=(6.5,4))
        plt.plot(favg_arr, dphi_arr, 'o', alpha=0.8)
        plt.axhline(0, color='k', lw=1)
        plt.xlabel('Frequency (μHz)'); plt.ylabel('Phase difference (deg)')
        plt.title('Phase(after) - Phase(before) at matched f_avg')
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(PLOT_PHI, dpi=200); plt.close()
        print(f"Saved -> {PLOT_PHI}")
else:
    print("No matched rows to plot.")

