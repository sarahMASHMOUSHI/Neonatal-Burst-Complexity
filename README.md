# Neonatal-Burst-Complexity

This repository contains scripts associated with the manuscript:

"Spontaneous burst complexity in preterm neonates tracks cortical maturation along divergent spatiotemporal and temporal dimensions".

The repository includes scripts for computing the Perturbational Complexity Index (PCIst):

    1. calc_PCIst.py
        Core implementation of PCIst, adapted from:
        Comolatti et al. (2019), "A fast and general method to empirically estimate the complexity of brain responses to perturbations."

    2. calc_PCIst_burst.py
        Computes PCIst for burst-related EEG segments.
   
    3. calc_PCIst_moving.py
        Computes PCIst using a 2-second sliding window (−5s to +10s around burst onset, step size = 0.5s).

Important: calc_PCIst.py must be available and executed/imported before running scripts 2 and 3.

----------------------------------------------------------------------------------------------------------------------------
Notes:
- Ensure EEG data is preprocessed before running PCIst computations.
- Input formats (e.g., .fif, .xlsx) must match those expected by the scripts.
- All analyses are performed on a per-event and per-subject basis.

Requirements: 
- Python (≥ 3.8)
- Required packages: mne, numpy, pandas, matplotlib, scipy, pingouin (for statistical analysis). 

