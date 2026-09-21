# Neonatal-Burst-Complexity

This repository contains scripts associated with the manuscript:

"Spontaneous burst complexity in preterm neonates tracks cortical maturation along divergent spatiotemporal and temporal dimensions".

The repository includes scripts for computing the Perturbational Complexity Index based on state transitions (PCIst) and Multiscale Entropy (MSE) complexity indicies:

    1. calc_PCIst.py
        Core implementation of PCIst, adapted from:
        Comolatti et al. (2019), "A fast and general method to empirically estimate the complexity of brain responses to perturbations."

    2. calc_PCIst_burst.py
        Computes PCIst for isolated burst-related EEG segments on a per-event and per-subject basis.
   
    3. calc_PCIst_moving.py
        Computes PCIst using a 2-second sliding window (−5s to +10s around burst onset, step size = 0.5s).

    4. calc_MSE_burst.py
        Computes a single, channel-and-scale-averaged MSE value for each burst event.

    5. calc_MSE_moving.py
        Computes MSE using a 2-second sliding window (−5s to +10s around burst onset, step size = 0.5s).
----------------------------------------------------------------------------------------------------------------------------

Usage & Execution Notes

- Input Data Format: Scripts expect preprocessed MNE continuous EEG files (.fif) and burst onset/offset markers in Excel format (.xlsx).
- File Structure: Update the input/output directory paths at the top of each script ('input_folder_excel', 'input_folder_fif', 'output_folder') to match your local dataset layout.
- Dependency: Ensure 'calc_PCIst.py' is located in the same root folder when executing 'calc_PCIst_burst.py' or 'calc_PCIst_moving.py'.

Requirements: 
- Python (≥ 3.8)
- Required packages: mne, numpy, pandas, matplotlib, scipy, neurokit2, joblib, pingouin (for statistical analysis). 

