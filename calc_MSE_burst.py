### Script to compute MSE for bursts on a per-subject basis.

import os
import numpy as np
import pandas as pd
import mne
import neurokit2 as nk

input_folder_excel = r"D:\markers_burst_isolated"
input_folder_fif = r"D:\clean_fif"
output_folder = r"D:\MSE_burst"
os.makedirs(output_folder, exist_ok=True)

# ---------------- parameters ----------------
scale = 20                 # MSE scales 1–20
order = 2                  # embedding dimension
r_factor = 0.15             # tolerance = r * std
min_samples_factor = 30     # min length = scale * factor

subject_names = []

for subject in subject_names:
    excel_file = os.path.join(input_folder_excel, f"{subject}.xlsx")
    fif_file = os.path.join(input_folder_fif, f"{subject}.fif")

    if not os.path.exists(excel_file) or not os.path.exists(fif_file):
        print(f"Skipping {subject}")
        continue

    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
    raw.pick("eeg")

    data = raw.get_data()  # (n_channels, n_times)
    events = pd.read_excel(excel_file) # load burst markers (already isolated)
    mse_events = []

    for _, ev in events.iterrows():
        t_start, t_end = ev["burst_start"], ev["burst_end"]
        i_start, i_end = raw.time_as_index([t_start, t_end], use_rounding=True)
        burst = data[:, i_start:i_end]

        if burst.shape[1] < scale * min_samples_factor: # minimum length check
            continue

        channel_mse = []

        for ch in range(burst.shape[0]):
            sig = burst[ch]
            if np.std(sig) == 0:
                continue

            mse_values, _ = nk.entropy_multiscale(sig, scale=scale, dimension=order, tolerance=r_factor * np.std(sig))
            if mse_values is None or np.all(np.isnan(mse_values)):
                continue
        
            channel_mse.append(mse_values)
            
        if len(channel_mse) == 0:
            continue
        channel_mse = np.array(channel_mse)
        
        mse_events.append({
            "Start_time_s": t_start,
            "End_time_s": t_end,
            "Duration_s": t_end - t_start,
            "MSE_mean": np.nanmean(channel_mse),
            "N_channels": channel_mse.shape[0],
            "Burst_length_samples": burst.shape[1]
        })

    events_df = pd.DataFrame(mse_events)
    events_df.to_excel(os.path.join(output_folder, f"MSE_events_{subject}.xlsx"),index=False)
    
print("All subjects processed.")
