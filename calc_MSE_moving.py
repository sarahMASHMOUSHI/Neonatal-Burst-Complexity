### Script to compute MSE complexity using a 2-second sliding window (−5s to +10s around burst onset, 0.5s step).
### Analysis is performed per event and per subject.

import os
import numpy as np
import pandas as pd
import mne
import neurokit2 as nk
from joblib import Parallel, delayed

input_folder_excel = r"D:\Study_2\markers_burst_updated"
input_folder_fif = r"D:\Study_2\clean_fif"
output_folder = r"D:\Study_2\MSE_burst_moving"
os.makedirs(output_folder, exist_ok=True)

scale = 20             
order = 2
r_factor = 0.15
sfreq_resampled = 500
window_before = 5
window_after = 10
win_size = 2
step_size = 0.5

win_samples = int(win_size * sfreq_resampled)       
step_samples = int(step_size * sfreq_resampled)     
epoch_samples = int((window_before + window_after) * sfreq_resampled)  

n_windows = int((epoch_samples - win_samples) / step_samples) + 1  # 27
time_axis = np.array([
    -window_before + (i * step_size) + win_size / 2
    for i in range(n_windows)
])

# ---------------- channel MSE function ----------------
def compute_channel_mse(sig, scale, order, r_factor):
    if np.std(sig) == 0:
        return None
    try:
        mse_values, _ = nk.entropy_multiscale(
            sig, scale=scale, dimension=order,
            tolerance=r_factor * np.std(sig)
        )
        if mse_values is None or np.all(np.isnan(mse_values)):
            return None
        return np.nanmean(mse_values)
    except Exception:
        return None

subject_names = [] # "subject's-last-name_gestational-age" for example "MASHMOUSHI_29.6"

for subject in subject_names:
    
    out_path = os.path.join(output_folder, f"MSE_moving_{subject}.xlsx")
    excel_file = os.path.join(input_folder_excel, f"{subject}.xlsx")
    fif_file   = os.path.join(input_folder_fif,   f"{subject}.fif")

    if not os.path.exists(excel_file) or not os.path.exists(fif_file):
        print(f"Skipping (missing files): {subject}")
        continue

    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
    raw.pick("eeg")
    raw.resample(sfreq_resampled, verbose=False)

    data   = raw.get_data()
    events = pd.read_excel(excel_file)
    n_bursts = len(events)

    mse_events = []

    for event_idx, ev in events.iterrows():
        print(f"  {subject} | burst {event_idx+1}/{n_bursts} | {n_windows} windows", end="\r")

        t_start = ev["burst_start"] - window_before
        t_end   = ev["burst_start"] + window_after

        i_start, i_end = raw.time_as_index([t_start, t_end], use_rounding=True)
        epoch = data[:, i_start:i_end]
        window_mse = []

        for w in range(n_windows):
            ws = w * step_samples
            we = ws + win_samples
            win_data = epoch[:, ws:we]

            # parallel across channels
            results_ch = Parallel(n_jobs=-1)(
                delayed(compute_channel_mse)(win_data[ch], scale, order, r_factor)
                for ch in range(win_data.shape[0]))
            channel_mse = [v for v in results_ch if v is not None]
            window_mse.append(np.nanmean(channel_mse) if channel_mse else np.nan)

        row = {"Event_Index": event_idx, "Burst_start": ev["burst_start"]}
        for w_idx, mse_val in enumerate(window_mse):
            row[f"W{w_idx+1}_t{time_axis[w_idx]:.1f}s"] = mse_val
        mse_events.append(row)

    if mse_events:
        out_df = pd.DataFrame(mse_events)
        out_df.to_excel(out_path, index=False)
        print(f"\nSaved: {subject} — {len(mse_events)} bursts")
    else:
        print(f"\nNo valid bursts: {subject}")

print("Done.")
