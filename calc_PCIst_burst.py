### Script to compute PCIst for bursts on a per-subject basis.
### Ensure that calc_PCIst.py is executed beforehand.

import mne
import numpy as np
import pandas as pd
import os
import re

input_folder_excel = r"D:\markers_burst"
input_folder_fif = r"D:\clean_fif"
output_base_folder = r"D:\response_burst_isolated"
output_isolated_folder = r"D:\markers_burst_isolated"

subject_names = [] # "subject's-last-name_gestational-age" for example "MASHMOUSH_29.6"

for subject in subject_names:
    excel_file = os.path.join(input_folder_excel, f"markers_{subject}.xlsx")
    fif_file = os.path.join(input_folder_fif, f"{subject}.fif")
    output_dir = os.path.join(output_base_folder, subject)

    if not os.path.exists(excel_file) or not os.path.exists(fif_file):
        print(f"Skipping {subject}: Missing Excel or FIF file.")
        continue

    raw = mne.io.read_raw_fif(fif_file, preload=True)
    event_data = pd.read_excel(excel_file)

    # Filter isolated events
    isolated_events = []
    for i in range(len(event_data) - 1):
        if (event_data.iloc[i + 1]['burst_start'] - event_data.iloc[i]['burst_end']) >= 5:
            isolated_events.append(event_data.iloc[i])
    if len(event_data) > 0:
        isolated_events.append(event_data.iloc[-1])  # Always keeping the last event

    isolated_df = pd.DataFrame(isolated_events)
    os.makedirs(output_isolated_folder, exist_ok=True)
    isolated_df.to_excel(os.path.join(output_isolated_folder, f"markers_{subject}_isolated.xlsx"), index=False)

    results = []
    for _, event in isolated_df.iterrows():
        burst_start = event['burst_start']
        burst_end = event['burst_end']
        if burst_start - 10 < 0:
            continue
            
        # Selection of baseline window: 5-second window with lowest power window in the last 10s preceding the event
        raw_base = raw.copy().crop(tmin=burst_start - 10, tmax=burst_start)
        base_data = raw_base.get_data()
        sampling_rate = raw_base.info['sfreq']
        window_size = int(5 * sampling_rate) # 5-second baseline window

        min_power = float('inf')
        baseline_start = None
        for start_idx in range(0, base_data.shape[1] - window_size + 1):
            end_idx = start_idx + window_size
            window_data = base_data[:, start_idx:end_idx]
            window_power = (window_data ** 2).sum()
            if window_power < min_power:
                min_power = window_power
                baseline_start = start_idx / sampling_rate

        if baseline_start is None:
            continue

        baseline_start_idx = int(baseline_start * sampling_rate)
        baseline_end_idx = baseline_start_idx + window_size
        selected_baseline_data = base_data[:, baseline_start_idx:baseline_end_idx]
        min_amp_baseline = selected_baseline_data.min()
        max_amp_baseline = selected_baseline_data.max()

        raw_baseline = raw_base.copy().crop(tmin=baseline_start, tmax=min(baseline_start + 5, raw_base.times[-1])) # it's actually from baseline_start until baseline_start + 5, 
        # But baseline_start + 5 may exceed raw_base.times[-1] slightly due to floating-point precision, which causes mne.io.Raw.crop() to raise a ValueError.

        # Selection of response window, in this case it is the burst itself, from onset to offset
        raw_response = raw.copy().crop(tmin=burst_start, tmax=burst_end)
        raw_response_data = raw_response.get_data()
        min_amp_event = raw_response_data.min()
        max_amp_event = raw_response_data.max()

        raw_input = mne.concatenate_raws([raw_baseline, raw_response])  # Concatenate selected baseline and response windows
        par = {
            'baseline_window': (raw_input.times[0], raw_input.times[0] + 5),
            'response_window': (raw_input.times[0] + 5, raw_input.times[-1]),
            'k': 1.2, 'min_snr': 1, 'max_var': 99, 'embed': False, 'n_steps': 100
        }

        pci_result = calc_PCIst(raw_input.get_data(), raw_input.times, **par, full_return=True)
        results.append({
            'Start Time': burst_start,
            'End Time': burst_end,
            'Event Duration': burst_end - burst_start,
            'Baseline Window Start': baseline_start,
            'Baseline Window End': baseline_start + 5,
            'PCI Value': pci_result['PCI'],
            'Min Amp Baseline': min_amp_baseline,
            'Max Amp Baseline': max_amp_baseline,
            'Min Amp Event': min_amp_event,
            'Max Amp Event': max_amp_event,
            'Number of Principal Components': len(pci_result['signal_svd']),
            'Optimal Threshold Pair': pci_result.get('max_thresholds', None) if pci_result['PCI'] != 0 else None
        }) 
        
    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(output_base_folder, f"PCI_{subject}.xlsx"), index=False)
    print(f"PCI calculation completed for {subject} and results saved.")
