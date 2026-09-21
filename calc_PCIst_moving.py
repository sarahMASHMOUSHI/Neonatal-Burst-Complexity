### Script to compute PCIst using a 2-second sliding window (−5s to +10s around burst onset, 0.5s step).
### Analysis is performed per event and per subject.
### Ensure that calc_PCIst.py is executed beforehand.

import mne
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

input_folder_excel = r"D:\markers_burst_isolated_"
input_folder_fif = r"D:\clean_fif"
output_base_folder = r"D:\results_movingwindow"

subject_names = [] # "subject's-last-name_gestational-age" for example "MASHMOUSHI_29.6"

for subject in subject_names:
    excel_file = os.path.join(input_folder_excel, f"markers_{subject}.xlsx")
    fif_file = os.path.join(input_folder_fif, f"{subject}.fif")
    output_dir = os.path.join(output_base_folder, subject)

    if not os.path.exists(excel_file) or not os.path.exists(fif_file):
        print(f"Skipping {subject}: Missing Excel or FIF file.")
        continue

    os.makedirs(output_dir, exist_ok=True)

    raw = mne.io.read_raw_fif(fif_file, preload=True) # EEG data
    event_data = pd.read_excel(excel_file) # event data

    results = []

    for index in range(len(event_data)):
        burst_start = event_data.iloc[index]['burst_start']
        burst_end = event_data.iloc[index]['burst_end']

        if burst_start - 5 < 0:
            continue

        raw_base = raw.copy().crop(tmin=burst_start - 5, tmax=burst_start)
        base_data = raw_base.get_data()
        sampling_rate = raw_base.info['sfreq']
        window_size = int(2 * sampling_rate)

        min_power = float('inf')
        baseline_start = None

        for start_idx in range(0, base_data.shape[1] - window_size + 1):
            end_idx = start_idx + window_size
            window_data = base_data[:, start_idx:end_idx]
            window_power = (window_data ** 2).sum()
            if window_power < min_power:
                min_power = window_power
                baseline_start = start_idx / sampling_rate

        if baseline_start is not None:
            baseline_window = (baseline_start, baseline_start + 2)
        else:
            raise ValueError("Could not determine baseline window.")

        baseline_start_idx = int(baseline_start * sampling_rate)
        baseline_end_idx = baseline_start_idx + window_size
        selected_baseline_data = base_data[:, baseline_start_idx:baseline_end_idx]
        min_amp_baseline = selected_baseline_data.min()
        max_amp_baseline = selected_baseline_data.max()

        raw_baseline = raw_base.copy().crop(
            tmin=baseline_start,
            tmax=min(baseline_start + 2, raw_base.times[-1])
        )

        response_window_start = burst_start - 5 # start the moving window 5 seconds before event's onset
        response_window_end = burst_start + 10 # until 10 seconds after 
        step_size = 0.5
        window_length = 2

        moving_pci_results = []
        current_start = response_window_start

        while current_start + window_length <= response_window_end:
            current_end = current_start + window_length

            raw_response = raw.copy().crop(tmin=current_start, tmax=current_end)
            raw_response_data = raw_response.get_data()
            min_amp_response = raw_response_data.min()
            max_amp_response = raw_response_data.max()

            raw_input = mne.concatenate_raws([raw_baseline.copy(), raw_response.copy()])

            par = {
                'baseline_window': (raw_input.times[0], raw_input.times[0] + 2),
                'response_window': (raw_input.times[0] + 2, raw_input.times[-1]),
                'k': 1.2, 'min_snr': 1, 'max_var': 99, 'embed': False, 'n_steps': 100
            }

            pci_result = calc_PCIst(raw_input.get_data(), raw_input.times, **par, full_return=True)
            pci_value = pci_result['PCI']
            principal_components = pci_result['signal_svd']

            moving_pci_results.append({
                'Response Window Start': current_start,
                'Response Window End': current_end,
                'PCI Value': pci_value,
                'Min Amp Response': min_amp_response,
                'Max Amp Response': max_amp_response,
                'Number of Principal Components': len(principal_components),
                'Optimal Threshold Pair': pci_result.get('max_thresholds', None) if pci_value != 0 else None
            })

            current_start += step_size

        for pci_entry in moving_pci_results:
            results.append({
                'Event Index': index + 1,
                'Start Time': burst_start,
                'End Time': burst_end,
                'Event Duration': burst_end - burst_start,
                'Baseline Window Start': baseline_window[0],
                'Baseline Window End': baseline_window[1],
                **pci_entry
            })

    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(output_base_folder, f"PCI_{subject}.xlsx"), index=False)
    print(f"\nPCI calculation completed for {subject}. Results saved to {output_dir}")
