import pyxdf
import numpy as np
import mne

# Load XDF
streams, header = pyxdf.load_xdf("data/sub-P001_ses-S001_run-001_task-training.xdf")

# Find EEG stream
eeg_stream = None
for s in streams:
    if s["info"]["type"][0].lower() == "eeg":
        eeg_stream = s
        break

assert eeg_stream is not None, "EEG stream not found"

# Extract data
data = np.array(eeg_stream["time_series"]).T  # shape: (channels, samples)
sfreq = float(eeg_stream["info"]["nominal_srate"][0])

# Channel names (mock-safe)
n_ch = data.shape[0]
ch_names = [f"EEG{idx+1}" for idx in range(n_ch)]
ch_types = ["eeg"] * n_ch

# Create MNE Raw object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(data, info)

# Find marker stream
marker_stream = None
for s in streams:
    if s["info"]["type"][0].lower() == "markers":
        marker_stream = s
        break

if marker_stream is not None:
    marker_times = marker_stream["time_stamps"]
    eeg_times = eeg_stream["time_stamps"]

    events = []
    for t in marker_times:
        idx = np.argmin(np.abs(eeg_times - t))
        events.append([idx, 0, 1])  # event id = 1

    events = np.array(events, dtype=int)

raw.set_annotations(
    mne.Annotations(
        onset=events[:, 0] / raw.info["sfreq"],
        duration=[0] * len(events),
        description=["trial"] * len(events)
    )
)
raw.plot(block=True)
