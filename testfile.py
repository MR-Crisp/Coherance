from one.api import ONE
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
import spikeglx
import numpy as np
from neurodsp.timefrequency
from neurodsp.plts import plot_time_series, plot_spectral_hist
from ibllib.io import spikeglx
from ibllib.ephys import neuropixel


# Set up ONE instance and retrieve session data
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)
eid = one.search(subject='*', date='*', number='*', uuid='1a60a6e1-da99-4d4e-a734-39b1d4544fad')[0]

# Set up SpikeSortingLoader instance and load data
ba = AllenAtlas()
sl = SpikeSortingLoader(pid=eid, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Set up SpikeGLX reader instance and load data
lf_path = one.load(eid, 'ephysData', collection=f'raw_ephys_data/probe00', download_only=True)[0]
lf_file = spikeglx.Reader(lf_path)
fs = neuropixel.SAMPLING_RATE

# Retrieve LFP data and filter in the 300-3000 Hz band
t0 = 0
win_dur = 60
nwin = 50
start_idx = int(t0 * fs)
end_idx = int((t0 + win_dur) * fs)
lfp = lf_file[start_idx:end_idx, :-lf_file.nsync].T
f_lfp = neuropixel.proc_bandpass(lfp, fs, [300, 3000], 3, 0.1, 'bandpass')

# Compute spike-field coherence using NeuroDSP amp_by_time function
time_bins = np.linspace(0, win_dur, nwin)
amps, times, freqs = amp_by_time(clusters.spike_times, f_lfp, fs, f_range=[4, 200], time_bins=time_bins)
coh = np.abs(amps) / np.sqrt(np.mean(np.abs(amps)**2) * np.mean(np.abs(f_lfp)**2))

# Plot results
plot_time_series(lfp, fs)
plot_spectral_hist(coh, freqs, times)
