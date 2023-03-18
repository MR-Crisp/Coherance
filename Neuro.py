from one.api import ONE
from ibllib.atlas import AllenAtlas
from pprint import pprint
from brainbox.io.one import SpikeSortingLoader
import spikeglx
from ibllib.plots import Density
import matplotlib.pyplot as plt
from neurodsp.voltage import destripe
import numpy as np
#from ibllib.io import spikeglx



def ephysToLfp():
    lf_file = "C:/Users/akaam/Downloads/ONE/openalyx.internationalbrainlab.org/angelakilab/Subjects/NYU-45/2021-07-19/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.lf.cbin"
    return spikeglx.Reader(lf_file)

   



PID = '1a60a6e1-da99-4d4e-a734-39b1d4544fad'
# we are interested in the raw ephys data and the spike sorting data

one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True)


ba = AllenAtlas()
 # this is for the spike sorting data
sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)



#turns raw data to volts(lfp)
sr = ephysToLfp()
time0 = 100 # timepoint in recording to stream
time_win = 1 # number of seconds to stream
s0 = time0 * sr.fs
tsel = slice(int(s0), int(s0) + int(time_win * sr.fs))
raw = sr[tsel, :-sr.nsync].T
destriped = destripe(raw, fs=sr.fs)

pprint(destriped[240])

# plt.rcParams['figure.figsize']=(12,3)                   # Change the default figure size



# y = destriped[0]                               # ... get the LFP data,
# n = spikes['amps']                                # ... get the spike data,
# t = spikes['times']                # ... get the time axis,
# plt.plot(t,y[1,5])                               # ... and visualize the data, for the first trial.
# plt.plot(t,n[1,5])
# plt.xlabel('Time [s]')
# plt.autoscale(tight=True)                        # ... with white space minimized.



# plt.show() 




'''
This is the format of the lfp data
Units: samples
Type: int16
Dimensions: [number of samples, number of channels]
'''



