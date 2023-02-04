from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas


PID = '1a60a6e1-da99-4d4e-a734-39b1d4544fad'
# we are interested in the raw ephys data and the spike sorting data

one = ONE()
ba = AllenAtlas()

eid, pname = one.pid2eid(PID)
sl = SpikeSortingLoader(eid=eid, pname=pname, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)