import scipy.io as sio
from pylab import *
from matplotlib import *
from matplotlib.pyplot import *
from numbers import *
from scipy.fft import *

rcParams['figure.figsize']=(12,3)                   # Change the default figure size

# Load the data and plot it.
data = sio.loadmat('matfiles/spikes-LFP-1.mat')     # Load the multiscale data,
y = data['y']                                       # ... get the LFP data,
n = data['n']                                       # ... get the spike data,
t = data['t'].reshape(-1)                           # ... get the time axis,
K = shape(n)[0]                                     # Get the number of trials,
N = shape(n)[1]                                     # ... and the number of data points in each trial,
dt = t[1]-t[0]                                      # Get the sampling interval.

SYY = zeros(int(N/2+1))                             # Variable to store field spectrum.
SNN = zeros(int(N/2+1))                             # Variable to store spike spectrum.
SYN = zeros(int(N/2+1), dtype=complex)              # Variable to store cross spectrum.

for k in arange(K):                                 # For each trial,
    yf = rfft((y[k,:]-mean(y[k,:])) *hanning(N))    # Hanning taper the field,
    nf = rfft((n[k,:]-mean(n[k,:])))                # ... but do not taper the spikes.
    SYY = SYY + ( real( yf*conj(yf) ) )/K           # Field spectrum
    SNN = SNN + ( real( nf*conj(nf) ) )/K           # Spike spectrum
    SYN = SYN + (          yf*conj(nf)   )/K        # Cross spectrum

cohr = abs(SYN) / sqrt(SYY) / sqrt(SNN)             # Spike-field coherence
f = rfftfreq(N, dt)                                 # Frequency axis for plotting