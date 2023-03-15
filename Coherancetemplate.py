import scipy.io as sio
from pylab import *
from matplotlib import *
from matplotlib.pyplot import *
from scipy.fft import *
from scipy import signal
from scipy import stats
import statsmodels.api as sm
from pprint import pprint

rcParams['figure.figsize']=(12,3)                   # Change the default figure size



data = sio.loadmat('spikes-LFP-1.mat')  # Load the multiscale data,
y = data['y']                                # ... get the LFP data,
n = data['n']                                # ... get the spike data,
pprint(n)
t = data['t'].reshape(-1)                    # ... get the time axis,
#plot(t,y[1,:])                               # ... and visualize the data, for the first trial.
#plot(t,n[1,:])
#xlabel('Time [s]')
#autoscale(tight=True)                        # ... with white space minimized.

win = 100                                   # Define a temporal window to examine around each spike.
K = shape(n)[0]                             # Get the number of trials,
N = shape(n)[1]                             # ... and the number of data points in each trial.
STA = zeros([K,2*win+1])                    # Define a variable to hold the STA.
for k in arange(K):                         # For each trial,
    spike_times = where(n[k,:]==1)[0]       # ... find the spikes.
    counter=0
    for spike_t in spike_times:             # For each spike,
        if win < spike_t < N-win-1:         # ... add the LFP to the STA.
            STA[k,:] = STA[k,:] + y[k,spike_t-win:spike_t+win+1]
            counter += 1
    STA[k, :] = STA[k, :]/counter


dt = t[1]-t[0]                    # Get the sampling interval.
lags = arange(-win, win+1) * dt   # Make a time axis for plotting.




def FTA_function(y,n,t,Wn):                  #INPUTS: y=field, n=spikes, t=time, Wn=passband [low,high]
    dt = t[1]-t[0]                           #Define the sampling interval.
    fNQ = 1/dt/2                             #Define the Nyquist frequency.
    ord  = 100                               #...and filter order,
    b = signal.firwin(ord, Wn, nyq=fNQ, pass_zero=False, window='hamming'); #...build bandpass filter.
    FTA=zeros([K,N])                      #Create a variable to hold the FTA.
    for k in arange(K):                   #For each trial,
        Vlo = signal.filtfilt(b, 1, y[k,:])  # ... apply the filter.
        phi = angle(signal.hilbert(Vlo))  # Compute the phase of low-freq signal
        indices = argsort(phi)            #... get indices of sorted phase,
        FTA[k,:] = n[k,indices]              #... and store the sorted spikes.
    phi_axis = linspace(-pi,pi,N)   #Compute phase axis for plotting.
    return mean(FTA,0), phi_axis

Wn = [9,11]                                  #Define the passband, here from 9-11 Hz.
FTA, phi_axis = FTA_function(y,n,t,Wn)       #Compute the FTA.





SYY = zeros(int(N/2+1))                                # Variable to store field spectrum.
SNN = zeros(int(N/2+1))                                # Variable to store spike spectrum.
SYN = zeros(int(N/2+1), dtype=complex)                 # Variable to store cross spectrum.

for k in arange(K):                                    # For each trial,
    yf = rfft((y[k,:]-mean(y[k,:])) * hanning(N))      # Hanning taper the field,
    nf = rfft((n[k,:]-mean(n[k,:])))                   # ... but do not taper the spikes.
    SYY = SYY + real( yf*conj(yf) )/K                  # Field spectrum
    SNN = SNN + real( nf*conj(nf) )/K                  # Spike spectrum
    SYN = SYN + ( yf*conj(nf) )/K                      # Cross spectrum

cohr = abs(SYN) / sqrt(SYY) / sqrt(SNN)                # Coherence

f = rfftfreq(N, dt)                                    # Frequency axis for plotting





y_scaled = 0.1*y





def coherence(n,y,t):                           #INPUT (spikes, fields, time)
    K = shape(n)[0]                          #... where spikes and fields are arrays [trials, time]
    N = shape(n)[1]
    T = t[-1]
    SYY = zeros(int(N/2+1))
    SNN = zeros(int(N/2+1))
    SYN = zeros(int(N/2+1), dtype=complex)
    
    for k in arange(K):
        yf = rfft((y[k,:]-mean(y[k,:])) *hanning(N))    # Hanning taper the field,
        nf = rfft((n[k,:]-mean(n[k,:])))                # ... but do not taper the spikes.
        SYY = SYY + ( real( yf*conj(yf) ) )/K           # Field spectrum
        SNN = SNN + ( real( nf*conj(nf) ) )/K           # Spike spectrum
        SYN = SYN + ( yf*conj(nf) )/K                   # Cross spectrum

    cohr = abs(SYN) / sqrt(SYY) / sqrt(SNN)             # Coherence
    f = rfftfreq(N, dt)                                 # Frequency axis for plotting
    
    return (cohr, f, SYY, SNN, SYN)



def thinned_spike_train(n, thinning_factor):              # Thin the spike train (n) by the thinning_factor.
    n_thinned = copy(n)                                # Make a copy of the spike train data.
    for k in arange(K):                                # For each trial,
        spike_times = where(n[k,:]==1)                 # ...find the spikes.
        n_spikes = size(spike_times)                   # ...determine number of spikes.
        spike_times_random = spike_times[0][permutation(n_spikes)]    # ...permute spikes indices,
        n_remove=int(floor(thinning_factor*n_spikes))  # ... determine number of spikes to remove,
        n_thinned[k,spike_times_random[0:n_remove-1]]=0   # remove the spikes.
    return n_thinned

dt = t[1]-t[0]                     # Define the sampling interval.
fNQ = 1/dt/2                       # Define Nyquist frequency.
Wn = [9,11]                        # Set the passband
ord  = 100                         # ...and filter order,
b = signal.firwin(ord, Wn, nyq=fNQ, pass_zero=False, window='hamming');

phi=zeros([K,N])                # Create variable to hold phase.
for k in arange(K):             # For each trial,
    Vlo = signal.filtfilt(b, 1, y[k,:])       # ... apply the filter,
    phi[k,:] = angle(signal.hilbert(Vlo))  # ... and compute the phase.

n_reshaped   = copy(n)                     # Make a copy of the spike data.
n_reshaped   = reshape(n_reshaped,-1)      # Convert spike matrix to vector.
phi_reshaped = reshape(phi, -1)            # Convert phase matrix to vector.
                                              # Create a matrix of predictors [1, cos(phi), sin(phi)]
X            = transpose([ones(shape(phi_reshaped)), cos(phi_reshaped), sin(phi_reshaped)])
Y            = transpose([n_reshaped])     # Create a vector of responses.

model = sm.GLM(Y,X,family=sm.families.Poisson())    # Build the GLM model,
res   = model.fit()                                 # ... and fit it.


phi_predict = linspace(-pi, pi, 100)
X_predict   = transpose([ones(shape(phi_predict)), cos(phi_predict), sin(phi_predict)])
Y_predict   = res.get_prediction(X_predict, linear='False')
FTA, phi_axis = FTA_function(y,n,t,Wn)       #Compute the FTA.



Wn = [44,46]                       # Set the passband --------------------------------------------------This is very specific for the file at hand. note is to change this
b = signal.firwin(ord, Wn, nyq=fNQ, pass_zero=False, window='hamming');

del phi
phi=zeros([K,N])                # Create variable to hold phase.
for k in arange(K):             # For each trial,
    Vlo = signal.filtfilt(b, 1, y[k,:])       # ... apply the filter, ----------------------------------------going to have to make my own filter
    phi[k,:] = angle(signal.hilbert(Vlo))  # ... and compute the phase.

n_reshaped   = copy(n)
n_reshaped   = reshape(n_reshaped,-1)   # Convert spike matrix to vector.
phi_reshaped = reshape(phi, -1)         # Convert phase matrix to vector.
                                           # Create a matrix of predictors [1, cos(phi), sin(phi)]
X            = transpose([ones(shape(phi_reshaped)), cos(phi_reshaped), sin(phi_reshaped)])
Y            = transpose([n_reshaped])  # Create a vector of responses.

model = sm.GLM(Y,X,family=sm.families.Poisson())    # Build the GLM model,
res   = model.fit()                                 # ... and fit it,

phi_predict = linspace(-pi, pi, 100)       # ... and evaluate the model results.
X_predict   = transpose([ones(shape(phi_predict)), cos(phi_predict), sin(phi_predict)])
Y_predict   = res.get_prediction(X_predict, linear='False')

FTA, phi_axis = FTA_function(y,n,t,Wn)       #Compute the FTA, in the new frequency interval

plot(phi_axis, FTA)                          #... and plot it, along with the model fit.
plot(phi_predict, Y_predict.predicted_mean, 'k')
plot(phi_predict, Y_predict.conf_int(), 'k:')
xlabel('Phase')
ylabel('Probability of a spike')

show()