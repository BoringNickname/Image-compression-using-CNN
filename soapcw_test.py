#%%
"""RUN THE SOAP SEARCH SECTION. NO MODULE NAMED CW OR TRANSITION MATRIX, which is pretty important

soap.line_aware_stat.gen_lookup_python.LineAwareStatistic has no keyword arg ndet
"""
import soapcw as soap
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from soapcw import cw

#SEARCH FOR A SINUSOID IN RANDOM NOISE
simp_nsft = 100
simp_freqrange = 70
spect = np.random.normal(size = (simp_nsft, simp_freqrange))
signal_track = np.linspace(20,50,simp_nsft).astype(np.int32)
for t,f in enumerate(signal_track):
    spect[t,f]+=2

simp_tr = np.log([0,1,0])
simp_track = soap.single_detector(simp_tr, spect)
fg = soap.plots.plot_single(spect, soapout=simp_track)
#%%
#TRANSITION MATRIX
import soapcw as soap
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

powers = np.linspace(1,400,10)
v1d = soap.line_aware_stat.gen_lookup_python.LineAwareStatistic(powers, pvs = 1.5, pvl = 1.5, ratio = 0.3)

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(powers, v1d.signoiseline)
ax.set_xlabel('normalized spectrogram power')
ax.set_ylabel('line aware statistic')
ax.grid()
# %%
# v2d = soap.line_aware_stat.gen_lookup_python.LineAwareStatistic(powers, ndet =2, pvs=1.5, pvl=1.5, ratio=0.3)

# fig, ax = plt.subplots(figsize = (12,8))
# ax.imshow(np.log(v2d.signoiseline), origin = 'lower', extent = [powers.min(), powers.max(), powers.min(), powers.max()])
# ax.set_xlabel("normalised spectogram power (det1)")
# ax.set_ylabel("normalised spectogram power (det2)")
# %%
#a way to generate signals?
data = np.abs(np.random.normal(scale = 4, size = (20,30)))
data[:,11] +=8

transition_matrix = np.log([0,1,0])
print(transition_matrix)
track = soap.single_detector(transition_matrix, data)


fig, ax = plt.subplots(figsize = (15,7))
sns.heatmap(data.T,ax=ax,cbar=True,cmap="YlGnBu",cbar_kws={'label': 'FFT power'})
# 0.5 added so at center of bin not at edge
# sns.scatterplot(np.arange(len(track.opt_path)) + 0.5,track.opt_path+0.5,color="red")
ax.invert_yaxis()
ax.set_ylabel("Frequency")
ax.set_xlabel("Time")
# %%
soap.tools.transition_matrix(1.1,1e400,1e400)
# %%
# change the number of elements to reduce generation time (500 -> 100)
powers = np.linspace(1,200,500)
v1d = soap.line_aware_stat.gen_lookup_python.LineAwareStatistic(powers,ndet=1,pvs=2.0,pvl=9.0,ratio=0.0)
v1d.save_lookup("./")
# %%
from soapcw import cw
import matplotlib.pyplot as plt
import numpy as np

sig = cw.GenerateSignal()
# define signal parameters
sig.alpha = 3.310726752188296
sig.delta = -0.8824241920781501
sig.cosi = -0.63086
sig.phi0 = 4.007
sig.psi = 0.52563
sig.f = [100.05,-1e-17,0]
sig.tref = 946339148.816094
sig.h0 = 3e-24

nsft, tstart, tsft, flow, fhigh = 22538, 931042949, 1800., 100.0,100.1
snr= 200
spect = sig.get_spectrogram(tstart = tstart, nsft=nsft,tsft=tsft,fmin=flow,fmax=fhigh,dets=["H1"],snr=snr)
spect.sum_sfts()

fig, ax = plt.subplots(nrows=2,figsize=(14,10))
ax[0].imshow(spect.H1.norm_sft_power.T,aspect="auto",origin="lower",extent=[spect.epochs.min(),spect.epochs.max(),spect.frequencies.min(),spect.frequencies.max()],cmap="YlGnBu")
ax[1].imshow(spect.H1.summed_norm_sft_power.T,aspect="auto",origin="lower",extent=[spect.epochs.min(),spect.epochs.max(),spect.frequencies.min(),spect.frequencies.max()],cmap="YlGnBu")
ax[0].set_xlabel("GPS time [s]",fontsize=20)
ax[0].set_ylabel("Frequency [Hz]",fontsize=20)
ax[1].set_xlabel("GPS time [s]",fontsize=20)
ax[1].set_ylabel("Frequency [Hz]",fontsize=20)

# define ephemeredies (optional the default is below)
#sig.earth_ephem = "earth00-19-DE405.dat.gz"
#sig.sun_ephem = "sun00-19-DE405.dat.gz"

# %%
