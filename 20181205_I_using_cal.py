import mass
import numpy as np
import pylab as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import h5py
from analyze_switcher_spectra_class import MassSidesAnalysis
import devel
from devel import Side, Sides, RepeatedLinePlotter, WorstSpectra, PredictedVsAchieved
from collections import OrderedDict
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
import pprint
import time
import ebit
import collections
plt.ion()
plt.close("all")

calHdf5Filename = "/Users/oneilg/Documents/GB EBIT/data/20181205_H/20181205_H_mass.hdf5"
toAnalyzeLJHFilename = "/Users/oneilg/Documents/GB EBIT/data/20181205_I/20181205_I_chan1.ljh" # point to one specific channel, it will find all channels
deleteHDF5BeforeAnalysis = True


dataCal = mass.TESGroupHDF5(calHdf5Filename, read_only=True)
basename,_ = mass.ljh_util.ljh_basename_channum(toAnalyzeLJHFilename)
channels = mass.ljh_util.ljh_get_channels(toAnalyzeLJHFilename)
filenames = [basename+"_chan%i.ljh"%i for i in channels]
toAnalyzeHDF5Filename = basename+"_mass_external_cal.hdf5"
if os.path.isfile(toAnalyzeHDF5Filename) and deleteHDF5BeforeAnalysis:
    os.remove(toAnalyzeHDF5Filename)
data = mass.TESGroup(filenames[:],hdf5_filename=toAnalyzeHDF5Filename)
data.summarize_data()
data.set_chan_bad(dataCal.why_chan_bad.keys(),"was bad in calHdf5File %s"%calHdf5Filename)
for ds in data:
    if not ds.channum in dataCal.channel:
        data.set_chan_bad(ds.channum, "cannot load channel %i from dataCal"%ds.channum)
        continue
    dsCal = dataCal.channel[ds.channum]
    ds.filter = dsCal.filter
    # ds.apply_cuts(dsCal.saved_auto_cuts)
    for key in dsCal.p_filt_value_dc.attrs.keys():
        ds.p_filt_value_dc.attrs[key]=dsCal.p_filt_value_dc.attrs[key]
    if len(ds.p_filt_value_dc.attrs) != 3:
        data.set_chan_bad(ds.channum,"failed to load drift correction from cal file")
        continue
    ds.calibration = dsCal.calibration
data.filter_data()
data.drift_correct(forceNew=True)
data.convert_to_energy("p_filt_value_dc",calname="p_filt_value")


# filter channels based on energy resolution at a specific line
# and line position
fitters = collections.OrderedDict()
resolutions = collections.OrderedDict()
energyResolutionBinsize = 0.2
energyResolutionThreshold = 4.8
for ds in data:
    fitter = ds.linefit("O H-Like 2p",dlo=15,dhi=10,holdvals={"dP_dE":1},plot=False)
    fitters[ds.channum]=fitter
    resolutions[ds.channum]=fitter.last_fit_params_dict["resolution"][0]
plt.figure()
plt.hist(resolutions.values(),np.arange(0,energyResolutionThreshold+energyResolutionBinsize,energyResolutionBinsize),label="kept")
plt.hist(resolutions.values(),np.arange(energyResolutionThreshold,10,energyResolutionBinsize),label="excluded")
plt.xlabel("energy resolution (eV)")
plt.ylabel("pixels per bin")
plt.vlines([energyResolutionThreshold],plt.ylim()[0],plt.ylim()[1])
for (channum, res) in resolutions.items():
    if res>energyResolutionThreshold:
        data.set_chan_bad(channum, "fwhm resolution at O H-Like 2p > {} eV".format(energyResolutionThreshold))
linePositionDeviations = [fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy for fitter in fitters.values()]
deviationThreshold = 2
for (channum, fitter) in fitters.items():
    absDeviation = np.abs(fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy)
    if absDeviation>deviationThreshold:
        data.set_chan_bad(channum, "line position of O H-Like 2p wrong by more than > {} eV".format(deviationThreshold))

# make plots of coadded spectrum, as well as coadded line fits
data.plot_hist(np.arange(4000),attr="p_energy")
fitter = data.linefit("O H-Like 3p",dlo=15,dhi=10,holdvals={"dP_dE":1})
fitter = data.linefit("O H-Like 2p",dlo=15,dhi=10,holdvals={"dP_dE":1})
fitter = data.linefit("O He-Like 1s2p + 1s2s",dlo=15,dhi=10,holdvals={"dP_dE":1})
fitter = data.linefit("O He-Like 1s3p",dlo=6,dhi=10,holdvals={"dP_dE":1})
fitter = data.linefit("O He-Like 1s4p",dlo=5,dhi=8,holdvals={"dP_dE":1})

# plot all remaining channels, inspect this plot to make sure all channels
# appear aligned
ws=devel.WorstSpectra(data,np.arange(4000))
ws.plot()

# plot pretrig mean and energy vs time, combined across datasets
ds = data.channel[3]
dsCal = dataCal.channel[3]
plt.figure()
plt.plot(ds.p_timestamp[ds.good()]-ds.p_timestamp[0],ds.p_pretrig_mean[ds.good()],".",label=ds.shortname())
plt.plot(dsCal.p_timestamp[dsCal.good()]-ds.p_timestamp[0],dsCal.p_pretrig_mean[dsCal.good()],".",label=dsCal.shortname())
plt.xlabel("timestamp (s)")
plt.ylabel("p_pretrig_mean")
plt.title(ds.shortname()+dsCal.shortname())

plt.figure()
plt.plot(ds.p_timestamp[ds.good()]-ds.p_timestamp[0],ds.p_energy[ds.good()],".",label=ds.shortname())
plt.plot(dsCal.p_timestamp[dsCal.good()]-ds.p_timestamp[0],dsCal.p_energy[dsCal.good()],".",label=dsCal.shortname())
plt.xlabel("timestamp (s)")
plt.ylabel("p_energy")
plt.title(ds.shortname()+dsCal.shortname())


# save coadded data in easy to read formats
outputDirName = data.shortname().split(",")[0]
if not os.path.isdir(outputDirName):
    os.mkdir(outputDirName)
with h5py.File(os.path.join(outputDirName,outputDirName+"_prelim.h5"),"w") as h5:
    bin_centers, counts = data.hist(np.arange(4000))
    h5["bin_centers"]=bin_centers
    h5["counts"]=counts
np.savetxt(os.path.join(outputDirName,outputDirName+"_prelim.yuri"),
np.vstack((bin_centers,counts)).T,
header="#bin_centers (eV), counts")

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(os.path.join(outputDirName,'figure%d.png' % i)  )
data.hdf5_file.flush()
