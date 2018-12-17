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
import fastdtw
import scipy
import collections
import ebit

dirname = "/Users/oneilg/Documents/GB EBIT/dataH/20181205_H" # pulse file
ndirname = "/Users/oneilg/Documents/GB EBIT/dataH/20181205_A" # noise file
maxChans = 244
delete_hdf5_file_before_analysis = True
filtValueChoice = "p_filt_value" # use p_filt_value mostly, try p_filt_value_dc on longer, higher count-rate dataHsets
referenceChannelNumber = 3
referenceLines = collections.OrderedDict() # map filt value or either energy or line name
referenceLines[2002]="O He-Like 1s2p + 1s2s"
referenceLines[2278]="O H-Like 2p"
referenceLines[2693]="O H-Like 3p"
energyResolutionThreshold = 4.8
energyResolutionThresholdLine = "O H-Like 2p"
binEdgesFiltValue = np.arange(100,8000,4)
deleteHDF5BeforeAnalysis = True

basename,_ = mass.ljh_util.ljh_basename_channum(dirname)
available_chans = mass.ljh_util.ljh_get_channels_both(dirname, ndirname)
chan_nums = available_chans[:min(maxChans, len(available_chans))]
pulse_files = [s+".ljh" for s in mass.ljh_util.ljh_chan_names(dirname, chan_nums)]
noise_files = [s+".ljh" for s in mass.ljh_util.ljh_chan_names(ndirname, chan_nums)]

plt.close("all")
plt.interactive(True)

hdf5filename = mass.core.channel_group._generate_hdf5_filename(pulse_files[0])
if delete_hdf5_file_before_analysis:
    if os.path.isfile(hdf5filename): os.remove(hdf5filename)
dataH = mass.TESGroup(pulse_files,noise_files,hdf5_filename=hdf5filename)
dataH.set_chan_good(dataH.why_chan_bad.keys())
dataH.summarize_data(forceNew=False)
dataH.auto_cuts()
dataH.avg_pulses_auto_masks(forceNew=False)
# for ds in dataH: ds._use_new_filters=False
dataH.compute_filters(forceNew=False)
dataH.filter_data(forceNew=False)

toAnalyzeLJHFilename = "/Users/oneilg/Documents/GB EBIT/dataI/20181205_I/20181205_I_chan1.ljh" # point to one specific channel, it will find all channels


basename,_ = mass.ljh_util.ljh_basename_channum(toAnalyzeLJHFilename)
channels = mass.ljh_util.ljh_get_channels(toAnalyzeLJHFilename)
filenames = [basename+"_chan%i.ljh"%i for i in channels]
toAnalyzeHDF5Filename = basename+"_mass_external_cal.hdf5"
if os.path.isfile(toAnalyzeHDF5Filename) and deleteHDF5BeforeAnalysis:
    os.remove(toAnalyzeHDF5Filename)
dataI = mass.TESGroup(filenames[:],hdf5_filename=toAnalyzeHDF5Filename)
dataI.summarize_dataI()
dataI.set_chan_bad(dataH.why_chan_bad.keys(),"was bad in calHdf5File %s"%calHdf5Filename)
for ds in dataI:
    if not ds.channum in dataH.channel:
        dataI.set_chan_bad(ds.channum, "cannot load channel %i from dataH"%ds.channum)
        continue
    dsCal = dataH.channel[ds.channum]
    ds.filter = dsCal.filter
    # ds.apply_cuts(dsCal.saved_auto_cuts)
    for key in dsCal.p_filt_value_dc.attrs.keys():
        ds.p_filt_value_dc.attrs[key]=dsCal.p_filt_value_dc.attrs[key]
    if len(ds.p_filt_value_dc.attrs) != 3:
        dataI.set_chan_bad(ds.channum,"failed to load drift correction from cal file")
        continue
    ds.calibration = dsCal.calibration
dataI.filter_dataI()
dataI.drift_correct(forceNew=True)
dataI.convert_to_energy("p_filt_value",calname="p_filt_value")


# filter channels based on energy resolution at a specific line
# and line position
fitters = collections.OrderedDict()
resolutions = collections.OrderedDict()
energyResolutionBinsize = 0.2
energyResolutionThreshold = 4.8
for ds in dataI:
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
        dataI.set_chan_bad(channum, "fwhm resolution at O H-Like 2p > {} eV".format(energyResolutionThreshold))
linePositionDeviations = [fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy for fitter in fitters.values()]
deviationThreshold = 2
for (channum, fitter) in fitters.items():
    absDeviation = np.abs(fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy)
    if absDeviation>deviationThreshold:
        dataI.set_chan_bad(channum, "line position of O H-Like 2p wrong by more than > {} eV".format(deviationThreshold))

# make plots of coadded spectrum, as well as coadded line fits
dataI.plot_hist(np.arange(4000),attr="p_energy")
fitter = dataI.linefit("O H-Like 3p",dlo=15,dhi=10,holdvals={"dP_dE":1})
fitter = dataI.linefit("O H-Like 2p",dlo=15,dhi=10,holdvals={"dP_dE":1})
fitter = dataI.linefit("O He-Like 1s2p + 1s2s",dlo=15,dhi=10,holdvals={"dP_dE":1})
fitter = dataI.linefit("O He-Like 1s3p",dlo=6,dhi=10,holdvals={"dP_dE":1})
fitter = dataI.linefit("O He-Like 1s4p",dlo=5,dhi=8,holdvals={"dP_dE":1})

# plot all remaining channels, inspect this plot to make sure all channels
# appear aligned
ws=devel.WorstSpectra(dataI,np.arange(4000))
ws.plot()

# plot pretrig mean and energy vs time, combined across dataIsets
ds = dataI.channel[3]
dsCal = dataH.channel[3]
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


# save coadded dataI in easy to read formats
outputDirName = dataI.shortname().split(",")[0]
if not os.path.isdir(outputDirName):
    os.mkdir(outputDirName)
with h5py.File(os.path.join(outputDirName,outputDirName+"_prelim.h5"),"w") as h5:
    bin_centers, counts = dataI.hist(np.arange(4000))
    h5["bin_centers"]=bin_centers
    h5["counts"]=counts
np.savetxt(os.path.join(outputDirName,outputDirName+"_prelim.yuri"),
np.vstack((bin_centers,counts)).T,
header="#bin_centers (eV), counts")

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(os.path.join(outputDirName,'figure%d.png' % i)  )
dataI.hdf5_file.flush()
