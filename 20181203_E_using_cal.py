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
plt.ion()

calHdf5Filename = "/data/20181203_C/20181203_C_mass.hdf5"
toAnalyzeLJHFilename = "/data/20181203_E/20181203_E_chan1.ljh" # point to one specific channel, it will find all channels
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
    ds.apply_cuts(dsCal.saved_auto_cuts)
    for key in dsCal.p_filt_value_dc.attrs.keys():
        ds.p_filt_value_dc.attrs[key]=dsCal.p_filt_value_dc.attrs[key]
    if len(ds.p_filt_value_dc.attrs) != 3:
        data.set_chan_bad(ds.channum,"failed to load drift correction from cal file")
        continue
    ds.calibration = dsCal.calibration
data.filter_data()
for ds in data:
    ds._apply_drift_correction()
data.convert_to_energy("p_filt_value_dc","p_filt_value_phc")


# lets select channels that agree with each other fairly well
ws=devel.WorstSpectra(data,np.arange(4500))
ws.plot()
plt.title(data.shortname()+", before exluding worst spectra")
medianChisq = np.nanmedian(ws.chisqdict.values())
marked = []
for ds in data:
    if ws.chisqdict[ds.channum] > 4*medianChisq:
        data.set_chan_bad(ds.channum,"WorstSpectra chisq too high")
        marked.append(ds.channum)
    elif np.isnan(ws.chisqdict[ds.channum]):
        data.set_chan_bad(ds.channum,"chisq = nan")
        marked.append(ds.channum)

ws.plot()
plt.title(data.shortname()+", after excluding worst spectra")

data.plot_hist(np.arange(6000),"p_energy")
plt.xlabel("energy (eV)")
plt.ylabel("counts per bin")
plt.yscale("log")
plt.ylim(0.9,plt.ylim()[1])


ds = data.first_good_dataset
plt.figure()
plt.plot(ds.p_timestamp[ds.good()]-ds.p_timestamp[0],ds.p_pretrig_mean[ds.good()],".")
plt.xlabel("time since start of file (s)")
plt.ylabel("pretrigger mean (arb)")

# data.linefit("AlKAlpha")
