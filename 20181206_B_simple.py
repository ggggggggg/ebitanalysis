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

dirname = "/data/20181206_D"
ndirname = "/data/20181206_C"
maxChans = 240
delete_hdf5_file_before_analysis = True

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
data = mass.TESGroup(pulse_files,noise_files,hdf5_filename=hdf5filename)
data.summarize_data(forceNew=False)
data.auto_cuts()
data.avg_pulses_auto_masks(forceNew=False)
# for ds in data: ds._use_new_filters=False
data.compute_filters(forceNew=False)
data.filter_data(forceNew=False)
data.drift_correct(forceNew=False)
# data.phase_correct(forceNew=False)

data.calibrate("p_filt_value_dc",["CuKAlpha"],diagnose=False,forceNew=True,
fit_range_ev=10,
size_related_to_energy_resolution=10)

# lets select channels that agree with each other fairly well
ws=devel.WorstSpectra(data,np.arange(4000))
ws.plot()
ymax = plt.ylim()[1]
plt.ylim(0,ymax)
plt.title(data.shortname()+", before exluding worst spectra")
# medianChisq = np.nanmedian(ws.chisqdict.values())
# marked = []
# if np.sum([1 for ds in data])>160:
#     for ds in data:
#         if ws.chisqdict[ds.channum] > np.percentile(ws.chisqdict.values(),.7):
#             data.set_chan_bad(ds.channum,"WorstSpectra chisq too high")
#             marked.append(ds.channum)
#         elif np.isnan(ws.chisqdict[ds.channum]):
#             data.set_chan_bad(ds.channum,"chisq = nan")
#             marked.append(ds.channum)
# ws.plot()
# plt.title(data.shortname()+", after excluding worst spectra")

# data.set_chan_bad([129, 133, 7, 137, 269, 17, 149, 151, 153, 27, 285, 289, 35, 165,
#  295, 39, 169, 193, 43, 257, 301, 47, 49, 307, 181, 265, 57, 317, 63, 65, 203, 197, 199,
#  75, 333, 79, 335, 291, 85, 343, 89, 207, 135, 221, 351, 379, 373, 103, 105, 167, 281, 367,
#  243, 117, 233, 377, 255, 247, 213],"spectra don't agree that well, need more cuts?")

data.plot_hist(np.arange(4000),"p_energy",label_lines=[])
plt.xlabel("energy (eV)")
plt.ylabel("counts per bin")
plt.yscale("log")
plt.ylim(0.9,plt.ylim()[1])


ds = data.first_good_dataset
plt.figure()
plt.plot(ds.p_timestamp[ds.good()]-ds.p_timestamp[0],ds.p_pretrig_mean[ds.good()],".")
plt.xlabel("time since start of file (s)")
plt.ylabel("pretrigger mean (arb)")
plt.title(ds.shortname())

ds.plot_hist(np.arange(4000),"p_energy",label_lines=[])
g = np.logical_and(ds.p_energy[:]<10,ds.good())
plt.find(g)
ds.plot_traces(plt.find(g)[:10])
# figured out that these bad pulses have really short rise time
# ds.cuts.cut("RISE_TIME_MS",g)

with h5py.File("hists/"+data.shortname()+"_prelim.h5","w") as h5:
    bin_centers, counts = data.hist(np.arange(4000))
    h5["bin_centers"]=bin_centers
    h5["counts"]=counts

np.savetxt("hists/"+data.shortname()+"_prelim.yuri",
np.vstack((bin_centers,counts)).T,
header="#bin_centers (eV), counts")