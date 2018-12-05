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

dirname = "/data/20181203_A"
ndirname = "/data/20181203_B"
maxChans = 244
delete_hdf5_file_before_analysis = True

basename,_ = mass.ljh_util.ljh_basename_channum(dirname)
available_chans = mass.ljh_util.ljh_get_channels_both(dirname, ndirname)
chan_nums = available_chans[:min(maxChans, len(available_chans))]
pulse_files = [s+".ljh" for s in mass.ljh_util.ljh_chan_names(dirname, chan_nums)]
noise_files = [s+".ljh" for s in mass.ljh_util.ljh_chan_names(ndirname, chan_nums)]

plt.close("all")
plt.interactive(True)

if delete_hdf5_file_before_analysis:
    hdf5filename = mass.core.channel_group._generate_hdf5_filename(pulse_files[0])+"tmp"
    if os.path.isfile(hdf5filename): os.remove(hdf5filename)
data = mass.TESGroup(pulse_files,noise_files,hdf5_filename=hdf5filename)
data.summarize_data(forceNew=False)
data.auto_cuts()
data.avg_pulses_auto_masks(forceNew=False)
# for ds in data: ds._use_new_filters=False
data.compute_filters(forceNew=False)
data.filter_data(forceNew=False)
data.drift_correct(forceNew=False)
data.phase_correct(forceNew=False)

data.calibrate("p_filt_value_phc",[1021.801114,1210.9164766667,1277.1110853333,1307.7473793333,1324.387893,
                                    915.0096, 1071.7814, 1126.2724],diagnose=False,forceNew=True,fit_range_ev=50)
data.convert_to_energy("p_filt_value_phc")




# lets select channels that agree with each other fairly well
ws=devel.WorstSpectra(data,np.arange(3000))
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


data.plot_hist(np.arange(2200),"p_energy",
label_lines=["OKAlpha",1022])
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

data.linefit("AlKAlpha")
