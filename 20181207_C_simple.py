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

# Inputs that can change at the start of each analysis
dirname = "/data/20181207_C" # directory where we find the data to be analyzed
ndirname = "/data/20181207_D" # the associated directory for the noise
maxChans = 384  # Max channels so you can limit for realitime anaylsis (maybe? confirm later)
delete_hdf5_file_before_analysis = False # Write a new file for the analysis


# Find the data from the directory given
basename,_ = mass.ljh_util.ljh_basename_channum(dirname)
available_chans = mass.ljh_util.ljh_get_channels_both(dirname, ndirname)
chan_nums = available_chans[:min(maxChans, len(available_chans))]
pulse_files = [s+".ljh" for s in mass.ljh_util.ljh_chan_names(dirname, chan_nums)]
noise_files = [s+".ljh" for s in mass.ljh_util.ljh_chan_names(ndirname, chan_nums)]

# close plots existing and allow changing of plots produced
plt.close("all")
plt.interactive(True)

# make an hdf5 which stores the information about the analysis
hdf5filename = mass.core.channel_group._generate_hdf5_filename(pulse_files[0])
if delete_hdf5_file_before_analysis:
    if os.path.isfile(hdf5filename): os.remove(hdf5filename)

# Pull data from files
data = mass.TESGroup(pulse_files,noise_files,hdf5_filename=hdf5filename)
data.summarize_data(forceNew=False)
data.auto_cuts()                            # cut data based on criteria given in ppt
data.avg_pulses_auto_masks(forceNew=False)
# for ds in data: ds._use_new_filters=False
data.compute_filters(forceNew=False)        # compute filters based on average of median pulses
data.filter_data(forceNew=False)            # apply the filter
data.drift_correct(forceNew=False)          # correct for drift of baseline during collection
# data.phase_correct(forceNew=False)        # correct for error between arrrival time and sampling rate

# Choose a channel (3) and identify it's peaks. From there, line up the peaks in every other channel to this one
def cal_b_to_a(ds_a,ds_b,npeaks,attr="p_filt_value_dc",diagnose_plots=False):
    bin_centers, counts_a = ds_a.hist(np.arange(0,16000,4),attr)
    bin_centers, counts_b = ds_b.hist(np.arange(0,16000,4),attr)   
    distance, path = fastdtw.fastdtw(counts_a, counts_b)
    i_a = np.array([x[0] for x in path])
    i_b = np.array([x[1] for x in path])
    cc = counts_a[i_a]*counts_b[i_b]

    peaks_inds_cc, properties = scipy.signal.find_peaks(cc,prominence=2)
    peakscounts = cc[peaks_inds_cc]
    peaks_inds_ccsorted = peaks_inds_cc[np.argsort(peakscounts)][::-1][:npeaks]
    peaks_inds_a = i_a[peaks_inds_ccsorted]
    peaks_inds_b = i_b[peaks_inds_ccsorted]    

    phs_a = bin_centers[peaks_inds_a]
    phs_b = bin_centers[peaks_inds_b]
    cal_b_to_a = mass.EnergyCalibration()

    for ph_a,ph_b in zip(phs_a, phs_b):
        cal_b_to_a.add_cal_point(ph_b,ph_a)

    attr_b_in_a_units = cal_b_to_a(getattr(ds_b,attr))
    newattr = attr+"_ch%i"%ds_a.channum
    setattr(ds_b, newattr, attr_b_in_a_units)
    ds.calibration[attr+"_to_"+newattr]=cal_b_to_a


    bin_centers,counts_b_in_a_units = ds_b.hist(np.arange(0,16000,4),newattr)

    if diagnose_plots:
        # plt.figure()
        # plt.plot(cc)
        # plt.plot(peaks_inds_ccsorted,cc[peaks_inds_ccsorted],".")
        # plt.xlabel(attr)
        # plt.ylabel("counts per %0.2f unit bin"%(bin_centers[1]-bin_centers[0]))

        # plt.figure()
        # plt.plot(bin_centers,counts_a,label="channel %i"%ds_a.channum)
        # for i,pi in enumerate(peaks_inds_a):
        #     plt.plot(bin_centers[pi],counts_a[pi],".",color=plt.cm.gist_ncar(float(i)/len(peaks_inds)))

        # plt.plot(bin_centers,counts_b,label="channel %i"%ds_b.channum)
        # for i,pi in enumerate(peaks_inds_b):
        #     plt.plot(bin_centers[pi],counts_b[pi],".",color=plt.cm.gist_ncar(float(i)/len(peaks_inds)))
        # plt.xlabel(attr+" channel %i"%ds_a.channum)
        # plt.ylabel("counts per %0.2f unit bin"%(bin_centers[1]-bin_centers[0]))
        # plt.legend()

        plt.figure(figsize=(20,10))
        plt.plot(bin_centers,counts_a,label="channel %i"%ds_a.channum)
        plt.plot(bin_centers,counts_b_in_a_units,label="channel %i"%ds_b.channum)
        plt.xlabel(attr+" channel %i"%ds_a.channum)
        plt.ylabel("counts per %0.2f unit bin"%(bin_centers[1]-bin_centers[0]))
        plt.legend()


# a metric for the distance between the matchings from cal_b_to_a
def dtw_distance(ds_a,ds_b,bin_edges,attr="p_filt_value_dc"):
    bin_centers, counts_a = ds_a.hist(bin_edges,attr)
    bin_centers, counts_b = ds_b.hist(bin_edges,attr)   
    distance, path = fastdtw.fastdtw(counts_a, counts_b)
    return distance

#data.set_chan_bad([83],"Error in extranneous peak")

# match each channel to channel 3
for ds in data:
        cal_b_to_a(ds_a=data.channel[3],ds_b=ds,npeaks=9,diagnose_plots=False)


for ds in data:
    ds.distance = dtw_distance(data.channel[3],ds,np.arange(0,16000,4),attr="p_filt_value_dc_ch3")

distance = [ds.distance for ds in data]
plt.plot(sorted(distance),".")

bin_centers, counts = data.hist(np.arange(0,16000,4),"p_filt_value_dc_ch3")
peaks_inds, properties = scipy.signal.find_peaks(counts,prominence=2)
peakscounts = counts[peaks_inds]
peaks_inds_sorted = peaks_inds[np.argsort(peakscounts)][::-1][:9]

plt.figure()
plt.plot(bin_centers,counts)
plt.plot(bin_centers[peaks_inds_sorted], counts[peaks_inds_sorted],".")
for pi in peaks_inds_sorted:
    print("ph = %f, counts = %g"%(bin_centers[pi],counts[pi]))

calR = mass.EnergyCalibration()
calR.add_cal_point(2486,653,"O H-like 2p")
calR.add_cal_point(2185.4,574,"O He-like 1s2p")
calR.add_cal_point(2945,774,"O H-like 3p")


for ds in data:
    ds.calibration["p_filt_value_dc_ch3"]=calR
    cal = mass.EnergyCalibration()
    for ph, e, name in zip(calR._ph,calR._energies, calR._names):
        cal.add_cal_point(ds.calibration["p_filt_value_dc_to_p_filt_value_dc_ch3"].energy2ph(ph),e,name)
    ds.calibration["p_filt_value_dc"]=cal
    cal.save_to_hdf5(ds.hdf5_group["calibration"],"p_filt_value_dc")




data.convert_to_energy("p_filt_value_dc")
data.plot_hist(np.arange(4000),label_lines=["OKAlpha"])
plt.xlabel("energy (eV)")
plt.ylabel("counts per bin")


def ksdist(counts_a,counts_b):
    counts_a_norm_cumsum = np.cumsum(counts_a/float(np.sum(counts_a)))
    counts_b_norm_cumsum = np.cumsum(counts_a/float(np.sum(counts_b)))
    ydiff = counts_a_norm_cumsum-counts_b_norm_cumsum
    return np.amin(ydiff), np.amax(ydiff)

# lets select channels that agree with each other fairly well
ws=devel.WorstSpectraDTW(data,np.arange(4000))
ws.plot()
ymax = plt.ylim()[1]
for e in calR._energies:
    plt.plot([e,e],[0,ymax],"k--")
plt.ylim(0,ymax)
plt.title(data.shortname()+", before exluding worst spectra")

with h5py.File("hists/"+data.shortname()+"_prelim.h5","w") as h5:
    bin_centers, counts = data.hist(np.arange(4000))
    h5["bin_centers"]=bin_centers
    h5["counts"]=counts
