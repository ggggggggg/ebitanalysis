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

dirname = "/data/20181207_A"
ndirname = "/data/20181207_B"
maxChans = 244
delete_hdf5_file_before_analysis = False

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

def cal_b_to_a(ds_a,ds_b,npeaks,attr="p_filt_value_dc",diagnose_plots=False, bin_edges = np.arange(100,20000,4)):
    bin_centers, counts_a = ds_a.hist(bin_edges,attr)
    bin_centers, counts_b = ds_b.hist(bin_edges,attr)   
    if counts_b.sum() < 100:
        print("too few counts_b, sum=%g"%counts_b.sum())
        return
    distance, path = fastdtw.fastdtw(counts_a, counts_b)
    i_a = np.array([x[0] for x in path])
    i_b = np.array([x[1] for x in path])
    cc = counts_a[i_a]*counts_b[i_b]

    peaks_inds_cc, properties = scipy.signal.find_peaks(cc,prominence=4,distance=10)
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
    ds_b.calibration[attr+"_to_"+newattr]=cal_b_to_a


    bin_centers,counts_b_in_a_units = ds_b.hist(bin_edges,newattr)

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
        plt.plot(bin_centers[peaks_inds_a],counts_a[peaks_inds_a],"s")
        plt.plot(bin_centers[peaks_inds_a],counts_b_in_a_units[peaks_inds_a],"*")
        plt.xlabel(attr+" channel %i"%ds_a.channum)
        plt.ylabel("counts per %0.2f unit bin"%(bin_centers[1]-bin_centers[0]))
        plt.legend()

for ds in data:
    cal_b_to_a(ds_a=data.channel[3],ds_b=ds,npeaks=10,diagnose_plots=False)

goodchans = []
badchans = []
for ds in data:
    if ds.good().sum()<300:
        data.set_chan_bad(ds.channum,"<300 good counts")
    #modify each ds to have an attribute called ds.p_filt_value_dc_ch3 
    cal_b_to_a(ds_a=data.channel[3],ds_b=ds,npeaks=10,diagnose_plots=True)
    while True:
        v = raw_input("does this match look good?")
        if v == "y":
            goodchans.append(ds.channum)
            plt.close()
            break
        if v == "n":
            badchans.append(ds.channum)
            plt.close()
            break
        else:
            print("y or n")

goodchans = [3, 5, 7, 11, 13, 15, 17, 21, 23, 25, 27, 29, 31, 33, 37, 41, 43, 49, 53, 55, 59, 61, 65, 73, 75, 77, 79, 81, 83, 85, 89, 93, 95, 97, 99, 101, 105, 109, 117, 121, 123, 125, 127, 133, 135, 149, 153, 155, 159, 163, 165, 167, 173, 175, 177, 185, 191, 199, 201, 205, 211, 213, 215, 217, 219, 221, 223, 227, 229, 231, 235, 237, 239, 241, 247, 251, 253, 257, 259, 261, 263, 267, 273, 275, 279, 281, 285, 287, 291, 293, 299, 305, 307, 309, 313, 319, 321, 323, 325, 331, 335, 337, 339, 341, 343, 353, 355, 361, 365, 367, 371, 377]
badchans = [1, 9, 19, 39, 45, 47, 51, 57, 67, 69, 71, 91, 103, 107, 111, 115, 119, 129, 131, 137, 141, 143, 145, 147, 151, 157, 161, 169, 179, 181, 183, 187, 193, 195, 197, 203, 207, 209, 225, 233, 249, 255, 269, 277, 283, 289, 295, 297, 301, 303, 311, 315, 327, 329, 363]

cals = {5: (41691,41029),
7:(38109,37596),
9:(37518.5,36462)}

bin_edges = np.arange(12800,13200,2)
counts = np.zeros(len(bin_edges)-1)
for (k,v) in cals.items():
    cal = mass.EnergyCalibration()
    cal.add_cal_point(v[0],13114.5)
    cal.add_cal_point(v[1],12979.3)
    ds = data.channel[k]
    ds.p_energy[:] = cal(ds.p_filt_value_dc[:])
    bin_centers, countsDS = ds.hist(bin_edges)
    counts+=countsDS

plt.plot(bin_centers, counts)


for badchan in badchans:
    data.set_chan_bad(badchan,"manually marked bad when compared to channel 3")

bin_centers, counts = data.hist(np.arange(0,16000,4),"p_filt_value_dc_ch3")
peaks_inds, properties = scipy.signal.find_peaks(counts,prominence=2)
peakscounts = counts[peaks_inds]
peaks_inds_sorted = peaks_inds[np.argsort(peakscounts)][::-1][:7]

plt.figure()
plt.plot(bin_centers,counts)
plt.plot(bin_centers[peaks_inds_sorted], counts[peaks_inds_sorted],".")
for pi in peaks_inds_sorted:
    print("ph = %f, counts = %g"%(bin_centers[pi],counts[pi]))

calR = mass.EnergyCalibration()
calR.add_cal_point(2278,653,"O H-like 2p")
calR.add_cal_point(2002,574,"O He-like 1s2p")
calR.add_cal_point(2693,774,"O H-like 3p")

for ds in data:
    ds.calibration["p_filt_value_dc_ch3"]=calR
    cal = mass.EnergyCalibration()
    for ph, e, name in zip(calR._ph,calR._energies, calR._names):
        cal.add_cal_point(ds.calibration["p_filt_value_dc_to_p_filt_value_dc_ch3"].energy2ph(ph),e,name)
    ds.calibration["p_filt_value_dc"]=cal
    cal.save_to_hdf5(ds.hdf5_group["calibration"],"p_filt_value_dc")

data.convert_to_energy("p_filt_value_dc")
data.plot_hist(np.arange(4000))

fitter = ds.linefit(653,dlo=10,dhi=10)
fitter = ds.linefit(574,dlo=10,dhi=10)
fitter = ds.linefit(774,dlo=10,dhi=10)

# guess_params = fitter.last_fit_params[:]
# guess_params[fitter.param_meaning["bg_slope"]]=0
# guess_params[fitter.param_meaning["background"]]=0
# fitter.fit(fitter.last_fit_contents, fitter.last_fit_bins, vary_bg=False,vary_bg_slope=False,label="full")

# lets select channels that agree with each other fairly well
ws=devel.WorstSpectra(data,np.arange(4000))
ws.plot()
ymax = plt.ylim()[1]
for e in calR._energies:
    plt.plot([e,e],[0,ymax],"k--")
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


with h5py.File("hists/"+data.shortname()+"_prelim.h5","w") as h5:
    bin_centers, counts = data.hist(np.arange(4000))
    h5["bin_centers"]=bin_centers
    h5["counts"]=counts

np.savetxt("hists/"+data.shortname()+"_prelim.yuri",
np.vstack((bin_centers,counts)).T,
header="#bin_centers (eV), counts")