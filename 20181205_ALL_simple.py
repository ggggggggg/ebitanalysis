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
import shutil

maxChans = 240
delete_hdf5_file_before_analysis = True
filtValueChoice = "p_filt_value_dc" # use p_filt_value mostly, try p_filt_value_dc on longer, higher count-rate datasets
referenceChannelNumber = 3
referenceLines = collections.OrderedDict() # map filt value or either energy or line name
referenceLines[2002]="O He-Like 1s2p + 1s2s"
referenceLines[2278]="O H-Like 2p"
referenceLines[2693]="O H-Like 3p"
energyResolutionThreshold = 4.8
energyResolutionThresholdLine = "O H-Like 2p"
binEdgesFiltValue = np.arange(100,8000,4)
switchTimestamp = 1544064607.7870846

class AreadyExistsException(Exception):
    pass

def ljh_combine(destination, src1, srcs2):
    print("ljh_combine:")
    print(destination)
    print(src1)
    print(srcs2)
    if os.path.isfile(destination):
        raise AreadyExistsException("output file already exists")
    nPulses = 0
    ljh1 = mass.LJHFile(src1)
    timestamp_offsets = [ljh1.timestamp_offset]
    nPulses += ljh1.nPulses
    shutil.copy(src1, destination)
    for src2 in srcs2:
        ljh2 = mass.LJHFile(src2)
        assert ljh1.nSamples == ljh2.nSamples
        assert ljh1.nPresamples == ljh2.nPresamples
        assert ljh1.timebase == ljh2.timebase
        with open(destination, "ab") as dest_fp:
            with open(src2,"rb") as src2_fp:
                src2_fp.seek(ljh2.header_size)
                dest_fp.write(src2_fp.read())
        nPulses += ljh2.nPulses
        timestamp_offsets.append(ljh2.timestamp_offset)
    ljhALL = mass.LJHFile(destination)
    if not nPulses == ljhALL.nPulses:
        raise Exception("ljh_combine want {}, has {} nPulses in {}".format(nPulses, ljhAll.nPulses, destination))
    if not sorted(timestamp_offsets) == timestamp_offsets:
        raise Exception("ljh combine did not find sorted timestamp offsets\ntimestamp_offsets = {}".format(timestamp_offsets))

dirname1 = "/Users/oneilg/Documents/EBIT/data/20181205_B"
source_letter = "B"
addition_letters = ["C","D","E","F","G","H","I"]
ndirname = "/Users/oneilg/Documents/EBIT/data/20181205_A" # noise file
dirname = "/Users/oneilg/Documents/EBIT/data/20181205_ALL"
available_chans = mass.ljh_util.ljh_get_channels_both(dirname1, ndirname)
chan_nums = available_chans[:maxChans]
pulse_files1 = [s+".ljh" for s in mass.ljh_util.ljh_chan_names(dirname1, chan_nums)]
pulse_files = [s+".ljh" for s in mass.ljh_util.ljh_chan_names(dirname, chan_nums)]
fb = "/Users/oneilg/Documents/EBIT/data/20181205_%s/20181205_%s_chan%i.ljh"
if not os.path.isdir(dirname):
    os.mkdir(dirname)
for channum in chan_nums:
    try:
        ljh_combine(fb%("ALL", "ALL", channum),
                    fb%(source_letter, source_letter, channum),
                    [fb%(c, c, channum) for c in addition_letters])
    except IOError as ex:
        print("channel %i failed with Exception:\n"%channum)
        print(ex)
    except AreadyExistsException as ex:
        print("channel %i output already exists"%channum)

timestampDict = collections.OrderedDict()
for c in [source_letter]+addition_letters:
    ljh=mass.LJHFile(fb%(c,c,1))
    ljh.read_trace(0)
    timestampDict[c]=ljh.datatimes_float[0]
timestampLoHiDict = collections.OrderedDict()
for i in range(len(timestampDict)):
    k = timestampDict.keys()[i]
    lo = timestampDict.values()[i]
    if i+1 == len(timestampDict):
        hi = np.inf
    else:
        hi = timestampDict.values()[i+1]
    timestampLoHiDict[k] = lo,hi

ebitDict = collections.OrderedDict()
ebitDict["B"] = "Ne, 4.0 kV"
ebitDict["C"] = "W, 4.1 kV"
ebitDict["D"] = "Os, 4.5 kV"
ebitDict["E"] = "Ar, 9.5 kV"
ebitDict["F"] = "Re, 4.3 kV"
ebitDict["G"] = "W, 4.1 kV"
ebitDict["H"] = "CO2, 2.0 kV"
ebitDict["I"] = "Ir, 15.06 kV"






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
data.set_chan_good(data.why_chan_bad.keys())
data.summarize_data(forceNew=False, peak_time_microsec=250)
data.register_categorical_cut_field("file",timestampLoHiDict.keys())
for ds in data:
    ds.cuts.cut_categorical("file", {k: np.logical_and(ds.p_timestamp[:]<hi,ds.p_timestamp[:]>=lo) for (k, (lo,hi)) in timestampLoHiDict.items()})
data.register_categorical_cut_field("injection",list(set(ebitDict.values())))
for ds in data:
    cutDict = {v:ds.good(file=k) for (k,v) in ebitDict.items()}
    cutDict["W, 4.1 kV"] = np.logical_or(ds.good(file="C"), ds.good(file="G"))
    ds.cuts.cut_categorical("injection",cutDict)

data.auto_cuts()
## auto cuts is confused by all the false triggers from the MeVVA so I need to help it out some
for ds in data:
    ds.clear_cuts()
    ds.saved_auto_cuts.cuts_prm["peak_time_ms"]=(0.1,0.4)
    ds.saved_auto_cuts.cuts_prm["rise_time_ms"]=(0.1,0.4)
    ds.apply_cuts(ds.saved_auto_cuts)

# check that file labels lok right
plt.figure()
plt.plot(ds.p_timestamp[ds.good()]-ds.p_timestamp[0],ds.p_pretrig_mean[ds.good()],".",label="all")
for c in timestampLoHiDict.keys():
    plt.plot(ds.p_timestamp[ds.good(file=c)]-ds.p_timestamp[0],ds.p_pretrig_mean[ds.good(file=c)],".",label=c)
plt.title(ds.shortname())
plt.xlabel("p_timestamp")
plt.ylabel("p_pretrig_mean")
plt.legend()

# check that injection labels lok right
plt.figure()
plt.plot(ds.p_timestamp[ds.good()]-ds.p_timestamp[0],ds.p_pretrig_mean[ds.good()],".",label="all")
for c in ebitDict.values():
    plt.plot(ds.p_timestamp[ds.good(injection=c)]-ds.p_timestamp[0],ds.p_pretrig_mean[ds.good(injection=c)],".",label=c)
plt.title(ds.shortname())
plt.xlabel("p_timestamp")
plt.ylabel("p_pretrig_mean")
plt.legend()

# for ds in data: ds._use_new_filters=False
data.compute_filters(forceNew=False)
data.filter_data(forceNew=False)

data.drift_correct(forceNew=True,category={"injection":"W, 4.1 kV"})
# data.phase_correct(forceNew=False)

# Plot data from reference channel,
# confirm that lines are correct, or manualy adjust `referenceLines` dict
dsRef = data.channel[referenceChannelNumber]
binCenters, counts = dsRef.hist(attr=filtValueChoice, bin_edges = binEdgesFiltValue, category={"injection":"CO2, 2.0 kV"})
dsRef.plot_hist(attr=filtValueChoice, bin_edges = binEdgesFiltValue, category={"injection":"CO2, 2.0 kV"})
plt.ylim(0.1,plt.ylim()[1])
plt.title(dsRef.shortname()+"\nidentify lines and populate `referenceLines` dictionary")
for (fv, lineName) in referenceLines.items():
        plt.axvline(fv,linestyle="dashed",color="r")
        if isinstance(lineName,str):
            plt.text(fv, plt.ylim()[1]/2.0,
                     "{}, {:0.2f}".format(lineName, ebit.lineNameOrEnergyToEnergy(lineName)),
                     rotation=90, verticalalignment='center')
        else:
            plt.text(fv, plt.ylim()[1]/2.0, "{:0.2f}".format(lineName), rotation=90, verticalalignment='center')

# check if drift correction looks needed
plt.figure()
plt.plot(dsRef.p_pretrig_mean[dsRef.good()],dsRef.p_filt_value[dsRef.good()],".",label="filt_value")
plt.plot(dsRef.p_pretrig_mean[dsRef.good()],dsRef.p_filt_value_dc[dsRef.good()],".",label="filt_value_dc")
plt.xlabel("p_pretrig_mean")
plt.ylabel("various filt_values")
plt.title(dsRef.shortname()+"\nzoom in on strongest line, is filt_value_dc is flatter?")
plt.legend()

# coarsley align all channels to referenceChannel
# adds a calibration with name like p_filt_value_dc_to_p_filt_value_dc_ch3 to each ds
hasPlotted = False
for ds_b in data:
    try:
        aligner = ebit.AlignBToA(dsRef, ds_b, referenceLines.keys(),
                                 binEdgesFiltValue, "p_filt_value_dc",
                                 category={"injection":"CO2, 2.0 kV"})
    except ValueError:
        data.set_chan_bad(ds_b.channum, "failed alignment")
    if not aligner.testForGoodnessBasedOnCalCurvature():
        data.set_chan_bad(ds_b.channum, "failed goodness test")
    elif ds_b != dsRef and not hasPlotted:
        aligner.samePeaksPlot()
        hasPlotted=True

# coarsley calibrate each channel, does not fit lines
for ds in data:
    cal = mass.EnergyCalibration()
    for (ph, lineNameOrEnergy) in referenceLines.items():
        e = ebit.lineNameOrEnergyToEnergy(lineNameOrEnergy)
        name = "{}".format(lineNameOrEnergy)
        cal.add_cal_point(ds.calibration[aligner.newcalname].energy2ph(ph),e,name)
    ds.calibration[filtValueChoice+"_rough"]=cal
    cal.save_to_hdf5(ds.hdf5_group["calibration"],filtValueChoice+"_rough")
    ds.p_energy_rough = cal(getattr(ds,filtValueChoice))
data.plot_hist(np.arange(4000),attr="p_energy_rough")




# carefully calibrate each channel
# fits a line for each entry in referenceLines
# uses the line shape defined in ebit.py for lineNameOrEnergy that are strings, like "O He-Like 1s2p + 1s2s"
# uses a GaussianFit for lineNameOrEnergy that are numbers like 505.5 or 707
for ds in data:
    cal_refined = mass.calibration.EnergyCalibration(curvetype="gain")
    for (ph, lineNameOrEnergy) in referenceLines.items():
        e = ebit.lineNameOrEnergyToEnergy(lineNameOrEnergy)
        name = "{}".format(lineNameOrEnergy)
        if isinstance(lineNameOrEnergy, str):
            fitter = ds.linefit(name,attr='p_energy_rough',dlo=10,dhi=10,plot=False,holdvals={"dP_dE":1}, category={"injection":"CO2, 2.0 kV"})
        else:
            fitter = ds.linefit(e,attr='p_energy_rough',dlo=10,dhi=10,plot=False, category={"injection":"CO2, 2.0 kV"})
        if not fitter.fit_success or np.abs(fitter.last_fit_params_dict["peak_ph"][0]-e)>10:
            data.set_chan_bad(ds.channum,"failed fit")
            continue
        ph_refined = ds.calibration[filtValueChoice+"_rough"].energy2ph(fitter.last_fit_params_dict["peak_ph"][0])
        cal_refined.add_cal_point(ph_refined,e, name)
        ds.calibration[filtValueChoice]=cal_refined
        cal_refined.save_to_hdf5(ds.hdf5_group["calibration"],filtValueChoice)
data.convert_to_energy(attr=filtValueChoice,calname=filtValueChoice)

# filter channels based on energy resolution at a specific line
resolutions = collections.OrderedDict()
fitters = collections.OrderedDict()
energyResolutionBinsize = 0.2
for ds in data:
    fitter = ds.linefit("O H-Like 2p",dlo=15,dhi=10,holdvals={"dP_dE":1},plot=False, category={"injection":"CO2, 2.0 kV"})
    resolutions[ds.channum]=fitter.last_fit_params_dict["resolution"][0]
    fitters[ds.channum]=fitter
plt.figure()
plt.hist(resolutions.values(),np.arange(0,energyResolutionThreshold+energyResolutionBinsize,energyResolutionBinsize),label="kept")
plt.hist(resolutions.values(),np.arange(energyResolutionThreshold,10,energyResolutionBinsize),label="excluded")
plt.xlabel("energy resolution (eV)")
plt.ylabel("pixels per bin")
plt.vlines([energyResolutionThreshold],plt.ylim()[0],plt.ylim()[1])
for (channum, res) in resolutions.items():
    if res>energyResolutionThreshold:
        data.set_chan_bad(channum, "fwhm resolution at O H-Like 2p > {} eV".format(energyResolutionThreshold))
for (channum, fitter) in fitters.items():
    print fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy
    if np.abs(fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy)>2:
        print("oh no %g"%channum)

# make plots of coadded spectrum, as well as coadded line fits
data.plot_hist(np.arange(4000),attr="p_energy", category={"injection":"CO2, 2.0 kV"})
fitter = data.linefit("O H-Like 3p",dlo=15,dhi=10,holdvals={"dP_dE":1}, category={"injection":"CO2, 2.0 kV"})
plt.title(plt.gca().get_title()+": CO2, 2.0 kV")
fitter = data.linefit("O H-Like 2p",dlo=15,dhi=10,holdvals={"dP_dE":1}, category={"injection":"CO2, 2.0 kV"})
plt.title(plt.gca().get_title()+": CO2, 2.0 kV")
fitter = data.linefit("O He-Like 1s2p + 1s2s",dlo=15,dhi=10,holdvals={"dP_dE":1}, category={"injection":"CO2, 2.0 kV"})
plt.title(plt.gca().get_title()+": CO2, 2.0 kV")
fitter = data.linefit("O He-Like 1s3p",dlo=6,dhi=10,holdvals={"dP_dE":1}, category={"injection":"CO2, 2.0 kV"})
plt.title(plt.gca().get_title()+": CO2, 2.0 kV")
fitter = data.linefit("O He-Like 1s4p",dlo=5,dhi=8,holdvals={"dP_dE":1}, category={"injection":"CO2, 2.0 kV"})
plt.title(plt.gca().get_title()+": CO2, 2.0 kV")

data.plot_hist(np.arange(4000),attr="p_energy", category={"injection":"Ir, 15.06 kV"})
fitter = data.linefit("O H-Like 3p",dlo=15,dhi=10,holdvals={"dP_dE":1}, category={"injection":"Ir, 15.06 kV"})
plt.title(plt.gca().get_title()+": Ir, 15.06 kV")
fitter = data.linefit("O H-Like 2p",dlo=15,dhi=10,holdvals={"dP_dE":1}, category={"injection":"Ir, 15.06 kV"})
plt.title(plt.gca().get_title()+": Ir, 15.06 kV")
fitter = data.linefit("O He-Like 1s2p + 1s2s",dlo=15,dhi=10,holdvals={"dP_dE":1}, category={"injection":"Ir, 15.06 kV"})
plt.title(plt.gca().get_title()+": Ir, 15.06 kV")
fitter = data.linefit("O He-Like 1s3p",dlo=6,dhi=10,holdvals={"dP_dE":1}, category={"injection":"Ir, 15.06 kV"})
plt.title(plt.gca().get_title()+": Ir, 15.06 kV")
fitter = data.linefit("O He-Like 1s4p",dlo=5,dhi=8,holdvals={"dP_dE":1}, category={"injection":"Ir, 15.06 kV"})
plt.title(plt.gca().get_title()+": Ir, 15.06 kV")

# plot all remaining channels, inspect this plot to make sure all channels
# appear aligned
ws=devel.WorstSpectra(data,np.arange(4000),category={"injection":"CO2, 2.0 kV"})
ws.plot()

ws=devel.WorstSpectra(data,np.arange(4000),category={"injection":"Ir, 15.06 kV"})
ws.plot()
# save coadded data in easy to read formats
outputDirName = data.shortname().split(",")[0]
if not os.path.isdir(outputDirName):
    os.mkdir(outputDirName)
with h5py.File(os.path.join(outputDirName,outputDirName+"_prelim.h5"),"w") as h5:
    for injection in data.cut_field_categories("injection"):
        bin_centers, counts = data.hist(np.arange(4000),category={"injection":injection})
        h5["bin_centers_{}".format(injection)]=bin_centers
        h5["counts_{}".format(injection)]=counts
        np.savetxt(os.path.join(outputDirName,outputDirName+"_prelim_{}.yuri".format(injection)),
        np.vstack((bin_centers,counts)).T,
        header="#bin_centers (eV), counts")

for i in plt.get_fignums():
    plt.figure(i)
    plt.savefig(os.path.join(outputDirName,'figure%d.png' % i)  )
data.hdf5_file.flush()
