
import os
import mass
import numpy as np, scipy as sp, pylab as plt
import collections
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
import devel
import ebit

plt.ion()

'''
INPUTS
'''
# User input parameters
forceNew = False
forceNewSummarize = False

# DTW Inputs
filtValueChoice = "p_filt_value" # use p_filt_value mostly, try p_filt_value_dc on longer, higher count-rate datasets
referenceChannelNumber = 3
referenceLines = collections.OrderedDict() # map filt value or either energy or line name
referenceLines[3283]='Ne He-Like 1s2s' # He-like 1s^2-1s2s
referenceLines[3341]='Ne He-Like 1s2p' # He-like 1s^2-1s2p
referenceLines[3698]='Ne H-Like 2p' # H-like 1s-2p
referenceLines[3886]='Ne He-Like 1s3p' # He-like 1s^2-1s3p
referenceLines[4374]='Ne H-Like 3p' # H-like 1s-3p
referenceLines[4608]='Ne H-Like 4p' # H-like 1s-4p
energyResolutionThreshold = 5.0
energyResolutionThresholdLine = 'Ne H-Like 2p'
binEdgesFiltValue = np.arange(100,8000,4)

# Pulse and noise file name and directory information
noisefile = "20181203_B"
pulsefile = "20181203_A"
EBITBaseDirectory = r"/Users/oneilg/Documents/EBIT/data"
outputDirectory = r"C:\Users\pns12\Desktop\EBIT_Analysis\Ne"

noiseDirectory = os.path.join(EBITBaseDirectory, noisefile)
pulseDirectory = os.path.join(EBITBaseDirectory, pulsefile)

'''
LOAD AND SUMMARIZE DATA
'''
# Check to see if directories exist, load in noi and ljh files
assert(os.path.isdir(noiseDirectory))
assert(os.path.isdir(pulseDirectory))
noisePattern = os.path.join(noiseDirectory, noisefile + "_chan*.ljh")
pulsePattern = os.path.join(pulseDirectory, pulsefile + "_chan*.ljh")


data = mass.TESGroup(pulsePattern, noisePattern)

# Summarize data, currently using python version instead of cython
data.summarize_data(forceNew=forceNewSummarize)

'''
CUTS
'''
# Apply auto_cuts
data.auto_cuts(forceNew=forceNew)

# Cut out channels with too few pulses
pulsesPerChannel = []
for ds in data:
    pulsesPerChannel.append(ds.nPulses)
medianPulses = np.nanmedian(pulsesPerChannel)
stdPulses = np.nanstd(pulsesPerChannel)
for ds in data:
    if ds.nPulses < medianPulses - 3.0*stdPulses:
        data.set_chan_bad(ds.channum, "Too few pulses")

# Drop devices with outlier fractions of pulses marked bad
frac = []
for ds in data:
    frac.append(1.0 - 1.0*np.sum(ds.good())/ds.nPulses)
medianFrac = np.nanmedian(frac)
stdFrac = np.nanstd(frac)
for ds in data:
    if (1.0 - 1.0*np.sum(ds.good())/ds.nPulses) > medianFrac + 3.0*stdFrac:
        data.set_chan_bad(ds.channum, "Too many bad pulses")

'''
FILTERING
'''
# Make average pulses
data.avg_pulses_auto_masks(forceNew=forceNew)

# Compute noise spectra
print 'Computing noise spectra...'
data.compute_noise_spectra(forceNew=forceNew)

# Compute filters
data.compute_filters(forceNew=forceNew)

# Summarize filters
data.summarize_filters(std_energy=mass.spectrum_classes["Ne H-Like 2p"]().peak_energy) # Most intense, H-like line

# Filter data
data.filter_data(forceNew=forceNew)

'''
CORRECTIONS
'''
# Drift and phase correct
data.drift_correct(forceNew=forceNew)
#data.phase_correct(forceNew=forceNew)

'''
FAST DTW
'''
# Plot data from reference channel,
# confirm that lines are correct, or manualy adjust `referenceLines` dict
dsRef = data.channel[referenceChannelNumber]
binCenters, counts = dsRef.hist(attr=filtValueChoice, bin_edges = binEdgesFiltValue, )
dsRef.plot_hist(attr=filtValueChoice, bin_edges = binEdgesFiltValue, )
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
        aligner = ebit.AlignBToA(dsRef, ds_b, referenceLines.keys(),binEdgesFiltValue, filtValueChoice)
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
            fitter = ds.linefit(name,attr='p_energy_rough',dlo=10,dhi=10,plot=False,holdvals={"dP_dE":1})
        else:
            fitter = ds.linefit(e,attr='p_energy_rough',dlo=10,dhi=10,plot=False)
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
energyResolutionBinsize = 0.2
for ds in data:
    fitter = ds.linefit("Ne H-Like 2p",dlo=15,dhi=10,holdvals={"dP_dE":1},plot=False)
    resolutions[ds.channum]=fitter.last_fit_params_dict["resolution"][0]
plt.figure()
plt.hist(resolutions.values(),np.arange(0,energyResolutionThreshold+energyResolutionBinsize,energyResolutionBinsize),label="kept")
plt.hist(resolutions.values(),np.arange(energyResolutionThreshold,10,energyResolutionBinsize),label="excluded")
plt.xlabel("energy resolution (eV)")
plt.ylabel("pixels per bin")
plt.vlines([energyResolutionThreshold],plt.ylim()[0],plt.ylim()[1])
for (channum, res) in resolutions.items():
    if res>energyResolutionThreshold:
        data.set_chan_bad(channum, "fwhm resolution at Ne H-Like 2p > {} eV".format(energyResolutionThreshold))

# make plots of coadded spectrum, as well as coadded line fits
data.plot_hist(np.arange(4000),attr="p_energy")
fitter = data.linefit('Ne He-Like 1s2s',dlo=10,dhi=10,holdvals={"dP_dE":1})
fitter = data.linefit('Ne He-Like 1s2p',dlo=10,dhi=10,holdvals={"dP_dE":1})
fitter = data.linefit('Ne H-Like 2p',dlo=10,dhi=10,holdvals={"dP_dE":1})
fitter = data.linefit('Ne He-Like 1s3p',dlo=10,dhi=10,holdvals={"dP_dE":1})
fitter = data.linefit('Ne H-Like 3p',dlo=15,dhi=10,holdvals={"dP_dE":1})
fitter = data.linefit('Ne H-Like 4p',dlo=15,dhi=10,holdvals={"dP_dE":1})

# plot all remaining channels, inspect this plot to make sure all channels
# appear aligned
ws=devel.WorstSpectra(data,np.arange(4000))
ws.plot()

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

'''
data.hdf5_file.close()
data.hdf5_noisefile.close()
'''
