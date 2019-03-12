import mass
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, OffFile, labelPeak, labelPeaks
import h5py
import os
import numpy as np
import pylab as plt
import h5py
import collections

# requires mass commit 7e83fe0
try:
    d = os.path.dirname(os.path.realpath(__file__))
except:
    d = os.getcwd()

def locateCalPoint(self,lineNameOrEnergy):
    if isinstance(lineNameOrEnergy, str):
        name = lineNameOrEnergy
        if name in mass.spectrum_classes:
            energy = mass.spectrum_classes[name]().peak_energy
        elif name in mass.STANDARD_FEATURES:
            energy = mass.STANDARD_FEATURES[name]
    else:
        energy = lineNameOrEnergy
    ph = self.getRoughCalibration().energy2ph(energy)
    return ph

mass.off.channels.CalibrationPlan.locateCalPoint = locateCalPoint

plt.close("all")
filename = "/Users/oneilg/Documents/EBIT/data/20181205_BCDEFGHI/20181205_BCDEFGHI_chan1.off"
data = ChannelGroup(getOffFileListFromOneFile(filename, maxChans=244))
data.setOutputDir(baseDir=d, deleteAndRecreate=True, suffix="_2spline_output")
data.experimentStateFile.aliasState("B","Ne")
data.experimentStateFile.aliasState("C","W 1")
data.experimentStateFile.aliasState("D","Os")
data.experimentStateFile.aliasState("E","Ar")
data.experimentStateFile.aliasState("F","Re")
data.experimentStateFile.aliasState("G","W 2")
data.experimentStateFile.aliasState("H","CO2")
data.experimentStateFile.aliasState("I","Ir")
data.learnStdDevResThresholdUsingRatioToNoiseStd(ratioToNoiseStd=5)
data.learnDriftCorrection()
ds=data.firstGoodChannel()
ds.plotAvsB("relTimeSec", "residualStdDev",  includeBad=True)
ds.plotAvsB("relTimeSec", "pretriggerMean", includeBad=True)
ds.plotAvsB("relTimeSec", "filtValue", includeBad=False)
ds.plotHist(np.arange(0,40000,4),"filtValue")
ds.plotHist(np.arange(0,40000,4),"filtValue", coAddStates=False)
ds.plotResidualStdDev()
driftCorrectInfo = ds.learnDriftCorrection(states=["W 1","W 2"])
ds.plotCompareDriftCorrect()

import scipy as sp
import lmfit

@mass.off.add_group_loop
def learnTDC(self):
    g = self.choose(states=["W 1","W 2"])
    indicator0 = self.relTimeSec[g]
    uncorrected = self.filtValueDC[g]
    limit = None
    indicator_median = np.median(indicator0)
    indicator = indicator0-indicator_median

    if limit is None:
        pct99 = sp.stats.scoreatpercentile(uncorrected, 99)
        limit = 1.25 * pct99

    smoother = mass.analysis_algorithms.HistogramSmoother(0.5, [0, limit])

    def entropy(slope):
        corrected = uncorrected * (1+indicator*slope)
        hsmooth = smoother(corrected)
        w = hsmooth > 0
        return -(np.log(hsmooth[w])*hsmooth[w]).sum()

    slope = sp.optimize.brent(entropy, brack=[0, .001])

    def gain_correct2(params, indicator, uncorrected):
        a = params["a"].value
        b = params["b"].value
        c = params["c"].value
        gain = (1+indicator*a) * (1+(uncorrected-c)*b)
        corrected = gain*uncorrected
        return corrected

    def entropy2(params):
        corrected = gain_correct2(params, indicator, uncorrected)
        hsmooth = smoother(corrected)
        w = hsmooth > 0
        return -(np.log(hsmooth[w])*hsmooth[w]).sum()

    params = lmfit.Parameters()
    params.add("a", -slope)
    params.add("b", 1/(1000.0*np.median(uncorrected)))
    params.add("c", np.median(uncorrected))


    result = lmfit.minimize(entropy2, params, method="nelder")

    corrected = gain_correct2(params, indicator, uncorrected)

    self.filtValueTDCParams = params
    self.filtValueTDC = gain_correct2(params, self.relTimeSec, self.filtValueDC)
    return params
Channel.learnTDC = learnTDC

data.learnTDC()

ds.calibrationPlanInit("filtValueTDC")
ds.calibrationPlanAddPoint(2128, "O He-Like 1s2p + 1s2s", states="CO2")
ds.calibrationPlanAddPoint(2421, "O H-Like 2p", states="CO2")
ds.calibrationPlanAddPoint(2864, "O H-Like 3p", states="CO2")
ds.calibrationPlanAddPoint(3404, "Ne He-Like 1s2p", states="Ne")
ds.calibrationPlanAddPoint(3768, "Ne H-Like 2p", states="Ne")
ds.calibrationPlanAddPoint(5716, "W Ni-2", states="W 1")
ds.calibrationPlanAddPoint(6287.8, 'W Ni-Like 3s^2,3p^6,3s^3_3/2,3d^6_5/2,4p_1/2', states="W 1")
ds.calibrationPlanAddPoint(6413, "W Ni-4", states="W 1")
ds.calibrationPlanAddPoint(7641, "W Ni-7", states="W 1")
ds.calibrationPlanAddPoint(10256, "W Ni-17", states="W 1")
# ds.calibrationPlanAddPoint(6287.8, 'W Ni-Like 3s^2,3p^6,3s^3_3/2,3d^6_5/2,4p_1/2', states="W 2")
# ds.calibrationPlanAddPoint(6413, "W Ni-4", states="W 2")
# ds.calibrationPlanAddPoint(7641, "W Ni-7", states="W 2")
# ds.calibrationPlanAddPoint(10256, "W Ni-17", states="W 2")
# ds.calibrationPlanAddPoint(10700, "W Ni-20", states=["W 1", "W 2"])
ds.calibrationPlanAddPoint(11125, "Ar He-Like 1s2s+1s2p", states="Ar")
ds.calibrationPlanAddPoint(11728, "Ar H-Like 2p", states="Ar")
# at this point energyRough should work
ds.plotHist(np.arange(0,4000,1),"energyRough", coAddStates=False)
fitters = ds.calibrateFollowingPlan("filtValueDC")
ds.linefit("Ne H-Like 2p",attr="energy",states="Ne")
ds.linefit("Ne He-Like 1s2p",attr="energy",states="Ne")
ds.linefit("W Ni-7",attr="energy",states="W 1")
ds.plotHist(np.arange(0,4000,4),"energy", coAddStates=False)


ds.diagnoseCalibration()

ds3 = data[3]
data.alignToReferenceChannel(referenceChannel=ds,
                             binEdges=np.arange(500,20000,4), attr="filtValueTDC")
aligner = ds3.aligner
aligner.samePeaksPlot()
aligner.samePeaksPlotWithAlignmentCal()

fitters = data.calibrateFollowingPlan("filtValueTDC", _rethrow=False, dlo=10,dhi=10)
data.qualityCheckDropOneErrors(thresholdAbsolute=2.5, thresholdSigmaFromMedianAbsoluteValue=6)
with data.outputHDF5 as h5:
    fitters = data.qualityCheckLinefit("Ne H-Like 3p", positionToleranceAbsolute=2,
                worstAllowedFWHM=4.5, states="Ne", _rethrow=False,
                resolutionPlot=True, hdf5Group=h5)
    # data.histsToHDF5(h5, np.arange(0,4000,0.25))
    # data.recipeToHDF5(h5)
    # data.energyTimestampLabelToHDF5(h5)

data.hist(np.arange(0,4000,1), "energy")
data.plotHist(np.arange(0,4000,1),"energy", coAddStates=False)
data.plotHists(np.arange(0,16000,4),"arbsInRefChannelUnits")
data.plotHists(np.arange(0,4000,1),"energy")



plt.figure(figsize=(12,6))
ax = plt.gca()
data.plotHist(np.arange(1000,4000,1),"energy", coAddStates=False, states=["W 1","Os"], axis=ax)
ax.set_ylim(0,1.2*np.amax([np.amax(l.get_ydata()) for l in ax.lines]))
names = ["W Ni-{}".format(i) for i in range(1,27)]
names += ['W Ni-Like 3s^2,3p^6,3s^3_3/2,3d^6_5/2,4p_1/2']
n = collections.OrderedDict()
l=ax.lines[0]
for name in names:
    n[name] = mass.spectrum_classes[name].nominal_peak_energy
labelPeak(ax, "W Ni-8", n["W Ni-8"])
labelPeaks(axis=ax, names=n.keys(), energies=n.values(), line=ax.lines[0])

nos = collections.OrderedDict()
nos["Os Ni-2"]=1680
nos["Os Ni-3"]=1755
nos["Os Ni-4"]=1902
nos["Os Ni-5"]=1975
nos["Os Ni-6"]=2155
nos["Os Ni-7"]=2268
nos["Os Ni-8"]=2342
nos["Os Ni-16"]=3032
nos["Os Ni-17"]=3102
labelPeaks(ax, names=nos.keys(), energies=nos.values(), line=ax.lines[1])




data.fitterPlot("W Ni-20", states=["W 1"])


lineNames = collections.OrderedDict()
lineNames["Ne"] = ["Ne He-Like 1s2p", "Ne H-Like 2p"]
lineNames["W 1"] = ["W Ni-8","W Ni-11","W Ni-16","W Ni-20","W Ni-21","W Ni-23"]
lineNames["Ar"] = ["Ar He-Like 1s2s+1s2p", "Ar H-Like 2p"]
lineNames["W 2"] = ["W Ni-8","W Ni-11","W Ni-16","W Ni-20","W Ni-21","W Ni-23"]
lineNames["CO2"] = ["O He-Like 1s2p + 1s2s", "O H-Like 2p", "O H-Like 3p"]
lineNames["Ir"] = ["O He-Like 1s2p + 1s2s", "O H-Like 2p", "O H-Like 3p"]



fitterDict = collections.OrderedDict()
for state in lineNames.keys():
    fitterDict[state] = collections.OrderedDict()
    for lineName in lineNames[state]:
        fitter = data.linefit(lineName, states=state, plot=False)
        fitterDict[state][lineName]=fitter

fig=plt.figure(figsize=(12,8))
for (i,state) in enumerate(lineNames.keys()):
    for (j,lineName) in enumerate(lineNames[state]):
        fitter=fitterDict[state][lineName]
        x=i+float(j)/len(lineNames[state])
        lines = plt.errorbar(x,fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy,
                     yerr=fitter.last_fit_params_dict["peak_ph"][1],
                     label="{}:{}".format(state,lineName),fmt='o')
        plt.annotate("{}:{}".format(state,lineName), (x,fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy),
                     rotation=90, verticalalignment='right', horizontalalignment="center", color=lines[0].get_color(),
                     xytext = (5,10), textcoords='offset points')
# plt.legend(loc="upper left")
plt.xlabel("state")
plt.ylabel("fit energy and uncertainty - database energy (eV)")
plt.title(data.shortName)
plt.grid(True)
plt.xticks(np.arange(len(lineNames.keys())), lineNames.keys())

fig=plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
z = np.zeros((ds.calibration.npts, len(data)))
for (j,ds) in enumerate(data.values()):
    x,y = ds.calibration.drop_one_errors()
    plt.plot(x,y,".")
    for (i,_y) in enumerate(y):
        z[i,j]=_y
plt.suptitle(data.shortName)
plt.grid(True)
plt.xlabel("energy (eV)")
plt.ylabel("drop on errors (eV)")
plt.subplot(2,1,2)
plt.errorbar(x, np.median(z,axis=1), yerr=np.std(z,axis=1),fmt="o")
# plt.title(data.shortName)
plt.grid(True)
plt.xlabel("energy (eV)")
plt.ylabel("drop one errors (eV)")

lineNames = collections.OrderedDict()
lineNames["W 1"] = ["W Ni-{}".format(i) for i in range(1,27)]
lineNames["W 2"] = ["W Ni-{}".format(i) for i in range(1,27)]

fitterDict = collections.OrderedDict()
for state in lineNames.keys():
    fitterDict[state] = collections.OrderedDict()
    for lineName in lineNames[state]:
        fitter = data.linefit(lineName, states=state, plot=False)
        fitterDict[state][lineName]=fitter

fig=plt.figure(figsize=(12,8))
for (j,lineName) in enumerate(lineNames["W 1"]):
    fitter=fitterDict["W 1"][lineName]
    fitter2=fitterDict["W 2"][lineName]
    x=fitter.spect.peak_energy
    y = fitter.last_fit_params_dict["peak_ph"][0] - fitter2.last_fit_params_dict["peak_ph"][0]
    yerr = np.sqrt(fitter.last_fit_params_dict["peak_ph"][1]**2 + fitter2.last_fit_params_dict["peak_ph"][1]**2)
    lines = plt.errorbar(x,y,
                 yerr=yerr,
                 label="{}:{}".format(state,lineName),fmt='o')
    plt.annotate("{}:{}".format(state,lineName), (x,y),
                 rotation=90, verticalalignment='right', horizontalalignment="center", color=lines[0].get_color(),
                 xytext = (5,10), textcoords='offset points')
# plt.legend(loc="upper left")
plt.xlabel("energy (eV)")
plt.ylabel("line position in W1 minus W2 (eV)")
plt.title(data.shortName)
plt.grid(True)
# plt.xticks(np.arange(len(lineNames.keys())), lineNames.keys())

xs=[]
ys=[]
fig=plt.figure(figsize=(12,8))
for (j,lineName) in enumerate(lineNames["W 1"]):
    fitter=fitterDict["W 1"][lineName]
    fitter2=fitterDict["W 2"][lineName]
    x=fitter.spect.peak_energy
    y = fitter.last_fit_params_dict["peak_ph"][0]/fitter2.last_fit_params_dict["peak_ph"][0]
    xs.append(x)
    ys.append(y)
    # yerr = np.sqrt(fitter.last_fit_params_dict["peak_ph"][1]**2 + fitter2.last_fit_params_dict["peak_ph"][1]**2)
    lines = plt.plot(x,y,"o",                 label="{}:{}".format(state,lineName))
    plt.annotate("{}:{}".format(state,lineName), (x,y),
                 rotation=90, verticalalignment='right', horizontalalignment="center", color=lines[0].get_color(),
                 xytext = (5,10), textcoords='offset points')
# plt.legend(loc="upper left")
plt.xlabel("energy (eV)")
plt.ylabel("line positions ratio W1/W2 (eV)")
plt.title(data.shortName)
plt.grid(True)
# plt.xticks(np.arange(len(lineNames.keys())), lineNames.keys())
pfit = np.polyfit(xs,ys,1)
ys_pfit = np.polyval(pfit,xs)
plt.plot(xs,ys_pfit)




with h5py.File(data.outputHDF5.filename,"r") as h5:
    print(h5.keys())
    newds = Channel(ds.offFile, ds.experimentStateFile)
    # newds.recipeFromHDF5(h5)


lineNames = collections.OrderedDict()
lineNames["W 1"] = ["W Ni-{}".format(i) for i in range(1,27)]
lineNames["W 2"] = ["W Ni-{}".format(i) for i in range(1,27)]


def choose(self, states=None, good=True):
    """ return boolean indicies of "choose" pulses
    if state is none, all states are chosen
    ds.choose("A") selects state A
    ds.choose(["A","B"]) selects states A and B
    """
    g = self.offFile["residualStdDev"]<self.stdDevResThreshold
    g = np.logical_and(self.filtValue>500,g)
    if not good:
        g = np.logical_not(g)
    if isinstance(states, str):
        states = [states]
    if states is not None:
        z = np.zeros(self.nRecords,dtype="bool")
        for state in states:
            z = np.logical_or(z,self.statesDict[state])
        g = np.logical_and(g,z)
    return g
ds.choose = choose.__get__(ds, Channel)




lineName = "O H-Like 3p"
binsizes = np.logspace(-1,.301,10)
fitters = []
datasource = data
for binsize in binsizes:
    fitter = datasource.linefit(lineName, attr="energy", states="CO2", dlo=30, dhi=30, plot=False, binsize=binsize)
    fitters.append(fitter)


plt.figure()
res = [fitter.last_fit_params_dict["resolution"][0] for fitter in fitters]
res_sigma = [fitter.last_fit_params_dict["resolution"][1] for fitter in fitters]
plt.errorbar(binsizes, res, res_sigma,fmt=".")
plt.xlabel("binsize (eV)")
plt.ylabel("fwhm energy resolution (eV)")
plt.title(datasource.shortName+" fit to {}".format(lineName))


plt.figure()
peak_ph = [fitter.last_fit_params_dict["peak_ph"][0] for fitter in fitters]
peak_ph_sigma = [fitter.last_fit_params_dict["peak_ph"][1] for fitter in fitters]
plt.errorbar(binsizes, peak_ph, peak_ph_sigma,fmt=".")
plt.xlabel("binsize (eV)")
plt.ylabel("peak location (eV)")
plt.title(datasource.shortName+" fit to {}".format(lineName))
