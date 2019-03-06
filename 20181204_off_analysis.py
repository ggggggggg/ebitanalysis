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

plt.close("all")
filename = "/Users/oneilg/Documents/EBIT/data/20181204_DEHIJKL/20181204_DEHIJKL_chan1.off"
data = ChannelGroup(getOffFileListFromOneFile(filename, maxChans=240))
data.setOutputDir(baseDir=d, deleteAndRecreate=True)

data.experimentStateFile.aliasState("D","Fe")
data.experimentStateFile.aliasState("E","W 1")
data.experimentStateFile.aliasState("H","W 2")
data.experimentStateFile.aliasState("I","Ir")
data.experimentStateFile.aliasState("J","Ar")
data.experimentStateFile.aliasState("K","Ir 15.6 kV")
data.experimentStateFile.aliasState("L","Ne")
data.learnStdDevResThresholdUsingRatioToNoiseStd(ratioToNoiseStd=5)
data.learnDriftCorrection()
ds=data.firstGoodChannel()
ds.plotAvsB("relTimeSec", "residualStdDev",  includeBad=True)
ds.plotAvsB("relTimeSec", "pretriggerMean", includeBad=True)
ds.plotAvsB("relTimeSec", "filtValue", includeBad=False)
ds.plotHist(np.arange(0,40000,4),"filtValue")
ds.plotHist(np.arange(0,40000,4),"filtValueDC", coAddStates=False)
ds.plotResidualStdDev()
driftCorrectInfo = ds.learnDriftCorrection(states=["W 2"])
ds.plotCompareDriftCorrect()

ds.calibrationPlanInit("filtValueDC")
# ds.calibrationPlanAddPoint(2128, "O He-Like 1s2p + 1s2s", states="CO2")
# ds.calibrationPlanAddPoint(2421, "O H-Like 2p", states="CO2")
# ds.calibrationPlanAddPoint(2864, "O H-Like 3p", states="CO2")
ds.calibrationPlanAddPoint(3413, "Ne He-Like 1s2p", states="Ne")
ds.calibrationPlanAddPoint(3777, "Ne H-Like 2p", states="Ne")
ds.calibrationPlanAddPoint(5726, "W Ni-2", states="W 2")
ds.calibrationPlanAddPoint(6437, "W Ni-4", states="W 2")
ds.calibrationPlanAddPoint(7648, "W Ni-7", states="W 2")
# ds.calibrationPlanAddPoint(10271, "W Ni-17", states="W")
# ds.calibrationPlanAddPoint(10700, "W Ni-20", states=["W 1", "W 2"])
ds.calibrationPlanAddPoint(11151, "Ar He-Like 1s2s+1s2p", states="Ar")
ds.calibrationPlanAddPoint(11759, "Ar H-Like 2p", states="Ar")
# at this point energyRough should work
ds.plotHist(np.arange(0,4000,1),"energyRough", coAddStates=False)
# fitters = ds.calibrateFollowingPlan("filtValueDC")
# ds.linefit("Ne H-Like 2p",attr="energy",states="Ne")
# ds.linefit("Ne He-Like 1s2p",attr="energy",states="Ne")
# ds.linefit("W Ni-7",attr="energy",states="W")
# ds.plotHist(np.arange(0,4000,4),"energy", coAddStates=False)


# ds.diagnoseCalibration()

ds3 = data[3]
data.alignToReferenceChannel(referenceChannel=ds,
                             binEdges=np.arange(500,20000,4), attr="filtValueDC")
aligner = ds3.aligner
aligner.samePeaksPlot()
aligner.samePeaksPlotWithAlignmentCal()

fitters = data.calibrateFollowingPlan("filtValueDC", _rethrow=False, dlo=10,dhi=10)
data.qualityCheckDropOneErrors(thresholdAbsolute=2.5, thresholdSigmaFromMedianAbsoluteValue=6)
with data.outputHDF5 as h5:
    fitters = data.qualityCheckLinefit("Ne H-Like 3p", positionToleranceAbsolute=2,
                worstAllowedFWHM=4.5, states="Ne", _rethrow=False,
                resolutionPlot=True, hdf5Group=h5)
    data.histsToHDF5(h5, np.arange(4000))
    data.recipeToHDF5(h5)
    # data.energyTimestampLabelToHDF5(h5)

data.hist(np.arange(0,4000,1), "energy")
data.plotHist(np.arange(0,4000,1),"energy", coAddStates=False)
data.plotHists(np.arange(0,16000,4),"arbsInRefChannelUnits")
data.plotHists(np.arange(0,4000,1),"energy")


#
# plt.figure(figsize=(12,6))
# ax = plt.gca()
# data.plotHist(np.arange(1000,4000,1),"energy", coAddStates=False, states=["W 1","Os"], axis=ax)
# ax.set_ylim(0,1.2*np.amax([np.amax(l.get_ydata()) for l in ax.lines]))
# names = ["W Ni-{}".format(i) for i in range(1,27)]
# n = collections.OrderedDict()
# l=ax.lines[0]
# for name in names:
#     n[name] = mass.spectrum_classes[name].nominal_peak_energy
# labelPeak(ax, "W Ni-8", n["W Ni-8"])
# labelPeaks(axis=ax, names=n.keys(), energies=n.values(), line=ax.lines[0])
#
# nos = collections.OrderedDict()
# nos["Os Ni-2"]=1680
# nos["Os Ni-3"]=1755
# nos["Os Ni-4"]=1902
# nos["Os Ni-5"]=1975
# nos["Os Ni-6"]=2155
# nos["Os Ni-7"]=2268
# nos["Os Ni-8"]=2342
# nos["Os Ni-16"]=3032
# nos["Os Ni-17"]=3102
# labelPeaks(ax, names=nos.keys(), energies=nos.values(), line=ax.lines[1])
#
#
#
#
# data.fitterPlot("W Ni-20", states=["W 1"])
#
#
# lineNames = collections.OrderedDict()
# lineNames["Ne"] = ["Ne He-Like 1s2p", "Ne H-Like 2p"]
# lineNames["W 1"] = ["W Ni-8","W Ni-11","W Ni-16","W Ni-20","W Ni-21","W Ni-23"]
# lineNames["Ar"] = ["Ar He-Like 1s2s+1s2p", "Ar H-Like 2p"]
# lineNames["W 2"] = ["W Ni-8","W Ni-11","W Ni-16","W Ni-20","W Ni-21","W Ni-23"]
# lineNames["CO2"] = ["O He-Like 1s2p + 1s2s", "O H-Like 2p", "O H-Like 3p"]
# lineNames["Ir"] = ["O He-Like 1s2p + 1s2s", "O H-Like 2p", "O H-Like 3p"]
#
#
#
# fitterDict = collections.OrderedDict()
# for state in lineNames.keys():
#     fitterDict[state] = collections.OrderedDict()
#     for lineName in lineNames[state]:
#         fitter = data.linefit(lineName, states=state, plot=False)
#         fitterDict[state][lineName]=fitter
#
# fig=plt.figure(figsize=(12,8))
# for (i,state) in enumerate(lineNames.keys()):
#     for (j,lineName) in enumerate(lineNames[state]):
#         fitter=fitterDict[state][lineName]
#         x=i+float(j)/len(lineNames[state])
#         lines = plt.errorbar(x,fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy,
#                      yerr=fitter.last_fit_params_dict["peak_ph"][1],
#                      label="{}:{}".format(state,lineName),fmt='o')
#         plt.annotate("{}:{}".format(state,lineName), (x,fitter.last_fit_params_dict["peak_ph"][0]-fitter.spect.peak_energy),
#                      rotation=90, verticalalignment='right', horizontalalignment="center", color=lines[0].get_color(),
#                      xytext = (5,10), textcoords='offset points')
# # plt.legend(loc="upper left")
# plt.xlabel("state")
# plt.ylabel("fit energy and uncertainty - database energy (eV)")
# plt.title(data.shortName)
# plt.grid(True)
# plt.xticks(np.arange(len(lineNames.keys())), lineNames.keys())
#
# fig=plt.figure(figsize=(12,8))
# plt.subplot(2,1,1)
# z = np.zeros((ds.calibration.npts, len(data)))
# for (j,ds) in enumerate(data.values()):
#     x,y = ds.calibration.drop_one_errors()
#     plt.plot(x,y,".")
#     for (i,_y) in enumerate(y):
#         z[i,j]=_y
# plt.suptitle(data.shortName)
# plt.grid(True)
# plt.xlabel("energy (eV)")
# plt.ylabel("drop on errors (eV)")
# plt.subplot(2,1,2)
# plt.errorbar(x, np.median(z,axis=1), yerr=np.std(z,axis=1),fmt="o")
# # plt.title(data.shortName)
# plt.grid(True)
# plt.xlabel("energy (eV)")
# plt.ylabel("drop one errors (eV)")
#
# lineNames = collections.OrderedDict()
# lineNames["W 1"] = ["W Ni-{}".format(i) for i in range(1,27)]
# lineNames["W 2"] = ["W Ni-{}".format(i) for i in range(1,27)]
#
# fitterDict = collections.OrderedDict()
# for state in lineNames.keys():
#     fitterDict[state] = collections.OrderedDict()
#     for lineName in lineNames[state]:
#         fitter = data.linefit(lineName, states=state, plot=False)
#         fitterDict[state][lineName]=fitter
#
# fig=plt.figure(figsize=(12,8))
# for (j,lineName) in enumerate(lineNames["W 1"]):
#     fitter=fitterDict["W 1"][lineName]
#     fitter2=fitterDict["W 2"][lineName]
#     x=fitter.spect.peak_energy
#     y = fitter.last_fit_params_dict["peak_ph"][0] - fitter2.last_fit_params_dict["peak_ph"][0]
#     yerr = np.sqrt(fitter.last_fit_params_dict["peak_ph"][1]**2 + fitter2.last_fit_params_dict["peak_ph"][1]**2)
#     lines = plt.errorbar(x,y,
#                  yerr=yerr,
#                  label="{}:{}".format(state,lineName),fmt='o')
#     plt.annotate("{}:{}".format(state,lineName), (x,y),
#                  rotation=90, verticalalignment='right', horizontalalignment="center", color=lines[0].get_color(),
#                  xytext = (5,10), textcoords='offset points')
# # plt.legend(loc="upper left")
# plt.xlabel("energy (eV)")
# plt.ylabel("line position in W1 minus W2 (eV)")
# plt.title(data.shortName)
# plt.grid(True)
# # plt.xticks(np.arange(len(lineNames.keys())), lineNames.keys())
#
#
#
#
#
# with h5py.File(data.outputHDF5.filename,"r") as h5:
#     print(h5.keys())
#     newds = Channel(ds.offFile, ds.experimentStateFile)
#     newds.recipeFromHDF5(h5)
#
#
# lineNames = collections.OrderedDict()
# lineNames["W 1"] = ["W Ni-{}".format(i) for i in range(1,27)]
# lineNames["W 2"] = ["W Ni-{}".format(i) for i in range(1,27)]
#
#
# def choose(self, states=None, good=True):
#     """ return boolean indicies of "choose" pulses
#     if state is none, all states are chosen
#     ds.choose("A") selects state A
#     ds.choose(["A","B"]) selects states A and B
#     """
#     g = self.offFile["residualStdDev"]<self.stdDevResThreshold
#     g = np.logical_and(self.filtValue>500,g)
#     if not good:
#         g = np.logical_not(g)
#     if isinstance(states, str):
#         states = [states]
#     if states is not None:
#         z = np.zeros(self.nRecords,dtype="bool")
#         for state in states:
#             z = np.logical_or(z,self.statesDict[state])
#         g = np.logical_and(g,z)
#     return g
# ds.choose = choose.__get__(ds, Channel)
#
# ds.plotAvsB("filtValue","coef3")
# g = ds.choose()
# x = ds.filtValue[g]
# y = ds.coef3[g]
# pfit3 = np.polyfit(x,y,4)
# _x = np.linspace(0,np.amax(x),100)
# plt.plot(_x, np.polyval(pfit3,_x))
#
# ds.plotAvsB("filtValue","coef4")
# g = ds.choose()
# x = ds.filtValue[g]
# y = ds.coef4[g]
# pfit4 = np.polyfit(x,y,4)
# _x = np.linspace(0,np.amax(x),100)
# plt.plot(_x, np.polyval(pfit4,_x))
#
# ds.plotAvsB("filtValue","derivativeLike")
# g = ds.choose()
# x = ds.filtValue[g]
# y = ds.derivativeLike[g]
# pfitd = np.polyfit(x,y,4)
# _x = np.linspace(0,np.amax(x),100)
# plt.plot(_x, np.polyval(pfitd,_x))
#
# def params(fv):
#     p3 = np.polyval(pfit3, fv)
#     p4 = np.polyval(pfit4, fv)
#     pd = np.polyval(pfitd, fv)
#
# def recordxy(fv):
#     p3, p4, pd =
