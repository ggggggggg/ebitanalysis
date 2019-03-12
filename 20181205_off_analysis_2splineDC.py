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
data = ChannelGroup(getOffFileListFromOneFile(filename, maxChans=4))
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

ds.calibrationPlanInit("filtValueDC")
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

ph = ds.calibration._ph
energies = ds.calibration._energies
names = ds.calibration._names
plan = ds.calibrationPlan
timesdict = {}
for state in ds.stateLabels:
    timesdict[state]=np.mean(ds.relTimeSec[ds.choose(states=state)])
times = [timesdict[state] for state in plan.states]

t0 = timesdict["Ne"]
t1 = timesdict["CO2"]


class Cal2Spline():
    def __init__(self, t0, t1, s0, s1):
        self.t0 = t0
        self.t1 = t1
        self.s0 = s0
        self.s1 = s1

    def __call__(self, ph, t):
        e0 = self.s0(ph)
        e1 = self.s1(ph)
        a0 = (self.t1-t)/(self.t1-self.t0)
        a1 = (t-self.t0)/(self.t1-self.t0)
        e = a0*e0+a1*e1
        return e

def cal2spline(t0,t1,s0,s1,ph,t):
    e0 = s0(ph)
    e1 = s1(ph)
    a0 = (t1-t)/(t1-t0)
    a1 = (t-t0)/(t1-t0)
    e = a0*e0+a1*e1
    return e

s0 = ds.calibration
s1 = lambda x: 0*x
s2 = Cal2Spline(t0, t1, s0, s1)
assert(np.abs(s2(1000, t0)-286.37599318810084)<20)
assert(s2(1000, t1)==0)
assert(np.abs(cal2spline(t0, t1, s0, s1, 1000, t0)-s2(1000,t0))<1e-7)
assert(cal2spline(t0, t1, s0, s1, 1000, t1)==0)
assert(all(plan.energies==energies))

import lmfit
params = lmfit.Parameters()
knot_energies = np.linspace(0,np.amax(energies),len(energies)-3)[1:]
for i,e in enumerate(knot_energies):
    _ph = ds.calibration.energy2ph(e)
    params.add("ph0_{}".format(i), _ph, vary=True, max=_ph+100, min=_ph-100)
params.add("a",1.0)
params.add("b",1/100.0**2)
params.add("c",-1/4000.0**3)

def Cal2SplineFromParams(params, t0, t1, knot_energies):
    c0 = mass.EnergyCalibration(curvetype="gain")
    c1 = mass.EnergyCalibration(curvetype="gain")
    a=params["a"].value
    b=params["b"].value
    c=params["c"].value
    for k,p in params.items():
        if not k.startswith("ph"):
            continue
        j,i = map(int,k[2:].split("_"))
        c0.add_cal_point(p.value,knot_energies[i])
        poly_energy = a*knot_energies[i] + b*knot_energies[i]**2 + c*knot_energies[i]**3
        c1.add_cal_point(p.value,poly_energy)
    return Cal2Spline(t0, t1, c0, c1)

def err(params,ph,energies,times, t0, t1, knot_energies):
    s2 = Cal2SplineFromParams(params, t0, t1, knot_energies)
    calc_energies = s2(ph, times)
    residuals = calc_energies-energies
    return residuals

result = lmfit.minimize(err, params.copy(), args=(ph, energies, times, t0, t1, knot_energies))

a=result.params["a"].value
b=result.params["b"].value
c=result.params["c"].value
poly_energy = a*knot_energies + b*knot_energies**2 + c*knot_energies**3
plt.plot(knot_energies, poly_energy-knot_energies, label="b_guess={}".format(params["b"].value))

params["b"].value*=-1
result2 = lmfit.minimize(err, params.copy(), args=(ph, energies, times, t0, t1, knot_energies))

a2=result2.params["a"].value
b2=result2.params["b"].value
c2=result2.params["c"].value
poly_energy2 = a2*knot_energies + b2*knot_energies**2 + c2*knot_energies**3
plt.plot(knot_energies, poly_energy2-knot_energies, label="b_guess={}".format(params["b"].value))
plt.legend()

s2 = Cal2SplineFromParams(result.params, t0, t1, knot_energies)

ds.energyCal2Spline = s2(ds.filtValueDC, ds.relTimeSec)


def calibrate2Spline(self):
    ph = self.calibration._ph
    energies = self.calibration._energies
    names = self.calibration._names
    plan = self.calibrationPlan
    timesdict = {}
    for state in self.stateLabels:
        timesdict[state]=np.mean(self.relTimeSec[self.choose(states=state)])
    times = [timesdict[state] for state in plan.states]
    t0 = timesdict["Ne"]
    t1 = timesdict["CO2"]

    def err(params,ph,energies,times, t0, t1, knot_energies):
        s2 = Cal2SplineFromParams(params, t0, t1, knot_energies)
        calc_energies = s2(ph, times)
        residuals = calc_energies-energies
        return residuals

    params = lmfit.Parameters()
    knot_energies = np.linspace(0,np.amax(energies),len(energies)-3)[1:]
    for i,e in enumerate(knot_energies):
        _ph = self.calibration.energy2ph(e)
        params.add("ph0_{}".format(i), _ph, vary=True, max=_ph+100, min=_ph-100)
    params.add("a",1.0)
    params.add("b",1/100.0**2)
    params.add("c",-1/4000.0**3)

    result = lmfit.minimize(err, params.copy(), args=(ph, energies, times, t0, t1, knot_energies))
    s2 = Cal2SplineFromParams(result.params, t0, t1, knot_energies)
    self.calibration2Spline = s2
    self.energyCal2Spline = s2(self.filtValueDC, self.relTimeSec)

Channel.calibrate2Spline = calibrate2Spline

for ds in data.values():
    ds.calibrate2Spline()

plt.figure(figsize=(12,10))
ax1 = plt.subplot(2,1,1)
data.plotHists(np.arange(4000), "energy", axis=ax1)
ax2 = plt.subplot(2,1,2,sharex=ax1)
data.plotHists(np.arange(4000), "energyCal2Spline", axis=ax2)
