import mass
import off
import ebit
import devel
import collections
import os

import numpy as np
import pylab as plt

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

dirname = "/Users/oneilg/Documents/EBIT/data/20181205_BC"
filename = "/Users/oneilg/Documents/EBIT/data/20181205_BC/20181205_BC_chan1.off"

class ExperimentStateFile():
    def __init__(self, filename=None, offFilename=None):
        if filename is not None:
            self.filename = filename
        elif offFilename is not None:
            self.filename = self.experimentStateFilenameFromOffFilename(offFilename)
        else:
            raise Exception("provide filename or offFilename")
        self.parse()

    def experimentStateFilenameFromOffFilename(self,offFilename):
        basename, channum = mass.ljh_util.ljh_basename_channum(offFilename)
        return basename+"_experiment_state.txt"

    def parse(self):
        with open(self.filename,"r") as f:
            lines = f.readlines()
        if len(lines) < 1:
            raise Exception("zero lines in file")
        if not lines[0][0] == "#":
            raise Exception("first line should start with #, was %s"%lines[0])
        unixnanos = []
        labels = []
        for line in lines[1:]:
            a,b = line.split(",")
            a=a.strip()
            b=b.strip()
            unixnano = int(a)
            label = b
            unixnanos.append(unixnano)
            labels.append(label)
        self.labels = labels
        self.unixnanos = np.array(unixnanos)





states = ExperimentStateFile(offFilename=filename)

f = off.OFFFile(filename)


def annotate_lines(axis,labelLines, labelLines_color2=[],color1 = "k",color2="r"):
    """Annotate plot on axis with line names.
    labelLines -- eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    labelLines_color2 -- optional,eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    color1 -- text color for labelLines
    color2 -- text color for labelLines_color2
    """
    n=len(labelLines)+len(labelLines_color2)
    yscale = plt.gca().get_yscale()
    for (i,labelLine) in enumerate(labelLines):
        energy = mass.STANDARD_FEATURES[labelLine]
        if yscale=="linear":
            axis.annotate(labelLine, (energy, (1+i)*plt.ylim()[1]/float(1.5*n)), xycoords="data",color=color1)
        elif yscale=="log":
            axis.annotate(labelLine, (energy, np.exp((1+i)*np.log(plt.ylim()[1])/float(1.5*n))), xycoords="data",color=color1)
    for (j,labelLine) in enumerate(labelLines_color2):
        energy = mass.STANDARD_FEATURES[labelLine]
        if yscale=="linear":
            axis.annotate(labelLine, (energy, (2+i+j)*plt.ylim()[1]/float(1.5*n)), xycoords="data",color=color2)
        elif yscale=="log":
            axis.annotate(labelLine, (energy, np.exp((2+i+j)*np.log(plt.ylim()[1])/float(1.5*n))), xycoords="data",color=color2)

class DriftCorrection():
    def __init__(self, indicatorName, uncorrectedName, medianIndicator, slope):
        self.indicatorName = indicatorName
        self.uncorrectedName = uncorrectedName
        self.medianIndicator = medianIndicator
        self.slope = slope

    def apply(self, indicator, uncorrected):
        gain = 1+(indicator-self.medianIndicator)*self.slope
        return gain*uncorrected

    def toHDF5(self, h5):
        pass

    # this can't be an instance method, maybe a class method? or just a function
    def fromHDF5(self):
        pass

# wrap up an off file with some conviencine functions
# like a TESChannel
class Channel():
    def __init__(self, offFile, experimentStateFile, stdDevResThreshold):
        self.offFile = offFile
        self.experimentStateFile = experimentStateFile
        self.stdDevResThreshold = stdDevResThreshold
        self.injestLabelsAndTimestamps(experimentStateFile.labels, experimentStateFile.unixnanos)
        self.learnChannumAndShortname()

    def learnChannumAndShortname(self):
        basename, self.channum = mass.ljh_util.ljh_basename_channum(self.offFile.filename)
        self.shortName = os.path.split(basename)[-1] + " chan%g"%self.channum

    def injestLabelsAndTimestamps(self, labels, unixnanos, excludeStart = True, excludeEnd = True):
        self.statesDict = {}
        inds = np.searchsorted(f["unixnano"],states.unixnanos)
        for i, label in enumerate(labels):
            if label == "START" and excludeStart:
                continue
            if label == "END" and excludeEnd:
                continue
            if not label in self.statesDict:
                self.statesDict[label] = np.zeros(self.nRecords,dtype="bool")
            if i+1 == len(labels):
                self.statesDict[label][inds[i]:self.nRecords]=True
            else:
                self.statesDict[label][inds[i]:inds[i+1]]=True

    def choose(self, states=None, good=True):
        """ return boolean indicies of "choose" pulses
        if state is none, all states are chosen
        ds.choose("A") selects state A
        ds.choose(["A","B"]) selects states A and B
        """
        g = self.offFile["residualStdDev"]<self.stdDevResThreshold
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

    def __repr__(self):
        return "Channel based on %s"%self.offFile

    @property
    def nRecords(self):
        return len(self.offFile)

    @property
    def stateLabels(self):
        return self.statesDict.keys()

    @property
    def residualStdDev(self):
        return self.offFile["residualStdDev"]

    @property
    def pretriggerMean(self):
        return self.offFile["pretriggerMean"]

    @property
    def relTimeSec(self):
        t = self.offFile["unixnano"]
        return (t-t[0])/1e9

    @property
    def filtValue(self):
        return self.offFile["coefs"][:,2]

    @property
    def filtValueDC(self):
        indicator = getattr(self, self.driftCorrection.indicatorName)
        uncorrected = getattr(self, self.driftCorrection.uncorrectedName)
        return self.driftCorrection.apply(indicator, uncorrected)

    @property
    def energy(self):
        uncalibrated = getattr(self, self.calibration.uncalibratedName)
        return self.calibration(uncalibrated)

    def plotAvsB(self, nameA, nameB, axis=None, states=None, includeBad=False):
        if axis == None:
            plt.figure()
            axis = plt.gca()
        A = getattr(self,nameA)
        B = getattr(self,nameB)
        if states == None:
            states = self.stateLabels
        for state in states:
            g = self.choose(state)
            axis.plot(A[g], B[g], ".", label=state)
            if includeBad:
                b = self.choose(state, good=False)
                axis.plot(A[b], B[b], "x", label=state+" bad")
        plt.xlabel(nameA)
        plt.ylabel(nameB)
        plt.title(self.shortName)
        plt.legend()
        return axis

    def hist(self, binEdges, attr, states=None, g_func=None):
        """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute). automatically filtes out nan values
        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram eg "filt_value"
        g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.choose() would return
            This vector is anded with the vector from ds.choose
         """
        binEdges = np.array(binEdges)
        binCenters = 0.5*(binEdges[1:]+binEdges[:-1])
        vals = getattr(self, attr)
        g = self.choose(states)
        if g_func is not None:
            g=np.logical_and(g,g_func(self))
        counts, _ = np.histogram(vals[g],binEdges)
        return binCenters, counts

    def plotHist(self,binEdges,attr,axis=None,labelLines=[],states=None,g_func=None, coAddStates=True):
        """plot a coadded histogram from all good datasets and all good pulses
        binEdges -- edges of bins unsed for histogram
        attr -- which attribute to histogram "p_energy" or "p_filt_value"
        axis -- if None, then create a new figure, otherwise plot onto this axis
        annotate_lines -- enter lines names in STANDARD_FEATURES to add to the plot, calls annotate_lines
        g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
            This vector is anded with the vector calculated by the histogrammer    """
        if axis is None:
            plt.figure()
            axis=plt.gca()
        if states == None:
            states = self.stateLabels
        if coAddStates:
            x,y = self.hist(binEdges, attr, states=states, g_func=g_func)
            axis.plot(x,y,drawstyle="steps-mid", label=states)
        else:
            for state in states:
                x,y = self.hist(binEdges, attr, states=state, g_func=g_func)
                axis.plot(x,y,drawstyle="steps-mid", label=state)
        axis.set_xlabel(attr)
        axis.set_ylabel("counts per %0.1f unit bin"%(binEdges[1]-binEdges[0]))
        plt.legend()
        axis.set_title(self.shortName)
        annotate_lines(axis, labelLines)

    def learnDriftCorrection(self, states = None):
        g = self.choose(states)
        indicator = self.pretriggerMean[g]
        uncorrected = self.filtValue[g]
        slope, info = mass.core.analysis_algorithms.drift_correct(indicator, uncorrected)
        self.driftCorrection = DriftCorrection("pretriggerMean", "filtValue", info["median_pretrig_mean"], slope)
        # we dont want to storeFiltValueDC in memory, we simply store a DriftCorrection object
        return self.driftCorrection

    def loadDriftCorrection(self):
        pass

    def hasDriftCorrection(self):
        return hasattr(self, driftCorrection)

    def plotCompareDriftCorrect(self, axis=None, states=None, includeBad=False):
        if axis == None:
            plt.figure()
            axis = plt.gca()
        A = getattr(self,self.driftCorrection.indicatorName)
        B = getattr(self,self.driftCorrection.uncorrectedName)
        C = getattr(self,"filtValueDC")
        if states == None:
            states = self.stateLabels
        for state in states:
            g = self.choose(state)
            axis.plot(A[g], B[g], ".", label=state)
            axis.plot(A[g], C[g], ".", label=state+" DC")
            if includeBad:
                b = self.choose(state, good=False)
                axis.plot(A[b], B[b], "x", label=state+" bad")
                axis.plot(A[b], C[b], "x", label=state+" bad DC")
        plt.xlabel(self.driftCorrection.indicatorName)
        plt.ylabel(self.driftCorrection.uncorrectedName +",filtValueDC")
        plt.title(self.shortName+" drift correct comparison")
        plt.legend()
        return axis




# I want to trim off the START and END
# and make an api for choosing the choose inds off f with a certain label or labels
# f.filt_value[f.choose("A")]
# f.filt_value[f.choose("A","B","Ir 12 kV")]
inds = np.searchsorted(f["unixnano"],states.unixnanos)

ds = Channel(f,states,30)
ds.plotAvsB("relTimeSec", "residualStdDev",  includeBad=True)
ds.plotAvsB("relTimeSec", "pretriggerMean", includeBad=True)
ds.plotAvsB("relTimeSec", "filtValue", includeBad=False)
ds.plotHist(np.arange(0,40000,4),"filtValue")
ds.plotHist(np.arange(0,40000,4),"filtValue", coAddStates=False)


# drift correction
driftCorrectInfo = ds.learnDriftCorrection()
ds.plotCompareDriftCorrect()
plt.show()

# calibration
class Calibrator():
    def __init__(self):
        self.uncalibratedVals = np.zeros(0)
        self.calibratedVals = np.zeros(0)
        self.states = []
        self.names = []

    def addCalPoint(self, uncalibratedVal, nameOrCalibratedVal, states=None, name=""):
        self.uncalibratedVals = np.hstack((self.uncalibratedVals, uncalibratedVal))
        if isinstance(nameOrCalibratedVal, str):
            name = nameOrCalibratedVal
            if name in mass.spectrum_classes:
                calibratedVal = mass.spectrum_classes[name]().peak_energy
            elif name in mass.STANDARD_FEATURES:
                calibratedVal = mass.STANDARD_FEATURES[name]
            self.names.append(name)
            self.calibratedVals = np.hstack((self.calibratedVals, calibratedVal))
        else:
            calibratedVal = nameOrCalibratedVal
            self.calibratedVals = np.hstack((self.calibratedVals, calibratedVal))
            self.names.append(name)
        self.states.append(states)

    def __repr__(self):
        s = """Calibrator with {} entries
        x: {}
        y: {}
        states: {}
        names: {}""".format(len(self.names),self.uncalibratedVals, self.calibratedVals, self.states, self.names)
        return s

    def getRoughCalibration(self):
        cal = mass.EnergyCalibration()
        for (x,y,name) in zip(self.uncalibratedVals, self.calibratedVals, self.names):
            cal.add_cal_point(x,y,name)
        return cal

# filt value, line name or energy, states, name
cal = Calibrator()
cal.addCalPoint(3404, "Ne He-Like 1s2p", states="B")
cal.addCalPoint(3768, "Ne H-Like 2p", states="B")
cal.addCalPoint(7869, 2181.4, states="C", name = "W Ni-8")
roughCal = cal.getRoughCalibration()
roughCal.uncalibratedName = "filtValueDC"
ds.calibration = roughCal
ds.energy

ds.plotHist(np.arange(0,4000,1),"energy", coAddStates=False)
