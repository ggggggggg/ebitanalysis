import mass
import off
import ebit
import devel
import collections
import os

import numpy as np
import pylab as plt
import progress.bar
import inspect
import fastdtw

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

    def __repr__(self):
        return "ExperimentStateFile: "+self.filename


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

class GroupLooper(object):
    """A mixin class to allow TESGroup objects to hold methods that loop over
    their constituent channels. (Has to be a mixin, in order to break the import
    cycle that would otherwise occur.)"""
    pass


def _add_group_loop(method):
    """Add MicrocalDataSet method `method` to GroupLooper (and hence, to TESGroup).

    This is a decorator to add before method definitions inside class MicrocalDataSet.
    Usage is:

    class MicrocalDataSet(...):
        ...

        @_add_group_loop
        def awesome_fuction(self, ...):
            ...
    """
    method_name = method.__name__

    def wrapper(self, *args, **kwargs):
        bar = SilenceBar(method_name, max=len(self.offFileNames), silence=not self.verbose)
        for (channum,ds) in self.items():
            try:
                method(ds, *args, **kwargs)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print("channel {} failed with {}".format(channum,e))
            bar.next()
        bar.finish()
    wrapper.__name__ = method_name

    # Generate a good doc-string.
    lines = ["Loop over self, calling the %s(...) method for each channel." % method_name]
    arginfo = inspect.getargspec(method)
    argtext = inspect.formatargspec(*arginfo)
    if method.__doc__ is None:
        lines.append("\n%s%s has no docstring" % (method_name, argtext))
    else:
        lines.append("\n%s%s docstring reads:" % (method_name, argtext))
        lines.append(method.__doc__)
    wrapper.__doc__ = "\n".join(lines)

    setattr(GroupLooper, method_name, wrapper)
    return method

# wrap up an off file with some conviencine functions
# like a TESChannel
class Channel():
    def __init__(self, offFile, experimentStateFile):
        self.offFile = offFile
        self.experimentStateFile = experimentStateFile
        self.markedBadBool = False
        self.injestLabelsAndTimestamps(experimentStateFile.labels, experimentStateFile.unixnanos)
        self.learnChannumAndShortname()
        self.learnStdDevResThresholdUsingMedianAbsoluteDeviation()

    def learnChannumAndShortname(self):
        basename, self.channum = mass.ljh_util.ljh_basename_channum(self.offFile.filename)
        self.shortName = os.path.split(basename)[-1] + " chan%g"%self.channum

    @_add_group_loop
    def learnStdDevResThresholdUsingMedianAbsoluteDeviation(self, nSigma = 7):
        median = np.median(self.residualStdDev)
        mad = np.median(np.abs(self.residualStdDev-median))
        k = 1.4826 # for gaussian distribution, ratio of sigma to median absolution deviation
        sigma = mad*k
        self.stdDevResThreshold = median+nSigma*sigma

    def learnStdDevResThresholdUsingRatioToNoiseStd(self, ratioToNoiseStd=1.5):
        self.stdDevResThreshold = self.offFile.header["ModelInfo"]["NoiseStandardDeviation"]*ratioToNoiseStd

    def injestLabelsAndTimestamps(self, labels, unixnanos, excludeStart = True, excludeEnd = True):
        self.statesDict = {}
        inds = np.searchsorted(self.offFile["unixnano"],unixnanos)
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

    @property
    def energyRough(self):
        uncalibrated = getattr(self, self.calibrationRough.uncalibratedName)
        return self.calibrationRough(uncalibrated)

    @property
    def arbsInRefChannelUnits(self):
        uncalibrated = getattr(self, self.calibrationArbsInRefChannelUnits.uncalibratedName)
        return self.calibrationArbsInRefChannelUnits(uncalibrated)

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

    @_add_group_loop
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

    def linefit(self,lineNameOrEnergy="MnKAlpha", attr="energy", states=None, axis=None,dlo=50,dhi=50,
                   binsize=1,binEdges=None,label="full",plot=True,
                   guessParams=None, g_func=None, holdvals=None):
        """Do a fit to `lineNameOrEnergy` and return the fitter. You can get the params results with fitter.last_fit_params_dict or any other way you like.
        lineNameOrEnergy -- A string like "MnKAlpha" will get "MnKAlphaFitter", your you can pass in a fitter like a mass.GaussianFitter().
        attr -- default is "energyRough". you must pass binEdges if attr is other than "energy" or "energyRough"
        states -- will be passed to hist, coAddStates will be True
        axis -- if axis is None and plot==True, will create a new figure, otherwise plot onto this axis
        dlo and dhi and binsize -- by default it tries to fit with bin edges given by np.arange(fitter.spect.nominal_peak_energy-dlo, fitter.spect.nominal_peak_energy+dhi, binsize)
        binEdges -- pass the binEdges you want as a numpy array
        label -- passed to fitter.plot
        plot -- passed to fitter.fit, determine if plot happens
        guessParams -- passed to fitter.fit, fitter.fit will guess the params on its own if this is None
        category -- pass {"side":"A"} or similar to use categorical cuts
        g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        holdvals -- a dictionary mapping keys from fitter.params_meaning to values... eg {"background":0, "dP_dE":1}
            This vector is anded with the vector calculated by the histogrammer
        """
        if isinstance(lineNameOrEnergy, mass.LineFitter):
            fitter = lineNameOrEnergy
            nominal_peak_energy = fitter.spect.nominal_peak_energy
        elif isinstance(lineNameOrEnergy,str):
            fitter = mass.fitter_classes[lineNameOrEnergy]()
            nominal_peak_energy = fitter.spect.nominal_peak_energy
        else:
            fitter = mass.GaussianFitter()
            nominal_peak_energy = float(lineNameOrEnergy)
        if binEdges is None:
            if attr == "energy" or attr == "energyRough":
                binEdges = np.arange(nominal_peak_energy-dlo, nominal_peak_energy+dhi, binsize)
            else:
                raise Exception("must pass binEdges if attr is other than energy or energyRough")
        if axis is None and plot:
            plt.figure()
            axis = plt.gca()
        bin_centers, counts = self.hist(binEdges, attr, states, g_func)
        if guessParams is None:
            guessParams = fitter.guess_starting_params(counts,bin_centers)
        if holdvals is None:
            holdvals = {}
        if (attr == "energy" or attr == "energyRough") and "dP_dE" in fitter.param_meaning:
            holdvals["dP_dE"]=1.0
        hold = []
        for (k,v) in holdvals.items():
            i = fitter.param_meaning[k]
            guessParams[i]=v
            hold.append(i)

        params, covar = fitter.fit(counts, bin_centers,params=guessParams,axis=axis,label=label,plot=plot, hold=hold)
        if plot:
            axis.set_title(self.shortName+", {}, states = {}".format(lineNameOrEnergy,states))
            if attr == "energy" or attr == "energyRough":
                plt.xlabel(attr+" (eV)")
            else:
                plt.xlabel(attr+ "(arbs)")

        return fitter

    def learnCalibration(self, attr, calibrationPlan, curvetype = "gain", dlo=50,dhi=50, binsize=1):
        self.learnCalibrationRough(attr,calibrationPlan)
        self.calibration = mass.EnergyCalibration(curvetype=curvetype)
        self.calibration.uncalibratedName = attr
        fitters = []
        for (ph, energy, name, states) in zip(calibrationPlan.uncalibratedVals, calibrationPlan.calibratedVals,
                                      calibrationPlan.names, calibrationPlan.states):
            if name in mass.fitter_classes:
                fitter = self.linefit(name, "energyRough", states, dlo=dlo, dhi=dhi,
                                plot=False, binsize=binsize)
            else:
                fitter = self.linefit(energy, "energyRough", states, dlo=dlo, dhi=dhi,
                                plot=False, binsize=binsize)
            fitters.append(fitter)
            if not fitter.fit_success or np.abs(fitter.last_fit_params_dict["peak_ph"][0]-energy)>10:
                self.markSelfBad("failed fit", fitter, extraInfo = fitter)
                continue
            phRefined = self.calibrationRough.energy2ph(fitter.last_fit_params_dict["peak_ph"][0])
            self.calibration.add_cal_point(phRefined, energy, name)
        return fitters

    def learnCalibrationRough(self, attr, calibrationPlan):
        self.calibrationRough = calibrationPlan.getRoughCalibration()
        assert(hasattr(self, attr))
        self.calibrationRough.uncalibratedName = attr
        self.calibrationRoughPlan = calibrationPlan

    def learnCalibrationInRefChannelUnits(self, calibrationPlan):
        pass

    def markSelfBad(self, reason, extraInfo = None):
        self.markedBadReson = reason
        self.markedBadExtraInfo = extraInfo
        self.markedBadBool = True
        print("MARK SELF BAD REQUESTED, BUT NOT IMPLMENTED: \nreason: {}\n self: {}\nextraInfo: {}".format(self,
                                                                                               reason, extraInfo))

    def plotResidualStdDev(self, axis = None):
        if axis is None:
            plt.figure()
            ax = plt.gca()
        x = np.sort(ds.residualStdDev)/self.offFile.header["ModelInfo"]["NoiseStandardDeviation"]
        y = np.linspace(0,1,len(self))
        inds = x>(self.stdDevResThreshold/self.offFile.header["ModelInfo"]["NoiseStandardDeviation"])
        plt.plot(x,y, label="<threshold")
        plt.plot(x[inds], y[inds], "r", label=">threshold")
        plt.vlines(self.stdDevResThreshold/self.offFile.header["ModelInfo"]["NoiseStandardDeviation"], 0, 1)
        plt.xlabel("residualStdDev/noiseStdDev")
        plt.ylabel("fraction of pulses with equal or lower residualStdDev")
        plt.title("{}, {} total pulses, {:0.3f} cut".format(
            self.shortName, len(self), inds.sum()/float(len(self)) ))
        plt.legend()
        plt.xlim(max(0,x[0]), 1.5)
        plt.ylim(0,1)

    def __len__(self):
        return len(self.offFile)

    def alignToReferenceChannel(self, referenceChannel):
        # is referenceChannel calibrated?
        refPlan = referenceChannel.calibrationRoughPlan


class AlignBToA():
    cm = plt.cm.gist_ncar
    def __init__(self,ds_a, ds_b, peak_xs_a, bin_edges, attr, states = None,
                 scale_by_median = True, normalize_before_dtw = True):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.bin_edges = bin_edges
        self.bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        self.peak_xs_a = peak_xs_a
        self.attr = attr
        self.scale_by_median = scale_by_median
        self.normalize_before_dtw = normalize_before_dtw
        self.states = states
        self.peak_inds_b = self.samePeaks()
        self.addCalToB()

    def samePeaks(self):
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.choose(self.states)]
        ph_b = getattr(self.ds_b,self.attr)[self.ds_b.choose(self.states)]
        if self.scale_by_median:
            median_ratio_a_over_b = np.median(ph_a)/np.median(ph_b)
        else:
            median_ratio_a_over_b = 1.0
        ph_b_median_scaled = ph_b*median_ratio_a_over_b
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b_median_scaled, _ = np.histogram(ph_b_median_scaled, self.bin_edges)
        self.peak_inds_a = self.findPeakIndsA(counts_a)
        if self.normalize_before_dtw:
            distance, path = fastdtw.fastdtw(self.normalize(counts_a), self.normalize(counts_b_median_scaled))
        else:
            distance, path = fastdtw.fastdtw(counts_a, counts_b_median_scaled)
        i_a = [x[0] for x in path]
        i_b_median_scaled = [x[1] for x in path]
        peak_inds_b_median_scaled = np.array([i_b_median_scaled[i_a.index(pia)] for pia in self.peak_inds_a])
        peak_xs_b_median_scaled = self.bin_edges[peak_inds_b_median_scaled]
        peak_xs_b = peak_xs_b_median_scaled/median_ratio_a_over_b
        min_bin = self.bin_edges[0]
        bin_spacing = self.bin_edges[1]-self.bin_edges[0]
        peak_inds_b = map(int,(peak_xs_b-min_bin)/bin_spacing)
        return peak_inds_b

    def findPeakIndsA(self, counts_a):
        peak_inds_a = np.searchsorted(self.bin_edges, self.peak_xs_a)-1
        return peak_inds_a

    def samePeaksPlot(self):
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.choose(self.states)]
        ph_b = getattr(self.ds_b,self.attr)[self.ds_b.choose(self.states)]
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b, _ = np.histogram(ph_b, self.bin_edges)
        plt.figure()
        plt.plot(self.bin_centers,counts_a,label="a: channel %i"%self.ds_a.channum)
        for i,pi in enumerate(self.peak_inds_a):
            plt.plot(self.bin_centers[pi],counts_a[pi],"o",color=self.cm(float(i)/len(self.peak_inds_a)))

        plt.plot(self.bin_centers,counts_b,label="b: channel %i"%self.ds_b.channum)
        for i,pi in enumerate(self.peak_inds_b):
            plt.plot(self.bin_centers[pi],counts_b[pi],"o",color=self.cm(float(i)/len(self.peak_inds_b)))
        plt.xlabel(self.attr)
        plt.ylabel("counts per %0.2f unit bin"%(self.bin_centers[1]-self.bin_centers[0]))
        plt.legend()
        plt.title(self.ds_a.shortName+" + "+self.ds_b.shortName+"\nwith same peaks noted, peaks not expected to be aligned in this plot")

    def samePeaksPlotWithAlignmentCal(self):
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.choose(self.states)]
        ph_b = self.ds_b.arbsInRefChannelUnits[self.ds_b.choose(self.states)]
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b, _ = np.histogram(ph_b, self.bin_edges)
        plt.figure()
        plt.plot(self.bin_centers,counts_a,label="a: channel %i"%self.ds_a.channum)
        for i,pi in enumerate(self.peak_inds_a):
            plt.plot(self.bin_centers[pi],counts_a[pi],"o",color=self.cm(float(i)/len(self.peak_inds_a)))
        plt.plot(self.bin_centers,counts_b,label="b: channel %i"%self.ds_b.channum)
        for i,pi in enumerate(self.peak_inds_a):
            plt.plot(self.bin_centers[pi],counts_b[pi],"o",color=self.cm(float(i)/len(self.peak_inds_a)))
        plt.xlabel("arbsInRefChannelUnits (ref channel = {})".format(self.ds_a.channum))
        plt.ylabel("counts per %0.2f unit bin"%(self.bin_centers[1]-self.bin_centers[0]))
        plt.legend()

    def normalize(self,x):
        return x/float(np.sum(x))

    def addCalToB(self):
        cal_b_to_a = mass.EnergyCalibration(curvetype="gain")
        for pi_a,pi_b in zip(self.peak_inds_a, self.peak_inds_b):
            cal_b_to_a.add_cal_point(self.bin_centers[pi_b], self.bin_centers[pi_a])
        cal_b_to_a.uncalibratedName = self.attr
        self.ds_b.calibrationArbsInRefChannelUnits=cal_b_to_a
        self.cal_b_to_a = cal_b_to_a

    def testForGoodnessBasedOnCalCurvature(self, threshold_frac = .1):
        assert threshold_frac > 0
        threshold_hi = 1+threshold_frac
        threshold_lo = 1/threshold_hi
        # here we test the "curvature" of cal_b_to_a
        # by comparing the most extreme sloped segment to the median slope
        derivatives = self.cal_b_to_a.energy2dedph(self.cal_b_to_a._energies)
        diff_frac_hi = np.amax(derivatives)/np.median(derivatives)
        diff_frac_lo = np.amin(derivatives)/np.median(derivatives)
        return diff_frac_hi < threshold_hi and diff_frac_lo > threshold_lo

    def _laplaceEntropy(self, w=None):
        if w == None:
            w = self.bin_edges[1]-self.bin_edges[0]
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.choose(self.states)]
        ph_b = getattr(self.ds_b,self.newattr)[self.ds_b.choose(self.states)]
        entropy = mass.entropy.laplace_cross_entropy(ph_a[ph_a>self.bin_edges[0]],
                     ph_b[ph_b>self.bin_edges[0]], w=w)
        return entropy

    def _ksStatistic(self):
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.choose(self.states)]
        ph_b = getattr(self.ds_b,self.newattr)[self.ds_b.choose(self.states)]
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b, _ = np.histogram(ph_b, self.bin_edges)
        cdf_a = np.cumsum(counts_a)/np.sum(a)
        cdf_b = np.cumsum(counts_b)/np.sum(b)
        ks_statistic = np.amax(np.abs(cdf_a-cdf_b))
        return ks_statistic


# calibration
class CalibrationPlan():
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
        s = """CalibrationPlan with {} entries
        x: {}
        y: {}
        states: {}
        names: {}""".format(len(self.names),self.uncalibratedVals, self.calibratedVals, self.states, self.names)
        return s

    def getRoughCalibration(self):
        cal = mass.EnergyCalibration(curvetype="gain")
        for (x,y,name) in zip(self.uncalibratedVals, self.calibratedVals, self.names):
            cal.add_cal_point(x,y,name)
        return cal


def getOffFileListFromOneFile(filename, maxChans=None):
    basename, _ = mass.ljh_util.ljh_basename_channum(filename)
    z = mass.ljh_util.filename_glob_expand(basename+"_chan*.off")
    if maxChans is not None:
        z = z[:min(maxChans, len(z))]
    return z


class SilenceBar(progress.bar.Bar):
    "A progres bar that can be turned off by passing silence=True"
    def __init__(self,message, max, silence):
        self.silence = silence
        if not self.silence:
            progress.bar.Bar.__init__(self, message, max=max)

    def next(self):
        if not self.silence:
            progress.bar.Bar.next(self)

    def finish(self):
        if not self.silence:
            progress.bar.Bar.finish(self)


class ChannelGroup(GroupLooper, collections.OrderedDict):
    def __init__(self, offFileNames, verbose=True):
        collections.OrderedDict.__init__(self)
        self.verbose = verbose
        self.offFileNames = offFileNames
        self.experimentStateFile = ExperimentStateFile(offFilename=self.offFileNames[0])
        self.loadChannels()

    def loadChannels(self):
        bar = SilenceBar('Parse OFF File Headers', max=len(self.offFileNames), silence=not self.verbose)
        for name in self.offFileNames:
            _, channum = mass.ljh_util.ljh_basename_channum(name)
            self[channum] = Channel(off.OFFFile(name), self.experimentStateFile)
            bar.next()
        bar.finish()

    def __repr__(self):
        return "ChannelGroup with {} channels".format(len(self))

    def alignToReferenceChannel(self, referenceChannelNumber=None):
        if referenceChannelNumber is None:
            ref = self.firstGoodChannel()
        else:
            ref = self[referenceChannelNumber]
        bar = SilenceBar('alignToReferenceChannel', max=len(self.offFileNames), silence=not self.verbose)
        for (channum, ds) in self.items():
            ds.alignToReferenceChannel(ds)
            bar.next()
        bar.finish()

    def firstGoodChannel(self):
        return self[1]






data = ChannelGroup(getOffFileListFromOneFile(filename, maxChans=4))
data.learnDriftCorrection()
ds=data.firstGoodChannel()
ds.plotAvsB("relTimeSec", "residualStdDev",  includeBad=True)
ds.plotAvsB("relTimeSec", "pretriggerMean", includeBad=True)
ds.plotAvsB("relTimeSec", "filtValue", includeBad=False)
ds.plotHist(np.arange(0,40000,4),"filtValue")
ds.plotHist(np.arange(0,40000,4),"filtValue", coAddStates=False)
ds.plotResidualStdDev()
driftCorrectInfo = ds.learnDriftCorrection()
ds.plotCompareDriftCorrect()

# filt value, line name or energy, states, name
calibrationPlan = CalibrationPlan()
calibrationPlan.addCalPoint(3404, "Ne He-Like 1s2p", states="B")
calibrationPlan.addCalPoint(3768, "Ne H-Like 2p", states="B")
calibrationPlan.addCalPoint(7869, 2181.4, states="C", name = "W Ni-8")
ds.learnCalibrationRough("filtValueDC",calibrationPlan)
fitters = ds.learnCalibration("filtValueDC",calibrationPlan) # overwrites calibrationRough




ds.plotHist(np.arange(0,4000,1),"energyRough", coAddStates=False)

ds.linefit("Ne H-Like 2p",attr="energy",states="B")
ds.linefit("Ne He-Like 1s2p",attr="energy",states="B")
ds.linefit(2181.4,attr="energy",states="C")
# bake in energyRough and arbUnitsOfReferenceChannel
# linefit
# execute calibration plan

ds.plotHist(np.arange(0,4000,4),"energy", coAddStates=False)


aligner = AlignBToA(ds, data[3], calibrationPlan.uncalibratedVals, np.arange(500,20000,4), "filtValueDC")
aligner.samePeaksPlot()
aligner.samePeaksPlotWithAlignmentCal()


plt.show()
