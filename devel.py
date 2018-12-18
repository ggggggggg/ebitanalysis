import mass
import numpy as np
import pylab as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import h5py
import unittest
import fastdtw
from uncertainties import ufloat
from uncertainties.umath import *
from collections import OrderedDict

def ds_shortname(self):
    """return a string containing part of the filename and the channel number, useful for labelling plots"""
    s = os.path.split(self.filename)[-1]
    chanstr = "chan%g"%self.channum
    if not chanstr in s:
        s+=chanstr
    return s

def data_shortname(self):
    """return a string containning part of the filename and the number of good channels"""
    ngoodchan = len([ds for ds in self])
    return mass.ljh_util.ljh_basename_channum(os.path.split(self.datasets[0].filename)[-1])[0]+", %g chans"%ngoodchan

def ds_hist(self,bin_edges,attr="p_energy",t0=0,tlast=1e20,category={},g_func=None):
    """return a tuple of (bin_centers, counts) of p_energy of good pulses (or another attribute). automatically filtes out nan values
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        This vector is anded with the vector calculated by the histogrammer    """
    bin_edges = np.array(bin_edges)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    vals = getattr(self, attr)[:]
    # sanitize the data bit
    tg = np.logical_and(self.p_timestamp[:]>t0,self.p_timestamp[:]<tlast)
    g = np.logical_and(tg,self.good(**category))
    g = np.logical_and(g,~np.isnan(vals))
    if g_func is not None:
        g=np.logical_and(g,g_func(self))

    counts, _ = np.histogram(vals[g],bin_edges)
    return bin_centers, counts

def data_hists(self,bin_edges,attr="p_energy",t0=0,tlast=1e20,category={},g_func=None):
    """return a tuple of (bin_centers, countsdict). automatically filters out nan values
    where countsdict is a dictionary mapping channel numbers to numpy arrays of counts
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        This vector is anded with the vector calculated by the histogrammer    """
    bin_edges = np.array(bin_edges)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    countsdict = {ds.channum:ds.hist(bin_edges, attr,t0,tlast,category,g_func)[1] for ds in self}
    return bin_centers, countsdict

def data_hist(self, bin_edges, attr="p_energy",t0=0,tlast=1e20,category={},g_func=None):
    """return a tuple of (bin_centers, counts) of p_energy of good pulses in all good datasets (use .hists to get the histograms individually). filters out nan values
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        This vector is anded with the vector calculated by the histogrammer    """
    bin_centers, countsdict = self.hists(bin_edges, attr,t0,tlast,category,g_func)
    counts = np.zeros_like(bin_centers, dtype="int")
    for (k,v) in countsdict.items():
        counts+=v
    return bin_centers, counts

def plot_hist(self,bin_edges,attr="p_energy",axis=None,label_lines=[],category={},g_func=None):
    """plot a coadded histogram from all good datasets and all good pulses
    bin_edges -- edges of bins unsed for histogram
    attr -- which attribute to histogram "p_energy" or "p_filt_value"
    axis -- if None, then create a new figure, otherwise plot onto this axis
    annotate_lines -- enter lines names in STANDARD_FEATURES to add to the plot, calls annotate_lines
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
        This vector is anded with the vector calculated by the histogrammer    """
    if axis is None:
        plt.figure()
        axis=plt.gca()
    x,y = self.hist(bin_edges, attr, category=category, g_func=g_func)
    axis.plot(x,y,drawstyle="steps-mid")
    axis.set_xlabel(attr)
    axis.set_ylabel("counts per %0.1f unit bin"%(bin_edges[1]-bin_edges[0]))
    axis.set_title(self.shortname())
    annotate_lines(axis, label_lines)

def annotate_lines(axis,label_lines, label_lines_color2=[],color1 = "k",color2="r"):
    """Annotate plot on axis with line names.
    label_lines -- eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    label_lines_color2 -- optional,eg ["MnKAlpha","TiKBeta"] list of keys of STANDARD_FEATURES
    color1 -- text color for label_lines
    color2 -- text color for label_lines_color2
    """
    n=len(label_lines)+len(label_lines_color2)
    yscale = plt.gca().get_yscale()
    for (i,label_line) in enumerate(label_lines):
        energy = mass.STANDARD_FEATURES[label_line]
        if yscale=="linear":
            axis.annotate(label_line, (energy, (1+i)*plt.ylim()[1]/float(1.5*n)), xycoords="data",color=color1)
        elif yscale=="log":
            axis.annotate(label_line, (energy, np.exp((1+i)*np.log(plt.ylim()[1])/float(1.5*n))), xycoords="data",color=color1)
    for (j,label_line) in enumerate(label_lines_color2):
        energy = mass.STANDARD_FEATURES[label_line]
        if yscale=="linear":
            axis.annotate(label_line, (energy, (2+i+j)*plt.ylim()[1]/float(1.5*n)), xycoords="data",color=color2)
        elif yscale=="log":
            axis.annotate(label_line, (energy, np.exp((2+i+j)*np.log(plt.ylim()[1])/float(1.5*n))), xycoords="data",color=color2)

def ds_linefit(self,line_name="MnKAlpha", t0=0,tlast=1e20,axis=None,dlo=50,dhi=50,
               binsize=1,bin_edges=None, attr="p_energy",label="full",plot=True,
               guess_params=None, ph_units="eV", category={}, g_func=None,holdvals={}):
    """Do a fit to `line_name` and return the fitter. You can get the params results with fitter.last_fit_params_dict or any other way you like.
    line_name -- A string like "MnKAlpha" will get "MnKAlphaFitter", your you can pass in a fitter like a mass.GaussianFitter().
    t0 and tlast -- cuts all pulses outside this timerange before fitting
    axis -- if axis is None and plot==True, will create a new figure, otherwise plot onto this axis
    dlo and dhi and binsize -- by default it tries to fit with bin edges given by np.arange(fitter.spect.nominal_peak_energy-dlo, fitter.spect.nominal_peak_energy+dhi, binsize)
    bin_edges -- pass the bin_edges you want as a numpy array
    attr -- default is "p_energy", you could pick "p_filt_value" or others. be sure to pass in bin_edges as well because the default calculation will probably fail for anything other than p_energy
    label -- passed to fitter.plot
    plot -- passed to fitter.fit, determine if plot happens
    guess_params -- passed to fitter.fit, fitter.fit will guess the params on its own if this is None
    ph_units -- passed to fitter.fit, used in plot label
    category -- pass {"side":"A"} or similar to use categorical cuts
    g_func -- a function a function taking a MicrocalDataSet and returnning a vector like ds.good() would return
    holdvals -- a dictionary mapping keys from fitter.params_meaning to values... eg {"background":0, "dP_dE":1}
        This vector is anded with the vector calculated by the histogrammer
    """
    if isinstance(line_name, mass.LineFitter):
        fitter = line_name
        nominal_peak_energy = fitter.spect.nominal_peak_energy
    elif isinstance(line_name,str):
        fitter = mass.fitter_classes[line_name]()
        nominal_peak_energy = fitter.spect.nominal_peak_energy
    else:
        fitter = mass.GaussianFitter()
        nominal_peak_energy = float(line_name)
    if bin_edges is None:
        bin_edges = np.arange(nominal_peak_energy-dlo, nominal_peak_energy+dhi, binsize)
    if axis is None and plot:
        plt.figure()
        axis = plt.gca()

    bin_centers, counts = self.hist(bin_edges, attr, t0, tlast, category, g_func)

    if guess_params is None:
        guess_params = fitter.guess_starting_params(counts,bin_centers)
    hold = []
    for (k,v) in holdvals.items():
        i = fitter.param_meaning[k]
        guess_params[i]=v
        hold.append(i)
    params, covar = fitter.fit(counts, bin_centers,params=guess_params,axis=axis,label=label, ph_units=ph_units,plot=plot, hold=hold)
    if plot:
        axis.set_title(self.shortname()+", {}".format(line_name))

    return fitter

class WorstSpectra():
    """
    Create this, then run this.plot() to get a plot ordered by how well the spectra align.
    this.channels_sorted_by_chisq is a list of channel number, the last elements are the "worst spectra" in that they look the least
    like the others
    this.chisqdict maps channel number to chisq, chisq is a measure of how "bad" the spectra is (mean sum of squares between normalized spectrum and normalized mean spectrum)
    this.output() prints out the worst spectra
    INPUTS
    data -- A TESChannelGroup
    bin_edges, attr, category -- passed on to data.hists
    """
    def __init__(self,data, bin_edges = np.arange(2000,10000),attr="p_energy",category={}):
        self.data = data
        self.attr = attr
        self.category = category
        self.bin_edges = bin_edges
        self.doit()

    def doit(self):
        self.bin_centers, self.countsdict = self.data.hists(bin_edges=self.bin_edges, category=self.category, attr=self.attr)
        self.chisqdict = self.rank_hists_chisq(self.countsdict)
        self.channels_sorted_by_chisq = self.keys_sorted_by_value(self.chisqdict)

    def output(self):
        print(self.worstn(len(self.channels_sorted_by_chisq),"\n"))

    def worstn(self,n,seperator=", "):
        n = min(n, len(self.channels_sorted_by_chisq))
        s=""
        for i in range(n):
            ch = self.channels_sorted_by_chisq[-(i+1)]
            s+="%g:%0.3e"%(ch,self.chisqdict[ch])+seperator
        return s

    def bestn(self,n,seperator=", "):
        n = min(n, len(self.channels_sorted_by_chisq))
        s=""
        for i in range(n):
            ch = self.channels_sorted_by_chisq[i]
            s+="%g:%0.3e"%(ch,self.chisqdict[ch])+seperator
        return s

    def plot(self):
        plt.figure(figsize=(10,5))
        offsetsize = 10*np.mean(self.normalizespectrum(self.countsdict[self.channels_sorted_by_chisq[0]]))
        for i,ch in enumerate(self.channels_sorted_by_chisq):
            plt.plot(self.bin_centers, offsetsize*i+self.normalizespectrum(self.countsdict[ch]), drawstyle="steps-mid",label=ch)
            plt.annotate("%g"%ch,(self.bin_centers[-1],offsetsize*i),xycoords="data")
        plt.xlabel(self.attr)
        plt.ylabel("normalized counts per bin (sum=1), arb offset")
        plt.annotate("worst "+self.worstn(5),(0.1,0.9), xycoords="axes fraction")
        plt.annotate("best  "+self.bestn(5),(0.1,0.85), xycoords="axes fraction")
        if self.category == {}:
            plt.title(self.data.shortname())
        else:
            plt.title("{}\ncateogry = {}".format(self.data.shortname(),self.category))


    def keys_sorted_by_value(self,d): return np.array([k for k, v in sorted(d.iteritems(), key=lambda (k,v): (v,k))])

    def rank_hists_chisq(self,countsdict):
        """Return chisqdict which maps channel number to chisq value. keys_sorted_by_value(chisqdict) may be useful.
        countsdict -- a dictionary mapping channel number to np arrays containing counts per bin
        """
        sumspectrum = np.zeros_like(countsdict.values()[0], dtype="int")
        for k,v in countsdict.items():
            sumspectrum+=v
        #normalize spectra
        normalizedsumspectrum = self.normalizespectrum(sumspectrum)
        chisqdict =  {ch:self.chisqspectrumcompare(self.normalizespectrum(spect),normalizedsumspectrum) for ch,spect in self.countsdict.items()}
        return chisqdict

    def chisqspectrumcompare(self,normcounts1,normcounts2): return  np.sum((normcounts1-normcounts2)**2)/len(normcounts1)
    def normalizespectrum(self,counts): return counts/(1.0*counts.sum())

import fastdtw
class WorstSpectraDTW():
    """
    Create this, then run this.plot() to get a plot ordered by how well the spectra align.
    this.channels_sorted_by_chisq is a list of channel number, the last elements are the "worst spectra" in that they look the least
    like the others
    this.chisqdict maps channel number to chisq, chisq is a measure of how "bad" the spectra is (mean sum of squares between normalized spectrum and normalized mean spectrum)
    this.output() prints out the worst spectra
    INPUTS
    data -- A TESChannelGroup
    bin_edges, attr, category -- passed on to data.hists
    """
    def __init__(self,data, bin_edges = np.arange(2000,10000),attr="p_energy",category={}):
        self.data = data
        self.attr = attr
        self.category = category
        self.bin_edges = bin_edges
        self.doit()

    def doit(self):
        self.bin_centers, self.countsdict = self.data.hists(bin_edges=self.bin_edges, category=self.category, attr=self.attr)
        self.chisqdict = self.rank_hists_chisq(self.countsdict)
        self.channels_sorted_by_chisq = self.keys_sorted_by_value(self.chisqdict)

    def output(self):
        print(self.worstn(len(self.channels_sorted_by_chisq),"\n"))

    def worstn(self,n,seperator=", "):
        n = min(n, len(self.channels_sorted_by_chisq))
        s=""
        for i in range(n):
            ch = self.channels_sorted_by_chisq[-(i+1)]
            s+="%g:%0.3e"%(ch,self.chisqdict[ch])+seperator
        return s

    def bestn(self,n,seperator=", "):
        n = min(n, len(self.channels_sorted_by_chisq))
        s=""
        for i in range(n):
            ch = self.channels_sorted_by_chisq[i]
            s+="%g:%0.3e"%(ch,self.chisqdict[ch])+seperator
        return s

    def plot(self):
        plt.figure(figsize=(10,5))
        offsetsize = 10*np.mean(self.normalizespectrum(self.countsdict[self.channels_sorted_by_chisq[0]]))
        for i,ch in enumerate(self.channels_sorted_by_chisq):
            plt.plot(self.bin_centers, offsetsize*i+self.normalizespectrum(self.countsdict[ch]), drawstyle="steps-mid",label=ch)
            plt.annotate("%g"%ch,(self.bin_centers[-1],offsetsize*i),xycoords="data")
        plt.xlabel(self.attr)
        plt.ylabel("normalized counts per bin (sum=1), arb offset")
        plt.annotate("worst "+self.worstn(5),(0.1,0.9), xycoords="axes fraction")
        plt.annotate("best  "+self.bestn(5),(0.1,0.85), xycoords="axes fraction")


    def keys_sorted_by_value(self,d): return np.array([k for k, v in sorted(d.iteritems(), key=lambda (k,v): (v,k))])

    def rank_hists_chisq(self,countsdict):
        """Return chisqdict which maps channel number to chisq value. keys_sorted_by_value(chisqdict) may be useful.
        countsdict -- a dictionary mapping channel number to np arrays containing counts per bin
        """
        sumspectrum = np.zeros_like(countsdict.values()[0], dtype="int")
        for k,v in countsdict.items():
            sumspectrum+=v
        #normalize spectra
        normalizedsumspectrum = self.normalizespectrum(sumspectrum)
        chisqdict =  {ch:self.chisqspectrumcompare(self.normalizespectrum(spect),normalizedsumspectrum) for ch,spect in self.countsdict.items()}
        return chisqdict

    def chisqspectrumcompare(self,normcounts1,normcounts2): return  fastdtw.fastdtw(normcounts1,normcounts2)[0]
    def normalizespectrum(self,counts): return counts/(1.0*counts.sum())

# def normalizespectrum(counts): return counts/(1.0*counts.sum())
# def chisqspectrumcompare(normcounts1,normcounts2): return  np.sum((normcounts1-normcounts2)**2)/len(normcounts1)
# def keys_sorted_by_value(d): return np.array([k for k, v in sorted(d.iteritems(), key=lambda (k,v): (v,k))])
# def rank_hists_chisq(countsdict):
#     """Return chisqdict which maps channel number to chisq value. keys_sorted_by_value(chisqdict) may be useful.
#     countsdict -- a dictionary mapping channel number to np arrays containing counts per bin
#     """
#     sumspectrum = np.zeros_like(countsdict.values()[0], dtype="int")
#     for k,v in countsdict.items():
#         sumspectrum+=v
#     #normalize spectra
#     normalizedsumspectrum = normalizespectrum(sumspectrum)
#     chisqdict =  {ch:chisqspectrumcompare(normalizespectrum(spect),normalizedsumspectrum) for ch,spect in countsdict.items()}
#     return chisqdict
#
# def plot_ranked_hists(bin_centers, countsdict, chisqdict):
#     channels_sorted_by_chisq = keys_sorted_by_value(chisqdict)
#     plt.figure(figsize=(10,5))
#     offsetsize = np.mean(normalizespectrum(countsdict[channels_sorted_by_chisq[0]]))
#     for i,ch in enumerate(channels_sorted_by_chisq):
#         plt.plot(bin_centers, offsetsize*i+normalizespectrum(countsdict[ch]), drawstyle="steps-mid")

def samepeaks(bin_centers, countsdict, npeaks, refchannel, gaussian_fwhm):
    raise ValueError("Not done!")
    refcounts = countsdict[refchannel]
    peak_locations, peak_intensities = mass.find_local_maxima(refcounts, peak_intensities)



class DriftCheck():
    """
    Slice the data into time steps, then fit the line at each of those time steps.
    INPUTS
    tstep_approx -- approx time in seconds for each fit (gets corrected to use all data)
    line_name,attr,dlo,dhi,binsize,bin_edges,category,guess_params -- passed directly to linefit, see linefit docs
    """
    def __init__(self,ds,tstep_approx,line_name="MnKAlpha",attr="p_energy",dlo=50,dhi=50,binsize=1,bin_edges=None,category={},guess_params=None):
        self.ds = ds
        self.tstep_approx= tstep_approx
        self.line_name=line_name
        self.attr=attr
        self.dlo=dlo
        self.dhi=dhi
        self.binsize=binsize
        self.bin_edges=bin_edges
        self.category=category
        self.guess_params=guess_params
        self.doit()

    def doit(self):
        g=self.ds.good(**self.category)
        t0 = self.ds.p_timestamp[g][0]
        tend = self.ds.p_timestamp[g][-1]
        nsteps = np.floor((tend-t0)/self.tstep_approx)
        tstep = (tend-t0)/nsteps
        self.r = np.linspace(t0,tend,np.int(nsteps))
        self.fitters = []
        ts = []
        for i in xrange(len(self.r)-1):
            t1,t2 = self.r[i],self.r[i+1]
            ts.append(0.5*(t1+t2))
            fitter = self.ds.linefit(self.line_name, t1, t2, attr=self.attr, dlo=self.dlo,dhi=self.dhi,binsize=self.binsize,bin_edges=self.bin_edges,category=self.category,guess_params=self.guess_params,plot=False)
            self.fitters.append(fitter)

        self.peak_ph_values = [fitter.last_fit_params_dict["peak_ph"][0] for fitter in self.fitters]
        self.peak_ph_errs = [fitter.last_fit_params_dict["peak_ph"][1] for fitter in self.fitters]
        self.peak_ph_mean_err = np.mean(self.peak_ph_errs)
        self.peak_ph_std_mean = np.std(self.peak_ph_values)
        self.X = self.peak_ph_std_mean/self.peak_ph_mean_err
        return self.X, self.fitters, self.r

    def plot(self, plotattr=True, ylim_pm=None):
        """
        Plot the results of the drift check.
        plotattr -- pass false to only plot the fit results, and not a point for each pulse used
        ylim_pm -- pass a number to make the ylim bounds be defined by the nominal_peak_energy of the fitter plus or mius this number
        """
        plt.figure()
        tcenters = 0.5*(self.r[1:]+self.r[:-1])
        if plotattr:
            g = self.ds.good(**self.category)
            vals= getattr(self.ds,self.attr)
            for i in xrange(len(self.r)-1):
                t1,t2 = self.r[i],self.r[i+1]
                g2 = np.logical_and(self.ds.p_timestamp>=t1, self.ds.p_timestamp<t2)
                g3 = np.logical_and(g,g2)
                color = "k" if i%2==0 else '0.75'
                plt.plot(self.ds.p_timestamp[g3]-tcenters[0], vals[g3],".",color=color)

        peak_ph_values = [fitter.last_fit_params_dict["peak_ph"][0] for fitter in self.fitters]
        peak_ph_errs = [fitter.last_fit_params_dict["peak_ph"][1] for fitter in self.fitters]
        plt.errorbar(tcenters-tcenters[0], peak_ph_values, yerr=peak_ph_errs,fmt="r.",markersize=12,elinewidth=5)
        plt.xlabel("time (fisrt pulse = 0) (s)")
        plt.ylabel("peak ph with fit uncertainty (?)")
        plt.title(self.ds.shortname()+" %s has drift X = %0.2f"%(self.line_name,self.X))
        fitter = self.fitters[0]
        if ylim_pm is None:
            plt.ylim(fitter.last_fit_bins[0],fitter.last_fit_bins[-1])
        else:
            plt.ylim(fitter.spect.nominal_peak_energy-ylim_pm,fitter.spect.nominal_peak_energy+ylim_pm)

def ds_driftcheck(self,tstep_approx,line_name="MnKAlpha",attr="p_energy",dlo=50,dhi=50,binsize=1,bin_edges=None,category={},guess_params=None):
    """
    Slice the data into time steps, then fit the line at each of those time steps.
    returns a DriftCheck objects, which has a .plot() method and has all the fitters in it's .fitters variable
    INPUTS
    tstep_approx -- approx time in seconds for each fit (gets corrected to use all data)
    line_name,attr,dlo,dhi,binsize,bin_edges,category,guess_params -- passed directly to linefit, see linefit docs
    """
    return DriftCheck(self,tstep_approx,line_name,attr,dlo,dhi,binsize,bin_edges,category,guess_params)


def ds_rowtime(self):
    """
    Return the row time in seconds. The row time required to make a single sample for a simple row, and the frame time is equal to the row time times the number of rows.
    """
    nrow = self.pulse_records.datafile.number_of_rows
    rowtime = self.timebase/nrow
    return rowtime

def ds_cut_calculated(ds):
    """
    If you open a pope hdf5 file there will be no cuts applied, but there will be ranges for cuts. This function
    looks up those ranges, and uses them to do actual cuts.
    """
    ds.cuts.clear_cut("pretrigger_rms")
    ds.cuts.clear_cut("postpeak_deriv")
    ds.cuts.clear_cut("pretrigger_mean")
    ds.cuts.cut_parameter(ds.p_pretrig_rms, ds.hdf5_group["calculated_cuts"]["pretrig_rms"][:], 'pretrigger_rms')
    ds.cuts.cut_parameter(ds.p_postpeak_deriv, ds.hdf5_group["calculated_cuts"]["postpeak_deriv"][:], 'postpeak_deriv')

def ds_plot_ptmean_vs_time(ds,t0,tlast):
    plt.figure()
    plt.plot(ds.p_timestamp[ds.good()]-ds.p_timestamp[0],ds.p_pretrig_mean[ds.good()])
    plt.xlabel("time after first pulse (s)")
    plt.ylabel("p_pretrig_mean (arb)")



mass.TESGroup.plot_hist = plot_hist
mass.TESGroup.hist = data_hist
mass.TESGroup.hists = data_hists
mass.TESGroup.shortname = data_shortname
mass.TESGroup.linefit = ds_linefit

mass.MicrocalDataSet.hist = ds_hist
mass.MicrocalDataSet.plot_hist = plot_hist
mass.MicrocalDataSet.shortname = ds_shortname
mass.MicrocalDataSet.linefit = ds_linefit
mass.MicrocalDataSet.driftcheck = ds_driftcheck
mass.MicrocalDataSet.rowtime = ds_rowtime
mass.MicrocalDataSet.cut_calculated = ds_cut_calculated
mass.MicrocalDataSet.plot_ptmean_vs_time = ds_plot_ptmean_vs_time

def expand_cal_lines(s):
    """Return a list of line names, eg ["MnKAlpha","MnKBeta"]
    s -- a list containing line names and/or element symbols eg ["Mn","TiKAlpha"]
    """
    out = []
    for symbol in s:
        if symbol == "": continue
        if len(symbol) == 2:
            out.append(symbol+"KBeta")
            out.append(symbol+"KAlpha")
        elif not symbol in out:
            out.append(symbol)
    return out

class Side():
    """
    name -- name of side, eg "A"
    tlo, thi -- in seconds, times after last external trigger beteween these are included with this side
    elements -- list of strings of lines from elements on that side ["MnKAlpha","MnKbeta","TiKAlpha","TiKBeta"]
    """
    def __init__(self,name,lines,tlo,thi):

        self.name = name
        self.tlo = tlo
        self.thi = thi
        self.lines = lines

    def __repr__(self):
        return "Side %s: %s %0.2f-%0.2fs"%(self.name, self.lines,self.tlo,self.thi)

    def good(self,ds):
        vals = ds.rows_after_last_external_trigger[:]*ds.rowtime()
        return np.logical_and(vals>=self.tlo,vals<self.thi)

class Sides():
    """
    Class for organizing plots related to samples on different sides of the XES reflection switcher.
    sides -- something like [Side("A",["MnKAlpha","MnKBeta"],0,5.6),Side("B",["MnKAlpha","MnKBeta"],5.8,12)]
    datasource -- data source, can be a MicrocalDataSet or TESGroup
    """
    def __init__(self,sides,datasource):
        self.sides = sides
        self.datasource = datasource
        self.fitters = None

    def sidesfits(self):
        """
        For each side, for each line, fit a spectrum. Return a dict mapping names to spectra.
        Dict keys look like 'A,MnKAlpha' or 'B,TiKBeta'
        After running this you can run `sidesfitsplots` to get plots.
        """
        self.fitters = OrderedDict()
        for side in self.sides:
            for linename in side.lines:
                category = {"side":side.name}
                self.fitters[side.name+","+linename] = self.datasource.linefit(linename,plot=False,category=category)
        return self.fitters

    def sidesfitsplots(self):
        if self.fitters is None:
            raise ValueError("self.fitters is None, try running sidesfits first")
        for (k,v) in self.fitters.items():
            plt.figure()
            v.plot(ph_units="eV",label="full")
            plt.title(self.datasource.shortname()+", "+k)

    def manifest(self, paramname="peak_ph", write_dir = None):
        s=""
        for (k,v) in self.fitters.items():
            val, uncertainty = v.last_fit_params_dict[paramname]
            s+="%s %s %f +/- %f\n"%(paramname,k,val,uncertainty)
        if write_dir is not None:
            write_name = os.path.join(write_dir, self.datasource.shortname()+"+manifest_%s.txt"%paramname)
            with open(write_name,"w") as f:
                f.write(s)
        return s

    def alllines(self):
        lines = []
        for side in self.sides:
            for linename in side.lines:
                if not linename in lines:
                    lines.append(linename)
        return lines

    def sidesfullplot(self,bin_edges=np.arange(10000),attr="p_energy",yscale="log",annotate_extras = ["FeKAlpha","FeKBeta","CuKAlpha","CuKBeta","ZnKAlpha","CrKAlpha","CrKBeta"]):
        plt.figure(figsize=(10,5))
        for side in self.sides:
            category = {"side":side.name}
            x,counts = self.datasource.hist(bin_edges=bin_edges,attr=attr,category=category)
            plt.plot(x,counts,label="Side "+side.name)
        plt.title(self.datasource.shortname())
        plt.xlabel(attr)
        plt.ylabel("counts per %0.f unit bin"%(x[1]-x[0]))
        plt.yscale(yscale)
        annotate_lines(plt.gca(),self.alllines(),annotate_extras)
        plt.legend(loc="best")

    def sidesindividualplots(self,bin_edges=np.arange(10000),attr="p_energy",yscale="log",annotate_extras = ["FeKAlpha","FeKBeta","CuKAlpha","CuKBeta","ZnKAlpha","CrKAlpha","CrKBeta"]):
        for side in self.sides:
            plt.figure(figsize=(10,5))
            category = {"side":side.name}
            bin_centers,counts = self.datasource.hist(bin_edges=bin_edges,attr=attr,category=category)
            side.counts_sum = counts.sum()
            side.spectral_mean = (bin_centers*counts).sum()/side.counts_sum
            plt.plot(bin_centers,counts,label="Side "+side.name)
            plt.title(self.datasource.shortname()+": Side "+side.name+"\nSpectral Center %0.2f eV"%side.spectral_mean)
            plt.xlabel(attr)
            plt.ylabel("counts per %0.f unit bin"%(bin_centers[1]-bin_centers[0]))
            plt.yscale(yscale)
            annotate_lines(plt.gca(),self.alllines(),annotate_extras)

class RepeatedLinePlotter():
    """
    Class to make a plot of the peah ph and uncertainty vs channel of the same line on different sides.
    INPUTS
    dfitters -- a dictionary mapping channel number to the output of Side.sidesfits()
    keys -- something like ["A,MnKAlpha","B,MnKAlpha"]
    """
    def __init__(self,dsfitters, keys):
        self.chs = [k for k in dsfitters.keys()]
        self.dout = OrderedDict()
        for key in keys:
            # key is something like "A,MnKAlpha"
            phs = [dsfitters[ch][key].last_fit_params_dict["peak_ph"][0] for ch in self.chs]
            phs_err = [dsfitters[ch][key].last_fit_params_dict["peak_ph"][1] for ch in self.chs]
            self.dout[key]=(phs,phs_err)

    def plot(self):
        plt.figure()
        for k,v in self.dout.items():
            phs,phs_err=v
            plt.errorbar(self.chs, phs, phs_err, label = k,fmt=".",markersize=12,lw=2)
        med = np.median(np.array(phs)[~np.isnan(phs)])
        plt.ylim(med-0.5,med+.5)
        plt.xlabel("channel number (arb)")
        plt.ylabel("peak ph (eV)")
        plt.legend(loc="best")




class PredictedVsAchieved():
    def __init__(self, data, calibration, fitters):
        """
        Call this.plot() after initialization. For each ds in data calculated the predicted energy resolution at the average pulse (using the
        calibration to get the average pulse energy, does no nonlinearity correction). Then looks up the
        achieved energy resolution in the fitters. And plots it all.
        data -- a TESChannelGroup
        calibration -- a calibraiton name, eg "p_filt_value_tdc"
        fitters -- a dictionary mapping channel number to a fitter at line
        """
        self.data = data
        self.calibration = calibration
        self.fitters = fitters

    @property
    def vdvs(self):
        preds = []
        for ds in self.data:
            d = ds.filter.predicted_v_over_dv
            if d.has_key("filt_noconst"):
                preds.append(ds.filter.predicted_v_over_dv["filt_noconst"])
            elif d.has_key("noconst"):
                preds.append(ds.filter.predicted_v_over_dv["noconst"])

        return np.array(preds)
    @property
    def average_pulse_energies(self):
        energies = []
        for ds in self.data:
            calibration = ds.calibration[self.calibration]
            energy = calibration(np.amax(ds.average_pulse)-np.amin(ds.average_pulse))
            energies.append(energy)
        return np.array(energies)
    @property
    def channels(self):
        return [ds.channum for ds in self.data]
    @property
    def predicted_at_average_pulse(self):
        return self.average_pulse_energies/self.vdvs
    @property
    def achieved(self):
        return np.array([self.fitters[ds.channum].last_fit_params_dict["resolution"][0] for ds in self.data])
    @property
    def fitter_line_name(self):
        fitter = self.fitters.values()[0]
        return fitter.spect.name
    def plot(self):
        plt.figure()
        predicted = self.predicted_at_average_pulse
        achieved = self.achieved
        med_pred = np.median(predicted)
        med_ach = np.median(achieved)
        xlim_max = min(2*med_pred, np.amax(predicted)*1.05)
        ylim_max = min(2*med_ach, np.amax(achieved)*1.05)
        lim_max = max(xlim_max, ylim_max)
        plt.plot(predicted, achieved,"o")
        plt.plot([0, lim_max], [0, lim_max],"k")
        plt.xlim(0,lim_max)
        plt.ylim(0,lim_max)
        plt.xlabel("predicted res at avg pulse (eV)")
        plt.ylabel("achieved at %s"%self.fitter_line_name)
        plt.title("median predicted %0.2f, median achieved %0.2f\n%s"%(med_pred, med_ach,self.data.shortname()))







class TestPlotAndHistMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data = mass.TESGroupHDF5("/Users/oneilg/Documents/molecular movies/Horton Wiring/horton_2017_07/20171006_B.ljh_pope.hdf5")
        self.data.set_chan_good(self.data.why_chan_bad.keys())
        for ds in self.data:
            ds_cut_calculated(ds)
            print("Chan %s, %g bad of %g"%(ds.channum, ds.bad().sum(), ds.nPulses))
        self.ds = self.data.first_good_dataset

        self.bin_edges  = np.arange(0,10000,1)

    def test_ds_hist(self):
        x,y = self.ds.hist(self.bin_edges)
        self.assertEqual(x[0], 0.5*(self.bin_edges[1]+self.bin_edges[0]))
        self.assertEqual(np.argmax(y),4511)

    def test_data_hists(self):
        x,countsdict = self.data.hists(self.bin_edges)
        self.assertTrue(all(countsdict[self.ds.channum]==self.ds.hist(self.bin_edges)[1]))
        self.assertTrue(len(x)==len(self.bin_edges)-1)
        len(countsdict.keys())==len([ds for ds in self.data])

    def test_plots(self):
        self.ds.plot_hist(self.bin_edges, label_lines = ["MnKAlpha","MnKBeta"])
        self.data.plot_hist(self.bin_edges, label_lines = ["MnKAlpha","MnKBeta"])

    def test_linefit(self):
        fitter = self.ds.linefit("MnKAlpha")
        self.assertTrue(fitter.success)

    def test_linefit_pass_fitter(self):
        fitter = self.ds.linefit(mass.MnKAlphaFitter(), bin_edges = np.arange(5850,5950), attr="p_energy")
        self.assertTrue(fitter.success)

    def test_rank_hists_chisq(self):
        ws=WorstSpectra(self.data)
        ws.output()
        ws.plot()

    def test_ds_driftcheck(self):
        driftchecker = self.ds.driftcheck(600)
        driftchcker.plot()

if __name__ == "__main__":
    unittest.findTestCases("__main__").debug()
    unittest.main()
    plt.show()
