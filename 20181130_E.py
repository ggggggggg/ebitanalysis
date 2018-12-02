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


class SideInfo():
    t0 = None
    period = None
    # fields from pickle
    # begin_transition_s
    # dwell_s
    # end_transition_s
    # masetpoint
    # move_duration_s
    # move_to_amount
    # position_pulse
    # settle_s
    # uxman_settle_s
    def __repr__(self):
        s="SideInfo with:\n"
        s+=pprint.pformat(self.__dict__)
        return s
    def good(self,ds):
        vals = (ds.p_timestamp[:]-self.t0)%self.period
        if not hasattr(self,"tlo"):
            self.tlo = self.end_transition_s
        if not hasattr(self,"thi"):
            self.thi = self.end_transition_s+self.dwell_s
        return np.logical_and(vals>=self.tlo,vals<self.thi)



dirname = "/data/20181130_E/20181130_E_chan1.ljh"
ndirname = "/data/20181130_B/20181130_B_chan1.ljh"
basename,_ = mass.ljh_util.ljh_basename_channum(dirname)

# emulate side with one side
side = SideInfo()
side.name="A"
side.elements=["Al","Mg"]
side.lines = ["AlKAlpha","MgKAlpha"]
side.tlo = 0
side.thi = np.inf
side.t0 = time.time()/2
side.period = side.t0*4
side.begin_transition_s = 1
side.end_transition_s = 1
side.dwell_s = 1
side.masetpoint = 1
side.move_duration_s = 1
side.move_to_amount = 1
side.position_pulse = 1
side.settle_s = 1
side.uxman_settle_s = 1

sidesdict= {"A":side}
sides=sidesdict.values()


alllines = set([line for side in sidesdict.values() for line in side.lines])
alllines = sorted(alllines)
repeated_line_specs = [[",".join([side.name,line]) for side in sides if line in side.lines] for line in alllines]
def getuniquelines(element,maxenergy=10000.0):
    positions = set([v for (k,v) in mass.STANDARD_FEATURES.items() if k.startswith(element) and v<maxenergy])
    names = [sorted([k for (k,v) in mass.STANDARD_FEATURES.items() if v==p],key=len)[-1] for p in sorted(positions)]
    return names


plt.close("all")
plt.interactive(True)
switcher = MassSidesAnalysis(dirname,ndirname,
roughcal_lines=["AlKAlpha","MgKAlpha"],
forceCalibrationNew=True,
calibrationCategory={"side":"A"},
sides=sides,
maxchans = 240, cps_time_binsize_s=0.1,
repeated_lines_specs=repeated_line_specs,
delete_hdf5_file_before_analysis=True,
figdirname_extra="output"
#tlast=1525309886.159698,#15897+sidesdict["A"].t0,
)
switcher.doit()

dsfitters = switcher.dsfitters_roughcal
plt.close("all")
p = PredictedVsAchieved(switcher.data, "p_filt_value_tdc", {ds.channum:dsfitters[ds.channum]["A,AlKAlpha"] for ds in switcher.data})
p.plot()

data=switcher.data
data.calibrate("p_filt_value_phc",["MgKAlpha","AlKAlpha","OKAlpha","ZnLAlpha"],diagnose=False,forceNew=True,fit_range_ev=50)
data.convert_to_energy("p_filt_value_phc")
data.plot_hist(np.arange(2200),label_lines=["SiKAlpha","AlKAlpha","MgKAlpha","FeLAlpha","OKAlpha","NiLAlpha","CoLAlpha","CuLAlpha","ZnLAlpha","ZnLBeta","AlKBeta","MgKBeta","CKAlpha","NKAlpha"])
plt.yscale("log")
