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



dirname = "/data/20181017/20181017_Mn_pulses/20181017_150616_chan275.ljh"
ndirname = "/data/20181017/20181017_Mn_noise/20181017_150522_chan25.noi"
basename,_ = mass.ljh_util.ljh_basename_channum(dirname)

# emulate side with one side
side = SideInfo()
side.name="A"
side.elements=["Mn"]
side.lines = ["MnKAlpha","MnKBeta"]
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
roughcal_lines=["MnKAlpha","MnKBeta"],
forceCalibrationNew=False,
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
p = PredictedVsAchieved(switcher.data, "p_filt_value_tdc", {ds.channum:dsfitters[ds.channum]["A,MnKAlpha"] for ds in switcher.data})
p.plot()

ds = switcher.data.first_good_dataset
plt.figure()
if ds.has_external_trigger_data:
    timestamps = ds.rows_after_last_external_trigger[:]*ds.rowtime()
else:
    timestamps = (ds.p_timestamp[:]-sides[0].t0)%sides[0].period
plt.hist2d(timestamps[ds.good()],ds.p_energy[ds.good()],bins=[200,np.arange(4000,6000,0.5)],normed=True)
plt.title(ds.shortname())
plt.xlabel("time after last external trigger (s)")
plt.ylabel("energy (eV)")
plt.ylim(5860,5940)

plt.figure()
med = np.median(ds.p_pretrig_mean[ds.good()])
plt.hist2d(timestamps[ds.good()],ds.p_pretrig_mean[ds.good()],bins=[200,np.linspace(med-100,med+100,500)],normed=True)
plt.title(ds.shortname())
plt.xlabel("time after last external trigger (s)")
plt.ylabel("p_pretrig_mean (arb)")


switcher.save_and_close_all_plots("resolution_all")

with h5py.File(os.path.join(switcher.figdirname,switcher.data.shortname()+"_hists.h5"), "w") as h5:
    for side in switcher.sides:
        bin_centers, counts = switcher.data.hist(np.arange(0,10000,0.1), category={"side":side.name})
        h5[side.name+"counts"] = counts
        h5[side.name+"bin_centers"] = bin_centers
    h5["ljh_filename"] = switcher.data.shortname()

with h5py.File(os.path.join(switcher.figdirname,switcher.data.shortname()+"_hists_by_channel.h5"), "w") as h5:
    for ds in switcher.data:
        g = h5.create_group("%i"%ds.channum)
        for side in switcher.sides:
            bin_centers, counts = ds.hist(np.arange(0,10000,0.1), category={"side":side.name})
            g[side.name+"counts"] = counts
            g[side.name+"bin_centers"] = bin_centers
        fitters = dsfitters[ds.channum]
        for (side_and_line,fitter) in fitters.items():
            gg = g.create_group(side_and_line)
            for param in sorted(fitter.last_fit_params_dict.keys()):
                gg[param]=fitter.last_fit_params_dict[param][0]
                gg[param+"_uncertainty"]=fitter.last_fit_params_dict[param][1]
        gg = g.create_group("predicted_vs_achieved")
        gg["line_name"]=p.fitter_line_name
        i=p.channels.index(ds.channum)
        gg["predicted"]=p.predicted_at_average_pulse[i]
        gg["achieved"]=p.achieved[i]
        gg["vdv"]=p.vdvs[i]
        g["calibration/ph"]=ds.last_used_calibration._ph
        g["calibration/energies"]=ds.last_used_calibration._energies
        g["calibration/line_names"]=ds.last_used_calibration._names


epm = 50 # energy to add and subtract around each line center
switcher.line_zoom_plots({side.name:side.elements for side in sides}, alllines, epm)

sidesdata = switcher.sidesdata

repeatedlinesd = OrderedDict()
repeatedlinesd_res = OrderedDict()
for repeated_line_spec in switcher.repeated_lines_specs:
    positions = []
    resolutions = []
    for line in repeated_line_spec:
        bareline = line.split(",")[1]
        fitter = sidesdata.fitters[line]
        v=ufloat(fitter.last_fit_params_dict["peak_ph"][0],fitter.last_fit_params_dict["peak_ph"][1])
        r=ufloat(fitter.last_fit_params_dict["resolution"][0],fitter.last_fit_params_dict["resolution"][1])
        positions.append(v)
        resolutions.append(r)
        if line!=repeated_line_spec[0]:
            print("{}-{}={}".format(line,repeated_line_spec[0], v-positions[0]))
    repeatedlinesd[bareline]=np.array(positions)
    repeatedlinesd_res[bareline]=np.array(resolutions)

side_counts_sums = [side.counts_sum for side in sidesdict.values()]
side_durations = [side.thi-side.tlo for side in sidesdict.values()]
side_cps_in_arbs = np.array(side_counts_sums)/np.array(side_durations)
side_cps_ratios = side_cps_in_arbs/float(np.amin(side_cps_in_arbs))
for i,side in enumerate(sidesdict.values()):
    side.cps_ratio = side_cps_ratios[i]


plt.figure()
for ((k,v), repeated_line_spec) in zip(repeatedlinesd.items(), switcher.repeated_lines_specs):
    if len(repeated_line_spec)<2: continue
    specnames = [spec.split(",")[0] for spec in repeated_line_spec]
    x = [sidesdict[specname].spectral_mean*sidesdict[specname].cps_ratio for specname in specnames]
    plt.errorbar(x,unumpy.nominal_values(v-v[0]),unumpy.std_devs(v-v[0]),label=k,fmt="-o",
    capsize=10,capthick=2,lw=2)
plt.xlabel("spectral center (eV) * cps ratio (arb)")
plt.ylabel("line position first occurance of line (eV)")
plt.legend()
plt.grid(True)
plt.title(switcher.data.shortname())
plt.show()

plt.figure()
for ((k,v), repeated_line_spec) in zip(repeatedlinesd.items(), switcher.repeated_lines_specs):
    if len(repeated_line_spec)<2: continue
    specnames = [spec.split(",")[0] for spec in repeated_line_spec]
    x = [sidesdict[specname].spectral_mean for specname in specnames]
    plt.errorbar(x,unumpy.nominal_values(v-v[0]),unumpy.std_devs(v-v[0]),label=k,fmt="-o",
    capsize=10,capthick=2,lw=2)
plt.xlabel("spectral center (eV)")
plt.ylabel("line position first occurance of line (eV)")
plt.legend()
plt.grid(True)
plt.title(switcher.data.shortname())
plt.show()


tstep = 1
tstarts=[]
tends=[]
tstart_sides=[]
for side in sides:
    r=np.arange(side.end_transition_s,side.end_transition_s+side.dwell_s,tstep)
    for x in r:
        tstarts.append(x)
        tends.append(x+tstep)
        tstart_sides.append(side)
def make_my_g_func(tstart,tend):
    def my_g_func(ds):
        trel = (ds.p_timestamp[:]-side.t0)%side.period
        g = np.logical_and(trel>tstart,trel<tend)
        return g
    return my_g_func


plotdict = {}
fitterdict = {}
for i,line_name in enumerate(set([line for side in sidesdict.values() for line in side.lines])):
    print("line number %g"%i)
    max_err = 0.5
    peak_phs = []
    peak_ph_errs = []
    tstart_plot = []
    fitters = []
    for tstart,tend,side in zip(tstarts,tends,tstart_sides):
        # elo,ehi = mass.STANDARD_FEATURES[line_name]-20,mass.STANDARD_FEATURES[line_name]+20
        # bin_centers, counts = ds.hist(np.arange(elo,ehi,1),g_func=make_my_g_func(tstart,tend))
        if not line_name in side.lines:
            continue
        fitter = switcher.data.linefit(line_name,g_func=make_my_g_func(tstart,tend),plot=False)
        err = fitter.last_fit_params_dict["peak_ph"][1]
        if err < max_err:
            peak_phs.append(fitter.last_fit_params_dict["peak_ph"][0])
            peak_ph_errs.append(fitter.last_fit_params_dict["peak_ph"][1])
            tstart_plot.append(tstart)
            fitters.append(fitter)
    plotdict[line_name] = (tstart_plot, peak_phs, peak_ph_errs)
    fitterdict[line_name] = fitters
print("done fits")
for line_type in ["KAlpha","KBeta"]:
    plt.figure()
    for line_name, v in plotdict.items():
        tstart_plot, peak_phs, peak_ph_errs = v
        if line_name.endswith(line_type) and len(peak_phs)>2:
            plt.errorbar(tstart_plot,np.array(peak_phs)-mass.STANDARD_FEATURES[line_name],yerr=peak_ph_errs,fmt=".",capthick=2,lw=2,label=line_name)
    plt.vlines([side.begin_transition_s for side in sides],-1,1)
    plt.vlines([side.end_transition_s for side in sides],-1,1,color="grey")
    plt.xlabel("time since start of motor rotation (s)")
    plt.ylabel("line position (eV with arb offset)")
    plt.legend(loc="best")
    plt.xlim(0,side.period)
    plt.ylim(-1,1)
    plt.title(switcher.data.shortname()+"\nblack vertical=start of motor rotation\ngrey vertical=start of data used for a given side")
print("made plots")

switcher.save_and_close_all_plots("repeated_lines_shift")
print("saved plots")
