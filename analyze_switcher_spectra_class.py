import mass
import numpy as np
import pylab as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import h5py
import sys
import devel
from devel import Side, Sides, RepeatedLinePlotter, WorstSpectra
import pickle
from collections import OrderedDict
# from relative_arrival_cut import relative_arrival_cut_tdm_preceeding_row

class MassSidesAnalysis():
    def __init__(self, pulse_file, noise_file, roughcal_lines, sides,
    maxchans = 4,do_tdc=True, nextra=3,fit_range_ev=200,forceNew=False,
    n_datasets_for_plots=4, drift_line_names=None,cps_time_binsize_s=0.1,
    repeated_lines_specs=[["A,MnKBeta","B,MnKBeta"],["A,MnKAlpha","B,MnKAlpha"]],
    delete_hdf5_file_before_analysis=False, forceCalibrationNew=False,
    calibrationCategory={},t0=0,tlast=1e20, crosstalkVetoWindow=None,
    figdirname_extra=""):
        self.pulse_file = pulse_file
        self.noise_file = noise_file
        self.roughcal_lines = roughcal_lines
        self.sides = sides
        self.maxchans = maxchans
        self.do_tdc = do_tdc
        self.nextra = nextra
        self.fit_range_ev = fit_range_ev
        self.forceNew = forceNew
        self.forceCalibrationNew = forceCalibrationNew
        self.calibrationCategory = calibrationCategory
        self.n_datasets_for_plots = n_datasets_for_plots
        self.drift_line_names = drift_line_names
        self.cps_time_binsize_s = cps_time_binsize_s
        self.repeated_lines_specs=repeated_lines_specs
        self.delete_hdf5_file_before_analysis=delete_hdf5_file_before_analysis
        self.t0=t0
        self.tlast=tlast
        self.replace_default_Nones()
        self.crosstalkVetoWindow=crosstalkVetoWindow # should be like (-0.01,0.01) in s
        self.figdirname_extra=figdirname_extra


    def crosstalkVeto(self):
        self.data.register_boolean_cut_fields("crosstalk")
        for ds in self.data: ds.cuts.clear_cut("crosstalk")
        if self.crosstalkVetoWindow is not None:
            print("RUNNING XTALK CUT\n"*10)
            cut_perp_preceeds_s, cut_perp_follows_s = self.crosstalkVetoWindow
            relative_arrival_cut_tdm_preceeding_row(self.data,
                cut_perp_preceeds_s,cut_perp_follows_s)
        else:
            print("SKIPPING XTALK CUT\n"*10)

    def replace_default_Nones(self):
        if self.drift_line_names is None:
            self.drift_line_names = self.sides[0].lines

    def doit(self):
        self.do_mass_analysis()
        self.do_post_mass_analysis()
        self.group_plots()
        self.repeated_line_plots()
        self.dataset_plots()

    def do_post_mass_analysis(self):
        self.dsfitters_roughcal = OrderedDict()
        for ds in self.data:
            sidesds = Sides(self.sides,ds)
            sidesds.sidesfits()
            self.dsfitters_roughcal[ds.channum] = sidesds.fitters
        with open(os.path.join(self.figdirname,self.data.shortname()+"_channel_fitters.pkl"),"w") as f:
            pickle.dump(self.dsfitters_roughcal,f)

    def do_mass_analysis(self):
        self.data = self.open_data_through_filter_and_external_timing()
        self.do_sidescut()
        self.data.drift_correct(forceNew=self.forceNew)
        self.data.phase_correct(forceNew=self.forceNew)
        self.time_drift_correct_and_roughcalibration_and_convert_to_energy()


    def open_data_through_filter_and_external_timing(self):
        if os.path.splitext(self.pulse_file)[-1] == ".hdf5":
            shortname = self.pulse_file[:10]
            if not os.path.exists(self.pulse_file):
                print(self.pulse_file+" does not exist, quitting")
                sys.exit()
            ishdf5=True
        elif len(self.noise_file)>0:
            shortname = os.path.split(mass.ljh_util.ljh_basename_channum(self.pulse_file)[0])[-1]
            ishdf5=False
        else:
            raise ValueError("invalid pulse_file spec")
        self.figdirname=shortname+self.figdirname_extra
        if not os.path.isdir(self.figdirname):
            os.mkdir(self.figdirname)

        self.roughcal_lines = devel.expand_cal_lines(self.roughcal_lines)

        if ishdf5:
            self.data = mass.TESGroupHDF5(self.pulse_file)
            self.data.set_chan_good(data.why_chan_bad.keys())
            self.data.updater = mass.core.utilities.NullUpdater
            for ds in data:
                devel.ds_cut_calculated(ds)
                print("Chan %s, %g bad of %g"%(ds.channum, ds.bad().sum(), ds.nPulses))
        else:
            dir_p = self.pulse_file
            dir_n = self.noise_file
            available_chans = mass.ljh_util.ljh_get_channels_both(dir_p, dir_n)
            chan_nums = available_chans[:min(self.maxchans, len(available_chans))]
            print chan_nums
            pulse_files = mass.ljh_util.ljh_chan_names(dir_p, chan_nums)
            noise_files = mass.ljh_util.ljh_chan_names(dir_n, chan_nums)
            # pulse_files = [p+".ljh" for p in pulse_files]
            # noise_files = [p+".ljh" for p in noise_files]
            if self.delete_hdf5_file_before_analysis:
                hdf5filename = mass.core.channel_group._generate_hdf5_filename(pulse_files[0])
                if os.path.isfile(hdf5filename): os.remove(hdf5filename)
            self.data = mass.TESGroup(pulse_files, noise_files)
            self.data.set_chan_good(self.data.why_chan_bad.keys())
            self.data.updater = mass.core.utilities.NullUpdater
            self.data.summarize_data(forceNew=self.forceNew)
            self.data.auto_cuts()
            for ds in self.data:
                ds.cuts.cut("timestamp_sec",
                ~np.logical_and(ds.p_timestamp[:]>self.t0, ds.p_timestamp[:]<self.tlast))
            self.crosstalkVeto()
            self.data.avg_pulses_auto_masks(forceNew=self.forceNew)
            # for ds in data: ds._use_new_filters=False
            self.data.compute_filters(forceNew=self.forceNew)
            self.data.filter_data(forceNew=self.forceNew)

        try:
            self.data.calc_external_trigger_timing(after_last=True)
            self.has_external_trigger_data = True
        except IOError:
            print("no external trigger data found!\nself.has_external_trigger_data=False")
            self.has_external_trigger_data = False
        for ds in self.data:
            ds.has_external_trigger_data = self.has_external_trigger_data
        return self.data

    def time_drift_correct_and_roughcalibration_and_convert_to_energy(self):
        if not self.do_tdc:
            self.data.calibrate("p_filt_value_phc",self.roughcal_lines,
            size_related_to_energy_resolution=40,forceNew=self.forceNew or self.forceCalibrationNew,
            nextra=self.nextra,fit_range_ev=self.fit_range_ev, category=self.calibrationCategory)
            self.roughcal_name = "p_filt_value_phc"
            # self.data.convert_to_energy("p_filt_value_phc","p_filt_value_phc")
        else:
            self.data.time_drift_correct()
            self.data.calibrate("p_filt_value_tdc",self.roughcal_lines,
            size_related_to_energy_resolution=40,forceNew=self.forceNew or self.forceCalibrationNew,
            nextra=self.nextra,fit_range_ev=self.fit_range_ev, category=self.calibrationCategory)
            self.roughcal_name="p_filt_value_tdc"
            # self.data.convert_to_energy("p_filt_value_tdc","p_filt_value_tdc")
    def do_sidescut(self):
        self.data.register_categorical_cut_field("side",[side.name for side in self.sides])
        for ds in self.data:
            self.sidescut_ds(ds)
    def sidescut_ds(self,ds):
        nrow = ds.pulse_records.datafile.number_of_rows
        rowtime = ds.timebase/nrow
        cutdict = {side.name:side.good(ds) for side in self.sides}
        ds.cuts.cut_categorical("side",cutdict)
        if self.has_external_trigger_data:
            MAXINT=9223372036854775807
            ds.cuts.cut("timestamp_sec",ds.rows_after_last_external_trigger[:]==MAXINT)

    def group_plots(self, name_ext=""):
        sidesdata = Sides(self.sides,self.data)
        sidesdata.sidesfits()
        sidesdata.sidesfitsplots()
        manifest = sidesdata.manifest(write_dir=self.figdirname)
        print(manifest)
        sidesdata.sidesfullplot()
        sidesdata.sidesindividualplots()
        self.sidesdata = sidesdata
        with open(os.path.join(self.figdirname,self.figdirname+"_coadded_fitters%s.pkl"%name_ext),"w") as f:
            pickle.dump(sidesdata.fitters,f)

        ws=WorstSpectra(self.data)
        ws.plot()
        self.save_and_close_all_plots("coadded_figure")

    def save_and_close_all_plots(self,prefix):
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig(os.path.join(self.figdirname,self.figdirname+"_%s%d.png" %(prefix,i) ))
            plt.close(i)

    def repeated_line_plots(self):
        for repeated_lines_spec in self.repeated_lines_specs:
            repeatedlineplotter = RepeatedLinePlotter(self.dsfitters_roughcal, repeated_lines_spec)
            repeatedlineplotter.plot()
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig(os.path.join(self.figdirname,self.figdirname+"repeated_lines%d.png" % i))
            plt.close(i)

    def dataset_plots(self):
        imax = min(self.n_datasets_for_plots,len([ds for ds in self.data]))
        for (i,ds) in enumerate(self.data):
            self.single_ds_plots(ds)
            if i >= imax: break

    def single_ds_plots(self,ds):
        ds.plot_traces(np.where(ds.good())[0][:10])
        plt.title(ds.shortname())
        plt.savefig(os.path.join(self.figdirname,'ch%g_traces.png' % ds.channum))
        plt.close()
        ds.plot_ptmean_vs_time(self.t0,self.tlast)
        plt.title(ds.shortname())
        plt.savefig(os.path.join(self.figdirname,'ch%g_ptmean.png' % ds.channum))
        plt.close()
        ds.last_used_calibration.diagnose()
        plt.title(ds.shortname())
        plt.savefig(os.path.join(self.figdirname,'ch%g_cal_diagnose.png' % ds.channum))
        plt.close()
        # sidesplot(ds,self.sides)
        # plt.savefig(os.path.join(self.figdirname,'ch%g_sides_p_energy.png' % ds.channum))
        # plt.close()
        ds.noise_records.plot_power_spectrum()
        plt.savefig(os.path.join(self.figdirname,'ch%g_sides_power_spectrum.png' % ds.channum))
        plt.close()
        if ds.last_used_calibration is None:
            assert len(ds.calibration)==1
            ds.last_used_calibration = ds.calibration.values()[0]
        ds.last_used_calibration.plot()
        plt.savefig(os.path.join(self.figdirname,'ch%g_last_used_calibration.png' % ds.channum))
        plt.title(ds.shortname())
        plt.close()
        # sidesplot_hist(ds,self.sides,self.cps_time_binsize_s)
        # plt.savefig(os.path.join(self.figdirname,'ch%g_sides_countrate.png' % ds.channum))
        # plt.close()
        for drift_line_name in self.drift_line_names:
            driftchecker = ds.driftcheck(150,line_name=drift_line_name)
            driftchecker.plot(ylim_pm=10)
            plt.savefig(os.path.join(self.figdirname,'ch%g_s%s_driftcheck.png' % (ds.channum, drift_line_name)))
            plt.close()

    def sidecal_ds(self,ds):
        sidecal = mass.EnergyCalibration(approximate=True)
        roughcal = ds.calibration[self.roughcal_name]
        for line in self.sidecal_lines:
            fitter = self.dsfitters_roughcal[ds.channum][line]
            peak_ph_E, peak_ph_err_E = fitter.last_fit_params_dict["peak_ph"] # in energy units of 'p_filt_value_tdc' calibration
            peak_ph_filt_value = roughcal.energy2ph(peak_ph_E)
            peak_ph_err_filt_value = peak_ph_err_E/roughcal.energy2dedph(peak_ph_E)
            peak_energy_reference = fitter.spect.peak_energy
            sidecal.add_cal_point(peak_ph_filt_value, peak_energy_reference, name=line,
                pht_error=peak_ph_err_filt_value, e_error=None)
        ds.calibration[self.sidecal_name]=sidecal

    def sidecal(self,sidecal_lines):
        self.sidecal_lines = sidecal_lines
        self.sidecal_name = "sidecal_"+self.roughcal_name
        for ds in self.data:
            self.sidecal_ds(ds)
        self.data.convert_to_energy("p_filt_value_tdc",self.sidecal_name)

    def do_post_sidecal_analysis(self):
        self.dsfitters_sidecal = OrderedDict()
        for ds in self.data:
            sidesds = Sides(self.sides,ds)
            sidesds.sidesfits()
            self.dsfitters_sidecal[ds.channum] = sidesds.fitters
        with open(os.path.join(self.figdirname,self.data.shortname()+"_channel_fitters_sidecal.pkl"),"w") as f:
            pickle.dump(self.dsfitters_sidecal,f)

    def repeated_line_plots_sidecal(self):
        for repeated_lines_spec in self.repeated_lines_specs:
            repeatedlineplotter = RepeatedLinePlotter(self.dsfitters_sidecal, repeated_lines_spec)
            repeatedlineplotter.plot()
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig(os.path.join(self.figdirname,self.figdirname+"repeated_lines_sidecal%d.png" % i))
            plt.close(i)

    def line_zoom_plots(self, side_element_dict, lines, epm,filename_extra=""):
        """
            For each `line` in `lines` make a plot showing the spectrum from each side in the energy
            range line_center-epm to line_center+epm where both line_center and epm are in eV.
        """
        for line in lines:
            for normalize in [True, False]:
                plt.figure()
                elo,ehi = mass.STANDARD_FEATURES[line]-epm,mass.STANDARD_FEATURES[line]+epm
                for side in self.sides:
                    bin_centers, counts = self.data.hist(np.arange(elo,ehi,1), category={"side":side.name})
                    label = side.name + ": "
                    for element in side_element_dict[side.name]:
                        label += element + ", "
                    label = label[:-2]
                    if normalize:
                        plt.plot(bin_centers, counts/float(counts.sum()), label=label + " /%3g"%counts.sum())
                        plt.ylabel("counts per bin %0.2f eV bin normalized"%(bin_centers[1]-bin_centers[0]))
                    else:
                        plt.semilogy(bin_centers, counts, label=label)
                        plt.ylabel("counts per bin %0.2f eV bin"%(bin_centers[1]-bin_centers[0]))
                plt.xlabel("energy (eV)")
                plt.legend(loc="best")
                plt.title(self.data.shortname()+ " %s"%line)
            self.save_and_close_all_plots("line_zoom%s_%s_"%(filename_extra,line))

def sidesplot(ds, sides, attr="p_energy"):
    """
    Plot the attribute attr vs rows_after_last_external_trigger, and color code by the "side" categorical cut.
    """
    rowtime = ds.rowtime()
    vals = getattr(ds,attr)
    plt.figure()
    for side in sides+[Side("uncategorized",[],-1,-1)]:
        g = ds.good(side=side.name)
        if hasattr(side,"period"):
            period = side.period
            t0 = side.t0
        if ds.has_external_trigger_data:
            period = np.amax(ds.rows_after_last_external_trigger[g])*rowtime
            N_side_visits = ds.external_trigger_rowcount.size
        else:
            N_side_visits = round((ds.p_timestamp[g][-1]-t0)/period)

        ncounts = g.sum()
        if ncounts == 0:
            total_time = 0
            cps=0
        else:
            if side.name == "uncategorized":
                duration = [sides[i].thi-sides[i].tlo for i in range(len(sides))]
                t_uncategorized = period-np.sum(duration)
                side_time = t_uncategorized*N_side_visits
            else:
                if ds.has_external_trigger_data:
                    timestamps = ds.rows_after_last_external_trigger[:]*rowtime
                else:
                    timestamps = ds.p_timestamp[:]
                side_time = (side.thi-side.tlo)*N_side_visits
            cps = ncounts/side_time
        assert len(timestamps)==len(vals)
        plt.plot(timestamps[g], vals[g],".",label="Side "+side.name+repr(side.lines)+", %0.2fcps"%cps)
    plt.legend(loc="best")
    plt.xlabel("time after last external trigger (s)")
    plt.ylabel(attr)
    plt.title(ds.shortname())
def sidesplot_hist(ds, sides, tstep):
    """
    Plot the count rate vs time after last external trigger. Color by sides.
    tstep -- bin size for calculating count rate
    """
    g=ds.good()
    rowtime = ds.rowtime()
    tlo = 0
    if ds.has_external_trigger_data:
        thi = np.amax(ds.rows_after_last_external_trigger[g])*rowtime
        N_side_visits = ds.external_trigger_rowcount.size
        timestamps = ds.rows_after_last_external_trigger[:]*rowtime
    else:
        thi = sides[0].period
        N_side_visits = round((ds.p_timestamp[g][-1]-sides[0].t0)/sides[0].period)
        timestamps = (ds.p_timestamp[:]-sides[0].t0)%sides[0].period
    total_time = N_side_visits*(thi-tlo)

    bin_edges = np.arange(tlo,thi,tstep)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    counts,_ = np.histogram(timestamps[ds.good()], bin_edges)
    cps=counts/tstep/N_side_visits
    plt.figure()
    plt.plot(bin_centers, cps,drawstyle="steps-mid",label="all")
    for side in sides+[Side("uncategorized",[],-1,-1)]:
        counts,_ = np.histogram(timestamps[ds.good(side=side.name)], bin_edges)
        cps = counts/tstep/N_side_visits
        plt.plot(bin_centers, cps,drawstyle="steps-mid",label="Side "+side.name)
    plt.legend()
    plt.xlabel("time after last external trigger (s)")
    plt.ylabel("counts per second")
    plt.title(ds.shortname())



if __name__ == "__main__":
    plt.close("all")
    slow_switcher = MassSidesAnalysis("20171101/20171101_B","20171101/20171101_A",
    roughcal_lines=["MnKAlpha","MnKBeta"],
    sides=[Side("A",["MnKAlpha","MnKBeta"],0,5.6),Side("B",["MnKAlpha","MnKBeta"],5.8,12)],
    maxchans = 240)
    slow_switcher.doit()
