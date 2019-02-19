import mass
import pylab as plt
import numpy as np
import fastdtw

class AlignBToA():
    cm = plt.cm.gist_ncar
    def __init__(self,ds_a, ds_b, peak_xs_a, bin_edges, attr, category = {},
                 scale_by_median = True, normalize_before_dtw = True):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.bin_edges = bin_edges
        self.bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        self.peak_xs_a = peak_xs_a
        self.peak_inds_a = np.searchsorted(self.bin_centers, self.peak_xs_a)
        self.attr = attr
        self.scale_by_median = scale_by_median
        self.normalize_before_dtw = normalize_before_dtw
        self.category = category
        self.peak_inds_b = self.samePeaks()
        self.addCalToB()

    def samePeaks(self):
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.good(**self.category)]
        ph_b = getattr(self.ds_b,self.attr)[self.ds_b.good(**self.category)]
        if self.scale_by_median:
            median_ratio_a_over_b = np.median(ph_a)/np.median(ph_b)
        else:
            median_ratio_a_over_b = 1
        ph_b_median_scaled = ph_b*median_ratio_a_over_b
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b_median_scaled, _ = np.histogram(ph_b_median_scaled, self.bin_edges)
        if self.normalize_before_dtw:
            distance, path = fastdtw.fastdtw(self.normalize(counts_a), self.normalize(counts_b_median_scaled))
        else:
            distance, path = fastdtw.fastdtw(counts_a, counts_b_median_scaled)
        i_a = [x[0] for x in path]
        i_b_median_scaled = [x[1] for x in path]
        peak_inds_b_median_scaled = [i_b_median_scaled[i_a.index(pia)] for pia in self.peak_inds_a]
        peak_xs_b_median_scaled = self.bin_centers[peak_inds_b_median_scaled]
        peak_xs_b = peak_xs_b_median_scaled/median_ratio_a_over_b
        peak_inds_b = np.searchsorted(self.bin_centers, peak_xs_b)
        return peak_inds_b

    def samePeaksPlot(self):
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.good(**self.category)]
        ph_b = getattr(self.ds_b,self.attr)[self.ds_b.good(**self.category)]
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
        plt.title(self.ds_a.shortname()+" + "+self.ds_b.shortname()+"\nwith same peaks noted, peaks not expected to be aligned in this plot")

    def samePeaksPlotWithAlignmentCal(self):
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.good(**self.category)]
        ph_b = getattr(self.ds_b,self.newattr)[self.ds_b.good(**self.category)]
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b, _ = np.histogram(ph_b, self.bin_edges)
        plt.figure()
        plt.plot(self.bin_centers,counts_a,label="a: channel %i"%self.ds_a.channum)
        for i,pi in enumerate(self.peak_inds_a):
            plt.plot(self.bin_centers[pi],counts_a[pi],"o",color=self.cm(float(i)/len(self.peak_inds_a)))
        plt.plot(self.bin_centers,counts_b,label="b: channel %i"%self.ds_b.channum)
        for i,pi in enumerate(self.peak_inds_a):
            plt.plot(self.bin_centers[pi],counts_b[pi],"o",color=self.cm(float(i)/len(self.peak_inds_a)))
        plt.xlabel(self.newattr)
        plt.ylabel("counts per %0.2f unit bin"%(self.bin_centers[1]-self.bin_centers[0]))
        plt.legend()

    def normalize(self,x):
        return x/float(np.sum(x))

    def addCalToB(self):
        cal_b_to_a = mass.EnergyCalibration(curvetype="gain")
        for pi_a,pi_b in zip(self.peak_inds_a, self.peak_inds_b):
            cal_b_to_a.add_cal_point(self.bin_centers[pi_b], self.bin_centers[pi_a])
        attr_b_in_a_units = cal_b_to_a(getattr(self.ds_b,self.attr))
        self.newattr = self.attr+"_ch%i"%self.ds_a.channum
        self.newcalname = self.attr+"_to_"+self.newattr
        setattr(self.ds_b, self.newattr, attr_b_in_a_units)
        self.ds_b.calibration[self.newcalname ]=cal_b_to_a
        self.cal_b_to_a = cal_b_to_a
        cal_b_to_a.save_to_hdf5(self.ds_b.hdf5_group["calibration"], self.newcalname )

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
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.good(**self.category)]
        ph_b = getattr(self.ds_b,self.newattr)[self.ds_b.good(**self.category)]
        entropy = mass.entropy.laplace_cross_entropy(ph_a[ph_a>self.bin_edges[0]],
                     ph_b[ph_b>self.bin_edges[0]], w=w)
        return entropy

    def _ksStatistic(self):
        ph_a = getattr(self.ds_a,self.attr)[self.ds_a.good(**self.category)]
        ph_b = getattr(self.ds_b,self.newattr)[self.ds_b.good(**self.category)]
        counts_a, _ = np.histogram(ph_a, self.bin_edges)
        counts_b, _ = np.histogram(ph_b, self.bin_edges)
        cdf_a = np.cumsum(counts_a)/np.sum(a)
        cdf_b = np.cumsum(counts_b)/np.sum(b)
        ks_statistic = np.amax(np.abs(cdf_a-cdf_b))
        return ks_statistic

mass.addfitter(
element="O",
linetype=" He-Like 1s2p + 1s2s",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=573.94777,
energies=np.array([560.983, 568.551, 573.94777]), lorentzian_fwhm=np.array([0.1,0.1,0.1]),
reference_amplitude=np.array([1,1,10]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="O",
linetype=" H-Like 2p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(653.679946*2+653.493657*1)/3,
energies=np.array([653.493657, 653.679946]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="O",
linetype=" H-Like 3p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(774.634043*2+774.578843*1)/3,
energies=np.array([774.634043, 774.578843]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([2,1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="O",
linetype=" He-Like 1s3p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=665.61536,
energies=np.array([665.61536]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="O",
linetype=" He-Like 1s4p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=697.79546 ,
energies=np.array([697.79546 ]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="Ne",
linetype=" H-Like 2p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(1021.952896*2+1021.497550*1)/3,
energies=np.array([1021.497550, 1021.952896]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="Ne",
linetype=" H-Like 3p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(1210.961453*2+1210.826524*1)/3,
energies=np.array([1210.826524, 1210.961453]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="Ne",
linetype=" H-Like 4p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(1277.130058*2+1277.073140*1)/3,
energies=np.array([1277.073140, 1277.130058]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="Ne",
linetype=" He-Like 1s2s",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=905.0772 ,
energies=np.array([905.0772]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="Ne",
linetype=" He-Like 1s2p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=922.0159 ,
energies=np.array([922.0159]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

mass.addfitter(
element="Ne",
linetype=" He-Like 1s3p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=1073.7689 ,
energies=np.array([1073.7689]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

'''
O LINES
'''
# H-like
mass.addfitter(
element="O",
linetype=" H-Like 2p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(653.679946*2+653.493657*1)/3,
energies=np.array([653.493657, 653.679946]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None
)

mass.addfitter(
element="O",
linetype=" H-Like 3p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(774.634043*2+774.578843*1)/3,
energies=np.array([774.634043, 774.578843]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([2,1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None
)

mass.addfitter(
element="O",
linetype=" H-Like 4p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(816.974368*2+816.951082*1)/3,
energies=np.array([816.951082, 816.974368]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([2,1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None
)

# He-like
mass.addfitter(
element="O",
linetype=" He-Like 1s2s+1s2p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=573.94777,
energies=np.array([560.983, 568.551, 573.94777]), lorentzian_fwhm=np.array([0.1,0.1,0.1]),
reference_amplitude=np.array([500,300,1000]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="O",
linetype=" He-Like 1s2s 3S1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=560.98386,
energies=np.array([560.98386]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="O",
linetype=" He-Like 1s2p 1P1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=573.94777,
energies=np.array([573.94777]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="O",
linetype=" He-Like 1s3p 1P1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=665.61536,
energies=np.array([665.61536]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="O",
linetype=" He-Like 1s4p 1P1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=697.79546 ,
energies=np.array([697.79546 ]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

'''
Ne LINES
'''
# H-like
mass.addfitter(
element="Ne",
linetype=" H-Like 2p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(1021.952896*2+1021.497550*1)/3,
energies=np.array([1021.497550, 1021.952896]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

mass.addfitter(
element="Ne",
linetype=" H-Like 3p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(1210.961453*2+1210.826524*1)/3,
energies=np.array([1210.826524, 1210.961453]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

mass.addfitter(
element="Ne",
linetype=" H-Like 4p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(1277.130058*2+1277.073140*1)/3,
energies=np.array([1277.073140, 1277.130058]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

# He-like
mass.addfitter(
element="Ne",
linetype=" He-Like 1s2s+1s2p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=922.0159,
energies=np.array([905.0772, 914.8174, 922.0159]), lorentzian_fwhm=np.array([0.1,0.1,0.1]),
reference_amplitude=np.array([500,150,1000]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="Ne",
linetype=" He-Like 1s2s 3S1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=905.0772 ,
energies=np.array([905.0772]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="Ne",
linetype=" He-Like 1s2p 1P1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=922.0159 ,
energies=np.array([922.0159]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="Ne",
linetype=" He-Like 1s3p 1P1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=1073.7689 ,
energies=np.array([1073.7689]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

'''
Ar LINES
'''
# H-like
mass.addfitter(
element="Ar",
linetype=" H-Like 2p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(3322.9921*2+3318.1762*1)/3,
energies=np.array([3318.1762, 3322.9921]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

mass.addfitter(
element="Ar",
linetype=" H-Like 3p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(3935.72070*2+3934.29336*1)/3,
energies=np.array([3934.29336, 3935.72070]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

mass.addfitter(
element="Ar",
linetype=" H-Like 4p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=(4150.33999*2+4149.73807*1)/3,
energies=np.array([4149.73807, 4150.33999]), lorentzian_fwhm=np.array([0.1,0.1]),
reference_amplitude=np.array([1,2]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

# He-like
mass.addfitter(
element="Ar",
linetype=" He-Like 1s2s+1s2p",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3139.5824,
energies=np.array([3104.1486, 3123.5346, 3139.5824]), lorentzian_fwhm=np.array([0.1,0.1,0.1]),
reference_amplitude=np.array([100,55,200]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

mass.addfitter(
element="Ar",
linetype=" He-Like 1s2s 3S1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3104.1486,
energies=np.array([3104.1486]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

mass.addfitter(
element="Ar",
linetype=" He-Like 1s2p 1P1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3139.5824,
energies=np.array([3139.5824]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

mass.addfitter(
element="Ar",
linetype=" He-Like 1s3p 1P1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3683.848,
energies=np.array([3683.848]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

mass.addfitter(
element="Ar",
linetype=" He-Like 1s4p 1P1",
reference_short='NIST ASD',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3874.886,
energies=np.array([3874.886]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Theory"
)

'''
W Lines
'''
# Ni-like
mass.addfitter(
element="W",
linetype=" Ni-1",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=1488.2,
energies=np.array([1488.2]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.4,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-2",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=1562.9,
energies=np.array([1562.9]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.3,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-3",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=1629.8,
energies=np.array([1629.8]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.3,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-4",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=1764.6,
energies=np.array([1764.6]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.3,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-5",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=1829.6,
energies=np.array([1829.6]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.4,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-6",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2015.4,
energies=np.array([2015.4]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.3,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-7",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2112.2,
energies=np.array([2112.2]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.3,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-8",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2179.7,
energies=np.array([2179.7]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.4,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-9",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2320.3,
energies=np.array([2320.3]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.6,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-10",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2360.7,
energies=np.array([2360.7]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.7,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-11",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2384.2,
energies=np.array([2384.2]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.4,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-12",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2553.0,
energies=np.array([2553.0]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.4,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-13",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2651.3,
energies=np.array([2651.3]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.4,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-14",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2673.7,
energies=np.array([2673.7]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.6,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-15",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2760.7,
energies=np.array([2760.7]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.5,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-16",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2816.1,
energies=np.array([2816.1]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.3,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-17",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=2878.2,
energies=np.array([2878.2]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.3,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-18",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3182.7,
energies=np.array([3182.7]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.4,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-19",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3196.8,
energies=np.array([3196.8]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.3,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-20",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3259.9,
energies=np.array([3259.9]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.3,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-21",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3426.0,
energies=np.array([3426.0]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.4,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-22",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3480.9,
energies=np.array([3480.9]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.7,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-23",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3490.2,
energies=np.array([3490.2]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.4,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-24",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3574.1,
energies=np.array([3574.1]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.5,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-25",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3600.0,
energies=np.array([3600.0]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.6,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

mass.addfitter(
element="W",
linetype=" Ni-26",
reference_short='Clementson 2010',
fitter_type = mass.line_fits.GenericKBetaFitter,
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=3639.5,
energies=np.array([3639.5]), lorentzian_fwhm=np.array([0.1]),
position_uncertainty=0.6,
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT,
ka12_energy_diff=None,
reference_measurement_type = "Experiment"
)

def lineNameOrEnergyToEnergy(lineNameOrEnergy):
    if lineNameOrEnergy in mass.spectrum_classes:
        return mass.spectrum_classes[lineNameOrEnergy]().peak_energy
    elif isinstance(lineNameOrEnergy,float) or isinstance(lineNameOrEnergy, int):
        return lineNameOrEnergy
    else:
        raise Exception("could not convert {} to energy".format(lineNameOrEnergy))
