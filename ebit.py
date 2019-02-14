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
        self.peak_inds_a = np.searchsorted(bin_edges, self.peak_xs_a)-1
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
        min_bin = self.bin_edges[0]
        bin_spacing = self.bin_edges[1]-self.bin_edges[0]
        peak_inds_b = map(int,(peak_xs_b-min_bin)/bin_spacing)
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
reference_plot_gaussian_fwhm=0.5,
nominal_peak_energy=1073.7689 ,
energies=np.array([1073.7689]), lorentzian_fwhm=np.array([0.1]),
reference_amplitude=np.array([1]),
reference_amplitude_type=mass.calibration.LORENTZIAN_PEAK_HEIGHT
,ka12_energy_diff=None
)

def lineNameOrEnergyToEnergy(lineNameOrEnergy):
    if lineNameOrEnergy in mass.spectrum_classes:
        return mass.spectrum_classes[lineNameOrEnergy]().peak_energy
    elif isinstance(lineNameOrEnergy,float) or isinstance(lineNameOrEnergy, int):
        return lineNameOrEnergy
    else:
        raise Exception("could not convert {} to energy".format(lineNameOrEnergy))
