import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.PROMPI.PROMPI_data as pd
import UTILS.Calculus as calc
import UTILS.SetAxisLimit as al


class SpectrumTotalEnergyVarianceResolutionStudy(calc.Calculus, al.SetAxisLimit, uT.Tools, eR.Errors, object):

    def __init__(self, datadir, filename, data_prefix, ig, lhc):
        super(SpectrumTotalEnergyVarianceResolutionStudy, self).__init__(ig)

        # initialize
        ig = int(ig)

        # load data to a list
        block = []
        for file in filename:
            print(datadir + file)
            block.append(pd.PROMPI_bindata(datadir + file, ['energy']))

        # declare data lists
        xzn0, energy, xlm, ilhc = [], [], [], []
        nx, ny, nz = [], [], []
        sterad, steradtot = [], []

        for i in range(len(filename)):
            xzn0.append(block[i].datadict['xzn0'])
            energy.append(block[i].datadict['energy'])
            xlm.append(np.abs(np.asarray(xzn0[i]) - np.float(lhc)))
            ilhc.append(int(np.where(xlm[i] == xlm[i].min())[0][0]))
            nx.append(block[i].datadict['qqx'])
            ny.append(block[i].datadict['qqy'])
            nz.append(block[i].datadict['qqz'])

            if (ig == 1):
                sterad.append(np.ones((nz[i], ny[i])))
                steradtot.append(np.sum(sterad[i]))
            elif (ig == 2):
                print("ERROR (SpectrumTotalEnergyVarianceEnergyResolutionStudy.py): Spherical Geometry not implemented.")
                # sterad =
                # steradtot =
            else:
                print("ERROR (SpectrumTotalEnergyVarianceResolutionStudy.py): Geometry not implemented")

        henergy = []
        khh, spect_etvar_mean_res = [], []

        for i in range(len(filename)):

            # get horizontal data
            henergy = energy[i][ilhc[i]][:][:]

            # calculate horizontal mean value
            eh_et = np.sum(henergy * sterad[i]) / steradtot[i]

            # calculate Reynolds fluctuations
            etf_r = henergy - eh_et

            # calculate Fourier coefficients (a_ and b_)
            etfr_fft = np.fft.fft2(etf_r)

            a_etff = np.real(etfr_fft)
            b_etff = np.imag(etfr_fft)

            # calculate energy contribution to total variance
            # from real and imaginary parts of Fourier transforms

            energy_etff = a_etff * a_etff + b_etff * b_etff

            # define wave numbers (in 2pi/N units)
            nyy = ny[i]
            nzz = nz[i]
            if (nyy == nzz):
                nn = nyy
                nnmax = np.round(np.sqrt(2. * (nn ** 2.)))

            # array of horizontal wave numbers
            kh = np.arange((nnmax / 2))
            khmax = int(max(kh))
            khh.append(kh)

            # calculate distance from nearest corners in nn x nn matrix
            # and use it later to computer mask for integration over
            # spherical shells
            aa = self.dist(nn)

            # integrate over radial shells and calculate total et variance spectrum
            spect_etvar = []
            for ishell in kh:
                mask = np.where((aa >= float(ishell)) & (aa < float(ishell + 1)), 1., 0.)
                integrand_etvar = mask * 0.5 * (energy_etff)
                spect_etvar.append(np.sum(integrand_etvar) / (float(nn) * float(nn)))

            # calculate mean total energy energy variance spectrum
            spect_etvar_mean = []
            j = -1
            for ishell in kh:
                j += 1
                spect_etvar_mean.append(spect_etvar[j] / (float(ny[i]) * float(nz[i])))

            # check Parseval's theorem
            total_etvar = (etf_r * etf_r) / 2.0
            print(np.sum(total_etvar), np.sum(spect_etvar))

            spect_etvar_mean_res.append(np.asarray(spect_etvar_mean))

        # share stuff across class

        self.spect_etvar_mean_res = spect_etvar_mean_res
        self.khh = khh
        self.data_prefix = data_prefix

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def plot_ETspectrum(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot et spectrum"""

        # load horizontal wave number GRID
        grd = self.khh

        # load DATA to plot
        spect_etvar_mean_res = self.spect_etvar_mean_res

        # find maximum resolution data
        grd_maxres = self.maxresdata(grd)
        etvar_maxres = self.maxresdata(spect_etvar_mean_res)

        plt_interp = []
        for i in range(len(grd)):
            plt_interp.append(np.interp(grd_maxres, grd[i], spect_etvar_mean_res[i]))

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        LAXIS = int(LAXIS)
        if LAXIS != 2:
            print("ERROR(SpectrumTotalEnergyVarianceResolutionStudy.py): Only LAXIS=2 is supported.")
            sys.exit()

        spect_etvar_mean_res0_tmp = spect_etvar_mean_res[0]
        spect_etvar_mean_res1_tmp = spect_etvar_mean_res[0]

        spect_etvar_mean_res_foraxislimit = []
        spect_etvar_mean_resmax = np.max(spect_etvar_mean_res[0])
        for spect_etvar_mean_resi in spect_etvar_mean_res:
            if (np.max(spect_etvar_mean_resi) > spect_etvar_mean_resmax):
                spect_etvar_mean_res_foraxislimit = spect_etvar_mean_resi

        # set plot boundaries
        to_plot = [spect_etvar_mean_res_foraxislimit]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('total energy variance "energy" spectrum')

        for i in range(len(grd)):
            plt.plot(grd[i], spect_etvar_mean_res[i],
                     label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]))

        plt.loglog(grd[0], max(spect_etvar_mean_res[0]) * grd[0] ** (-5. / 3.), color='r', linestyle='--',
                   label=r"k$^{-5/3}$")

        setxlabel = r'k$_h$'
        setylabel = r"$\sigma_{\epsilon T}$ (K$^{2}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=0, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'etspectrumRes.png')

    # source: https://gist.github.com/abeelen/453de325dd9787ea2aa7fad495f4f018
    def dist(self, NAXIS):
        """Returns a rectangular array in which the value of each element is proportional to its frequency.
        >>> dist(3)
        array([[ 0.        ,  1.        ,  1.        ],
               [ 1.        ,  1.41421356,  1.41421356],
               [ 1.        ,  1.41421356,  1.41421356]])
        >>> dist(4)
        array([[ 0.        ,  1.        ,  2.        ,  1.        ],
               [ 1.        ,  1.41421356,  2.23606798,  1.41421356],
               [ 2.        ,  2.23606798,  2.82842712,  2.23606798],
               [ 1.        ,  1.41421356,  2.23606798,  1.41421356]])
        """
        axis = np.linspace(-NAXIS / 2 + 1, NAXIS / 2, NAXIS)
        result = np.sqrt(axis ** 2 + axis[:, np.newaxis] ** 2)
        return np.roll(result, NAXIS / 2 + 1, axis=(0, 1))

    # find data with maximum resolution
    def maxresdata(self, data):
        tmp = 0
        for idata in data:
            if idata.shape[0] > tmp:
                data_maxres = idata
            else:
                tmp = idata.shape[0]

        return data_maxres
