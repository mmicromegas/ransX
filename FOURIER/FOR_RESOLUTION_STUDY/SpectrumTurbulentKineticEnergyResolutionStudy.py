import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.PROMPI.PROMPI_data as pd
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

class SpectrumTurbulentKineticEnergyResolutionStudy(calc.CALCULUS, al.ALIMIT, object):

    def __init__(self, datadir, filename, data_prefix, ig, lhc):
        super(SpectrumTurbulentKineticEnergyResolutionStudy, self).__init__(ig)

        # initialize
        ig = int(ig)

        # load data to a list
        block = []
        for file in filename:
            print(datadir+file)
            block.append(pd.PROMPI_bindata(datadir+file, ['velx', 'vely', 'velz']))

        # declare data lists
        xzn0, velx, vely, velz, xlm, ilhc = [], [], [], [], [], []
        nx, ny, nz = [], [], []
        sterad, steradtot = [],[]

        for i in range(len(filename)):
            xzn0.append(block[i].datadict['xzn0'])
            velx.append(block[i].datadict['velx'])
            vely.append(block[i].datadict['vely'])
            velz.append(block[i].datadict['velz'])
            xlm.append(np.abs(np.asarray(xzn0[i]) - np.float(lhc)))
            ilhc.append(int(np.where(xlm[i] == xlm[i].min())[0][0]))
            nx.append(block[i].datadict['qqx'])
            ny.append(block[i].datadict['qqy'])
            nz.append(block[i].datadict['qqz'])

            if (ig == 1):
                sterad.append(np.ones((nz[i], ny[i])))
                steradtot.append(np.sum(sterad[i]))
            elif (ig == 2):
                print("ERROR (SpectrumTurbulentKineticEnergy.py): Spherical Geometry not implemented.")
                # sterad =
                # steradtot =
            else:
                print("ERROR (SpectrumTurbulentKineticEnergy.py): Geometry not implemented")

        hvelx, hvely, hvelz = [], [], []
        khh, spect_tke_mean_res = [], []

        for i in range(len(filename)):

            # get horizontal data
            hvelx = velx[i][ilhc[i]][:][:]
            hvely = vely[i][ilhc[i]][:][:]
            hvelz = velz[i][ilhc[i]][:][:]

            # calculate horizontal mean value
            eh_ux = np.sum(hvelx * sterad[i]) / steradtot[i]
            eh_uy = np.sum(hvely * sterad[i]) / steradtot[i]
            eh_uz = np.sum(hvelz * sterad[i]) / steradtot[i]

            # calculate Reynolds fluctuations
            uxf_r = hvelx - eh_ux
            uyf_r = hvely - eh_uy
            uzf_r = hvelz - eh_uz

            # calculate Fourier coefficients (a_ and b_)
            uxfr_fft = np.fft.fft2(uxf_r)
            uyfr_fft = np.fft.fft2(uyf_r)
            uzfr_fft = np.fft.fft2(uzf_r)

            a_uxff = np.real(uxfr_fft)
            b_uxff = np.imag(uxfr_fft)

            a_uyff = np.real(uyfr_fft)
            b_uyff = np.imag(uyfr_fft)

            a_uzff = np.real(uzfr_fft)
            b_uzff = np.imag(uzfr_fft)

            # calculate energy contribution to total variance
            # from real and imaginary parts of Fourier transforms

            energy_uxff = a_uxff * a_uxff + b_uxff * b_uxff
            energy_uyff = a_uyff * a_uyff + b_uyff * b_uyff
            energy_uzff = a_uzff * a_uzff + b_uzff * b_uzff

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

            # integrate over radial shells and calculate total TKE spectrum
            spect_tke = []
            for ishell in kh:
                mask = np.where((aa >= float(ishell)) & (aa < float(ishell + 1)), 1., 0.)
                integrand_tke = mask * 0.5 * (energy_uxff + energy_uyff + energy_uzff)
                spect_tke.append(np.sum(integrand_tke) / (float(nn) * float(nn)))
                # fig, (ax1) = plt.subplots(figsize=(3, 3))
                # pos = ax1.imshow(mask,interpolation='bilinear',origin='lower',extent=(0,nn,0,nn))
                # fig.colorbar(pos, ax=ax1)
                # plt.show(block=False)

            # calculate mean TKE spectrum
            spect_tke_mean = []
            j = -1
            for ishell in kh:
                j += 1
                spect_tke_mean.append(spect_tke[j] / (float(ny[i]) * float(nz[i])))

            # check Parseval's theorem

            total_tke = (uxf_r * uxf_r + uyf_r * uyf_r + uzf_r * uzf_r) / 2.0
            # print(uxf_r*uxf_r,uyf_r*uyf_r,uzf_r*uzf_r,uxf_r*uxf_r+uyf_r*uyf_r+uzf_r*uzf_r)
            # print(np.sum(uxf_r*uxf_r+uyf_r*uyf_r+uzf_r*uzf_r))
            print(np.sum(total_tke), np.sum(spect_tke))

            spect_tke_mean_res.append(np.asarray(spect_tke_mean))

        # share stuff across class

        self.spect_tke_mean_res = spect_tke_mean_res
        self.khh = khh
        self.data_prefix = data_prefix

        self.nx = nx
        self.ny = ny
        self.nz = nz

    def plot_TKEspectrum(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot TKE spectrum"""

        # load horizontal wave number GRID
        grd = self.khh

        # load DATA to plot
        spect_tke_mean_res = self.spect_tke_mean_res

        # find maximum resolution data
        grd_maxres = self.maxresdata(grd)
        tke_maxres = self.maxresdata(spect_tke_mean_res)

        plt_interp = []
        for i in range(len(grd)):
            plt_interp.append(np.interp(grd_maxres, grd[i], spect_tke_mean_res[i]))

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        LAXIS = int(LAXIS)
        if LAXIS != 2:
            print("ERROR(SpectrumTurbulentKineticEnergyResolutionStudy.py): Only LAXIS=2 is supported.")
            sys.exit()

        spect_tke_mean_res0_tmp = spect_tke_mean_res[0]
        spect_tke_mean_res1_tmp = spect_tke_mean_res[0]

        spect_tke_mean_res_foraxislimit = []
        spect_tke_mean_resmax = np.max(spect_tke_mean_res[0])
        for spect_tke_mean_resi in spect_tke_mean_res:
            if (np.max(spect_tke_mean_resi) > spect_tke_mean_resmax):
                spect_tke_mean_res_foraxislimit = spect_tke_mean_resi

        # set plot boundaries
        to_plot = [spect_tke_mean_res_foraxislimit]
        self.set_plt_axis(LAXIS, xbl, xbr, ybu, ybd, to_plot)

        # plot DATA 
        plt.title('turbulent kinetic energy spectrum')

        for i in range(len(grd)):
            plt.plot(grd[i], spect_tke_mean_res[i],
                     label=str(self.nx[i]) + ' x ' + str(self.ny[i]) + ' x ' + str(self.nz[i]))

        plt.loglog(grd[0], max(spect_tke_mean_res[0]) * grd[0] ** (-5. / 3.), color='r', linestyle='--',
                   label=r"k$^{-5/3}$")

        setxlabel = r'k$_h$'
        setylabel = r"$\widetilde{k}$ (erg g$^{-1}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=0, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'tkespectrumRes.png')

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
