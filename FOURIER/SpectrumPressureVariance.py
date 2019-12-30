import numpy as np
import sys
import matplotlib.pyplot as plt
import UTILS.PROMPI.PROMPI_data as pd


class SpectrumPressureVariance():

    def __init__(self, filename, data_prefix, ig, lhc):

        block = pd.PROMPI_bindata(filename, ['press'])

        xzn0 = block.datadict['xzn0']
        pressure = block.datadict['press']

        xlm = np.abs(np.asarray(xzn0) - np.float(lhc))
        ilhc = int(np.where(xlm == xlm.min())[0][0])

        ny = block.datadict['qqy']
        nz = block.datadict['qqz']

        # initialize 
        ig = int(ig)

        if (ig == 1):
            sterad = np.ones((nz, ny))
            steradtot = np.sum(sterad)
        elif (ig == 2):
            print("ERROR (SpectrumPressureVariance.py): Spherical Geometry not implemented.")
            # sterad =
            # steradtot =
        else:
            print("ERROR (SpectrumPressureVariance.py): Geometry not implemented")

        # get horizontal data
        hpressure = pressure[ilhc][:][:]

        # calculate horizontal mean value
        eh_pressure = np.sum(pressure * sterad) / steradtot

        # calculate Reynolds fluctuations
        pressuref_r = hpressure - eh_pressure

        # calculate Fourier coefficients (a_ and b_)
        pressurefr_fft = np.fft.fft2(pressuref_r)

        a_pressuref = np.real(pressurefr_fft)
        b_pressuref = np.imag(pressurefr_fft)

        # calculate energy contribution to total variance 
        # from real and imaginary parts of Fourier transforms  

        energy_pressuref = a_pressuref * a_pressuref + b_pressuref * b_pressuref

        # define wave numbers (in 2pi/N units)
        if (ny == nz):
            nn = ny
            nnmax = np.round(np.sqrt(2. * (nn ** 2.)))

        # array of horizontal wave numbers
        kh = np.arange((nnmax / 2))
        khmax = int(max(kh))

        # calculate distance from nearest corners in nn x nn matrix
        # and use it later to computer mask for integration over 
        # spherical shells
        aa = self.dist(nn)

        # integrate over radial shells and calculate total TKE spectrum
        spect_ppvar = []
        for ishell in kh:
            mask = np.where((aa >= float(ishell)) & (aa < float(ishell + 1)), 1., 0.)
            integrand_ppvar = mask * energy_pressuref
            spect_ppvar.append(np.sum(integrand_ppvar) / (float(nn) * float(nn)))

        # calculate spectrum
        spect_ppvar_mean = []
        i = -1
        for ishell in kh:
            i += 1
            spect_ppvar_mean.append(spect_ppvar[i] / (float(ny) * float(nz)))

        # check Parseval's theorem

        total_ppvar = (pressuref_r * pressuref_r)
        print(np.sum(total_ppvar), np.sum(spect_ppvar))

        # share stuff across class
        self.spect_ppvar = spect_ppvar
        self.spect_ppvar_mean = spect_ppvar_mean
        self.kh = kh
        self.data_prefix = data_prefix

    def plot_PPspectrum(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot pp spectrum"""

        # load horizontal wave number GRID
        grd1 = self.kh

        # load spectrum to plot
        plt1 = self.spect_ppvar_mean

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   						
        plt.axis([xbl, xbr, ybu, ybd])

        # plot DATA 
        plt.title('pressure variance "energy" spectrum')
        plt.loglog(grd1, plt1, color='brown', label=r"$P'P'$")
        plt.loglog(grd1, plt1[1] * grd1 ** (-5. / 3.), color='r', linestyle='--', label=r"k$^{-5/3}$")

        setxlabel = r'k$_h$'
        setylabel = r"$P$ variance (erg$^{2}$ cm$^{-6}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=0, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'ppvariancespectrum.png')

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
