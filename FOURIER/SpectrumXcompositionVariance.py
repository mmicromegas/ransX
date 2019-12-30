import numpy as np
import matplotlib.pyplot as plt
import UTILS.PROMPI.PROMPI_data as pd


class SpectrumXcompositionVariance():

    def __init__(self, filename, data_prefix, ig, lhc, inuc, element):

        block = pd.PROMPI_bindata(filename, [inuc,'density'])

        xzn0 = block.datadict['xzn0']

        xnuc = block.datadict[inuc]
        density = block.datadict['density']
        xrho = xnuc*density

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
            print("ERROR (SpectrumXrhoVariance.py): Spherical Geometry not implemented.")
            # sterad =
            # steradtot =
        else:
            print("ERROR (SpectrumXrhoVariance.py): Geometry not implemented")

        # get horizontal data
        hXrho = xrho[ilhc][:][:]

        # calculate horizontal mean value
        eh_Xrho = np.sum(xrho * sterad) / steradtot

        # calculate Reynolds fluctuations
        Xrhof_r = hXrho - eh_Xrho

        # calculate Fourier coefficients (a_ and b_)
        Xrhofr_fft = np.fft.fft2(Xrhof_r)

        a_Xrhof = np.real(Xrhofr_fft)
        b_Xrhof = np.imag(Xrhofr_fft)

        # calculate energy contribution to total variance 
        # from real and imaginary parts of Fourier transforms  

        energy_Xrhof = a_Xrhof * a_Xrhof + b_Xrhof * b_Xrhof

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
        spect_xrhovar = []
        for ishell in kh:
            mask = np.where((aa >= float(ishell)) & (aa < float(ishell + 1)), 1., 0.)
            integrand_xrhovar = mask * energy_Xrhof
            spect_xrhovar.append(np.sum(integrand_xrhovar) / (float(nn) * float(nn)))

        # calculate spectrum
        spect_xrhovar_mean = []
        i = -1
        for ishell in kh:
            i += 1
            spect_xrhovar_mean.append(spect_xrhovar[i] / (float(ny) * float(nz)))

        # check Parseval's theorem

        total_xrhovar = (Xrhof_r * Xrhof_r)
        print(np.sum(total_xrhovar), np.sum(spect_xrhovar))

        # share stuff across class
        self.spect_xrhovar = spect_xrhovar
        self.spect_xrhovar_mean = spect_xrhovar_mean
        self.kh = kh
        self.data_prefix = data_prefix
        self.element = element

    def plot_XrhoSpectrum(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot Xrho spectrum"""

        element = self.element

        # load horizontal wave number GRID
        grd1 = self.kh

        # load spectrum to plot
        plt1 = self.spect_xrhovar_mean

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   						
        plt.axis([xbl, xbr, ybu, ybd])

        # plot DATA 
        plt.title('xrho variance "energy" spectrum ' + element)
        plt.loglog(grd1, plt1, color='brown', label=r"$X'X'$")
        plt.loglog(grd1, plt1[1] * grd1 ** (-5. / 3.), color='r', linestyle='--', label=r"k$^{-5/3}$")

        setxlabel = r'k$_h$'
        setylabel = r"$\rho$X variance (g$^{2}$ cm$^{-6}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=0, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'xrhovariancespectrum' + element + '.png')

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
