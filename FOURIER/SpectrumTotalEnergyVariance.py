import numpy as np
import matplotlib.pyplot as plt
import UTILS.PROMPI.PROMPI_data as pd


class SpectrumTotalEnergyVariance():

    def __init__(self, filename, data_prefix, ig, lhc):

        block = pd.PROMPI_bindata(filename, ['energy'])

        xzn0 = block.datadict['xzn0']
        energy = block.datadict['energy']

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
            print("ERROR (SpectrumTotalEnergyVariance.py): Spherical Geometry not implemented.")
            # sterad =
            # steradtot =
        else:
            print("ERROR (SpectrumTotalEnergyVariance.py): Geometry not implemented")

        # get horizontal data
        henergy = energy[ilhc][:][:]

        # calculate horizontal mean value
        eh_energy = np.sum(energy * sterad) / steradtot

        # calculate Reynolds fluctuations
        energyf_r = henergy - eh_energy

        # calculate Fourier coefficients (a_ and b_)
        energyfr_fft = np.fft.fft2(energyf_r)

        a_energyf = np.real(energyfr_fft)
        b_energyf = np.imag(energyfr_fft)

        # calculate energy contribution to total variance 
        # from real and imaginary parts of Fourier transforms  

        energy_energyf = a_energyf * a_energyf + b_energyf * b_energyf

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
        spect_etvar = []
        for ishell in kh:
            mask = np.where((aa >= float(ishell)) & (aa < float(ishell + 1)), 1., 0.)
            integrand_etvar = mask * energy_energyf
            spect_etvar.append(np.sum(integrand_etvar) / (float(nn) * float(nn)))

        # calculate spectrum
        spect_etvar_mean = []
        i = -1
        for ishell in kh:
            i += 1
            spect_etvar_mean.append(spect_etvar[i] / (float(ny) * float(nz)))

        # check Parseval's theorem
        total_etvar = (energyf_r * energyf_r)
        print(np.sum(total_etvar), np.sum(spect_etvar))

        # share stuff across class
        self.spect_etvar = spect_etvar
        self.spect_etvar_mean = spect_etvar_mean
        self.kh = kh
        self.data_prefix = data_prefix

    def plot_ETspectrum(self, LAXIS, xbl, xbr, ybu, ybd, ilg):
        """Plot energy spectrum"""

        # load horizontal wave number GRID
        grd1 = self.kh

        # load spectrum to plot
        plt1 = self.spect_etvar_mean

        # create FIGURE
        plt.figure(figsize=(7, 6))

        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        # set plot boundaries   						
        plt.axis([xbl, xbr, ybu, ybd])

        # plot DATA 
        plt.title('total energy variance "energy" spectrum')
        plt.loglog(grd1, plt1, color='brown', label=r"$\epsilon'_T \epsilon'_T$")
        plt.loglog(grd1, plt1[1] * grd1 ** (-5. / 3.), color='r', linestyle='--', label=r"k$^{-5/3}$")

        setxlabel = r'k$_h$'
        setylabel = r"$\epsilon_T$ variance (erg$^{2}$ cm$^{-6}$)"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)

        # show LEGEND
        plt.legend(loc=0, prop={'size': 18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/' + self.data_prefix + 'etvariancespectrum.png')

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
