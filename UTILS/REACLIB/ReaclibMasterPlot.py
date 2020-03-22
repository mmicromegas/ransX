import EQUATIONS.REACLIB.XtransportVsNuclearTimescales as xtvsn

import matplotlib.pyplot as plt


class ReaclibMasterPlot():

    def __init__(self, params):
        self.params = params

    def execXtransportVSnuclearTimescales(self, inuc, element, x, bconv, tconv, tauL):
        params = self.params

        # instantiate 
        ransXtvsn = xtvsn.XtransportVsNuclearTimescales(params.getForProp('prop')['eht_data'],
                                                        params.getForProp('prop')['reaclib'],
                                                        params.getForProp('prop')['ig'],
                                                        inuc, element, bconv, tconv, tauL,
                                                        params.getForProp('prop')['intc'],
                                                        params.getForProp('prop')['prefix'],
                                                        params.getForProp('prop')['fext'],
                                                        params.getForProp('prop')['tnuc'],
                                                        params.getNetwork())

        ransXtvsn.plot_Xtimescales(params.getForProp('prop')['laxis'],
                                   params.getForEqs(x)['xbl'],
                                   params.getForEqs(x)['xbr'],
                                   params.getForEqs(x)['ybu'],
                                   params.getForEqs(x)['ybd'],
                                   params.getForEqs(x)['ilg'])

        ransXtvsn.plot_Xtransport_equation(params.getForProp('prop')['laxis'],
                                           params.getForEqs(x)['xbl'],
                                           params.getForEqs(x)['xbr'],
                                           params.getForEqs(x)['ybu'],
                                           params.getForEqs(x)['ybd'],
                                           params.getForEqs(x)['ilg'])

    def SetMatplotlibParams(self):
        """ This routine sets some standard values for matplotlib """
        """ to obtain publication-quality figures """

        # plt.rc('text',usetex=True)
        # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        plt.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
        plt.rc('font', size=16.)
        plt.rc('lines', linewidth=2, markeredgewidth=2., markersize=12)
        plt.rc('axes', linewidth=1.5)
        plt.rcParams['xtick.major.size'] = 8.
        plt.rcParams['xtick.minor.size'] = 4.
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.subplot.left'] = 0.17
        plt.rcParams['figure.subplot.right'] = 0.85
        plt.rcParams.update({'figure.max_open_warning': 0})
