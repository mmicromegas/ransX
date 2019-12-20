import EVOLUTION.FOR_RESOLUTION_STUDY.TurbulentKineticEnergyEquationEvolutionResolutionStudy as tkeevol
import EVOLUTION.FOR_RESOLUTION_STUDY.MachNumberMaxEvolutionResolutionStudy as machmxevol
import EVOLUTION.FOR_RESOLUTION_STUDY.MachNumberMeanEvolutionResolutionStudy as machmeevol

# import EVOLUTION.FOR_RESOLUTION_STUDY.ConvectionBoundariesPositionEvolutionResolutionStudy as cnvzpos
# import EVOLUTION.FOR_RESOLUTION_STUDY.EnergySourceTermEvolutionResolutionStudy as enesrc
# import EVOLUTION.FOR_RESOLUTION_STUDY.X0002EvolutionResolutionStudy as x2evol


import matplotlib.pyplot as plt


class ResEvolMasterPlot():

    def __init__(self, params):
        self.params = params

    def execEvolTKE(self):
        params = self.params

        # instantiate 		
        ransTkeEvol = tkeevol.TurbulentKineticEnergyEquationEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'], \
            params.getForProp('prop')['ig'], \
            params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy evolution	   
        ransTkeEvol.plot_tke_evolution(params.getForProp('prop')['laxis'], \
                                       params.getForEvol('tkeevol')['xbl'], \
                                       params.getForEvol('tkeevol')['xbr'], \
                                       params.getForEvol('tkeevol')['ybu'], \
                                       params.getForEvol('tkeevol')['ybd'], \
                                       params.getForEvol('tkeevol')['ilg'])

    def execEvolMachMax(self):
        params = self.params

        # instantiate
        ransMachMax = machmxevol.MachNumberMaxEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'], \
            params.getForProp('prop')['ig'], \
            params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy evolution
        ransMachMax.plot_machmax_evolution(params.getForProp('prop')['laxis'], \
                                       params.getForEvol('machmxevol')['xbl'], \
                                       params.getForEvol('machmxevol')['xbr'], \
                                       params.getForEvol('machmxevol')['ybu'], \
                                       params.getForEvol('machmxevol')['ybd'], \
                                       params.getForEvol('machmxevol')['ilg'])

    def execEvolMachMean(self):
        params = self.params

        # instantiate
        ransMachMean = machmeevol.MachNumberMeanEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'], \
            params.getForProp('prop')['ig'], \
            params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy evolution
        ransMachMean.plot_machmean_evolution(params.getForProp('prop')['laxis'], \
                                       params.getForEvol('machmeevol')['xbl'], \
                                       params.getForEvol('machmeevol')['xbr'], \
                                       params.getForEvol('machmeevol')['ybu'], \
                                       params.getForEvol('machmeevol')['ybd'], \
                                       params.getForEvol('machmeevol')['ilg'])


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
