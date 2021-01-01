from EVOLUTION.TurbulentKineticEnergyEquationEvolutionStudy import TurbulentKineticEnergyEquationEvolutionResolutionStudy
from EVOLUTION.MachNumberMaxEvolutionStudy import MachNumberMaxEvolutionResolutionStudy
from EVOLUTION.MachNumberMeanEvolutionStudy import MachNumberMeanEvolutionResolutionStudy
from EVOLUTION.ConvectiveRMSvelocityEvolutionStudy import ConvectiveRMSvelocityEvolutionResolutionStudy
from EVOLUTION.ConvectiveTurnoverTimescaleEvolutionStudy import ConvectiveTurnoverTimescaleEvolutionResolutionStudy
from EVOLUTION.ConvectionBoundariesPositionEvolutionStudy import ConvectionBoundariesPositionEvolutionResolutionStudy
from EVOLUTION.EnergySourceTermEvolutionStudy import EnergySourceTermEvolutionResolutionStudy
from EVOLUTION.ContResMaxEvolutionStudy import ContResMaxEvolutionResolutionStudy
from EVOLUTION.ContResMeanEvolutionStudy import ContResMeanEvolutionResolutionStudy
from EVOLUTION.TeeResMaxEvolutionStudy import TeeResMaxEvolutionResolutionStudy
from EVOLUTION.TeeResMeanEvolutionStudy import TeeResMeanEvolutionResolutionStudy

from EVOLUTION.X0002EvolutionResolutionStudy import X0002EvolutionResolutionStudy

import matplotlib.pyplot as plt


class EvolMasterPlot():

    def __init__(self, params):
        self.params = params

    def execEvolTKE(self):
        params = self.params

        # instantiate 		
        ransTkeEvol = TurbulentKineticEnergyEquationEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy evolution	   
        ransTkeEvol.plot_tke_evolution(params.getForProp('prop')['laxis'],
                                       params.getForEvol('tkeevol')['xbl'],
                                       params.getForEvol('tkeevol')['xbr'],
                                       params.getForEvol('tkeevol')['ybu'],
                                       params.getForEvol('tkeevol')['ybd'],
                                       params.getForEvol('tkeevol')['ilg'])

    def execEvolMachMax(self):
        params = self.params

        # instantiate
        ransMachMax = MachNumberMaxEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot maximum Mach number evolution
        ransMachMax.plot_machmax_evolution(params.getForProp('prop')['laxis'],
                                           params.getForEvol('machmxevol')['xbl'],
                                           params.getForEvol('machmxevol')['xbr'],
                                           params.getForEvol('machmxevol')['ybu'],
                                           params.getForEvol('machmxevol')['ybd'],
                                           params.getForEvol('machmxevol')['ilg'])

    def execEvolMachMean(self):
        params = self.params

        # instantiate
        ransMachMean = MachNumberMeanEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot mean mach number evolution
        ransMachMean.plot_machmean_evolution(params.getForProp('prop')['laxis'],
                                             params.getForEvol('machmeevol')['xbl'],
                                             params.getForEvol('machmeevol')['xbr'],
                                             params.getForEvol('machmeevol')['ybu'],
                                             params.getForEvol('machmeevol')['ybd'],
                                             params.getForEvol('machmeevol')['ilg'])

    def execEvolConvVelRMS(self):
        params = self.params

        # instantiate
        ransUrmsEvol = ConvectiveRMSvelocityEvolutionResolutionStudy(params.getForProp('prop')['eht_data'],
                                                                             params.getForProp('prop')['ig'],
                                                                             params.getForProp('prop')['prefix'])

        # plot convective rms velocity
        ransUrmsEvol.plot_turms_evolution(params.getForProp('prop')['laxis'],
                                          params.getForEvol('convelrms')['xbl'],
                                          params.getForEvol('convelrms')['xbr'],
                                          params.getForEvol('convelrms')['ybu'],
                                          params.getForEvol('convelrms')['ybd'],
                                          params.getForEvol('convelrms')['ilg'])

        # ransUrmsEvol.plot_u_vs_L()

    def execEvolConvTurnoverTime(self):
        params = self.params

        # instantiate
        ransConvTurnEvol = ConvectiveTurnoverTimescaleEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot convective turnover timescales
        ransConvTurnEvol.plot_tconvturn_evolution(params.getForProp('prop')['laxis'],
                                                  params.getForEvol('convturn')['xbl'],
                                                  params.getForEvol('convturn')['xbr'],
                                                  params.getForEvol('convturn')['ybu'],
                                                  params.getForEvol('convturn')['ybd'],
                                                  params.getForEvol('convturn')['ilg'])

    def execEvolConvTurnoverTime(self):
        params = self.params

        # instantiate
        ransConvTurnEvol = ConvectiveTurnoverTimescaleEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot convective convective turnover timescale
        ransConvTurnEvol.plot_tconvturn_evolution(params.getForProp('prop')['laxis'],
                                                  params.getForEvol('convturn')['xbl'],
                                                  params.getForEvol('convturn')['xbr'],
                                                  params.getForEvol('convturn')['ybu'],
                                                  params.getForEvol('convturn')['ybd'],
                                                  params.getForEvol('convturn')['ilg'])

    def execEvolCNVZbnry(self):
        params = self.params

        # instantiate
        ransCnvzPositionEvol = ConvectionBoundariesPositionEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot evolution of convection boundaries
        ransCnvzPositionEvol.plot_conv_bndry_location(params.getForProp('prop')['laxis'],
                                                      params.getForEvol('cnvzbndry')['xbl'],
                                                      params.getForEvol('cnvzbndry')['xbr'],
                                                      params.getForEvol('cnvzbndry')['ybu'],
                                                      params.getForEvol('cnvzbndry')['ybd'],
                                                      params.getForEvol('cnvzbndry')['ilg'])

    def execEvolTenuc(self):
        params = self.params

        # instantiate
        ransTenucEvol = EnergySourceTermEvolutionResolutionStudy(params.getForProp('prop')['eht_data'],
                                                         params.getForProp('prop')['ig'],
                                                         params.getForProp('prop')['prefix'])

        # plot total energy source
        ransTenucEvol.plot_tenuc_evolution(params.getForProp('prop')['laxis'],
                                           params.getForEvol('enesource')['xbl'],
                                           params.getForEvol('enesource')['xbr'],
                                           params.getForEvol('enesource')['ybu'],
                                           params.getForEvol('enesource')['ybd'],
                                           params.getForEvol('enesource')['ilg'])

    def execContResMax(self):
        params = self.params

        # instantiate
        ransContResMax = ContResMaxEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot max residual from continuity equation evolution
        ransContResMax.plot_resContMax_evolution(params.getForProp('prop')['laxis'],
                                                 params.getForEvol('contresmax')['xbl'],
                                                 params.getForEvol('contresmax')['xbr'],
                                                 params.getForEvol('contresmax')['ybu'],
                                                 params.getForEvol('contresmax')['ybd'],
                                                 params.getForEvol('contresmax')['ilg'])

    def execContResMean(self):
        params = self.params

        # instantiate
        ransContResMean = ContResMeanEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot mean residual from continuity equation evolution
        ransContResMean.plot_resContMean_evolution(params.getForProp('prop')['laxis'],
                                                   params.getForEvol('contresmean')['xbl'],
                                                   params.getForEvol('contresmean')['xbr'],
                                                   params.getForEvol('contresmean')['ybu'],
                                                   params.getForEvol('contresmean')['ybd'],
                                                   params.getForEvol('contresmean')['ilg'])

    def execTeeResMax(self):
        params = self.params

        # instantiate
        ransTeeResMax = TeeResMaxEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot max residual from total energy equation evolution
        ransTeeResMax.plot_resTeeMax_evolution(params.getForProp('prop')['laxis'],
                                                 params.getForEvol('teeresmax')['xbl'],
                                                 params.getForEvol('teeresmax')['xbr'],
                                                 params.getForEvol('teeresmax')['ybu'],
                                                 params.getForEvol('teeresmax')['ybd'],
                                                 params.getForEvol('teeresmax')['ilg'])

    def execTeeResMean(self):
        params = self.params

        # instantiate
        ransTeeResMean = TeeResMeanEvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot mean residual from total energy equation evolution
        ransTeeResMean.plot_resTeeMean_evolution(params.getForProp('prop')['laxis'],
                                                   params.getForEvol('teeresmean')['xbl'],
                                                   params.getForEvol('teeresmean')['xbr'],
                                                   params.getForEvol('teeresmean')['ybu'],
                                                   params.getForEvol('teeresmean')['ybd'],
                                                   params.getForEvol('teeresmean')['ilg'])

    def execEvolX0002(self):
        params = self.params

        # instantiate
        ransX0002 = X0002EvolutionResolutionStudy(
            params.getForProp('prop')['eht_data'],
            params.getForProp('prop')['ig'],
            params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy evolution
        ransX0002.plot_x0002_evolution(params.getForProp('prop')['laxis'],
                                       params.getForEvol('x0002evol')['xbl'],
                                       params.getForEvol('x0002evol')['xbr'],
                                       params.getForEvol('x0002evol')['ybu'],
                                       params.getForEvol('x0002evol')['ybd'],
                                       params.getForEvol('x0002evol')['ilg'])

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
