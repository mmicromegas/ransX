import EQUATIONS.FOR_RESOLUTION_STUDY.TemperatureResolutionStudy as tt
import EQUATIONS.FOR_RESOLUTION_STUDY.BruntVaisallaResolutionStudy as bruntv
import EQUATIONS.FOR_RESOLUTION_STUDY.TurbulentKineticEnergyResolutionStudy as tke
import EQUATIONS.FOR_RESOLUTION_STUDY.InternalEnergyFluxResolutionStudy as feix
import EQUATIONS.FOR_RESOLUTION_STUDY.EntropyFluxResolutionStudy as fssx
import EQUATIONS.FOR_RESOLUTION_STUDY.PressureFluxResolutionStudy as fppx
import EQUATIONS.FOR_RESOLUTION_STUDY.TemperatureFluxResolutionStudy as fttx
import EQUATIONS.FOR_RESOLUTION_STUDY.EnthalpyFluxResolutionStudy as fhhx
import EQUATIONS.FOR_RESOLUTION_STUDY.TurbulentMassFluxResolutionStudy as a
import EQUATIONS.FOR_RESOLUTION_STUDY.TurbulentRadialVelocityResolutionStudy as uxRms
import EQUATIONS.FOR_RESOLUTION_STUDY.TurbulentUyVelocityResolutionStudy as uyRms
import EQUATIONS.FOR_RESOLUTION_STUDY.TurbulentUzVelocityResolutionStudy as uzRms
import EQUATIONS.FOR_RESOLUTION_STUDY.DensityRmsResolutionStudy as ddRms
import EQUATIONS.FOR_RESOLUTION_STUDY.BuoyancyResolutionStudy as buoy

import EQUATIONS.FOR_RESOLUTION_STUDY.XResolutionStudy as xx
import EQUATIONS.FOR_RESOLUTION_STUDY.XdensityResolutionStudy as xrho
import EQUATIONS.FOR_RESOLUTION_STUDY.XfluxResolutionStudy as xflxx

import matplotlib.pyplot as plt


class ResMasterPlot():

    def __init__(self, params):
        self.params = params

    def execX(self, inuc, element, x):
        params = self.params

        # instantiate
        ransX = xx.XResolutionStudy(params.getForProp('prop')['eht_data'],
                                                params.getForProp('prop')['ig'],
                                                inuc, element,
                                                params.getForProp('prop')['intc'],
                                                params.getForProp('prop')['prefix'])

        ransX.plot_X(params.getForProp('prop')['laxis'],
                        params.getForEqs(x)['xbl'],
                        params.getForEqs(x)['xbr'],
                        params.getForEqs(x)['ybu'],
                        params.getForEqs(x)['ybd'],
                        params.getForEqs(x)['ilg'])

    def execXrho(self, inuc, element, x):
        params = self.params

        # instantiate 		
        ransXrho = xrho.XdensityResolutionStudy(params.getForProp('prop')['eht_data'],
                                                params.getForProp('prop')['ig'],
                                                inuc, element,
                                                params.getForProp('prop')['intc'],
                                                params.getForProp('prop')['prefix'])

        ransXrho.plot_Xrho(params.getForProp('prop')['laxis'],
                           params.getForEqs(x)['xbl'],
                           params.getForEqs(x)['xbr'],
                           params.getForEqs(x)['ybu'],
                           params.getForEqs(x)['ybd'],
                           params.getForEqs(x)['ilg'])

    def execXflxx(self, inuc, element, x):
        params = self.params

        # instantiate 		
        ransXflxx = xflxx.XfluxResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               inuc, element,
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransXflxx.plot_fxi(params.getForProp('prop')['laxis'],
                           params.getForEqs(x)['xbl'],
                           params.getForEqs(x)['xbr'],
                           params.getForEqs(x)['ybu'],
                           params.getForEqs(x)['ybd'],
                           params.getForEqs(x)['ilg'])

    def execBruntV(self):
        params = self.params

        # instantiate 		
        ransBruntV = bruntv.BruntVaisallaResolutionStudy(params.getForProp('prop')['eht_data'],
                                                         params.getForProp('prop')['ig'],
                                                         params.getForProp('prop')['ieos'],
                                                         params.getForProp('prop')['intc'],
                                                         params.getForProp('prop')['prefix'])

        ransBruntV.plot_bruntvaisalla(params.getForProp('prop')['laxis'],
                                      params.getForEqs('nsq')['xbl'],
                                      params.getForEqs('nsq')['xbr'],
                                      params.getForEqs('nsq')['ybu'],
                                      params.getForEqs('nsq')['ybd'],
                                      params.getForEqs('nsq')['ilg'])

    def execTemp(self):
        params = self.params

        # instantiate 		
        ransTT = tt.TemperatureResolutionStudy(params.getForProp('prop')['eht_data'], \
                                               params.getForProp('prop')['ig'], \
                                               params.getForProp('prop')['intc'], \
                                               params.getForProp('prop')['prefix'])

        ransTT.plot_tt(params.getForProp('prop')['laxis'], \
                       params.getForEqs('temp')['xbl'], \
                       params.getForEqs('temp')['xbr'], \
                       params.getForEqs('temp')['ybu'], \
                       params.getForEqs('temp')['ybd'], \
                       params.getForEqs('temp')['ilg'])

    def execTke(self):
        params = self.params
        kolmrate = 0.

        # instantiate 		
        ransTke = tke.TurbulentKineticEnergyResolutionStudy(params.getForProp('prop')['eht_data'], \
                                                            params.getForProp('prop')['ig'], \
                                                            params.getForProp('prop')['intc'], \
                                                            params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy			   
        ransTke.plot_tke(params.getForProp('prop')['laxis'], \
                         params.getForEqs('tkie')['xbl'], \
                         params.getForEqs('tkie')['xbr'], \
                         params.getForEqs('tkie')['ybu'], \
                         params.getForEqs('tkie')['ybd'], \
                         params.getForEqs('tkie')['ilg'])

    def execEiFlx(self):
        params = self.params

        # instantiate 		
        ransEiFlx = feix.InternalEnergyFluxResolutionStudy(params.getForProp('prop')['eht_data'], \
                                                           params.getForProp('prop')['ig'], \
                                                           params.getForProp('prop')['intc'], \
                                                           params.getForProp('prop')['prefix'])

        ransEiFlx.plot_feix(params.getForProp('prop')['laxis'], \
                            params.getForEqs('eintflx')['xbl'], \
                            params.getForEqs('eintflx')['xbr'], \
                            params.getForEqs('eintflx')['ybu'], \
                            params.getForEqs('eintflx')['ybd'], \
                            params.getForEqs('eintflx')['ilg'])

    def execSSflx(self):
        params = self.params

        # instantiate 		
        ransSSflx = fssx.EntropyFluxResolutionStudy(params.getForProp('prop')['eht_data'], \
                                                    params.getForProp('prop')['ig'], \
                                                    params.getForProp('prop')['intc'], \
                                                    params.getForProp('prop')['prefix'])

        ransSSflx.plot_fssx(params.getForProp('prop')['laxis'], \
                            params.getForEqs('entrflx')['xbl'], \
                            params.getForEqs('entrflx')['xbr'], \
                            params.getForEqs('entrflx')['ybu'], \
                            params.getForEqs('entrflx')['ybd'], \
                            params.getForEqs('entrflx')['ilg'])

    def execTTflx(self):
        params = self.params

        # instantiate 		
        ransTTflx = fttx.TemperatureFluxResolutionStudy(params.getForProp('prop')['eht_data'], \
                                                        params.getForProp('prop')['ig'], \
                                                        params.getForProp('prop')['intc'], \
                                                        params.getForProp('prop')['prefix'])

        ransTTflx.plot_fttx(params.getForProp('prop')['laxis'], \
                            params.getForEqs('tempflx')['xbl'], \
                            params.getForEqs('tempflx')['xbr'], \
                            params.getForEqs('tempflx')['ybu'], \
                            params.getForEqs('tempflx')['ybd'], \
                            params.getForEqs('tempflx')['ilg'])

    def execHHflx(self):
        params = self.params

        # instantiate 		
        ransHHflx = fhhx.EnthalpyFluxResolutionStudy(params.getForProp('prop')['eht_data'], \
                                                     params.getForProp('prop')['ig'], \
                                                     params.getForProp('prop')['intc'], \
                                                     params.getForProp('prop')['prefix'])

        ransHHflx.plot_fhhx(params.getForProp('prop')['laxis'], \
                            params.getForEqs('enthflx')['xbl'], \
                            params.getForEqs('enthflx')['xbr'], \
                            params.getForEqs('enthflx')['ybu'], \
                            params.getForEqs('enthflx')['ybd'], \
                            params.getForEqs('enthflx')['ilg'])

    def execTMSflx(self):
        params = self.params

        # instantiate 		
        ransTMSflx = a.TurbulentMassFluxResolutionStudy(params.getForProp('prop')['eht_data'], \
                                                        params.getForProp('prop')['ig'], \
                                                        params.getForProp('prop')['intc'], \
                                                        params.getForProp('prop')['prefix'])

        ransTMSflx.plot_fddx(params.getForProp('prop')['laxis'], \
                             params.getForEqs('tmsflx')['xbl'], \
                             params.getForEqs('tmsflx')['xbr'], \
                             params.getForEqs('tmsflx')['ybu'], \
                             params.getForEqs('tmsflx')['ybd'], \
                             params.getForEqs('tmsflx')['ilg'])

    def execPPxflx(self):
        params = self.params

        # instantiate 		
        ransPPxflx = fppx.PressureFluxResolutionStudy(params.getForProp('prop')['eht_data'], \
                                                      params.getForProp('prop')['ig'], \
                                                      params.getForProp('prop')['intc'], \
                                                      params.getForProp('prop')['prefix'])

        ransPPxflx.plot_fppx(params.getForProp('prop')['laxis'], \
                             params.getForEqs('pressxflx')['xbl'], \
                             params.getForEqs('pressxflx')['xbr'], \
                             params.getForEqs('pressxflx')['ybu'], \
                             params.getForEqs('pressxflx')['ybd'], \
                             params.getForEqs('pressxflx')['ilg'])

    def execUXrms(self):
        params = self.params

        # instantiate
        ransUXrms = uxRms.TurbulentRadialVelocityResolutionStudy(params.getForProp('prop')['eht_data'],
                                                                 params.getForProp('prop')['ig'],
                                                                 params.getForProp('prop')['intc'],
                                                                 params.getForProp('prop')['prefix'])

        ransUXrms.plot_uxrms(params.getForProp('prop')['laxis'],
                             params.getForEqs('uxrms')['xbl'],
                             params.getForEqs('uxrms')['xbr'],
                             params.getForEqs('uxrms')['ybu'],
                             params.getForEqs('uxrms')['ybd'],
                             params.getForEqs('uxrms')['ilg'])

    def execUYrms(self):
        params = self.params

        # instantiate
        ransUYrms = uyRms.TurbulentUyVelocityResolutionStudy(params.getForProp('prop')['eht_data'],
                                                                 params.getForProp('prop')['ig'],
                                                                 params.getForProp('prop')['intc'],
                                                                 params.getForProp('prop')['prefix'])

        ransUYrms.plot_uyrms(params.getForProp('prop')['laxis'],
                             params.getForEqs('uyrms')['xbl'],
                             params.getForEqs('uyrms')['xbr'],
                             params.getForEqs('uyrms')['ybu'],
                             params.getForEqs('uyrms')['ybd'],
                             params.getForEqs('uyrms')['ilg'])

    def execUZrms(self):
        params = self.params

        # instantiate
        ransUZrms = uzRms.TurbulentUzVelocityResolutionStudy(params.getForProp('prop')['eht_data'],
                                                                 params.getForProp('prop')['ig'],
                                                                 params.getForProp('prop')['intc'],
                                                                 params.getForProp('prop')['prefix'])

        ransUZrms.plot_uzrms(params.getForProp('prop')['laxis'],
                             params.getForEqs('uzrms')['xbl'],
                             params.getForEqs('uzrms')['xbr'],
                             params.getForEqs('uzrms')['ybu'],
                             params.getForEqs('uzrms')['ybd'],
                             params.getForEqs('uzrms')['ilg'])

    def execDDrms(self):
        params = self.params

        # instantiate
        ransDDrms = ddRms.DensityRmsResolutionStudy(params.getForProp('prop')['eht_data'], \
                                                      params.getForProp('prop')['ig'], \
                                                      params.getForProp('prop')['intc'], \
                                                      params.getForProp('prop')['prefix'])

        ransDDrms.plot_ddrms(params.getForProp('prop')['laxis'], \
                             params.getForEqs('ddrms')['xbl'], \
                             params.getForEqs('ddrms')['xbr'], \
                             params.getForEqs('ddrms')['ybu'], \
                             params.getForEqs('ddrms')['ybd'], \
                             params.getForEqs('ddrms')['ilg'])

    def execBuoyancy(self):
        params = self.params

        # instantiate
        ransBuoyancy = buoy.BuoyancyResolutionStudy(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['ieos'],
                                                    params.getForProp('prop')['intc'],
                                                    params.getForProp('prop')['prefix'])

        ransBuoyancy.plot_buoyancy(params.getForProp('prop')['laxis'], \
                             params.getForEqs('buoy')['xbl'], \
                             params.getForEqs('buoy')['xbr'], \
                             params.getForEqs('buoy')['ybu'], \
                             params.getForEqs('buoy')['ybd'], \
                             params.getForEqs('buoy')['ilg'])


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
