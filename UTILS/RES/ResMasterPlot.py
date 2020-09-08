import EQUATIONS.FOR_RESOLUTION_STUDY.TemperatureResolutionStudy as tt
import EQUATIONS.FOR_RESOLUTION_STUDY.DensityResolutionStudy as dd
import EQUATIONS.FOR_RESOLUTION_STUDY.MomentumXResolutionStudy as momex
import EQUATIONS.FOR_RESOLUTION_STUDY.TotalEnergyResolutionStudy as et
import EQUATIONS.FOR_RESOLUTION_STUDY.EntropyResolutionStudy as ss
import EQUATIONS.FOR_RESOLUTION_STUDY.EntropyVarianceResolutionStudy as ssvar
import EQUATIONS.FOR_RESOLUTION_STUDY.EnthalpyResolutionStudy as hh
import EQUATIONS.FOR_RESOLUTION_STUDY.PressureResolutionStudy as pp
import EQUATIONS.FOR_RESOLUTION_STUDY.AbarResolutionStudy as abar
import EQUATIONS.FOR_RESOLUTION_STUDY.AbarFluxResolutionStudy as abarflux
import EQUATIONS.FOR_RESOLUTION_STUDY.DensitySpecificVolumeCovarianceResolutionStudy as dsvc
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
import EQUATIONS.FOR_RESOLUTION_STUDY.XvarianceResolutionStudy as xvar
import EQUATIONS.FOR_RESOLUTION_STUDY.DivuResolutionStudy as divu
import EQUATIONS.FOR_RESOLUTION_STUDY.DivFrhoResolutionStudy as divfrho

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

    def execXvar(self, inuc, element, x):
        params = self.params

        # instantiate
        ransXvar = xvar.XvarianceResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               inuc, element,
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransXvar.plot_Xvariance(params.getForProp('prop')['laxis'],
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
        ransTT = tt.TemperatureResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransTT.plot_tt(params.getForProp('prop')['laxis'],
                       params.getForEqs('temp')['xbl'],
                       params.getForEqs('temp')['xbr'],
                       params.getForEqs('temp')['ybu'],
                       params.getForEqs('temp')['ybd'],
                       params.getForEqs('temp')['ilg'])


    def execRho(self):
        params = self.params

        # instantiate
        ransDD = dd.DensityResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransDD.plot_dd(params.getForProp('prop')['laxis'],
                       params.getForEqs('rho')['xbl'],
                       params.getForEqs('rho')['xbr'],
                       params.getForEqs('rho')['ybu'],
                       params.getForEqs('rho')['ybd'],
                       params.getForEqs('rho')['ilg'])

    def execMomex(self):
        params = self.params

        # instantiate
        ransMomx = momex.MomentumXResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransMomx.plot_momex(params.getForProp('prop')['laxis'],
                       params.getForEqs('momex')['xbl'],
                       params.getForEqs('momex')['xbr'],
                       params.getForEqs('momex')['ybu'],
                       params.getForEqs('momex')['ybd'],
                       params.getForEqs('momex')['ilg'])

    def execEt(self):
        params = self.params

        # instantiate
        ransEt = et.TotalEnergyResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransEt.plot_et(params.getForProp('prop')['laxis'],
                       params.getForEqs('toe')['xbl'],
                       params.getForEqs('toe')['xbr'],
                       params.getForEqs('toe')['ybu'],
                       params.getForEqs('toe')['ybd'],
                       params.getForEqs('toe')['ilg'])

    def execSS(self):
        params = self.params

        # instantiate
        ransSS = ss.EntropyResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransSS.plot_ss(params.getForProp('prop')['laxis'],
                       params.getForEqs('entr')['xbl'],
                       params.getForEqs('entr')['xbr'],
                       params.getForEqs('entr')['ybu'],
                       params.getForEqs('entr')['ybd'],
                       params.getForEqs('entr')['ilg'])

    def execSSvar(self):
        params = self.params

        # instantiate
        ransSSvar = ssvar.EntropyVarianceResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransSSvar.plot_ssvar(params.getForProp('prop')['laxis'],
                       params.getForEqs('entrvar')['xbl'],
                       params.getForEqs('entrvar')['xbr'],
                       params.getForEqs('entrvar')['ybu'],
                       params.getForEqs('entrvar')['ybd'],
                       params.getForEqs('entrvar')['ilg'])

    def execHH(self):
        params = self.params

        # instantiate
        ransHH = hh.EnthalpyResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransHH.plot_hh(params.getForProp('prop')['laxis'],
                       params.getForEqs('enth')['xbl'],
                       params.getForEqs('enth')['xbr'],
                       params.getForEqs('enth')['ybu'],
                       params.getForEqs('enth')['ybd'],
                       params.getForEqs('enth')['ilg'])

    def execPP(self):
        params = self.params

        # instantiate
        ransPP = pp.PressureResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransPP.plot_pp(params.getForProp('prop')['laxis'],
                       params.getForEqs('press')['xbl'],
                       params.getForEqs('press')['xbr'],
                       params.getForEqs('press')['ybu'],
                       params.getForEqs('press')['ybd'],
                       params.getForEqs('press')['ilg'])

    def execDSVC(self):
        params = self.params

        # instantiate
        ransDSVC = dsvc.DensitySpecificVolumeCovarianceResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransDSVC.plot_dsvc(params.getForProp('prop')['laxis'],
                       params.getForEqs('dsvc')['xbl'],
                       params.getForEqs('dsvc')['xbr'],
                       params.getForEqs('dsvc')['ybu'],
                       params.getForEqs('dsvc')['ybd'],
                       params.getForEqs('dsvc')['ilg'])

    def execAbar(self):
        params = self.params

        # instantiate
        ransAbar = abar.AbarResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransAbar.plot_abar(params.getForProp('prop')['laxis'],
                       params.getForEqs('abar')['xbl'],
                       params.getForEqs('abar')['xbr'],
                       params.getForEqs('abar')['ybu'],
                       params.getForEqs('abar')['ybd'],
                       params.getForEqs('abar')['ilg'])

    def execAbarFlux(self):
        params = self.params

        # instantiate
        ransAbar = abarflux.AbarFluxResolutionStudy(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        ransAbar.plot_abarflux(params.getForProp('prop')['laxis'],
                       params.getForEqs('abflx')['xbl'],
                       params.getForEqs('abflx')['xbr'],
                       params.getForEqs('abflx')['ybu'],
                       params.getForEqs('abflx')['ybd'],
                       params.getForEqs('abflx')['ilg'])

    def execTke(self):
        params = self.params
        kolmrate = 0.

        # instantiate 		
        ransTke = tke.TurbulentKineticEnergyResolutionStudy(params.getForProp('prop')['eht_data'],
                                                            params.getForProp('prop')['ig'],
                                                            params.getForProp('prop')['intc'],
                                                            params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy			   
        ransTke.plot_tke(params.getForProp('prop')['laxis'],
                         params.getForEqs('tkie')['xbl'],
                         params.getForEqs('tkie')['xbr'],
                         params.getForEqs('tkie')['ybu'],
                         params.getForEqs('tkie')['ybd'],
                         params.getForEqs('tkie')['ilg'])

    def execEiFlx(self):
        params = self.params

        # instantiate 		
        ransEiFlx = feix.InternalEnergyFluxResolutionStudy(params.getForProp('prop')['eht_data'],
                                                           params.getForProp('prop')['ig'],
                                                           params.getForProp('prop')['intc'],
                                                           params.getForProp('prop')['prefix'])

        ransEiFlx.plot_feix(params.getForProp('prop')['laxis'],
                            params.getForEqs('eintflx')['xbl'],
                            params.getForEqs('eintflx')['xbr'],
                            params.getForEqs('eintflx')['ybu'],
                            params.getForEqs('eintflx')['ybd'],
                            params.getForEqs('eintflx')['ilg'])

    def execSSflx(self):
        params = self.params

        # instantiate 		
        ransSSflx = fssx.EntropyFluxResolutionStudy(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    params.getForProp('prop')['prefix'])

        ransSSflx.plot_fssx(params.getForProp('prop')['laxis'],
                            params.getForEqs('entrflx')['xbl'],
                            params.getForEqs('entrflx')['xbr'],
                            params.getForEqs('entrflx')['ybu'],
                            params.getForEqs('entrflx')['ybd'],
                            params.getForEqs('entrflx')['ilg'])

    def execTTflx(self):
        params = self.params

        # instantiate 		
        ransTTflx = fttx.TemperatureFluxResolutionStudy(params.getForProp('prop')['eht_data'],
                                                        params.getForProp('prop')['ig'],
                                                        params.getForProp('prop')['intc'],
                                                        params.getForProp('prop')['prefix'])

        ransTTflx.plot_fttx(params.getForProp('prop')['laxis'],
                            params.getForEqs('tempflx')['xbl'],
                            params.getForEqs('tempflx')['xbr'],
                            params.getForEqs('tempflx')['ybu'],
                            params.getForEqs('tempflx')['ybd'],
                            params.getForEqs('tempflx')['ilg'])

    def execHHflx(self):
        params = self.params

        # instantiate 		
        ransHHflx = fhhx.EnthalpyFluxResolutionStudy(params.getForProp('prop')['eht_data'],
                                                     params.getForProp('prop')['ig'],
                                                     params.getForProp('prop')['intc'],
                                                     params.getForProp('prop')['prefix'])

        ransHHflx.plot_fhhx(params.getForProp('prop')['laxis'],
                            params.getForEqs('enthflx')['xbl'],
                            params.getForEqs('enthflx')['xbr'],
                            params.getForEqs('enthflx')['ybu'],
                            params.getForEqs('enthflx')['ybd'],
                            params.getForEqs('enthflx')['ilg'])

    def execTMSflx(self):
        params = self.params

        # instantiate 		
        ransTMSflx = a.TurbulentMassFluxResolutionStudy(params.getForProp('prop')['eht_data'],
                                                        params.getForProp('prop')['ig'],
                                                        params.getForProp('prop')['intc'],
                                                        params.getForProp('prop')['prefix'])

        ransTMSflx.plot_fddx(params.getForProp('prop')['laxis'],
                             params.getForEqs('tmsflx')['xbl'],
                             params.getForEqs('tmsflx')['xbr'],
                             params.getForEqs('tmsflx')['ybu'],
                             params.getForEqs('tmsflx')['ybd'],
                             params.getForEqs('tmsflx')['ilg'])

    def execPPxflx(self):
        params = self.params

        # instantiate 		
        ransPPxflx = fppx.PressureFluxResolutionStudy(params.getForProp('prop')['eht_data'],
                                                      params.getForProp('prop')['ig'],
                                                      params.getForProp('prop')['intc'],
                                                      params.getForProp('prop')['prefix'])

        ransPPxflx.plot_fppx(params.getForProp('prop')['laxis'],
                             params.getForEqs('pressxflx')['xbl'],
                             params.getForEqs('pressxflx')['xbr'],
                             params.getForEqs('pressxflx')['ybu'],
                             params.getForEqs('pressxflx')['ybd'],
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
        ransDDrms = ddRms.DensityRmsResolutionStudy(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    params.getForProp('prop')['prefix'])

        ransDDrms.plot_ddrms(params.getForProp('prop')['laxis'],
                             params.getForEqs('ddrms')['xbl'],
                             params.getForEqs('ddrms')['xbr'],
                             params.getForEqs('ddrms')['ybu'],
                             params.getForEqs('ddrms')['ybd'],
                             params.getForEqs('ddrms')['ilg'])

    def execBuoyancy(self):
        params = self.params

        # instantiate
        ransBuoyancy = buoy.BuoyancyResolutionStudy(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['ieos'],
                                                    params.getForProp('prop')['intc'],
                                                    params.getForProp('prop')['prefix'])

        ransBuoyancy.plot_buoyancy(params.getForProp('prop')['laxis'],
                                   params.getForEqs('buoy')['xbl'],
                                   params.getForEqs('buoy')['xbr'],
                                   params.getForEqs('buoy')['ybu'],
                                   params.getForEqs('buoy')['ybd'],
                                   params.getForEqs('buoy')['ilg'])


    def execDilatation(self):
        params = self.params

        # instantiate
        ransDilatation = divu.DivuResolutionStudy(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    params.getForProp('prop')['prefix'])

        ransDilatation.plot_divu(params.getForProp('prop')['laxis'],
                                 params.getForEqs('divu')['xbl'],
                                 params.getForEqs('divu')['xbr'],
                                 params.getForEqs('divu')['ybu'],
                                 params.getForEqs('divu')['ybd'],
                                 params.getForEqs('divu')['ilg'])

    def execDivFrho(self):
        params = self.params

        # instantiate
        ransDivFrho = divfrho.DivFrhoResolutionStudy(params.getForProp('prop')['eht_data'],
                                                    params.getForProp('prop')['ig'],
                                                    params.getForProp('prop')['intc'],
                                                    params.getForProp('prop')['prefix'])

        ransDivFrho.plot_divfrho(params.getForProp('prop')['laxis'],
                                 params.getForEqs('divfrho')['xbl'],
                                 params.getForEqs('divfrho')['xbr'],
                                 params.getForEqs('divfrho')['ybu'],
                                 params.getForEqs('divfrho')['ybd'],
                                 params.getForEqs('divfrho')['ilg'])

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
