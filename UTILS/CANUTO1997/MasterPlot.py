import CANUTO1997.ContinuityEquationWithFavrianDilatation as contfdil
import CANUTO1997.MomentumEquationX as momx
import CANUTO1997.MomentumEquationY as momy
import CANUTO1997.MomentumEquationZ as momz
import CANUTO1997.KineticEnergyFlux as keflx
import CANUTO1997.TurbulentMassFlux as a
import CANUTO1997.LuminosityEquation as lumi
import CANUTO1997.TurbulentKineticEnergyEquation as tke

import matplotlib.pyplot as plt

class MasterPlot():

    def __init__(self, params):
        self.params = params

    def execRho(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransCONT = contfdil.ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],
                                                                    params.getForProp('prop')['ig'],
                                                                    params.getForProp('prop')['fext'],
                                                                    params.getForProp('prop')['intc'],
                                                                    params.getForProp('prop')['prefix'])

        # plot density
        ransCONT.plot_rho(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('rho')['xbl'],
                          params.getForEqs('rho')['xbr'],
                          params.getForEqs('rho')['ybu'],
                          params.getForEqs('rho')['ybd'],
                          params.getForEqs('rho')['ilg'])

        # ransCONT.plot_mm_vs_MM(params.getForProp('prop')['laxis'],
        #                       params.getForEqs('rho')['xbl'],
        #                       params.getForEqs('rho')['xbr'],
        #                       params.getForEqs('rho')['ybu'],
        #                       params.getForEqs('rho')['ybd'],
        #                       params.getForEqs('rho')['ilg'])

    def execContEq(self, bconv, tconv):
        params = self.params

        # instantiate 
        ransCONT = contfdil.ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],
                                                                    params.getForProp('prop')['ig'],
                                                                    params.getForProp('prop')['fext'],
                                                                    params.getForProp('prop')['intc'],
                                                                    params.getForProp('prop')['prefix'])

        # plot continuity equation						       
        ransCONT.plot_continuity_equation(params.getForProp('prop')['laxis'],
                                          bconv, tconv,
                                          params.getForEqs('conteq')['xbl'],
                                          params.getForEqs('conteq')['xbr'],
                                          params.getForEqs('conteq')['ybu'],
                                          params.getForEqs('conteq')['ybd'],
                                          params.getForEqs('conteq')['ilg'])

    def execContEqBar(self):
        params = self.params

        # instantiate 
        ransCONT = contfdil.ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],
                                                                    params.getForProp('prop')['ig'],
                                                                    params.getForProp('prop')['fext'],
                                                                    params.getForProp('prop')['intc'],
                                                                    params.getForProp('prop')['prefix'])

        # plot continuity equation integral budget					       
        ransCONT.plot_continuity_equation_integral_budget(params.getForProp('prop')['laxis'],
                                                          params.getForEqsBar('conteqBar')['xbl'],
                                                          params.getForEqsBar('conteqBar')['xbr'],
                                                          params.getForEqsBar('conteqBar')['ybu'],
                                                          params.getForEqsBar('conteqBar')['ybd'])

    def execMomx(self, bconv, tconv):
        params = self.params

        # instantiate
        ransMomx = momx.MomentumEquationX(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        ransMomx.plot_momentum_x(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('momex')['xbl'],
                                 params.getForEqs('momex')['xbr'],
                                 params.getForEqs('momex')['ybu'],
                                 params.getForEqs('momex')['ybd'],
                                 params.getForEqs('momex')['ilg'])

    def execMomxEq(self, bconv, tconv):
        params = self.params

        # instantiate
        ransMomx = momx.MomentumEquationX(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        ransMomx.plot_momentum_equation_x(params.getForProp('prop')['laxis'],
                                          bconv, tconv,
                                          params.getForEqs('momxeq')['xbl'],
                                          params.getForEqs('momxeq')['xbr'],
                                          params.getForEqs('momxeq')['ybu'],
                                          params.getForEqs('momxeq')['ybd'],
                                          params.getForEqs('momxeq')['ilg'])

    def execMomy(self, bconv, tconv):
        params = self.params

        # instantiate
        ransMomy = momy.MomentumEquationY(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        ransMomy.plot_momentum_y(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('momey')['xbl'],
                                 params.getForEqs('momey')['xbr'],
                                 params.getForEqs('momey')['ybu'],
                                 params.getForEqs('momey')['ybd'],
                                 params.getForEqs('momey')['ilg'])

    def execMomyEq(self, bconv, tconv):
        params = self.params

        # instantiate
        ransMomy = momy.MomentumEquationY(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        ransMomy.plot_momentum_equation_y(params.getForProp('prop')['laxis'],
                                          bconv, tconv,
                                          params.getForEqs('momyeq')['xbl'],
                                          params.getForEqs('momyeq')['xbr'],
                                          params.getForEqs('momyeq')['ybu'],
                                          params.getForEqs('momyeq')['ybd'],
                                          params.getForEqs('momyeq')['ilg'])

    def execMomz(self, bconv, tconv):
        params = self.params

        # instantiate
        ransMomz = momz.MomentumEquationZ(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        ransMomz.plot_momentum_z(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('momez')['xbl'],
                                 params.getForEqs('momez')['xbr'],
                                 params.getForEqs('momez')['ybu'],
                                 params.getForEqs('momez')['ybd'],
                                 params.getForEqs('momez')['ilg'])

    def execMomzEq(self, bconv, tconv):
        params = self.params

        # instantiate
        ransMomz = momz.MomentumEquationZ(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        ransMomz.plot_momentum_equation_z(params.getForProp('prop')['laxis'],
                                          bconv, tconv,
                                          params.getForEqs('momzeq')['xbl'],
                                          params.getForEqs('momzeq')['xbr'],
                                          params.getForEqs('momzeq')['ybu'],
                                          params.getForEqs('momzeq')['ybd'],
                                          params.getForEqs('momzeq')['ilg'])

    def execKeflx(self, bconv, tconv):
        params = self.params

        # instantiate
        ransKeflx = keflx.KineticEnergyFlux(params.getForProp('prop')['eht_data'],
                                                                    params.getForProp('prop')['ig'],
                                                                    params.getForProp('prop')['fext'],
                                                                    params.getForProp('prop')['intc'],
                                                                    params.getForProp('prop')['prefix'])

        # plot density
        ransKeflx.plot_keflx(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('keflx')['xbl'],
                          params.getForEqs('keflx')['xbr'],
                          params.getForEqs('keflx')['ybu'],
                          params.getForEqs('keflx')['ybd'],
                          params.getForEqs('keflx')['ilg'])


    def execTMSflx(self, bconv, tconv, lc):
        params = self.params

        # instantiate
        ransTMSflx = a.TurbulentMassFlux(params.getForProp('prop')['eht_data'],
                                                 params.getForProp('prop')['ig'],
                                                 params.getForProp('prop')['intc'],
                                                 params.getForProp('prop')['prefix'],
                                                 lc)

        ransTMSflx.plot_a(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('tmsflx')['xbl'],
                          params.getForEqs('tmsflx')['xbr'],
                          params.getForEqs('tmsflx')['ybu'],
                          params.getForEqs('tmsflx')['ybd'],
                          params.getForEqs('tmsflx')['ilg'])


    def execLumiEq(self, tke_diss, bconv, tconv):
        params = self.params

        # instantiate
        ranslumi = lumi.LuminosityEquation(params.getForProp('prop')['eht_data'],
                                                       params.getForProp('prop')['ig'],
                                                       params.getForProp('prop')['ieos'],
                                                       params.getForProp('prop')['fext'],
                                                       params.getForProp('prop')['intc'],
                                                       tke_diss, bconv, tconv,
                                                       params.getForProp('prop')['prefix'])

        # plot luminosity equation exact
        ranslumi.plot_luminosity_equation_exact(params.getForProp('prop')['laxis'],
                                                    params.getForEqs('lueq')['xbl'],
                                                    params.getForEqs('lueq')['xbr'],
                                                    params.getForEqs('lueq')['ybu'],
                                                    params.getForEqs('lueq')['ybd'],
                                                    params.getForEqs('lueq')['ilg'])


    def execTKEeq(self, kolmrate, bconv, tconv):
        params = self.params

        # instantiate
        ransTke = tke.TurbulentKineticEnergyEquation(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['intc'],
                                          -kolmrate,
                                          params.getForProp('prop')['prefix'])

        # plot kinetic energy equation
        ransTke.plot_tke_equation(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('tkeq')['xbl'],
                                params.getForEqs('tkeq')['xbr'],
                                params.getForEqs('tkeq')['ybu'],
                                params.getForEqs('tkeq')['ybd'],
                                params.getForEqs('tkeq')['ilg'])

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


