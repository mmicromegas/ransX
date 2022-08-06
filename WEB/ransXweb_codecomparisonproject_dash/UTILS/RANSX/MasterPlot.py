from EQUATIONS.ContinuityEquationWithFavrianDilatation import ContinuityEquationWithFavrianDilatation
from EQUATIONS.ContinuityEquationWithMassFlux import ContinuityEquationWithMassFlux
from EQUATIONS.MomentumEquationX import MomentumEquationX
from EQUATIONS.XtransportEquation import XtransportEquation
from EQUATIONS.XvarianceEquation import XvarianceEquation
from EQUATIONS.TurbulentKineticEnergyEquation import TurbulentKineticEnergyEquation
from EQUATIONS.InternalEnergyEquation import InternalEnergyEquation
from EQUATIONS.SourceVel import SourceVel

from EQUATIONS.FOR_COMPARISON.VelTke import VelTke
from EQUATIONS.FOR_COMPARISON.CompositionFlux import CompositionFlux

class MasterPlot():

    def __init__(self, params):
        self.params = params
        #plt.close()

    def execContEq(self, bconv, tconv):
        params = self.params

        # instantiate
        ransCONT = ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['eht_data'],
                                                           params.getForProp('prop')['plabel'],
                                                           params.getForProp('prop')['code'],
                                                           params.getForProp('prop')['ig'],
                                                           params.getForProp('prop')['fext'],
                                                           params.getForProp('prop')['intc'],
                                                           params.getForProp('prop')['nsdim'],
                                                           params.getForProp('prop')['prefix'])

        # figRANS
        fig = ransCONT.plot_ContinuityEquationWithFavrianDilatation(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('rho')['xbl'],
                          params.getForEqs('rho')['xbr'],
                          params.getForEqs('rho')['ybu'],
                          params.getForEqs('rho')['ybd'],
                          params.getForEqs('conteq')['ybu'],
                          params.getForEqs('conteq')['ybd'],
                          params.getForEqsBar('conteqBar')['ybu'],
                          params.getForEqsBar('conteqBar')['ybd'],
                          params.getForEqs('rho')['ilg'])

        return fig


    def execContEqFdd(self, bconv, tconv):
        params = self.params

        # instantiate
        ransCONT = ContinuityEquationWithMassFlux(params.getForProp('prop')['eht_data'],
                                                           params.getForProp('prop')['plabel'],
                                                           params.getForProp('prop')['code'],
                                                           params.getForProp('prop')['ig'],
                                                           params.getForProp('prop')['fext'],
                                                           params.getForProp('prop')['intc'],
                                                           params.getForProp('prop')['nsdim'],
                                                           params.getForProp('prop')['prefix'])

        # figRANS
        fig = ransCONT.plot_ContinuityEquationWithMassFlux(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('rho')['xbl'],
                          params.getForEqs('rho')['xbr'],
                          params.getForEqs('rho')['ybu'],
                          params.getForEqs('rho')['ybd'],
                          params.getForEqs('conteqfdd')['ybu'],
                          params.getForEqs('conteqfdd')['ybd'],
                          params.getForEqsBar('conteqfddBar')['ybu'],
                          params.getForEqsBar('conteqfddBar')['ybd'],
                          params.getForEqs('rho')['ilg'])

        return fig

    def execMomex(self, bconv, tconv):
        params = self.params

        # instantiate
        ransMomx = MomentumEquationX(params.getForProp('prop')['eht_data'],
                                                           params.getForProp('prop')['plabel'],
                                                           params.getForProp('prop')['code'],
                                                           params.getForProp('prop')['ig'],
                                                           params.getForProp('prop')['fext'],
                                                           params.getForProp('prop')['intc'],
                                                           params.getForProp('prop')['nsdim'],
                                                           params.getForProp('prop')['prefix'])

        # figRANS
        fig = ransMomx.plot_MomentumEquationX(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('momex')['xbl'],
                          params.getForEqs('momex')['xbr'],
                          params.getForEqs('momex')['ybu'],
                          params.getForEqs('momex')['ybd'],
                          params.getForEqs('momxeq')['ybu'],
                          params.getForEqs('momxeq')['ybd'],
                          params.getForEqsBar('momxeqBar')['ybu'],
                          params.getForEqsBar('momxeqBar')['ybd'],
                          params.getForEqs('momex')['ilg'])

        return fig

    def execXtrsEq(self, inuc, element, x, bconv, tconv):
        params = self.params

        # instantiate
        ransXtra = XtransportEquation(params.getForProp('prop')['eht_data'],
                                      params.getForProp('prop')['plabel'],
                                      params.getForProp('prop')['code'],
                                      params.getForProp('prop')['ig'],
                                      params.getForProp('prop')['fext'],
                                      inuc, element, bconv, tconv,
                                      params.getForProp('prop')['intc'],
                                      params.getForProp('prop')['nsdim'],
                                      params.getForProp('prop')['prefix'])

        # figRANS
        fig = ransXtra.plot_XtransportEquation(params.getForProp('prop')['laxis'],bconv, tconv,
                        params.getForEqs(x)['xbl'],
                        params.getForEqs(x)['xbr'],
                        params.getForEqs('x_' + element)['ybu'],
                        params.getForEqs('x_' + element)['ybd'],
                        params.getForEqs('xtrseq_' + element)['ybu'],
                        params.getForEqs('xtrseq_' + element)['ybd'],
                        params.getForEqsBar('xtrseq_' + element + 'Bar')['ybu'],
                        params.getForEqsBar('xtrseq_' + element + 'Bar')['ybd'],
                        params.getForEqs(x)['ilg'])

        return fig

    def execXvarEq(self, inuc, element, x, bconv, tconv):
        params = self.params

        # instantiate
        ransXvar = XvarianceEquation(params.getForProp('prop')['eht_data'],
                                      params.getForProp('prop')['plabel'],
                                      params.getForProp('prop')['code'],
                                      params.getForProp('prop')['ig'],
                                      params.getForProp('prop')['fext'],
                                      inuc, element, bconv, tconv,
                                      params.getForProp('prop')['intc'],
                                      params.getForProp('prop')['nsdim'],
                                      params.getForProp('prop')['prefix'])

        # figRANS
        fig = ransXvar.plot_XvarianceEquation(params.getForProp('prop')['laxis'],bconv, tconv,
                        params.getForEqs(x)['xbl'],
                        params.getForEqs(x)['xbr'],
                        params.getForEqs('xvar_' + element)['ybu'],
                        params.getForEqs('xvar_' + element)['ybd'],
                        params.getForEqs('xvareq_' + element)['ybu'],
                        params.getForEqs('xvareq_' + element)['ybd'],
                        params.getForEqsBar('xvareq_' + element + 'Bar')['ybu'],
                        params.getForEqsBar('xvareq_' + element + 'Bar')['ybd'],
                        params.getForEqs(x)['ilg'])

        return fig



    def execTkeEq(self, bconv, tconv):
        params = self.params

        # instantiate
        ransTke = TurbulentKineticEnergyEquation(params.getForProp('prop')['eht_data'],
                                                 params.getForProp('prop')['plabel'],
                                                 params.getForProp('prop')['code'],
                                                 params.getForProp('prop')['ig'],
                                                 params.getForProp('prop')['intc'],
                                                 params.getForProp('prop')['nsdim'],
                                                 params.getForProp('prop')['prefix'])

        # plot turbulent kinetic energy
        fig = ransTke.plot_TurbulentKineticEnergyEquation(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('tkie')['xbl'],
                          params.getForEqs('tkie')['xbr'],
                          params.getForEqs('tkie')['ybu'],
                          params.getForEqs('tkie')['ybd'],
                          params.getForEqs('tkeeq')['ybu'],
                          params.getForEqs('tkeeq')['ybd'],
                          params.getForEqsBar('tkeeqBar')['ybu'],
                          params.getForEqsBar('tkeeqBar')['ybd'],
                          params.getForEqs('tkie')['ilg'])

        return fig


    def execEiEq(self, bconv, tconv, tke_diss):
        params = self.params

        # instantiate
        ransEi = InternalEnergyEquation(params.getForProp('prop')['eht_data'],
                                                 params.getForProp('prop')['plabel'],
                                                 params.getForProp('prop')['code'],
                                                 params.getForProp('prop')['ig'],
                                                 params.getForProp('prop')['intc'],
                                                 params.getForProp('prop')['nsdim'],
                                                 tke_diss,
                                                 params.getForProp('prop')['prefix'])

        # plot internal energy equation
        fig = ransEi.plot_InternalEnergyEquation(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('eint')['xbl'],
                          params.getForEqs('eint')['xbr'],
                          params.getForEqs('eint')['ybu'],
                          params.getForEqs('eint')['ybd'],
                          params.getForEqs('eieq')['ybu'],
                          params.getForEqs('eieq')['ybd'],
                          params.getForEqsBar('eieqBar')['ybu'],
                          params.getForEqsBar('eieqBar')['ybd'],
                          params.getForEqs('eint')['ilg'])

        return fig


    def execSrcvel(self, bconv, tconv):
        params = self.params

        # instantiate
        ransSrcvel = SourceVel(params.getForProp('prop')['eht_data'],
                                                           params.getForProp('prop')['plabel'],
                                                           params.getForProp('prop')['code'],
                                                           params.getForProp('prop')['ig'],
                                                           params.getForProp('prop')['fext'],
                                                           params.getForProp('prop')['intc'],
                                                           params.getForProp('prop')['nsdim'],
                                                           params.getForProp('prop')['prefix'])

        # figRANS
        fig = ransSrcvel.plot_SourceVel(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('velbgr')['xbl'],
                          params.getForEqs('velbgr')['xbr'],
                          params.getForEqs('velbgr')['ybu'],
                          params.getForEqs('velbgr')['ybd'],
                          params.getForEqs('enuc')['ybu'],
                          params.getForEqs('enuc')['ybd'],
                          params.getForEqs('velmlt')['ybu'],
                          params.getForEqs('velmlt')['ybd'],
                          params.getForEqs('velmlt')['ilg'])

        return fig

    def execUxComparison(self):
        params = self.params

        # instantiate
        ransX = VelTke(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['plabel'],
                                               params.getForProp('prop')['code_list'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        # figRANS
        fig = ransX.plot_velTke(params.getForProp('prop')['laxis'],
                          params.getForEqs('uxrms')['xbl'],
                          params.getForEqs('uxrms')['xbr'],
                          params.getForEqs('uxrms')['ybu'],
                          params.getForEqs('uxrms')['ybd'],
                          params.getForEqs('uyrms')['ybu'],
                          params.getForEqs('uyrms')['ybd'],
                          params.getForEqs('uzrms')['ybu'],
                          params.getForEqs('uzrms')['ybd'],
                          params.getForEqs('tke')['ybu'],
                          params.getForEqs('tke')['ybd'])


        return fig

    def execXfluxComparison(self):
        params = self.params

        # instantiate
        ransX = CompositionFlux(params.getForProp('prop')['eht_data'],
                                               params.getForProp('prop')['plabel'],
                                               params.getForProp('prop')['code_list'],
                                               params.getForProp('prop')['ig'],
                                               params.getForProp('prop')['intc'],
                                               params.getForProp('prop')['prefix'])

        # figRANS
        fig = ransX.plot_composition_flux(params.getForProp('prop')['laxis'],
                          params.getForEqs('xflux_fluid1')['xbl'],
                          params.getForEqs('xflux_fluid1')['xbr'],
                          params.getForEqs('xflux_fluid1')['ybu'],
                          params.getForEqs('xflux_fluid1')['ybd'],
                          params.getForEqs('xflux_fluid2')['ybu'],
                          params.getForEqs('xflux_fluid2')['ybd'])

        return fig