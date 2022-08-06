from EQUATIONS.ContinuityEquationWithMassFlux import ContinuityEquationWithMassFlux
from EQUATIONS.ContinuityEquationWithFavrianDilatation import ContinuityEquationWithFavrianDilatation

from EQUATIONS.MomentumEquationX import MomentumEquationX
from EQUATIONS.EntropyEquation import EntropyEquation
from EQUATIONS.InternalEnergyEquation import InternalEnergyEquation

from EQUATIONS.TurbulentKineticEnergyEquation import TurbulentKineticEnergyEquation

from EQUATIONS.RelativeRMSflct import RelativeRMSflct
from EQUATIONS.TemperatureEquation import TemperatureEquation
from EQUATIONS.NuclearEnergyProduction import NuclearEnergyProduction

from EQUATIONS.VelocitiesMeanExp import VelocitiesMeanExp
from EQUATIONS.VelocitiesMLTturb import VelocitiesMLTturb
from EQUATIONS.TemperatureGradients import TemperatureGradients

from EQUATIONS.XtransportEquation import XtransportEquation

import matplotlib.pyplot as plt
import mpld3
import sys


class MasterPlot():

    def __init__(self, params):
        self.params = params

    def execContEq(self, outputFile, bconv, tconv, plabel, nsdim):
        params = self.params

        # instantiate
        ransCONT = ContinuityEquationWithFavrianDilatation(outputFile, params.getForProp('prop')['eht_data'],
                                                           params.getForProp('prop')['ig'],
                                                           params.getForProp('prop')['fext'],
                                                           params.getForProp('prop')['intc'],
                                                           params.getForProp('prop')['nsdim'],
                                                           params.getForProp('prop')['prefix'])

        # create FIGURE
        fig = plt.figure(figsize=(13, 4))

        # fig.tight_layout()

        # display subplot
        plt.subplot(1, 3, 1)
        plt.subplots_adjust(left=0.053)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        if plabel == 'ccptwo' and nsdim == 3:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e28
        elif plabel == 'ccptwo' and nsdim == 2:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e28
        elif plabel == '3d-oburn-14ele' or plabel == '3d-oburn-25ele':
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e28
        elif plabel == '3d-neshellBoost10x-25ele':
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e25
        elif plabel == '3d-heflashBoost100x-6ele':
            xsc = 1.e8
            yscContEq = 1.e0
            yscRho = 1.e5
            fcIntegralBudget = 1.e28
        elif plabel == '3d-thpulse-15ele':
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e28
        else:
            print('MasterPlot.py - plabel unknown')
            sys.exit()

        # plot continuity equation
        ransCONT.plot_continuity_equation(params.getForProp('prop')['laxis'],
                                          bconv, tconv,
                                          params.getForEqs('conteq')['xbl'],
                                          params.getForEqs('conteq')['xbr'],
                                          params.getForEqs('conteq')['ybu'],
                                          params.getForEqs('conteq')['ybd'],
                                          params.getForEqs('conteq')['ilg'],
                                          xsc, yscContEq)

        # display subplot
        plt.subplot(1, 3, 2)

        # plot density
        ransCONT.plot_rho(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('rho')['xbl'],
                          params.getForEqs('rho')['xbr'],
                          params.getForEqs('rho')['ybu'],
                          params.getForEqs('rho')['ybd'],
                          params.getForEqs('rho')['ilg'],
                          xsc, yscRho)

        # display subplot
        ax = plt.subplot(1, 3, 3)

        # plot continuity equation integral budget
        ransCONT.plot_continuity_equation_integral_budget(plabel, ax, params.getForProp('prop')['laxis'],
                                                          params.getForEqsBar('conteqBar')['xbl'],
                                                          params.getForEqsBar('conteqBar')['xbr'],
                                                          params.getForEqsBar('conteqBar')['ybu'],
                                                          params.getForEqsBar('conteqBar')['ybd'],
                                                          fcIntegralBudget)

        html_str = mpld3.fig_to_html(fig)
        Html_file = open(outputFile, "w")
        Html_file.write(html_str)
        Html_file.close()

    def execContFddEq(self, outputFile, bconv, tconv, plabel, nsdim):
        params = self.params

        # instantiate
        ransCONTfdd = ContinuityEquationWithMassFlux(outputFile, params.getForProp('prop')['eht_data'],
                                                     params.getForProp('prop')['ig'],
                                                     params.getForProp('prop')['fext'],
                                                     params.getForProp('prop')['intc'],
                                                     params.getForProp('prop')['nsdim'],
                                                     params.getForProp('prop')['prefix'])

        if plabel == 'ccptwo' and nsdim == 3:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e28
        elif plabel == 'ccptwo' and nsdim == 2:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e28
        elif plabel == '3d-oburn-14ele' or plabel == '3d-oburn-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e28
        elif plabel == '3d-neshellBoost10x-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e25
        elif plabel == '3d-heflashBoost100x-6ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscContEq = 1.e0
            yscRho = 1.e5
            fcIntegralBudget = 1.e28
        elif plabel == '3d-thpulse-15ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscContEq = 1.e2
            yscRho = 1.e6
            fcIntegralBudget = 1.e28
        else:
            print('MasterPlot.py - plabel unknown')
            sys.exit()

        # create FIGURE
        fig = plt.figure(figsize=(13, 4))
        # fig.tight_layout()

        # display subplot
        plt.subplot(1, 3, 1)
        plt.subplots_adjust(left=0.053)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # plot continuity equation
        ransCONTfdd.plot_continuity_equation(params.getForProp('prop')['laxis'],
                                             bconv, tconv,
                                             params.getForEqs('conteqfdd')['xbl'],
                                             params.getForEqs('conteqfdd')['xbr'],
                                             params.getForEqs('conteqfdd')['ybu'],
                                             params.getForEqs('conteqfdd')['ybd'],
                                             params.getForEqs('conteqfdd')['ilg'],
                                             xsc, yscContEq)

        # display subplot
        plt.subplot(1, 3, 2)

        # plot density
        ransCONTfdd.plot_rho(params.getForProp('prop')['laxis'],
                             bconv, tconv,
                             params.getForEqs('rho')['xbl'],
                             params.getForEqs('rho')['xbr'],
                             params.getForEqs('rho')['ybu'],
                             params.getForEqs('rho')['ybd'],
                             params.getForEqs('rho')['ilg'],
                             xsc, yscRho)

        # display subplot
        ax = plt.subplot(1, 3, 3)

        # plot continuity equation integral budget
        ransCONTfdd.plot_continuity_equation_integral_budget(plabel, ax, params.getForProp('prop')['laxis'],
                                                             params.getForEqsBar('conteqBar')['xbl'],
                                                             params.getForEqsBar('conteqBar')['xbr'],
                                                             params.getForEqsBar('conteqBar')['ybu'],
                                                             params.getForEqsBar('conteqBar')['ybd'],
                                                             fcIntegralBudget)

        html_str = mpld3.fig_to_html(fig)
        Html_file = open(outputFile, "w")
        Html_file.write(html_str)
        Html_file.close()

    def execTkeEq(self, outputFile, kolmdissrate, bconv, tconv, super_ad_i, super_ad_o, plabel, nsdim):
        params = self.params

        # instantiate 		
        ransTke = TurbulentKineticEnergyEquation(outputFile, params.getForProp('prop')['eht_data'],
                                                 params.getForProp('prop')['ig'],
                                                 params.getForProp('prop')['intc'],
                                                 params.getForProp('prop')['nsdim'],
                                                 kolmdissrate, bconv, tconv,
                                                 super_ad_i, super_ad_o,
                                                 params.getForProp('prop')['prefix'])

        if plabel == 'ccptwo' and nsdim == 3:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e19
            yscTke = 1.e14
            fcIntegralBudget = 1.e44
        elif plabel == 'ccptwo' and nsdim == 2:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e19
            yscTke = 1.e14
            fcIntegralBudget = 1.e44
        elif plabel == '3d-oburn-14ele' or plabel == '3d-oburn-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e18
            yscTke = 1.e13
            fcIntegralBudget = 1.e45
        elif plabel == '3d-neshellBoost10x-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e18
            yscTke = 1.e12
            fcIntegralBudget = 1.e38
        elif plabel == '3d-heflashBoost100x-6ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e16
            yscTke = 1.e12
            fcIntegralBudget = 1.e43
        elif plabel == '3d-thpulse-15ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e2
            yscTke = 1.e6
            fcIntegralBudget = 1.e28
        else:
            print('MasterPlot.py - plabel unknown')
            sys.exit()

        # create FIGURE
        fig = plt.figure(figsize=(13, 4))
        # fig.tight_layout()

        # display subplot
        plt.subplot(1, 3, 1)
        plt.subplots_adjust(left=0.063)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)


        # plot turbulent kinetic energy equation
        ransTke.plot_tke_equation(params.getForProp('prop')['laxis'],
                                  params.getForEqs('tkeeq')['xbl'],
                                  params.getForEqs('tkeeq')['xbr'],
                                  params.getForEqs('tkeeq')['ybu'],
                                  params.getForEqs('tkeeq')['ybd'],
                                  params.getForEqs('tkeeq')['ilg'], xsc, yscTkeEq)

        # display subplot
        plt.subplot(1, 3, 2)

        # plot density
        ransTke.plot_tke(params.getForProp('prop')['laxis'],
                         bconv, tconv,
                         params.getForEqs('tkie')['xbl'],
                         params.getForEqs('tkie')['xbr'],
                         params.getForEqs('tkie')['ybu'],
                         params.getForEqs('tkie')['ybd'],
                         params.getForEqs('tkie')['ilg'],
                         xsc, yscTke)

        # display subplot
        ax = plt.subplot(1, 3, 3)

        # plot continuity equation integral budget
        ransTke.plot_tke_equation_integral_budget(ax, plabel, params.getForProp('prop')['laxis'],
                                                  params.getForEqsBar('tkeeqBar')['xbl'],
                                                  params.getForEqsBar('tkeeqBar')['xbr'],
                                                  params.getForEqsBar('tkeeqBar')['ybu'],
                                                  params.getForEqsBar('tkeeqBar')['ybd'],
                                                  fcIntegralBudget)

        html_str = mpld3.fig_to_html(fig)
        Html_file = open(outputFile, "w")
        Html_file.write(html_str)
        Html_file.close()

    def execXtrsEq(self, outputFile, inuc, element, x, bconv, tconv, super_ad_i, super_ad_o, plabel):
        params = self.params

        # instantiate
        ransXtra = XtransportEquation(params.getForProp('prop')['eht_data'],
                                      params.getForProp('prop')['plabel'],
                                      params.getForProp('prop')['ig'],
                                      params.getForProp('prop')['fext'],
                                      inuc, element, bconv, tconv, super_ad_i, super_ad_o,
                                      params.getForProp('prop')['intc'],
                                      params.getForProp('prop')['nsdim'],
                                      params.getForProp('prop')['prefix'])

        if plabel == 'ccptwo':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscXeq = 1.e3
            yscX = 1.e0
            fcIntegralBudget = 1.e29
        elif plabel == '3d-oburn-14ele' or plabel == '3d-oburn-25ele':
            xsc = 1.e8
            if element == 'neut':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
                yscXeq = 1.e-16
                yscX = 1.e0
                fcIntegralBudget = 1.e12
            if element == 'prot':
                yscXeq = 1.e-6
                yscX = 1.e0
                fcIntegralBudget = 1.e19
            if element == 'he4':
                yscXeq = 1.e-4
                yscX = 1.e0
                fcIntegralBudget = 1.e20
            if element == 'c12':
                yscXeq = 1.e-1
                yscX = 1.e0
                fcIntegralBudget = 1.e25
            if element == 'o16':
                yscXeq = 1.e2
                yscX = 1.e0
                fcIntegralBudget = 1.e28
            if element == 'ne20':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e28
            if element == 'na23':
                yscXeq = 1.e0
                yscX = 1.e0
                fcIntegralBudget = 1.e26
            if element == 'mg24':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e28
            if element == 'si28':
                yscXeq = 1.e2
                yscX = 1.e0
                fcIntegralBudget = 1.e28
            if element == 'p31':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e25
            if element == 's32':
                yscXeq = 1.e2
                yscX = 1.e0
                fcIntegralBudget = 1.e28
            if element == 's34':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e26
            if element == 'cl35':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e25
            if element == 'ar36':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e28
            if element == 'ar38':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e27
            if element == 'k39':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e27
            if element == 'ca40':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e27
            if element == 'ca42':
                yscXeq = 1.e-1
                yscX = 1.e0
                fcIntegralBudget = 1.e26
            if element == 'ti44':
                yscXeq = 1.e-2
                yscX = 1.e0
                fcIntegralBudget = 1.e24
            if element == 'ti46':
                yscXeq = 1.e-2
                yscX = 1.e0
                fcIntegralBudget = 1.e25
            if element == 'cr48':
                yscXeq = 1.e-5
                yscX = 1.e0
                fcIntegralBudget = 1.e22
            if element == 'cr50':
                yscXeq = 1.e-3
                yscX = 1.e0
                fcIntegralBudget = 1.e25
            if element == 'fe52':
                yscXeq = 1.e-7
                yscX = 1.e0
                fcIntegralBudget = 1.e21
            if element == 'fe54':
                yscXeq = 1.e-3
                yscX = 1.e0
                fcIntegralBudget = 1.e24
            if element == 'ni56':
                yscXeq = 1.e-8
                yscX = 1.e0
                fcIntegralBudget = 1.e19
        elif plabel == '3d-neshellBoost10x-25ele':
            xsc = 1.e8
            if element == 'neut':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
                yscXeq = 1.e-27
                yscX = 1.e0
                fcIntegralBudget = 1.e-4
            if element == 'prot':
                yscXeq = 1.e-6
                yscX = 1.e0
                fcIntegralBudget = 1.e16
            if element == 'he4':
                yscXeq = 1.e-4
                yscX = 1.e0
                fcIntegralBudget = 1.e15
            if element == 'c12':
                yscXeq = 1.e-9
                yscX = 1.e0
                fcIntegralBudget = 1.e13
            if element == 'o16':
                yscXeq = 1.e2
                yscX = 1.e0
                fcIntegralBudget = 1.e24
            if element == 'ne20':
                yscXeq = 1.e1
                yscX = 1.e0
                fcIntegralBudget = 1.e25
            if element == 'na23':
                yscXeq = 1.e-5
                yscX = 1.e0
                fcIntegralBudget = 1.e17
            if element == 'mg24':
                yscXeq = 1.e2
                yscX = 1.e0
                fcIntegralBudget = 1.e25
            if element == 'si28':
                yscXeq = 1.e2
                yscX = 1.e0
                fcIntegralBudget = 1.e23
            if element == 'p31':
                yscXeq = 1.e-4
                yscX = 1.e0
                fcIntegralBudget = 1.e17
            if element == 's32':
                yscXeq = 1.e-1
                yscX = 1.e0
                fcIntegralBudget = 1.e21
            if element == 's34':
                yscXeq = 1.e-10
                yscX = 1.e0
                fcIntegralBudget = 1.e11
            if element == 'cl35':
                yscXeq = 1.e-7
                yscX = 1.e0
                fcIntegralBudget = 1.e14
            if element == 'ar36':
                yscXeq = 1.e-6
                yscX = 1.e0
                fcIntegralBudget = 1.e16
            if element == 'ar38':
                yscXeq = 1.e-15
                yscX = 1.e0
                fcIntegralBudget = 1.e7
            if element == 'k39':
                yscXeq = 1.e-13
                yscX = 1.e0
                fcIntegralBudget = 1.e9
            if element == 'ca40':
                yscXeq = 1.e-12
                yscX = 1.e0
                fcIntegralBudget = 1.e10
            if element == 'ca42':
                yscXeq = 1.e-20
                yscX = 1.e0
                fcIntegralBudget = 1.e2
            if element == 'ti44':
                yscXeq = 1.e-21
                yscX = 1.e0
                fcIntegralBudget = 1.e2
            if element == 'ti46':
                yscXeq = 1.e-28
                yscX = 1.e0
                fcIntegralBudget = 1.e-6
            if element == 'cr48':
                yscXeq = 1.e-28
                yscX = 1.e0
                fcIntegralBudget = 1.e-6
            if element == 'cr50':
                yscXeq = 1.e-29
                yscX = 1.e0
                fcIntegralBudget = 1.e-6
            if element == 'fe52':
                yscXeq = 1.e-29
                yscX = 1.e0
                fcIntegralBudget = 1.e-6
            if element == 'fe54':
                yscXeq = 1.e-29
                yscX = 1.e0
                fcIntegralBudget = 1.e-6
            if element == 'ni56':
                yscXeq = 1.e-29
                yscX = 1.e0
                fcIntegralBudget = 1.e-6
        elif plabel == '3d-heflashBoost100x-6ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            if element == 'neut':
                yscXeq = 1.e-8
                yscX = 1.e0
                fcIntegralBudget = 1.e19
            if element == 'prot':
                yscXeq = 1.e-8
                yscX = 1.e0
                fcIntegralBudget = 1.e19
            if element == 'he4':
                yscXeq = 1.e0
                yscX = 1.e0
                fcIntegralBudget = 1.e26
            if element == 'c12':
                yscXeq = 1.e0
                yscX = 1.e0
                fcIntegralBudget = 1.e26
            if element == 'o16':
                yscXeq = 1.e-2
                yscX = 1.e0
                fcIntegralBudget = 1.e23
            if element == 'ne20':
                yscXeq = 1.e-1
                yscX = 1.e0
                fcIntegralBudget = 1.e24
        elif plabel == '3d-thpulse-15ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscXeq = 1.e2
            yscX = 1.e6
            fcIntegralBudget = 1.e28
        else:
            print('MasterPlot.py - plabel unknown')
            sys.exit()

        # create FIGURE
        fig = plt.figure(figsize=(13, 4))
        # fig.tight_layout()

        # display subplot
        plt.subplot(1, 3, 1)
        plt.subplots_adjust(left=0.063)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        ransXtra.plot_Xtransport_equation(params.getForProp('prop')['laxis'],
                                          params.getForEqs(x)['xbl'],
                                          params.getForEqs(x)['xbr'],
                                          params.getForEqs(x)['ybu'],
                                          params.getForEqs(x)['ybd'],
                                          params.getForEqs(x)['ilg'], xsc, yscXeq)

        # display subplot
        plt.subplot(1, 3, 2)

        x = 'x_' + element
        ransXtra.plot_X(params.getForProp('prop')['laxis'],
                        params.getForEqs(x)['xbl'],
                        params.getForEqs(x)['xbr'],
                        params.getForEqs(x)['ybu'],
                        params.getForEqs(x)['ybd'],
                        params.getForEqs(x)['ilg'], xsc, yscX)

        # display subplot
        ax = plt.subplot(1, 3, 3)

        x = 'xtrseq_' + element + 'Bar'
        # plot X transport equation integral budget
        ransXtra.plot_Xtransport_equation_integral_budget(ax, params.getForProp('prop')['laxis'],
                                                          params.getForEqsBar(x)['xbl'],
                                                          params.getForEqsBar(x)['xbr'],
                                                          params.getForEqsBar(x)['ybu'],
                                                          params.getForEqsBar(x)['ybd'],
                                                          fcIntegralBudget)

        html_str = mpld3.fig_to_html(fig)
        Html_file = open(outputFile, "w")
        Html_file.write(html_str)
        Html_file.close()

    def execVelNablas(self, outputFile, bconv, tconv, uconv, super_ad_i, super_ad_o, plabel, nsdim):
        params = self.params

        if plabel == 'ccptwo' and nsdim == 3:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscvT = 1.e7
            yscVexp = 1.e4
            yscNablas = 1.e-1
        elif plabel == 'ccptwo' and nsdim == 2:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscvT = 1.e7
            yscVexp = 1.e4
            yscNablas = 1.e-1
        elif plabel == '3d-oburn-14ele' or plabel == '3d-oburn-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscvT = 1.e7
            yscVexp = 1.e4
            yscNablas = 1.e-1
        elif plabel == '3d-neshellBoost10x-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscvT = 1.e6
            yscVexp = 1.e4
            yscNablas = 1.e-1
        elif plabel == '3d-heflashBoost100x-6ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscvT = 1.e6
            yscVexp = 1.e4
            yscNablas = 1.e-1
        elif plabel == '3d-thpulse-15ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e2
            yscTke = 1.e6
            fcIntegralBudget = 1.e28
        else:
            print('MasterPlot.py - plabel unknown')
            sys.exit()


        # create FIGURE
        fig = plt.figure(figsize=(13, 4))
        # fig.tight_layout()

        # display subplot
        plt.subplot(1, 3, 1)
        plt.subplots_adjust(left=0.053)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # instantiate
        ransVelMLTturb = VelocitiesMLTturb(params.getForProp('prop')['eht_data'],
                                           params.getForProp('prop')['ig'],
                                           params.getForProp('prop')['fext'],
                                           params.getForProp('prop')['ieos'],
                                           bconv, tconv, uconv, super_ad_i, super_ad_o,
                                           params.getForProp('prop')['intc'],
                                           params.getForProp('prop')['nsdim'],
                                           params.getForProp('prop')['prefix'])

        ransVelMLTturb.plot_velocities(params.getForProp('prop')['laxis'],
                                       params.getForEqs('velmlt')['xbl'],
                                       params.getForEqs('velmlt')['xbr'],
                                       params.getForEqs('velmlt')['ybu'],
                                       params.getForEqs('velmlt')['ybd'],
                                       params.getForEqs('velmlt')['ilg'], xsc, yscvT)

        # instantiate
        ransVelmeanExp = VelocitiesMeanExp(params.getForProp('prop')['eht_data'],
                                           params.getForProp('prop')['ig'],
                                           params.getForProp('prop')['fext'],
                                           params.getForProp('prop')['intc'],
                                           params.getForProp('prop')['nsdim'],
                                           params.getForProp('prop')['prefix'])

        # display subplot
        plt.subplot(1, 3, 2)
        ransVelmeanExp.plot_velocities(params.getForProp('prop')['laxis'],
                                       bconv, tconv,
                                       params.getForEqs('velbgr')['xbl'],
                                       params.getForEqs('velbgr')['xbr'],
                                       params.getForEqs('velbgr')['ybu'],
                                       params.getForEqs('velbgr')['ybd'],
                                       params.getForEqs('velbgr')['ilg'], xsc, yscVexp)

        # instantiate
        ransNablas = TemperatureGradients(params.getForProp('prop')['eht_data'],
                                          params.getForProp('prop')['ig'],
                                          params.getForProp('prop')['fext'],
                                          params.getForProp('prop')['ieos'],
                                          params.getForProp('prop')['intc'],
                                          params.getForProp('prop')['prefix'])

        # display subplot
        plt.subplot(1, 3, 3)
        ransNablas.plot_nablas(params.getForProp('prop')['laxis'],
                               bconv, tconv,
                               params.getForEqs('nablas')['xbl'],
                               params.getForEqs('nablas')['xbr'],
                               params.getForEqs('nablas')['ybu'],
                               params.getForEqs('nablas')['ybd'],
                               params.getForEqs('nablas')['ilg'], xsc, yscNablas)

        html_str = mpld3.fig_to_html(fig)
        Html_file = open(outputFile, "w")
        Html_file.write(html_str)
        Html_file.close()

    def execSrcTempFlct(self, outputFile, bconv, tconv, plabel, nsdim):
        params = self.params

        if plabel == 'ccptwo' and nsdim == 3:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscSource = 1.e14
            yscT = 1.e9
        elif plabel == 'ccptwo' and nsdim == 2:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscSource = 1.e14
            yscT = 1.e9
        elif plabel == '3d-oburn-14ele' or plabel == '3d-oburn-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscSource = 1.e14
            yscT = 1.e9
        elif plabel == '3d-neshellBoost10x-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscSource = 1.e14
            yscT = 1.e9
        elif plabel == '3d-heflashBoost100x-6ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscSource = 1.e12
            yscT = 1.e8
        elif plabel == '3d-thpulse-15ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e2
            yscTke = 1.e6
            fcIntegralBudget = 1.e28
        else:
            print('MasterPlot.py - plabel unknown')
            sys.exit()


        # create FIGURE
        fig = plt.figure(figsize=(13, 4))
        # fig.tight_layout()

        # display subplot
        plt.subplot(1, 3, 1)
        plt.subplots_adjust(left=0.063)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # instantiate
        ransEnuc = NuclearEnergyProduction(params.getForProp('prop')['eht_data'],
                                           params.getForProp('prop')['ig'],
                                           params.getForProp('prop')['intc'],
                                           params.getForProp('prop')['prefix'])

        ransEnuc.plot_enuc(params.getForProp('prop')['laxis'],
                           bconv, tconv,
                           params.getForEqs('enuc')['xbl'],
                           params.getForEqs('enuc')['xbr'],
                           params.getForEqs('enuc')['ybu'],
                           params.getForEqs('enuc')['ybd'],
                           params.getForEqs('enuc')['ilg'], xsc, yscSource)

        # ransEnuc.plot_enuc2(params.getForProp('prop')['laxis'],
        #                    bconv, tconv,
        #                    params.getForEqs('enuc')['xbl'],
        #                    params.getForEqs('enuc')['xbr'],
        #                    params.getForEqs('enuc')['ybu'],
        #                    params.getForEqs('enuc')['ybd'],
        #                    params.getForEqs('enuc')['ilg'])

        plt.subplot(1, 3, 2)
        # instantiate
        ransTT = TemperatureEquation(params.getForProp('prop')['eht_data'],
                                     params.getForProp('prop')['ig'],
                                     params.getForProp('prop')['fext'],
                                     params.getForProp('prop')['ieos'],
                                     params.getForProp('prop')['intc'],
                                     params.getForProp('prop')['nsdim'],
                                     params.getForProp('prop')['prefix'])

        ransTT.plot_tt(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('temp')['xbl'],
                       params.getForEqs('temp')['xbr'],
                       params.getForEqs('temp')['ybu'],
                       params.getForEqs('temp')['ybd'],
                       params.getForEqs('temp')['ilg'], xsc, yscT)

        plt.subplot(1, 3, 3)
        # instantiate
        ransRms = RelativeRMSflct(params.getForProp('prop')['eht_data'],
                                  params.getForProp('prop')['ig'],
                                  params.getForProp('prop')['ieos'],
                                  params.getForProp('prop')['intc'],
                                  params.getForProp('prop')['nsdim'],
                                  params.getForProp('prop')['prefix'])

        ransRms.plot_relative_rms_flct(params.getForProp('prop')['laxis'],
                                       bconv, tconv,
                                       params.getForEqs('relrmsflct')['xbl'],
                                       params.getForEqs('relrmsflct')['xbr'],
                                       params.getForEqs('relrmsflct')['ybu'],
                                       params.getForEqs('relrmsflct')['ybd'],
                                       params.getForEqs('relrmsflct')['ilg'], xsc)

        html_str = mpld3.fig_to_html(fig)
        Html_file = open(outputFile, "w")
        Html_file.write(html_str)
        Html_file.close()

    def execMomx(self, outputFile, bconv, tconv, plabel, nsdim):
        params = self.params

        if plabel == 'ccptwo' and nsdim == 3:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e12
            ysc = 2.e10
            fcIntegralBudget = 1.e38
        elif plabel == 'ccptwo' and nsdim == 2:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e12
            ysc = 2.e10
            fcIntegralBudget = 1.e38
        elif plabel == '3d-oburn-14ele' or plabel == '3d-oburn-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e12
            ysc = 1.e10
            fcIntegralBudget = 1.e38
        elif plabel == '3d-neshellBoost10x-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e12
            ysc = 1.e9
            fcIntegralBudget = 1.e34
        elif plabel == '3d-heflashBoost100x-6ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e10
            ysc = 1.e9
            fcIntegralBudget = 1.e36
        elif plabel == '3d-thpulse-15ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscXeq = 1.e2
            yscX = 1.e6
            fcIntegralBudget = 1.e28
        else:
            print('MasterPlot.py - plabel unknown')
            sys.exit()

        # create FIGURE
        fig = plt.figure(figsize=(13, 4))
        # fig.tight_layout()

        # display subplot
        plt.subplot(1, 3, 1)
        plt.subplots_adjust(left=0.053)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # instantiate
        ransMomx = MomentumEquationX(params.getForProp('prop')['eht_data'],
                                     params.getForProp('prop')['ig'],
                                     params.getForProp('prop')['fext'],
                                     params.getForProp('prop')['intc'],
                                     params.getForProp('prop')['nsdim'],
                                     params.getForProp('prop')['prefix'], plabel)

        ransMomx.plot_momentum_equation_x(params.getForProp('prop')['laxis'],
                                          bconv, tconv,
                                          params.getForEqs('momxeq')['xbl'],
                                          params.getForEqs('momxeq')['xbr'],
                                          params.getForEqs('momxeq')['ybu'],
                                          params.getForEqs('momxeq')['ybd'],
                                          params.getForEqs('momxeq')['ilg'], xsc, ysceq)

        plt.subplot(1, 3, 2)
        ransMomx.plot_momentum_x(params.getForProp('prop')['laxis'],
                                 bconv, tconv,
                                 params.getForEqs('momex')['xbl'],
                                 params.getForEqs('momex')['xbr'],
                                 params.getForEqs('momex')['ybu'],
                                 params.getForEqs('momex')['ybd'],
                                 params.getForEqs('momex')['ilg'], xsc, ysc)

        # display subplot
        ax = plt.subplot(1, 3, 3)

        # plot continuity equation integral budget
        ransMomx.plot_momentum_x_integral_budget(plabel, ax, params.getForProp('prop')['laxis'],
                                                 params.getForEqsBar('momxeqBar')['xbl'],
                                                 params.getForEqsBar('momxeqBar')['xbr'],
                                                 params.getForEqsBar('momxeqBar')['ybu'],
                                                 params.getForEqsBar('momxeqBar')['ybd'],
                                                 fcIntegralBudget)

        html_str = mpld3.fig_to_html(fig)
        Html_file = open(outputFile, "w")
        Html_file.write(html_str)
        Html_file.close()

    def execSSeq(self, outputFile, bconv, tconv, tke_diss, plabel, nsdim):
        params = self.params

        if plabel == 'ccptwo' and nsdim == 3:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e10
            ysc = 1.e8
            fcIntegralBudget = 1.e35
        elif plabel == 'ccptwo' and nsdim == 2:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e10
            ysc = 1.e8
            fcIntegralBudget = 1.e35
        elif plabel == '3d-oburn-14ele' or plabel == '3d-oburn-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e8
            ysc = 1.e8
            fcIntegralBudget = 1.e37
        elif plabel == '3d-neshellBoost10x-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e10
            ysc = 1.e8
            fcIntegralBudget = 1.e33
        elif plabel == '3d-heflashBoost100x-6ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            ysceq = 1.e9
            ysc = 1.e8
            fcIntegralBudget = 1.e35
        elif plabel == '3d-thpulse-15ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e2
            yscTke = 1.e6
            fcIntegralBudget = 1.e28
        else:
            print('MasterPlot.py - plabel unknown')
            sys.exit()



        # create FIGURE
        fig = plt.figure(figsize=(13, 4))
        # fig.tight_layout()

        # display subplot
        plt.subplot(1, 3, 1)
        plt.subplots_adjust(left=0.053)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # instantiate
        ransSS = EntropyEquation(params.getForProp('prop')['eht_data'],
                                 params.getForProp('prop')['ig'],
                                 params.getForProp('prop')['fext'],
                                 params.getForProp('prop')['intc'],
                                 params.getForProp('prop')['nsdim'],
                                 tke_diss,
                                 params.getForProp('prop')['prefix'])

        ransSS.plot_ss_equation(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('sseq')['xbl'],
                                params.getForEqs('sseq')['xbr'],
                                params.getForEqs('sseq')['ybu'],
                                params.getForEqs('sseq')['ybd'],
                                params.getForEqs('sseq')['ilg'], xsc, ysceq)

        plt.subplot(1, 3, 2)
        ransSS.plot_ss(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('entr')['xbl'],
                       params.getForEqs('entr')['xbr'],
                       params.getForEqs('entr')['ybu'],
                       params.getForEqs('entr')['ybd'],
                       params.getForEqs('entr')['ilg'], xsc, ysc)

        # display subplot
        ax = plt.subplot(1, 3, 3)

        # plot continuity equation integral budget
        ransSS.plot_ss_equation_integral_budget(ax, plabel, params.getForProp('prop')['laxis'],
                                                params.getForEqsBar('sseqBar')['xbl'],
                                                params.getForEqsBar('sseqBar')['xbr'],
                                                params.getForEqsBar('sseqBar')['ybu'],
                                                params.getForEqsBar('sseqBar')['ybd'],
                                                fcIntegralBudget)

        html_str = mpld3.fig_to_html(fig)
        Html_file = open(outputFile, "w")
        Html_file.write(html_str)
        Html_file.close()

    def execEiEq(self, outputFile, bconv, tconv, tke_diss, plabel, nsdim):
        params = self.params

        if plabel == 'ccptwo' and nsdim == 3:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscEiEq = 1.e19
            yscEi = 1.e17
            fcIntegralBudget = 1.e44
        elif plabel == 'ccptwo' and nsdim == 2:
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscEiEq = 1.e19
            yscEi = 1.e17
            fcIntegralBudget = 1.e44
        elif plabel == '3d-oburn-14ele' or plabel == '3d-oburn-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscEiEq = 1.e19
            yscEi = 1.e17
            fcIntegralBudget = 1.e46
        elif plabel == '3d-neshellBoost10x-25ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscEiEq = 1.e19
            yscEi = 1.e17
            fcIntegralBudget = 1.e42
        elif plabel == '3d-heflashBoost100x-6ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscEiEq = 1.e17
            yscEi = 1.e16
            fcIntegralBudget = 1.e43
        elif plabel == '3d-thpulse-15ele':
            # scale factors for axis labels
            # mpld3 support for scientific notation is limited
            xsc = 1.e8
            yscTkeEq = 1.e2
            yscTke = 1.e6
            fcIntegralBudget = 1.e28
        else:
            print('MasterPlot.py - plabel unknown')
            sys.exit()

        # create FIGURE
        fig = plt.figure(figsize=(13, 4))
        # fig.tight_layout()

        # display subplot
        plt.subplot(1, 3, 1)
        plt.subplots_adjust(left=0.053)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # instantiate
        ransEi = InternalEnergyEquation(params.getForProp('prop')['eht_data'],
                                        params.getForProp('prop')['ig'],
                                        params.getForProp('prop')['fext'],
                                        params.getForProp('prop')['intc'],
                                        params.getForProp('prop')['nsdim'],
                                        tke_diss,
                                        params.getForProp('prop')['prefix'])

        ransEi.plot_ei_equation(params.getForProp('prop')['laxis'],
                                bconv, tconv,
                                params.getForEqs('eieq')['xbl'],
                                params.getForEqs('eieq')['xbr'],
                                params.getForEqs('eieq')['ybu'],
                                params.getForEqs('eieq')['ybd'],
                                params.getForEqs('eieq')['ilg'], xsc, yscEiEq)

        plt.subplot(1, 3, 2)
        ransEi.plot_ei(params.getForProp('prop')['laxis'],
                       bconv, tconv,
                       params.getForEqs('eint')['xbl'],
                       params.getForEqs('eint')['xbr'],
                       params.getForEqs('eint')['ybu'],
                       params.getForEqs('eint')['ybd'],
                       params.getForEqs('eint')['ilg'], xsc, yscEi)

        # display subplot
        ax = plt.subplot(1, 3, 3)

        # plot continuity equation integral budget
        ransEi.plot_ei_equation_integral_budget(ax, plabel, params.getForProp('prop')['laxis'],
                                                params.getForEqsBar('eieqBar')['xbl'],
                                                params.getForEqsBar('eieqBar')['xbr'],
                                                params.getForEqsBar('eieqBar')['ybu'],
                                                params.getForEqsBar('eieqBar')['ybd'],
                                                fcIntegralBudget)

        html_str = mpld3.fig_to_html(fig)
        Html_file = open(outputFile, "w")
        Html_file.write(html_str)
        Html_file.close()

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
