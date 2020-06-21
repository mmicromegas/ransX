import CANUTO1997.ContinuityEquationWithFavrianDilatation as contfdil

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
