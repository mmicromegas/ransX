from EQUATIONS.XtransportEquation import XtransportEquation
from EQUATIONS.SourceVel import SourceVel
from EQUATIONS.TDC import TDC


class MasterPlot():

    def __init__(self, params):
        self.params = params
        #plt.close()



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

    def execTDC(self, bconv, tconv):
        params = self.params

        # instantiate
        ransTDC = TDC(params.getForProp('prop')['eht_data'],
                                                           params.getForProp('prop')['plabel'],
                                                           params.getForProp('prop')['code'],
                                                           params.getForProp('prop')['ig'],
                                                           params.getForProp('prop')['fext'],
                                                           params.getForProp('prop')['intc'],
                                                           params.getForProp('prop')['nsdim'],
                                                           params.getForProp('prop')['prefix'])

        # figRANS
        fig = ransTDC.plot_tempdensx(params.getForProp('prop')['laxis'],
                          bconv, tconv,
                          params.getForEqs('rho')['xbl'],
                          params.getForEqs('rho')['xbr'],
                          params.getForEqs('rho')['ybu'],
                          params.getForEqs('rho')['ybd'],
                          params.getForEqs('press')['ybu'],
                          params.getForEqs('press')['ybd'],
                          params.getForEqs('eint')['ybu'],
                          params.getForEqs('eint')['ybd'],
                          params.getForEqs('eint')['ilg'])

        return fig

    def execXtrsEq(self, inuc, element, x, bconv, tconv, tauL):
        params = self.params

        # instantiate
        ransXtra = XtransportEquation(params.getForProp('prop')['eht_data'],
                                      #params.getForProp('prop')['reaclib'],
                                      'DATA/PROMPI/REACLIB/netsu14_oburn_modified',
                                      params.getForProp('prop')['plabel'],
                                      params.getForProp('prop')['code'],
                                      params.getForProp('prop')['ig'],
                                      params.getForProp('prop')['fext'],
                                      inuc, element, bconv, tconv, tauL,
                                      params.getForProp('prop')['intc'],
                                      params.getForProp('prop')['nsdim'],
                                      params.getForProp('prop')['prefix'],
                                      '1',
                                      params.getNetwork)

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


    def execXtransportVSnuclearTimescales(self, inuc, element, x, bconv, tconv, tauL):
        params = self.params

        # instantiate
        ransXtvsn = XtransportEquation(params.getForProp('prop')['eht_data'],
                                      #params.getForProp('prop')['reaclib'],
                                      'DATA/PROMPI/REACLIB/netsu14_oburn_modified',
                                      params.getForProp('prop')['plabel'],
                                      'PROMPI',
                                      params.getForProp('prop')['ig'],
                                      params.getForProp('prop')['fext'],
                                      inuc, element, bconv, tconv, tauL,
                                      params.getForProp('prop')['intc'],
                                      params.getForProp('prop')['nsdim'],
                                      params.getForProp('prop')['prefix'],
                                      '1',
                                      params.getNetwork)

        fig = ransXtvsn.plot_Xtimescales(params.getForProp('prop')['laxis'],bconv, tconv,
                        params.getForEqs(x)['xbl'],
                        params.getForEqs(x)['xbr'],
                        params.getForEqs('xTimescales_' + element)['ybu'],
                        params.getForEqs('xTimescales_' + element)['ybd'],
                        params.getForEqs(x)['ilg'])

        #ransXtvsn.plot_Xtransport_equation(params.getForProp('prop')['laxis'],
        #                                   params.getForEqs(x)['xbl'],
        #                                   params.getForEqs(x)['xbr'],
        #                                   params.getForEqs(x)['ybu'],
        #                                   params.getForEqs(x)['ybd'],
        #                                   params.getForEqs(x)['ilg'])

        return fig
