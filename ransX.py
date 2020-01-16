###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# File: ransX.py
# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: January/2019
# Desc: controls selection, hence calculation and plotting of terms in RANS equations
# Usage: run ransX.py

import UTILS.RANSX.Properties as pRop
import UTILS.RANSX.ReadParamsRansX as rP
import UTILS.RANSX.MasterPlot as mPlot
import ast
import os


def main():
    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.ransx')
    params = rP.ReadParamsRansX(paramFile)

    # get input parameters
    filename = params.getForProp('prop')['eht_data']
    ig = params.getForProp('prop')['ig']
    ieos = params.getForProp('prop')['ieos']
    intc = params.getForProp('prop')['intc']
    laxis = params.getForProp('prop')['laxis']
    xbl = params.getForProp('prop')['xbl']
    xbr = params.getForProp('prop')['xbr']

    # calculate properties
    ransP = pRop.Properties(filename, ig, ieos, intc, laxis, xbl, xbr)
    prp = ransP.properties()

    # instantiate master plot
    plt = mPlot.MasterPlot(params)

    # obtain publication quality figures
    plt.SetMatplotlibParams()

    # TEMPERATURE AND DENSITY
    if str2bool(params.getForEqs('ttdd')['plotMee']):
        plt.execRhoTemp()

    # PRESSURE AND INTERNAL ENERGY
    if str2bool(params.getForEqs('ppei')['plotMee']):
        plt.execPressEi()

    # NUCLEAR ENERGY PRODUCTION
    if str2bool(params.getForEqs('enuc')['plotMee']):
        plt.execEnuc()

    # GRAVITY
    if str2bool(params.getForEqs('grav')['plotMee']):
        plt.execGrav()

    # TEMPERATURE GRADIENTS
    if str2bool(params.getForEqs('nablas')['plotMee']):
        plt.execNablas()

    # DEGENERACY PARAMETER
    if str2bool(params.getForEqs('psi')['plotMee']):
        plt.execDegeneracy()

    # MEAN AND EXPANSION VELOCITY
    if str2bool(params.getForEqs('velbgr')['plotMee']):
        plt.execVelocitiesMeanExp()

    # MLT AND TURBULENT VELOCITY
    if str2bool(params.getForEqs('velmlt')['plotMee']):
        plt.execVelocitiesMLTturb()

    # BRUNT-VAISALLA FREQUENCY
    if str2bool(params.getForEqs('nsq')['plotMee']):
        plt.execBruntV()

    # BUOYANCY
    if str2bool(params.getForEqs('buo')['plotMee']):
        plt.execBuoyancy()

    # RELATIVE RMS FLUCTUATIONS
    if str2bool(params.getForEqs('relrmsflct')['plotMee']):
        plt.execRelativeRmsFlct()

    # ABAR and ZBAR
    if str2bool(params.getForEqs('abzb')['plotMee']):
        plt.execAbarZbar()

    # CONTINUITY EQUATION
    if str2bool(params.getForEqs('rho')['plotMee']):
        plt.execRho()

    if str2bool(params.getForEqs('conteq')['plotMee']):
        plt.execContEq()

    if str2bool(params.getForEqsBar('conteqBar')['plotMee']):
        plt.execContEqBar()

    if str2bool(params.getForEqs('conteqfdd')['plotMee']):
        plt.execContFddEq()

    if str2bool(params.getForEqsBar('conteqfddBar')['plotMee']):
        plt.execContFddEqBar()

    # MOMENTUM X EQUATION
    if str2bool(params.getForEqs('momex')['plotMee']):
        plt.execMomx()

    if str2bool(params.getForEqs('momxeq')['plotMee']):
        plt.execMomxEq()

    # MOMENTUM Y EQUATION
    if str2bool(params.getForEqs('momey')['plotMee']):
        plt.execMomy()

    if str2bool(params.getForEqs('momyeq')['plotMee']):
        plt.execMomyEq()

    # MOMENTUM Z EQUATION
    if str2bool(params.getForEqs('momez')['plotMee']):
        plt.execMomz()

    if str2bool(params.getForEqs('momzeq')['plotMee']):
        plt.execMomzEq()

    # REYNOLDS STRESS XX EQUATION
    if str2bool(params.getForEqs('rxx')['plotMee']):
        plt.execRxx()

    if str2bool(params.getForEqs('rexxeq')['plotMee']):
        plt.execRxxEq(prp['kolm_tke_diss_rate'])

    # REYNOLDS STRESS YY EQUATION
    if str2bool(params.getForEqs('ryy')['plotMee']):
        plt.execRyy()

    if str2bool(params.getForEqs('reyyeq')['plotMee']):
        plt.execRyyEq(prp['kolm_tke_diss_rate'])

    # REYNOLDS STRESS ZZ EQUATION
    if str2bool(params.getForEqs('rzz')['plotMee']):
        plt.execRzz()

    if str2bool(params.getForEqs('rezzeq')['plotMee']):
        plt.execRzzEq(prp['kolm_tke_diss_rate'])

    # KINETIC ENERGY EQUATION
    if str2bool(params.getForEqs('kine')['plotMee']):
        plt.execKe()

    if str2bool(params.getForEqs('kieq')['plotMee']):
        plt.execKeEq(prp['kolm_tke_diss_rate'])

    if str2bool(params.getForEqs('tkie')['plotMee']):
        plt.execTke(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                    prp['xzn0outc'])

    if str2bool(params.getForEqs('tkeeq')['plotMee']):
        plt.execTkeEq(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                      prp['xzn0outc'])

    # TOTAL ENERGY EQUATION
    if str2bool(params.getForEqs('toe')['plotMee']):
        plt.execTe()

    if str2bool(params.getForEqs('teeq')['plotMee']):
        plt.execTeEq(prp['kolm_tke_diss_rate'])

    # INTERNAL ENERGY EQUATION
    if str2bool(params.getForEqs('eint')['plotMee']):
        plt.execEi()

    if str2bool(params.getForEqs('eieq')['plotMee']):
        plt.execEiEq(prp['tke_diss'])

    # INTERNAL ENERGY FLUX EQUATION
    if str2bool(params.getForEqs('eintflx')['plotMee']):
        plt.execEiFlx()

    if str2bool(params.getForEqs('eiflxeq')['plotMee']):
        plt.execEiFlxEq(prp['tke_diss'])

    # INTERNAL ENERGY VARIANCE EQUATION
    if str2bool(params.getForEqs('eintvar')['plotMee']):
        plt.execEiVar()

    if str2bool(params.getForEqs('eivareq')['plotMee']):
        plt.execEiVarEq(prp['tke_diss'], prp['tauL'])

    # PRESSURE EQUATION
    if str2bool(params.getForEqs('press')['plotMee']):
        plt.execPP()

    if str2bool(params.getForEqs('ppeq')['plotMee']):
        plt.execPPeq(prp['tke_diss'])

    # PRESSURE FLUX EQUATION in X
    if str2bool(params.getForEqs('pressxflx')['plotMee']):
        plt.execPPxflx()

    if str2bool(params.getForEqs('ppxflxeq')['plotMee']):
        plt.execPPxflxEq(prp['tke_diss'])

    # PRESSURE FLUX EQUATION in Y
    if str2bool(params.getForEqs('pressyflx')['plotMee']):
        plt.execPPyflx()

    if str2bool(params.getForEqs('ppyflxeq')['plotMee']):
        plt.execPPyflxEq(prp['tke_diss'])

    # PRESSURE FLUX EQUATION in Z
    if str2bool(params.getForEqs('presszflx')['plotMee']):
        plt.execPPzflx()

    if str2bool(params.getForEqs('ppzflxeq')['plotMee']):
        plt.execPPzflxEq(prp['tke_diss'])

    # PRESSURE VARIANCE EQUATION
    if str2bool(params.getForEqs('pressvar')['plotMee']):
        plt.execPPvar()

    if str2bool(params.getForEqs('ppvareq')['plotMee']):
        plt.execPPvarEq(prp['tke_diss'], prp['tauL'])

    # TEMPERATURE EQUATION
    if str2bool(params.getForEqs('temp')['plotMee']):
        plt.execTT()

    if str2bool(params.getForEqs('tteq')['plotMee']):
        plt.execTTeq(prp['tke_diss'])

    # TEMPERATURE FLUX EQUATION
    if str2bool(params.getForEqs('tempflx')['plotMee']):
        plt.execTTflx()

    if str2bool(params.getForEqs('ttflxeq')['plotMee']):
        plt.execTTflxEq(prp['tke_diss'])

    # TEMPERATURE VARIANCE EQUATION
    if str2bool(params.getForEqs('tempvar')['plotMee']):
        plt.execTTvar()

    if str2bool(params.getForEqs('ttvareq')['plotMee']):
        plt.execTTvarEq(prp['tke_diss'], prp['tauL'])

    # ENTHALPY EQUATION
    if str2bool(params.getForEqs('enth')['plotMee']):
        plt.execHH()

    if str2bool(params.getForEqs('hheq')['plotMee']):
        plt.execHHeq(prp['tke_diss'])

    # ENTHALPY FLUX EQUATION
    if str2bool(params.getForEqs('enthflx')['plotMee']):
        plt.execHHflx()

    if str2bool(params.getForEqs('hhflxeq')['plotMee']):
        plt.execHHflxEq(prp['tke_diss'])

    # ENTHALPY VARIANCE EQUATION
    if str2bool(params.getForEqs('enthvar')['plotMee']):
        plt.execHHvar()

    if str2bool(params.getForEqs('hhvareq')['plotMee']):
        plt.execHHvarEq(prp['tke_diss'], prp['tauL'])

    # ENTROPY EQUATION
    if str2bool(params.getForEqs('entr')['plotMee']):
        plt.execSS()

    if str2bool(params.getForEqs('sseq')['plotMee']):
        plt.execSSeq(prp['tke_diss'])

    # ENTROPY FLUX EQUATION
    if str2bool(params.getForEqs('entrflx')['plotMee']):
        plt.execSSflx()

    if str2bool(params.getForEqs('ssflxeq')['plotMee']):
        plt.execSSflxEq(prp['tke_diss'])

    # ENTROPY VARIANCE EQUATION
    if str2bool(params.getForEqs('entrvar')['plotMee']):
        plt.execSSvar()

    if str2bool(params.getForEqs('ssvareq')['plotMee']):
        plt.execSSvarEq(prp['tke_diss'], prp['tauL'])

    # DENSITY VARIANCE EQUATION
    if str2bool(params.getForEqs('densvar')['plotMee']):
        plt.execDDvar()

    if str2bool(params.getForEqs('ddvareq')['plotMee']):
        plt.execDDvarEq(prp['tauL'])

    # TURBULENT MASS FLUX EQUATION a.k.a A EQUATION
    if str2bool(params.getForEqs('tmsflx')['plotMee']):
        plt.execTMSflx()

    if str2bool(params.getForEqs('aeq')['plotMee']):
        plt.execAeq()

    # DENSITY-SPECIFIC VOLUME COVARIANCE a.k.a. B EQUATION
    if str2bool(params.getForEqs('dsvc')['plotMee']):
        plt.execDSVC()

    if str2bool(params.getForEqs('beq')['plotMee']):
        plt.execBeq()

    # MEAN NUMBER OF NUCLEONS PER ISOTOPE a.k.a ABAR EQUATION
    if str2bool(params.getForEqs('abar')['plotMee']):
        plt.execAbar()

    if str2bool(params.getForEqs('abreq')['plotMee']):
        plt.execAbarEq()

    # ABAR FLUX EQUATION
    if str2bool(params.getForEqs('abflx')['plotMee']):
        plt.execFabarx()

    if str2bool(params.getForEqs('fabxeq')['plotMee']):
        plt.execFabarxEq()

    # MEAN CHARGE PER ISOTOPE a.k.a ZBAR EQUATION
    if str2bool(params.getForEqs('zbar')['plotMee']):
        plt.execZbar()

    if str2bool(params.getForEqs('zbreq')['plotMee']):
        plt.execZbarEq()

    # ZBAR FLUX EQUATION
    if str2bool(params.getForEqs('zbflx')['plotMee']):
        plt.execFzbarx()

    if str2bool(params.getForEqs('fzbxeq')['plotMee']):
        plt.execFzbarxEq()

    # HYDRODYNAMIC CONTINUITY STELLAR STRUCTURE EQUATION
    if str2bool(params.getForEqs('cteqhsse')['plotMee']):
        plt.execHssContEq(prp['xzn0inc'],
                          prp['xzn0outc'])

    # HYDRODYNAMIC MOMENTUM STELLAR STRUCTURE EQUATION
    if str2bool(params.getForEqs('mxeqhsse')['plotMee']):
        plt.execHssMomxEq(prp['xzn0inc'],
                          prp['xzn0outc'])

    # HYDRODYNAMIC TEMPERATURE STELLAR STRUCTURE EQUATION
    if str2bool(params.getForEqs('tpeqhsse')['plotMee']):
        plt.execHssTempEq(prp['tke_diss'],
                          prp['xzn0inc'],
                          prp['xzn0outc'])

    # HYDRODYNAMIC LUMINOSITY STELLAR STRUCTURE EQUATION
    if str2bool(params.getForEqs('lueqhsse')['plotMee']):
        plt.execHssLumiEq(prp['tke_diss'],
                          prp['xzn0inc'],
                          prp['xzn0outc'])

    # FULL TURBULENT VELOCITY FIELD HYPOTHESIS
    if str2bool(params.getForEqs('ftvfh_x')['plotMee']):
        plt.execFtvfhX(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ftvfh_y')['plotMee']):
        plt.execFtvfhY(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ftvfh_z')['plotMee']):
        plt.execFtvfhZ(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('uxfpd')['plotMee']):
        plt.execUxfpd(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('uyfpd')['plotMee']):
        plt.execUyfpd(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('uzfpd')['plotMee']):
        plt.execUzfpd(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('divu')['plotMee']):
        plt.execDivu(prp['xzn0inc'], prp['xzn0outc'])

    # load network
    network = params.getNetwork()

    # COMPOSITION TRANSPORT, FLUX, VARIANCE EQUATIONS and EULERIAN DIFFUSIVITY
    for elem in network[1:]:  # skip network identifier in the list
        inuc = params.getInuc(network, elem)

        # COMPOSITION TRANSPORT EQUATION
        if str2bool(params.getForEqs('xrho_' + elem)['plotMee']):
            plt.execXrho(inuc, elem, 'xrho_' + elem,
                         prp['xzn0inc'],
                         prp['xzn0outc'])

        if str2bool(params.getForEqs('xtrseq_' + elem)['plotMee']):
            plt.execXtrsEq(inuc, elem, 'xtrseq_' + elem,
                           prp['xzn0inc'],
                           prp['xzn0outc'])

        if str2bool(params.getForEqsBar('xtrseq_' + elem + 'Bar')['plotMee']):
            plt.execXtrsEqBar(inuc, elem,
                              'xtrseq_' + elem + 'Bar',
                              prp['xzn0inc'],
                              prp['xzn0outc'])

        # COMPOSITION FLUX IN X
        if str2bool(params.getForEqs('xflxx_' + elem)['plotMee']):
            plt.execXflxx(inuc, elem, 'xflxx_' + elem,
                          prp['xzn0inc'],
                          prp['xzn0outc'],
                          prp['tke_diss'],
                          prp['tauL'])

        # COMPOSITION FLUX IN Y
        if str2bool(params.getForEqs('xflxy_' + elem)['plotMee']):
            plt.execXflxy(inuc, elem, 'xflxy_' + elem,
                          prp['xzn0inc'],
                          prp['xzn0outc'],
                          prp['tke_diss'],
                          prp['tauL'])

        # COMPOSITION FLUX IN Z
        if str2bool(params.getForEqs('xflxz_' + elem)['plotMee']):
            plt.execXflxz(inuc, elem, 'xflxz_' + elem,
                          prp['xzn0inc'],
                          prp['xzn0outc'],
                          prp['tke_diss'],
                          prp['tauL'])

        # COMPOSITION FLUX EQUATION IN X
        if str2bool(params.getForEqs('xflxxeq_' + elem)['plotMee']):
            plt.execXflxXeq(inuc, elem, 'xflxxeq_' + elem,
                            prp['xzn0inc'],
                            prp['xzn0outc'],
                            prp['tke_diss'],
                            prp['tauL'])

        # COMPOSITION FLUX EQUATION IN Y
        if str2bool(params.getForEqs('xflxyeq_' + elem)['plotMee']):
            plt.execXflxYeq(inuc, elem, 'xflxyeq_' + elem,
                            prp['xzn0inc'],
                            prp['xzn0outc'],
                            prp['tke_diss'],
                            prp['tauL'])

        # COMPOSITION FLUX EQUATION IN Z
        if str2bool(params.getForEqs('xflxzeq_' + elem)['plotMee']):
            plt.execXflxZeq(inuc, elem, 'xflxzeq_' + elem,
                            prp['xzn0inc'],
                            prp['xzn0outc'],
                            prp['tke_diss'],
                            prp['tauL'])

        # COMPOSITION VARIANCE
        if str2bool(params.getForEqs('xvar_' + elem)['plotMee']):
            plt.execXvar(inuc, elem, 'xvar_' + elem,
                         prp['xzn0inc'],
                         prp['xzn0outc'])

        # COMPOSITION VARIANCE EQUATION
        if str2bool(params.getForEqs('xvareq_' + elem)['plotMee']):
            plt.execXvarEq(inuc, elem, 'xvareq_' + elem,
                           prp['tauL'],
                           prp['xzn0inc'],
                           prp['xzn0outc'])

        # EULERIAN DIFFUSIVITY
        if str2bool(params.getForEqs('xdiff_' + elem)['plotMee']):
            plt.execDiff(inuc, elem, 'xdiff_' + elem,
                         prp['lc'], prp['uconv'],
                         prp['xzn0inc'],
                         prp['xzn0outc'])

        # HYDRODYNAMIC STELLAR STRUCTURE COMPOSITION TRANSPORT EQUATION
        if str2bool(params.getForEqs('coeqhsse_' + elem)['plotMee']):
            plt.execHssCompEq(inuc, elem, 'coeqhsse_' + elem,
                              prp['xzn0inc'],
                              prp['xzn0outc'])

        # DAMKOHLER NUMBER DISTRIBUTION
        if str2bool(params.getForEqs('xda_' + elem)['plotMee']):
            plt.execXda(inuc, elem, 'xda_' + elem,
                        prp['xzn0inc'],
                        prp['xzn0outc'])


# True/False strings to proper boolean
def str2bool(param):
    return ast.literal_eval(param)


# EXECUTE MAIN
if __name__ == "__main__":
    main()

# END
