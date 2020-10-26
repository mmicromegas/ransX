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
    plabel = params.getForProp('prop')['plabel']
    ig = params.getForProp('prop')['ig']
    nsdim = params.getForProp('prop')['nsdim']
    ieos = params.getForProp('prop')['ieos']
    intc = params.getForProp('prop')['intc']
    laxis = params.getForProp('prop')['laxis']
    xbl = params.getForProp('prop')['xbl']
    xbr = params.getForProp('prop')['xbr']

    # calculate properties
    ransP = pRop.Properties(filename, plabel, ig, nsdim, ieos, intc, laxis, xbl, xbr)
    prp = ransP.properties()

    # instantiate master plot
    plt = mPlot.MasterPlot(params)

    # obtain publication quality figures
    plt.SetMatplotlibParams()

    # TEMPERATURE AND DENSITY
    if str2bool(params.getForEqs('ttdd')['plotMee']):
        plt.execRhoTemp(prp['xzn0inc'], prp['xzn0outc'])

    # PRESSURE AND INTERNAL ENERGY
    if str2bool(params.getForEqs('ppei')['plotMee']):
        plt.execPressEi(prp['xzn0inc'], prp['xzn0outc'])

    # NUCLEAR ENERGY PRODUCTION
    if str2bool(params.getForEqs('enuc')['plotMee']):
        plt.execEnuc(prp['xzn0inc'], prp['xzn0outc'])

    # GRAVITY
    if str2bool(params.getForEqs('grav')['plotMee']):
        plt.execGrav(prp['xzn0inc'], prp['xzn0outc'])

    # TEMPERATURE GRADIENTS
    if str2bool(params.getForEqs('nablas')['plotMee']):
        plt.execNablas(prp['xzn0inc'], prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

    # DEGENERACY PARAMETER
    if str2bool(params.getForEqs('psi')['plotMee']):
        plt.execDegeneracy()

    # MEAN AND EXPANSION VELOCITY
    if str2bool(params.getForEqs('velbgr')['plotMee']):
        plt.execVelocitiesMeanExp(prp['xzn0inc'], prp['xzn0outc'])

    # MLT AND TURBULENT VELOCITY
    if str2bool(params.getForEqs('velmlt')['plotMee']):
        plt.execVelocitiesMLTturb(prp['xzn0inc'], prp['xzn0outc'],prp['uconv'],prp['super_ad_i'],prp['super_ad_o'])

    # BRUNT-VAISALLA FREQUENCY
    if str2bool(params.getForEqs('nsq')['plotMee']):
        plt.execBruntV(prp['xzn0inc'], prp['xzn0outc'])

    # BUOYANCY
    if str2bool(params.getForEqs('buo')['plotMee']):
        plt.execBuoyancy(prp['xzn0inc'], prp['xzn0outc'])

    # RELATIVE RMS FLUCTUATIONS
    if str2bool(params.getForEqs('relrmsflct')['plotMee']):
        plt.execRelativeRmsFlct(prp['xzn0inc'], prp['xzn0outc'])

    # ABAR and ZBAR
    if str2bool(params.getForEqs('abzb')['plotMee']):
        plt.execAbarZbar(prp['xzn0inc'], prp['xzn0outc'])

    # CONTINUITY EQUATION
    if str2bool(params.getForEqs('rho')['plotMee']):
        plt.execRho(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('conteq')['plotMee']):
        plt.execContEq(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqsBar('conteqBar')['plotMee']):
        plt.execContEqBar()

    if str2bool(params.getForEqs('conteqfdd')['plotMee']):
        plt.execContFddEq(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqsBar('conteqfddBar')['plotMee']):
        plt.execContFddEqBar()

    # MOMENTUM X EQUATION
    if str2bool(params.getForEqs('momex')['plotMee']):
        plt.execMomx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('momxeq')['plotMee']):
        plt.execMomxEq(prp['xzn0inc'], prp['xzn0outc'])

    # MOMENTUM Y EQUATION
    if str2bool(params.getForEqs('momey')['plotMee']):
        plt.execMomy(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('momyeq')['plotMee']):
        plt.execMomyEq(prp['xzn0inc'], prp['xzn0outc'])

    # MOMENTUM Z EQUATION
    if str2bool(params.getForEqs('momez')['plotMee']):
        plt.execMomz(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('momzeq')['plotMee']):
        plt.execMomzEq(prp['xzn0inc'], prp['xzn0outc'])

    # REYNOLDS STRESS XX EQUATION
    if str2bool(params.getForEqs('rxx')['plotMee']):
        plt.execRxx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('rexxeq')['plotMee']):
        plt.execRxxEq(prp['kolm_tke_diss_rate'], prp['xzn0inc'], prp['xzn0outc'])

    # REYNOLDS STRESS YY EQUATION
    if str2bool(params.getForEqs('ryy')['plotMee']):
        plt.execRyy(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('reyyeq')['plotMee']):
        plt.execRyyEq(prp['kolm_tke_diss_rate'], prp['xzn0inc'], prp['xzn0outc'])

    # REYNOLDS STRESS ZZ EQUATION
    if str2bool(params.getForEqs('rzz')['plotMee']):
        plt.execRzz(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('rezzeq')['plotMee']):
        plt.execRzzEq(prp['kolm_tke_diss_rate'], prp['xzn0inc'], prp['xzn0outc'])

    # KINETIC ENERGY EQUATION
    if str2bool(params.getForEqs('kine')['plotMee']):
        plt.execKe(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('kieq')['plotMee']):
        plt.execKeEq(prp['kolm_tke_diss_rate'], prp['xzn0inc'], prp['xzn0outc'])

    # TURBULENT KINETIC ENERGY EQUATION
    if str2bool(params.getForEqs('tkie')['plotMee']):
        plt.execTke(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                    prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

    if str2bool(params.getForEqs('tkeeq')['plotMee']):
        plt.execTkeEq(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                      prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

    if str2bool(params.getForEqsBar('tkeeqBar')['plotMee']):
        plt.execTkeEqBar(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                      prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

    # RADIAL TURBULENT KINETIC ENERGY EQUATION
    if str2bool(params.getForEqs('tkieR')['plotMee']):
        plt.execTkeRadial(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                    prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

    if str2bool(params.getForEqs('tkeReq')['plotMee']):
        plt.execTkeEqRadial(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                      prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

    if str2bool(params.getForEqsBar('tkeReqBar')['plotMee']):
        plt.execTkeEqRadialBar(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                      prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])


    # HORIZONTAL TURBULENT KINETIC ENERGY EQUATION
    if str2bool(params.getForEqs('tkieH')['plotMee']):
        plt.execTkeHorizontal(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                    prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

    if str2bool(params.getForEqs('tkeHeq')['plotMee']):
        plt.execTkeEqHorizontal(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                      prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

    if str2bool(params.getForEqsBar('tkeHeqBar')['plotMee']):
        plt.execTkeEqHorizontalBar(prp['kolm_tke_diss_rate'], prp['xzn0inc'],
                      prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

    # TOTAL ENERGY EQUATION
    if str2bool(params.getForEqs('toe')['plotMee']):
        plt.execTe(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('teeq')['plotMee']):
        plt.execTeEq(prp['kolm_tke_diss_rate'], prp['xzn0inc'], prp['xzn0outc'])

    # INTERNAL ENERGY EQUATION
    if str2bool(params.getForEqs('eint')['plotMee']):
        plt.execEi(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('eieq')['plotMee']):
        plt.execEiEq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # INTERNAL ENERGY FLUX EQUATION
    if str2bool(params.getForEqs('eintflx')['plotMee']):
        plt.execEiFlx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('eiflxeq')['plotMee']):
        plt.execEiFlxEq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # INTERNAL ENERGY VARIANCE EQUATION
    if str2bool(params.getForEqs('eintvar')['plotMee']):
        plt.execEiVar(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('eivareq')['plotMee']):
        plt.execEiVarEq(prp['tke_diss'], prp['tauL'], prp['xzn0inc'], prp['xzn0outc'])

    # PRESSURE EQUATION
    if str2bool(params.getForEqs('press')['plotMee']):
        plt.execPP(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ppeq')['plotMee']):
        plt.execPPeq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # PRESSURE FLUX EQUATION in X
    if str2bool(params.getForEqs('pressxflx')['plotMee']):
        plt.execPPxflx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ppxflxeq')['plotMee']):
        plt.execPPxflxEq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # PRESSURE FLUX EQUATION in Y
    if str2bool(params.getForEqs('pressyflx')['plotMee']):
        plt.execPPyflx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ppyflxeq')['plotMee']):
        plt.execPPyflxEq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # PRESSURE FLUX EQUATION in Z
    if str2bool(params.getForEqs('presszflx')['plotMee']):
        plt.execPPzflx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ppzflxeq')['plotMee']):
        plt.execPPzflxEq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # PRESSURE VARIANCE EQUATION
    if str2bool(params.getForEqs('pressvar')['plotMee']):
        plt.execPPvar(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ppvareq')['plotMee']):
        plt.execPPvarEq(prp['tke_diss'], prp['tauL'], prp['xzn0inc'], prp['xzn0outc'])

    # TEMPERATURE EQUATION
    if str2bool(params.getForEqs('temp')['plotMee']):
        plt.execTT(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('tteq')['plotMee']):
        plt.execTTeq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # TEMPERATURE FLUX EQUATION
    if str2bool(params.getForEqs('tempflx')['plotMee']):
        plt.execTTflx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ttflxeq')['plotMee']):
        plt.execTTflxEq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # TEMPERATURE VARIANCE EQUATION
    if str2bool(params.getForEqs('tempvar')['plotMee']):
        plt.execTTvar(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ttvareq')['plotMee']):
        plt.execTTvarEq(prp['tke_diss'], prp['tauL'], prp['xzn0inc'], prp['xzn0outc'])

    # ENTHALPY EQUATION
    if str2bool(params.getForEqs('enth')['plotMee']):
        plt.execHH(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('hheq')['plotMee']):
        plt.execHHeq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # ENTHALPY FLUX EQUATION
    if str2bool(params.getForEqs('enthflx')['plotMee']):
        plt.execHHflx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('hhflxeq')['plotMee']):
        plt.execHHflxEq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # ENTHALPY VARIANCE EQUATION
    if str2bool(params.getForEqs('enthvar')['plotMee']):
        plt.execHHvar(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('hhvareq')['plotMee']):
        plt.execHHvarEq(prp['tke_diss'], prp['tauL'], prp['xzn0inc'], prp['xzn0outc'])

    # ENTROPY EQUATION
    if str2bool(params.getForEqs('entr')['plotMee']):
        plt.execSS(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('sseq')['plotMee']):
        plt.execSSeq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # ENTROPY FLUX EQUATION
    if str2bool(params.getForEqs('entrflx')['plotMee']):
        plt.execSSflx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ssflxeq')['plotMee']):
        plt.execSSflxEq(prp['tke_diss'], prp['xzn0inc'], prp['xzn0outc'])

    # ENTROPY VARIANCE EQUATION
    if str2bool(params.getForEqs('entrvar')['plotMee']):
        plt.execSSvar(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ssvareq')['plotMee']):
        plt.execSSvarEq(prp['tke_diss'], prp['tauL'], prp['xzn0inc'], prp['xzn0outc'])

    # DENSITY VARIANCE EQUATION
    if str2bool(params.getForEqs('densvar')['plotMee']):
        plt.execDDvar(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('ddvareq')['plotMee']):
        plt.execDDvarEq(prp['tauL'], prp['xzn0inc'], prp['xzn0outc'])

    # TURBULENT MASS FLUX EQUATION a.k.a A EQUATION
    if str2bool(params.getForEqs('tmsflx')['plotMee']):
        plt.execTMSflx(prp['xzn0inc'], prp['xzn0outc'],prp['lc'])

    if str2bool(params.getForEqs('aeq')['plotMee']):
        plt.execAeq(prp['xzn0inc'], prp['xzn0outc'],prp['lc'])

    # DENSITY-SPECIFIC VOLUME COVARIANCE a.k.a. B EQUATION
    if str2bool(params.getForEqs('dsvc')['plotMee']):
        plt.execDSVC(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('beq')['plotMee']):
        plt.execBeq(prp['xzn0inc'], prp['xzn0outc'])

    # MEAN NUMBER OF NUCLEONS PER ISOTOPE a.k.a ABAR EQUATION
    if str2bool(params.getForEqs('abar')['plotMee']):
        plt.execAbar(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('abreq')['plotMee']):
        plt.execAbarEq(prp['xzn0inc'], prp['xzn0outc'])

    # ABAR FLUX EQUATION
    if str2bool(params.getForEqs('abflx')['plotMee']):
        plt.execFabarx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('fabxeq')['plotMee']):
        plt.execFabarxEq(prp['xzn0inc'], prp['xzn0outc'])

    # MEAN CHARGE PER ISOTOPE a.k.a ZBAR EQUATION
    if str2bool(params.getForEqs('zbar')['plotMee']):
        plt.execZbar(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('zbreq')['plotMee']):
        plt.execZbarEq(prp['xzn0inc'], prp['xzn0outc'])

    # ZBAR FLUX EQUATION
    if str2bool(params.getForEqs('zbflx')['plotMee']):
        plt.execFzbarx(prp['xzn0inc'], prp['xzn0outc'])

    if str2bool(params.getForEqs('fzbxeq')['plotMee']):
        plt.execFzbarxEq(prp['xzn0inc'], prp['xzn0outc'])

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

        hack = 0.0e9
        # COMPOSITION TRANSPORT EQUATION
        if str2bool(params.getForEqs('x_' + elem)['plotMee']):
            plt.execX(inuc, elem, 'x_' + elem,
                      prp['xzn0inc']+hack,
                      prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

        if str2bool(params.getForEqs('xrho_' + elem)['plotMee']):
            plt.execXrho(inuc, elem, 'xrho_' + elem,
                         prp['xzn0inc']+hack,
                         prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

        if str2bool(params.getForEqs('xtrseq_' + elem)['plotMee']):
            plt.execXtrsEq(inuc, elem, 'xtrseq_' + elem,
                           prp['xzn0inc']+hack,
                           prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

        if str2bool(params.getForEqsBar('xtrseq_' + elem + 'Bar')['plotMee']):
            plt.execXtrsEqBar(inuc, elem,
                              'xtrseq_' + elem + 'Bar',
                              prp['xzn0inc']+hack,
                              prp['xzn0outc'],prp['super_ad_i'],prp['super_ad_o'])

        # COMPOSITION FLUX IN X
        if str2bool(params.getForEqs('xflxx_' + elem)['plotMee']):
            plt.execXflxx(inuc, elem, 'xflxx_' + elem,
                          prp['xzn0inc'],
                          prp['xzn0outc'],
                          prp['tke_diss'],
                          prp['tauL'],
                          prp['cnvz_in_hp'])

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
                            prp['tauL'],
                            prp['cnvz_in_hp'])

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
                         prp['xzn0outc'], prp['tke_diss'], prp['tauL'],
                         prp['super_ad_i'],prp['super_ad_o'],
                         prp['cnvz_in_hp'])

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
