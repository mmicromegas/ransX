###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: January/2019

import UTILS.RES.ResReadParamsRansX as rP
import UTILS.RES.ResMasterPlot as mPlot
import UTILS.RANSX.Properties as pRop
import ast
import os


def main():
    # create os independent path and read parameter file
    paramFile = os.path.join('PARAMS', 'param.ransx.res')
    params = rP.ResReadParamsRansX(paramFile)

    # get list with data source files and central time index
    filename = params.getForProp('prop')['eht_data']
    plabel = params.getForProp('prop')['plabel']
    intc = params.getForProp('prop')['intc']

    # get input properties
    ig = params.getForProp('prop')['ig']
    ieos = params.getForProp('prop')['ieos']
    laxis = params.getForProp('prop')['laxis']
    xbl = params.getForProp('prop')['xbl']
    xbr = params.getForProp('prop')['xbr']

    # get and display simulation properties that you do resolution study of
    #for filee in filename:
    #    ransP = pRop.Properties(filee, plabel, ig, ieos, intc, laxis, xbl, xbr)
    #    ransP.properties()

    # instantiate master plot
    plt = mPlot.ResMasterPlot(params)

    # obtain publication quality figures
    plt.SetMatplotlibParams()

    # TEMPERATURE
    if str2bool(params.getForEqs('temp')['plotMee']):
        plt.execTemp()

    # DENSITY
    if str2bool(params.getForEqs('rho')['plotMee']):
        plt.execRho()

    # MOMENTUM X
    if str2bool(params.getForEqs('momex')['plotMee']):
        plt.execMomex()

    # TOTAL ENERGY
    if str2bool(params.getForEqs('toe')['plotMee']):
        plt.execEt()

    # ENTROPY
    if str2bool(params.getForEqs('entr')['plotMee']):
        plt.execSS()

    # ENTROPY VARIANCE
    if str2bool(params.getForEqs('entrvar')['plotMee']):
        plt.execSSvar()

    # DENSITY SPECIFIC VOLUME COVARIANCE
    if str2bool(params.getForEqs('dsvc')['plotMee']):
        plt.execDSVC()

    # ENTHALPY
    if str2bool(params.getForEqs('enth')['plotMee']):
        plt.execHH()

    # PRESSURE
    if str2bool(params.getForEqs('press')['plotMee']):
        plt.execPP()

    # MEAN MOLECULAR WEIGHT
    if str2bool(params.getForEqs('abar')['plotMee']):
        plt.execAbar()

    # MEAN MOLECULAR WEIGHT
    if str2bool(params.getForEqs('abflx')['plotMee']):
        plt.execAbarFlux()

    # BRUNT-VAISALLA FREQUENCY
    if str2bool(params.getForEqs('nsq')['plotMee']):
        plt.execBruntV()

    # TURBULENT KINETIC ENERGY
    if str2bool(params.getForEqs('tkie')['plotMee']):
        plt.execTke()

    # INTERNAL ENERGY FLUX
    if str2bool(params.getForEqs('eintflx')['plotMee']):
        plt.execEiFlx()

    # PRESSURE FLUX
    if str2bool(params.getForEqs('pressxflx')['plotMee']):
        plt.execPPxflx()

    # TEMPERATURE FLUX EQUATION
    if str2bool(params.getForEqs('tempflx')['plotMee']):
        plt.execTTflx()

    # ENTHALPY FLUX EQUATION
    if str2bool(params.getForEqs('enthflx')['plotMee']):
        plt.execHHflx()

    # ENTROPY FLUX
    if str2bool(params.getForEqs('entrflx')['plotMee']):
        plt.execSSflx()

    # TURBULENT MASS FLUX
    if str2bool(params.getForEqs('tmsflx')['plotMee']):
        plt.execTMSflx()

    # Turbulent Radial Velocity RMS
    if str2bool(params.getForEqs('uxrms')['plotMee']):
        plt.execUXrms()

    # Turbulent Uy Velocity RMS
    if str2bool(params.getForEqs('uyrms')['plotMee']):
        plt.execUYrms()

    # Turbulent Uz Velocity RMS
    if str2bool(params.getForEqs('uzrms')['plotMee']):
        plt.execUZrms()

    # Density RMS
    if str2bool(params.getForEqs('ddrms')['plotMee']):
        plt.execDDrms()

    # Buoyancy
    if str2bool(params.getForEqs('buoy')['plotMee']):
        plt.execBuoyancy()

    # Dilatation
    if str2bool(params.getForEqs('divu')['plotMee']):
        plt.execDilatation()

    # Div of Turbulent Mass Flux
    if str2bool(params.getForEqs('divfrho')['plotMee']):
        plt.execDivFrho()

    # load network
    network = params.getNetwork()

    # COMPOSITION MASS FRACTION AND FLUX
    for elem in network[1:]:  # skip network identifier in the list
        inuc = params.getInuc(network, elem)
        if str2bool(params.getForEqs('x_' + elem)['plotMee']):
            plt.execX(inuc, elem, 'x_' + elem)
        # if str2bool(params.getForEqs('xrho_' + elem)['plotMee']):
        #    plt.execXrho(inuc, elem, 'xrho_' + elem)
        if str2bool(params.getForEqs('xflxx_' + elem)['plotMee']):
            plt.execXflxx(inuc, elem, 'xflxx_' + elem)
        if str2bool(params.getForEqs('xvar_' + elem)['plotMee']):
            plt.execXvar(inuc, elem, 'xvar_' + elem)

# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)


# EXECUTE MAIN
if __name__ == "__main__":
    main()

# END
