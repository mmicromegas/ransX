# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: November/2019

import UTILS.FOURIER.FourierReadParams as rp
import UTILS.FOURIER.FourierMasterPlot as plot
import ast
import os

# create os independent path and read parameter file
paramFile = os.path.join('PARAMS', 'param.fourier')
params = rp.FourierReadParams(paramFile)

# instantiate master plot  	 
plt = plot.FourierMasterPlot(params)

# obtain publication quality figures
plt.SetMatplotlibParams()


# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)


# TURBULENT KINETIC ENERGY FOURIER SPECTRUM
if str2bool(params.getForEqs('fstke')['plotMee']): plt.execFourierTKE()

# ux FOURIER "ENERGY" SPECTRUM
if str2bool(params.getForEqs('fsux')['plotMee']): plt.execFourierUx()

# uy FOURIER "ENERGY" SPECTRUM
if str2bool(params.getForEqs('fsuy')['plotMee']): plt.execFourierUy()

# uz FOURIER "ENERGY" SPECTRUM
if str2bool(params.getForEqs('fsuz')['plotMee']): plt.execFourierUz()

# DENSITY FOURIER "ENERGY" SPECTRUM
if str2bool(params.getForEqs('fsdd')['plotMee']): plt.execFourierDD()

# PRESSURE FOURIER "ENERGY" SPECTRUM
if str2bool(params.getForEqs('fspp')['plotMee']): plt.execFourierPP()

# TEMPERATURE FOURIER "ENERGY" SPECTRUM
if str2bool(params.getForEqs('fstt')['plotMee']): plt.execFourierTT()

# TOTAL ENERGY FOURIER "ENERGY" SPECTRUM
if str2bool(params.getForEqs('fset')['plotMee']): plt.execFourierET()

# load network
network = params.getNetwork()

# COMPOSITION FOURIER "ENERGY" SPECTRUM
for elem in network[1:]:  # skip network identifier in the list
    inuc = params.getInuc(network, elem)
    if str2bool(params.getForEqs('fs_' + elem)['plotMee']): plt.execFourierXrho(inuc, elem, 'fs_' + elem)
