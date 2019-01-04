###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: January/2019

import ReadParams as rp
import EQUATIONS.Properties as prop
import MasterPlot as plot
import ast 

paramFile = 'param.file'
params = rp.ReadParams(paramFile)

# get simulation properties
ransP = prop.Properties(params)
properties = ransP.execute()

# instantiate master plot 								 
plt = plot.MasterPlot(params)								 

# define useful functions
def str2bool(param):
    # True/False strings to proper boolean
    return ast.literal_eval(param)

# PLOT

# CONTINUITY EQUATION
if str2bool(params.getForEqs('rho')['plotMee']): plt.execRho()					
if str2bool(params.getForEqs('conteq')['plotMee']): plt.execContEq()
if str2bool(params.getForEqsBar('conteqBar')['plotMee']): plt.execContEqBar()

# MOMENTUM X EQUATION
if str2bool(params.getForEqs('momx')['plotMee']): plt.execMomx()
if str2bool(params.getForEqs('momxeq')['plotMee']): plt.execMomxEq()

# TURBULENT KINETIC ENERGY EQUATION
if str2bool(params.getForEqs('tke')['plotMee']): plt.execTke()
if str2bool(params.getForEqs('tkeeq')['plotMee']): plt.execTkeEq(properties['kolm_tke_diss_rate'])

# INTERNAL ENERGY EQUATION
if str2bool(params.getForEqs('eint')['plotMee']): plt.execEi()
if str2bool(params.getForEqs('eieq')['plotMee']): plt.execEiEq(properties['tke_diss'])

# INTERNAL ENERGY FLUX EQUATION
if str2bool(params.getForEqs('eintflx')['plotMee']): plt.execEiFlx()
if str2bool(params.getForEqs('eiflxeq')['plotMee']): plt.execEiFlxEq(properties['tke_diss'])

# INTERNAL ENERGY VARIANCE EQUATION
if str2bool(params.getForEqs('eintvar')['plotMee']): plt.execEiVar()
if str2bool(params.getForEqs('eivareq')['plotMee']): plt.execEiVarEq(properties['tke_diss'],properties['tauL'])

# ENTROPY EQUATION
if str2bool(params.getForEqs('entr')['plotMee']): plt.execSS()
if str2bool(params.getForEqs('sseq')['plotMee']): plt.execSSeq(properties['tke_diss'])

# ENTROPY FLUX EQUATION
if str2bool(params.getForEqs('entrflx')['plotMee']): plt.execSSflx()
if str2bool(params.getForEqs('ssflxeq')['plotMee']): plt.execSSflxEq(properties['tke_diss'])

# ENTROPY VARIANCE EQUATION
if str2bool(params.getForEqs('entrvar')['plotMee']): plt.execSSvar()
if str2bool(params.getForEqs('ssvareq')['plotMee']): plt.execSSvarEq(properties['tke_diss'],properties['tauL'])

# DENSITY VARIANCE EQUATION
if str2bool(params.getForEqs('densvar')['plotMee']): plt.execDDvar()
if str2bool(params.getForEqs('ddvareq')['plotMee']): plt.execDDvarEq(properties['tauL'])

# TURBULENT MASS FLUX EQUATION a.k.a A EQUATION
if str2bool(params.getForEqs('tmsflx')['plotMee']): plt.execTMSflx()
if str2bool(params.getForEqs('aeq')['plotMee']): plt.execAeq()

# DENSITY-SPECIFIC VOLUME COVARIANCE a.k.a. B EQUATION
if str2bool(params.getForEqs('dsvc')['plotMee']): plt.execDSVC()
if str2bool(params.getForEqs('beq')['plotMee']): plt.execBeq()

# load network
network = params.getNetwork() 

for elem in network[1:-1]: # skip network identifier in the list 
    inuc = params.getInuc(network,elem) 	
    if str2bool(params.getForEqs('xrho_'+elem)['plotMee']): plt.execXrho(inuc,elem,'xrho_'+elem)
    if str2bool(params.getForEqs('xtrseq_'+elem)['plotMee']): plt.execXtrsEq(inuc,elem,'xtrseq_'+elem)
    if str2bool(params.getForEqs('xflx_'+elem)['plotMee']): plt.execXflx(inuc,elem,'xflx_'+elem)	
    if str2bool(params.getForEqs('xflxeq_'+elem)['plotMee']): plt.execXflxEq(inuc,elem,'xflxeq_'+elem)
    if str2bool(params.getForEqs('xvar_'+elem)['plotMee']): plt.execXvar(inuc,elem,'xvar_'+elem)	
    if str2bool(params.getForEqs('xvareq_'+elem)['plotMee']): plt.execXvarEq(inuc,elem,'xvareq_'+elem,properties['tauL'])	
    if str2bool(params.getForEqs('xdiff_'+elem)['plotMee']): plt.execDiff(inuc,elem,'xdiff_'+elem,properties['lc'],properties['uconv'])