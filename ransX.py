###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: January/2019

import EQUATIONS.Properties as prop
import UTILS.ReadParams as rp
import UTILS.MasterPlot as plot
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

# TEMPERATURE AND DENSITY 
if str2bool(params.getForEqs('ttdd')['plotMee']): plt.execRhoTemp()	

# PRESSURE AND INTERNAL ENERGY 
if str2bool(params.getForEqs('ppei')['plotMee']): plt.execPressEi()	

# NUCLEAR ENERGY PRODUCTION
if str2bool(params.getForEqs('enuc')['plotMee']): plt.execEnuc()	

# TEMPERATURE GRADIENTS
if str2bool(params.getForEqs('nablas')['plotMee']): plt.execNablas()

# DEGENERACY PARAMETER
if str2bool(params.getForEqs('psi')['plotMee']): plt.execDegeneracy()

# TURBULENT AND EXPANSION VELOCITY 
if str2bool(params.getForEqs('vel')['plotMee']): plt.execVelocities()

# BRUNT-VAISALLA FREQUENCY
if str2bool(params.getForEqs('nsq')['plotMee']): plt.execBruntV()

# BUOYANCY 
if str2bool(params.getForEqs('buo')['plotMee']): plt.execBuoyancy()

# CONTINUITY EQUATION
if str2bool(params.getForEqs('rho')['plotMee']): plt.execRho()					
if str2bool(params.getForEqs('conteq')['plotMee']): plt.execContEq()
if str2bool(params.getForEqsBar('conteqBar')['plotMee']): plt.execContEqBar()

# MOMENTUM X EQUATION
if str2bool(params.getForEqs('momex')['plotMee']): plt.execMomx()
if str2bool(params.getForEqs('momxeq')['plotMee']): plt.execMomxEq()

# MOMENTUM Y EQUATION
if str2bool(params.getForEqs('momey')['plotMee']): plt.execMomy()
if str2bool(params.getForEqs('momyeq')['plotMee']): plt.execMomyEq()

# MOMENTUM Z EQUATION
if str2bool(params.getForEqs('momez')['plotMee']): plt.execMomz()
if str2bool(params.getForEqs('momzeq')['plotMee']): plt.execMomzEq()

# REYNOLDS STRESS XX EQUATION
if str2bool(params.getForEqs('rxx')['plotMee']): plt.execRxx()
if str2bool(params.getForEqs('rexxeq')['plotMee']): plt.execRxxEq(properties['kolm_tke_diss_rate'])

# REYNOLDS STRESS YY EQUATION
if str2bool(params.getForEqs('ryy')['plotMee']): plt.execRyy()
if str2bool(params.getForEqs('reyyeq')['plotMee']): plt.execRyyEq(properties['kolm_tke_diss_rate'])

# REYNOLDS STRESS ZZ EQUATION
if str2bool(params.getForEqs('rzz')['plotMee']): plt.execRzz()
if str2bool(params.getForEqs('rezzeq')['plotMee']): plt.execRzzEq(properties['kolm_tke_diss_rate'])

# KINETIC ENERGY EQUATION
if str2bool(params.getForEqs('kine')['plotMee']): plt.execKe()
if str2bool(params.getForEqs('kieq')['plotMee']): plt.execKeEq(properties['kolm_tke_diss_rate'])

# TURBULENT KINETIC ENERGY EQUATION
if str2bool(params.getForEqs('tkie')['plotMee']): plt.execTke()
if str2bool(params.getForEqs('tkeeq')['plotMee']): plt.execTkeEq(properties['kolm_tke_diss_rate'])

# TOTAL ENERGY EQUATION
if str2bool(params.getForEqs('toe')['plotMee']): plt.execTe()
if str2bool(params.getForEqs('teeq')['plotMee']): plt.execTeEq(properties['kolm_tke_diss_rate'])

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

# MEAN NUMBER OF NUCLEONS PER ISOTOPE a.k.a ABAR EQUATION
if str2bool(params.getForEqs('abar')['plotMee']): plt.execAbar()
if str2bool(params.getForEqs('abreq')['plotMee']): plt.execAbarEq()

# MEAN CHARGE PER ISOTOPE a.k.a ZBAR EQUATION
if str2bool(params.getForEqs('zbar')['plotMee']): plt.execZbar()
if str2bool(params.getForEqs('zbreq')['plotMee']): plt.execZbarEq()

# load network
network = params.getNetwork() 

# COMPOSITION TRANSPORT, FLUX, VARIANCE EQUATIONS and EULERIAN DIFFUSIVITY
for elem in network[1:-1]: # skip network identifier in the list 
    inuc = params.getInuc(network,elem) 	
    if str2bool(params.getForEqs('xrho_'+elem)['plotMee']): plt.execXrho(inuc,elem,'xrho_'+elem)
    if str2bool(params.getForEqs('xtrseq_'+elem)['plotMee']): plt.execXtrsEq(inuc,elem,'xtrseq_'+elem)
    if str2bool(params.getForEqs('xflx_'+elem)['plotMee']): plt.execXflx(inuc,elem,'xflx_'+elem)	
    if str2bool(params.getForEqs('xflxeq_'+elem)['plotMee']): plt.execXflxEq(inuc,elem,'xflxeq_'+elem)
    if str2bool(params.getForEqs('xvar_'+elem)['plotMee']): plt.execXvar(inuc,elem,'xvar_'+elem)	
    if str2bool(params.getForEqs('xvareq_'+elem)['plotMee']): plt.execXvarEq(inuc,elem,'xvareq_'+elem,properties['tauL'])	
    if str2bool(params.getForEqs('xdiff_'+elem)['plotMee']): plt.execDiff(inuc,elem,'xdiff_'+elem,properties['lc'],properties['uconv'])