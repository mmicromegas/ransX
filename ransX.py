###############################################
# rans(eXtreme) https://arxiv.org/abs/1401.5176
###############################################

# Author: Miroslav Mocak 
# Email: miroslav.mocak@gmail.com 
# Date: January/2019

import EQUATIONS.Properties as prop
import UTILS.ReadParamsRansX as rp
import UTILS.MasterPlot as plot
import ast 

paramFile = 'param.ransx'
params = rp.ReadParamsRansX(paramFile)

# get data source file
filename = params.getForProp('prop')['eht_data']

# get simulation properties
ransP = prop.Properties(params,filename)
properties = ransP.execute()

# instantiate master plot 								 
plt = plot.MasterPlot(params)								 

# obtain publication quality figures
plt.SetMatplotlibParams()

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

# MEAN AND EXPANSION VELOCITY 
if str2bool(params.getForEqs('velbgr')['plotMee']): plt.execVelocitiesMeanExp()

# MLT AND TURBULENT VELOCITY 
if str2bool(params.getForEqs('velmlt')['plotMee']): plt.execVelocitiesMLTturb()

# BRUNT-VAISALLA FREQUENCY
if str2bool(params.getForEqs('nsq')['plotMee']): plt.execBruntV()

# BUOYANCY 
if str2bool(params.getForEqs('buo')['plotMee']): plt.execBuoyancy()

# RELATIVE RMS FLUCTUATIONS
if str2bool(params.getForEqs('relrmsflct')['plotMee']): plt.execRelativeRmsFlct()

# ABAR and ZBAR 
if str2bool(params.getForEqs('abzb')['plotMee']): plt.execAbarZbar()

# CONTINUITY EQUATION
if str2bool(params.getForEqs('rho')['plotMee']): plt.execRho()					
if str2bool(params.getForEqs('conteq')['plotMee']): plt.execContEq()
if str2bool(params.getForEqsBar('conteqBar')['plotMee']): plt.execContEqBar()
if str2bool(params.getForEqs('conteqfdd')['plotMee']): plt.execContFddEq()
if str2bool(params.getForEqsBar('conteqfddBar')['plotMee']): plt.execContFddEqBar()

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
if str2bool(params.getForEqs('tkie')['plotMee']): plt.execTke(properties['kolm_tke_diss_rate'],properties['xzn0inc'],properties['xzn0outc'])
if str2bool(params.getForEqs('tkeeq')['plotMee']): plt.execTkeEq(properties['kolm_tke_diss_rate'],properties['xzn0inc'],properties['xzn0outc'])

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

# PRESSURE EQUATION
if str2bool(params.getForEqs('press')['plotMee']): plt.execPP()
if str2bool(params.getForEqs('ppeq')['plotMee']): plt.execPPeq(properties['tke_diss'])

# PRESSURE FLUX EQUATION in X
if str2bool(params.getForEqs('pressxflx')['plotMee']): plt.execPPxflx()
if str2bool(params.getForEqs('ppxflxeq')['plotMee']): plt.execPPxflxEq(properties['tke_diss'])

# PRESSURE FLUX EQUATION in Y
if str2bool(params.getForEqs('pressyflx')['plotMee']): plt.execPPyflx()
if str2bool(params.getForEqs('ppyflxeq')['plotMee']): plt.execPPyflxEq(properties['tke_diss'])

# PRESSURE FLUX EQUATION in Z
if str2bool(params.getForEqs('presszflx')['plotMee']): plt.execPPzflx()
if str2bool(params.getForEqs('ppzflxeq')['plotMee']): plt.execPPzflxEq(properties['tke_diss'])

# PRESSURE VARIANCE EQUATION
if str2bool(params.getForEqs('pressvar')['plotMee']): plt.execPPvar()
if str2bool(params.getForEqs('ppvareq')['plotMee']): plt.execPPvarEq(properties['tke_diss'],properties['tauL'])

# TEMPERATURE EQUATION
if str2bool(params.getForEqs('temp')['plotMee']): plt.execTT()
if str2bool(params.getForEqs('tteq')['plotMee']): plt.execTTeq(properties['tke_diss'])

# TEMPERATURE FLUX EQUATION
if str2bool(params.getForEqs('tempflx')['plotMee']): plt.execTTflx()
if str2bool(params.getForEqs('ttflxeq')['plotMee']): plt.execTTflxEq(properties['tke_diss'])

# TEMPERATURE VARIANCE EQUATION
if str2bool(params.getForEqs('tempvar')['plotMee']): plt.execTTvar()
if str2bool(params.getForEqs('ttvareq')['plotMee']): plt.execTTvarEq(properties['tke_diss'],properties['tauL'])

# ENTHALPY EQUATION
if str2bool(params.getForEqs('enth')['plotMee']): plt.execHH()
if str2bool(params.getForEqs('hheq')['plotMee']): plt.execHHeq(properties['tke_diss'])

# ENTHALPY FLUX EQUATION
if str2bool(params.getForEqs('enthflx')['plotMee']): plt.execHHflx()
if str2bool(params.getForEqs('hhflxeq')['plotMee']): plt.execHHflxEq(properties['tke_diss'])

# ENTHALPY VARIANCE EQUATION
if str2bool(params.getForEqs('enthvar')['plotMee']): plt.execHHvar()
if str2bool(params.getForEqs('hhvareq')['plotMee']): plt.execHHvarEq(properties['tke_diss'],properties['tauL'])

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

# ABAR FLUX EQUATION
if str2bool(params.getForEqs('abflx')['plotMee']): plt.execFabarx()
if str2bool(params.getForEqs('fabxeq')['plotMee']): plt.execFabarxEq()

# MEAN CHARGE PER ISOTOPE a.k.a ZBAR EQUATION
if str2bool(params.getForEqs('zbar')['plotMee']): plt.execZbar()
if str2bool(params.getForEqs('zbreq')['plotMee']): plt.execZbarEq()

# ZBAR FLUX EQUATION
if str2bool(params.getForEqs('zbflx')['plotMee']): plt.execFzbarx()
if str2bool(params.getForEqs('fzbxeq')['plotMee']): plt.execFzbarxEq()

# HYDRODYNAMIC STELLAR STRUCTURE EQUATIONS
if str2bool(params.getForEqs('cteqhsse')['plotMee']): plt.execHssContEq(properties['xzn0inc'],properties['xzn0outc'])
if str2bool(params.getForEqs('mxeqhsse')['plotMee']): plt.execHssMomxEq(properties['xzn0inc'],properties['xzn0outc'])
if str2bool(params.getForEqs('tpeqhsse')['plotMee']): plt.execHssTempEq(properties['tke_diss'],properties['xzn0inc'],properties['xzn0outc'])
if str2bool(params.getForEqs('lueqhsse')['plotMee']): plt.execHssLumiEq(properties['tke_diss'],properties['xzn0inc'],properties['xzn0outc'])

# FULL TURBULENT VELOCITY FIELD HYPOTHESIS
if str2bool(params.getForEqs('ftvfh')['plotMee']): plt.execFtvfh()

# load network
network = params.getNetwork() 

# COMPOSITION TRANSPORT, FLUX, VARIANCE EQUATIONS and EULERIAN DIFFUSIVITY
for elem in network[1:]: # skip network identifier in the list 
    inuc = params.getInuc(network,elem) 	
    if str2bool(params.getForEqs('xrho_'+elem)['plotMee']): plt.execXrho(inuc,elem,'xrho_'+elem,properties['xzn0inc'],properties['xzn0outc'])
    if str2bool(params.getForEqs('xtrseq_'+elem)['plotMee']): plt.execXtrsEq(inuc,elem,'xtrseq_'+elem,properties['xzn0inc'],properties['xzn0outc'])
    if str2bool(params.getForEqsBar('xtrseq_'+elem+'Bar')['plotMee']): plt.execXtrsEqBar(inuc,elem,'xtrseq_'+elem+'Bar',properties['xzn0inc'],properties['xzn0outc'])
    if str2bool(params.getForEqs('xflxx_'+elem)['plotMee']): plt.execXflxx(inuc,elem,'xflxx_'+elem,properties['xzn0inc'],properties['xzn0outc'],properties['tke_diss'],properties['tauL'])	
    if str2bool(params.getForEqs('xflxy_'+elem)['plotMee']): plt.execXflxy(inuc,elem,'xflxy_'+elem,properties['xzn0inc'],properties['xzn0outc'],properties['tke_diss'],properties['tauL'])	
    if str2bool(params.getForEqs('xflxz_'+elem)['plotMee']): plt.execXflxz(inuc,elem,'xflxz_'+elem,properties['xzn0inc'],properties['xzn0outc'],properties['tke_diss'],properties['tauL'])		
    if str2bool(params.getForEqs('xflxxeq_'+elem)['plotMee']): plt.execXflxXeq(inuc,elem,'xflxxeq_'+elem,properties['xzn0inc'],properties['xzn0outc'],properties['tke_diss'],properties['tauL'])
    if str2bool(params.getForEqs('xflxyeq_'+elem)['plotMee']): plt.execXflxYeq(inuc,elem,'xflxyeq_'+elem,properties['xzn0inc'],properties['xzn0outc'],properties['tke_diss'],properties['tauL'])
    if str2bool(params.getForEqs('xflxzeq_'+elem)['plotMee']): plt.execXflxZeq(inuc,elem,'xflxzeq_'+elem,properties['xzn0inc'],properties['xzn0outc'],properties['tke_diss'],properties['tauL'])
    if str2bool(params.getForEqs('xvar_'+elem)['plotMee']): plt.execXvar(inuc,elem,'xvar_'+elem,properties['xzn0inc'],properties['xzn0outc'])	
    if str2bool(params.getForEqs('xvareq_'+elem)['plotMee']): plt.execXvarEq(inuc,elem,'xvareq_'+elem,properties['tauL'],properties['xzn0inc'],properties['xzn0outc'])	
    if str2bool(params.getForEqs('xdiff_'+elem)['plotMee']): plt.execDiff(inuc,elem,'xdiff_'+elem,properties['lc'],properties['uconv'],properties['xzn0inc'],properties['xzn0outc'])
    # HYDRODYNAMIC STELLAR STRUCTURE COMPOSITION TRANSPORT EQUATION	
    if str2bool(params.getForEqs('coeqhsse_'+elem)['plotMee']): plt.execHssCompEq(inuc,elem,'coeqhsse_'+elem,properties['xzn0inc'],properties['xzn0outc'])
    # DAMKOHLER NUMBER DISTRIBUTION	
    if str2bool(params.getForEqs('xda_'+elem)['plotMee']): plt.execXda(inuc,elem,'xda_'+elem,properties['xzn0inc'],properties['xzn0outc'])
	
	
