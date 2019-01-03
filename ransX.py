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

paramFile = 'param.oblrez'
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

# plot	
if str2bool(params.getForEqs('rho')['plotMee']): plt.execRho()					
if str2bool(params.getForEqs('conteq')['plotMee']): plt.execContEq()
if str2bool(params.getForEqsBar('conteqBar')['plotMee']): plt.execContEqBar()

if str2bool(params.getForEqs('tke')['plotMee']): plt.execTke()
if str2bool(params.getForEqs('tkeeq')['plotMee']): plt.execTkeEq(properties['kolmrate'])

# load network
network = params.getNetwork() 

for elem in network[1:-1]: # skip network identifier in the list 
    inuc = params.getInuc(network,elem) 	
    if str2bool(params.getForEqs('xrho_'+elem)['plotMee']): plt.execXrho(inuc,elem,'xrho_'+elem)
    if str2bool(params.getForEqs('xtrs_'+elem)['plotMee']): plt.execXtrs(inuc,elem,'xtrs_'+elem)
    #if str2bool(params.getForEqs('xflx_'+elem)['plotMee']): plt.execXflx(inuc,elem,'xflx_'+elem)
    #if str2bool(params.getForEqs('xsig_'+elem)['plotMee']): plt.execXvar(inuc,elem,'xsig_'+elem)	

			  
#ransXMOM    =    xmom.XmomentumEquation(eht_data,ig,intc,prefix)
#ransTKE     =     tken.TurbulentKineticEnergyEquation(eht_data,ig,intc,-properties['kolmrate'],prefix)
#ransEINT    =    eint.InternalEnergyEquation(eht_data,ig,intc,properties['tke_diss'],prefix)
#ransENTR    =    entr.EntropyEquation(eht_data,ig,intc,properties['tke_diss'],prefix)
#ransSIGMASS = sigmass.EntropyVarianceEquation(eht_data,ig,intc,properties['tke_diss'],properties['tauL'],prefix)
#ransSIGMAEI = sigmaei.InternalEnergyVarianceEquation(eht_data,ig,intc,properties['tke_diss'],properties['tauL'],prefix)
#ransSIGMADD = sigmadd.DensityVarianceEquation(eht_data,ig,intc,,prefix)
#ransB       =       b.DensitySpecificVolumeCovarianceEquation(eht_data,ig,intc,prefix)
#ransFEIX    = feix.InternalEnergyFluxEquation(eht_data,ig,intc,properties['tke_diss'],prefix)
#ransFSSX    = fssx.EntropyFluxEquation(eht_data,ig,intc,properties['tke_diss'],prefix)
#ransA       = a.TurbulentMassFluxEquation(eht_data,ig,intc,prefix)

#ransXtra = xtra.XtransportEquation(eht_data,ig,inuc,intc,prefix)
#ransXflx = xflx.XfluxEquation(eht_data,ig,inuc,intc,prefix)
#ransXvar = xvar.XvarianceEquation(eht_data,ig,inuc,intc,prefix)
#ransXdif = xdif.Xdiffusivity(eht_data,ig,inuc,intc,prefix) 


# PLOT UX MOMENTUM EQUATION

#ransXMOM.plot_ux(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransXMOM.plot_x_momentum_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT TURBULENT KINETIC ENERGY EQUATION

#xbl = 3.5e8; xbr = 3.9e8 ; ybu = 2.e2; ybd = -2.e2; ilg = 0 # Ne-shell Cyril
#xbl = 3.7e8; xbr = 9.8e8 ; ybu = 3.e13; ybd = 0.; ilg = 0 # O-burn Meakin/Arnett 2007

#ransTKE.plot_tke(LAXIS,xbl,xbr,ybu,ybd,ilg)

#xbl = 3.5e8; xbr = 3.9e8 ; ybu = 2.e2; ybd = -2.e2; ilg = 0 # Ne-shell Cyril
#xbl = 3.7e8; xbr = 9.8e8 ; ybu = +6.e18; ybd = -6.e18; ilg = 0 # O-burn Meakin/Arnett 2007


#ransTKE.plot_tke_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT INTERNAL ENERGY EQUATION
#ransEINT.plot_ei(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransEINT.plot_ei_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT ENTROPY EQUATION
#ransENTR.plot_ss(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransENTR.plot_ss_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT ENTROPY VARIANCE EQUATION 
#ransSIGMASS.plot_sigma_ss(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransSIGMASS.plot_sigma_ss_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT ENTROPY FLUX EQUATION 
#ransFSSX.plot_fss(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransFSSX.plot_fss_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT INTERNAL ENERGY FLUX EQUATION 
#ransFEIX.plot_fei(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransFEIX.plot_fei_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT DENSITY SPECIFIC VOLUME COVARIANCE
#ransB.plot_b(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransB.plot_b_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT INTERNAL ENERGY VARIANCE EQUATION
#ransSIGMAEI.plot_sigma_ei(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransSIGMAEI.plot_sigma_ei_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT DENSITY VARIANCE EQUATION
#ransSIGMADD.plot_sigma_dd(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransSIGMADD.plot_sigma_dd_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)

# PLOT TURBULENT MASS FLUX EQUATION
#ransA.plot_a(LAXIS,xbl,xbr,ybu,ybd,ilg)
#ransA.plot_a_equation(LAXIS,xbl,xbr,ybu,ybd,ilg)


