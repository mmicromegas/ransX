import UTILS.PROMPI.PROMPI_single as psg
import UTILS.SINGLE.ReadParamsSingle as rps
import warnings

warnings.filterwarnings("ignore")

# read input parameters
	   
paramFile = 'param.single'
params = rps.ReadParamsSingle(paramFile)	
 
datafile = params.getForSingle('single')['datafile']
endianness = params.getForSingle('single')['endianness']
precision = params.getForSingle('single')['precision']

xbl = params.getForSingle('single')['xbl']
xbr = params.getForSingle('single')['xbr']

q2plot = params.getForSingle('single')['q']

ransdat = psg.PROMPI_single(datafile,endianness,precision)

ransdat.SetMatplotlibParams()

for q in q2plot:
    ransdat.plot_lin_q1(xbl,xbr,q,r'r (10$^{8}$ cm)',q,q)

#for q in q2plot:
#    ransdat.plot_log_q1(xbl,xbr,q,r'r (10$^{8}$ cm)',q,q)
	
ransdat.plot_check_heq1()	   
ransdat.plot_check_heq2(xbl,xbr)
ransdat.plot_check_heq3()
ransdat.plot_check_ux(xbl,xbr)
#ransdat.plot_nablas(xbl,xbr)
#ransdat.plot_dx(xbl,xbr)
#ransdat.plot_mm(xbl,xbr)
#ransdat.PlotNucEnergyGen(xbl,xbr)

#ransdat.plot_lin_q1q2(xbl,xbr,'dd','tt',\
#                      r'r (10$^{8}$ cm)',\
#                      r'log $\rho$ (g cm$^{-3}$)',r'log T (K)',\
#                      r'$\rho$',r'$T$')

#ransdat.plot_log_q1q2(xbl,xbr,'enuc1','ei',\
#                       r'r (10$^{8}$ cm)',\
#                       r'$\varepsilon_{nuc}$ (erg $s^{-1}$)',\
#                       r'$\epsilon$ (ergs)',\
#                       r'$\varepsilon_{nuc}$',r'$\epsilon$')
