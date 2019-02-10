import PROMPI_single as psg
import warnings

warnings.filterwarnings("ignore")

fl_rans = 'C:\\Users\\mmocak\\Desktop\\GITDEV\\ransX\\RANSDAT\\he3d.45.nnuc6.lrez.00434.ransdat' 

ransdat = psg.PROMPI_single(fl_rans)

xbl = 3.e8
xbr = 1.e9

ransdat.SetMatplotlibParams()

# USAGE:

#ransdat.plot_lin_q1q2(xbl,xbr,'dd','tt',\
#                      r'r (10$^{8}$ cm)',\
#                      r'log $\rho$ (g cm$^{-3}$)',r'log T (K)',\
#                      r'$\rho$',r'$T$')

#ransdat.plot_log_q1q2(xbl,xbr,'enuc1','ei',\
#                       r'r (10$^{8}$ cm)',\
#                       r'$\varepsilon_{nuc}$ (erg $s^{-1}$)',\
#                       r'$\epsilon$ (ergs)',\
#                       r'$\varepsilon_{nuc}$',r'$\epsilon$')

#ransdat.plot_nablas(xbl,xbr)
#ransdat.plot_lin_q1(xbl,xbr,'psi',r'r (10$^{8}$ cm)','psi','psi')
#ransdat.plot_lin_q1(xbl,xbr,'ux',r'r (10$^{8}$ cm)','ux','ux')					   
#ransdat.plot_check_heq2(xbl,xbr)	   
ransdat.plot_lin_q1(xbl,xbr,'uyuz',r'r (10$^{8}$ cm)','ux','ux')
