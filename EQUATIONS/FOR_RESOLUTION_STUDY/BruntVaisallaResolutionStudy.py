import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import UTILS.CALCULUS as calc
import UTILS.ALIMIT as al

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

class BruntVaisallaResolutionStudy(calc.CALCULUS,al.ALIMIT,object):

    def __init__(self,filename,ig,intc,data_prefix):
        super(BruntVaisallaResolutionStudy,self).__init__(ig) 
	
        # load data to list of structured arrays
        eht = []		
        for file in filename:
            eht.append(np.load(file))
            print(file)			

        # declare data lists
        dd,pp,gg,gamma1,dlnrhodr,dlnpdr,dlnrhodrs,nsq,chim,chit,chid,mu,tt,gamma2,\
        alpha,delta,phi,hp,lntt,lnpp,lnmu,nabla,nabla_ad,nabla_mu,nsq_version2,xzn0 = \
        [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
			
        for i in range(len(filename)):			
            # load grid
            xzn0.append(np.asarray(eht[i].item().get('xzn0'))) 	

            # pick specific Reynolds-averaged mean fields according to:
            # https://github.com/mmicromegas/ransX/blob/master/DOCS/ransXimplementationGuide.pdf 		
		
            dd.append(np.asarray(eht[i].item().get('dd')[intc])) 
            pp.append(np.asarray(eht[i].item().get('pp')[intc])) 
            gg.append(np.asarray(eht[i].item().get('gg')[intc]))
            gamma1.append(np.asarray(eht[i].item().get('gamma1')[intc]))
		
            dlnrhodr.append(self.deriv(np.log(dd[i]),xzn0[i]))
            dlnpdr.append(self.deriv(np.log(pp[i]),xzn0[i]))
            dlnrhodrs.append((1./gamma1[i])*dlnpdr[i])
            nsq.append(gg[i]*(dlnrhodr[i]-dlnrhodrs[i]))
	
            chim.append(np.asarray(eht[i].item().get('chim')[intc])) 
            chit.append(np.asarray(eht[i].item().get('chit')[intc])) 
            chid.append(np.asarray(eht[i].item().get('chid')[intc]))
            mu.append(np.asarray(eht[i].item().get('abar')[intc])) 		
            tt.append(np.asarray(eht[i].item().get('tt')[intc]))
            gamma2.append(np.asarray(eht[i].item().get('gamma2')[intc]))
		
            alpha.append(1./chid[i])
            delta.append(-chit[i]/chid[i])
            phi.append(chid[i]/chim[i])
            hp.append(-pp[i]/self.Grad(pp[i],xzn0[i]))  		
	
            lntt.append(np.log(tt[i]))
            lnpp.append(np.log(pp[i]))
            lnmu.append(np.log(mu[i]))

            # calculate temperature gradients		
            nabla.append(self.deriv(lntt[i],lnpp[i])) 
            nabla_ad.append((gamma2[i]-1.)/gamma2[i])
            nabla_mu.append((chim[i]/chit[i])*self.deriv(lnmu[i],lnpp[i]))	
		
		    # Kippenhahn and Weigert, p.42 but with opposite (minus) sign at the (phi/delta)*nabla_mu
            nsq_version2.append((gg[i]*delta[i]/hp[i])*(nabla_ad[i] - nabla[i] - (phi[i]/delta[i])*nabla_mu[i])) 		

        # share data globally  			
        self.xzn0 = xzn0
        self.nsq = nsq
        self.nsq_version2 = nsq_version2		
			
    def plot_bruntvaisalla(self,LAXIS,xbl,xbr,ybu,ybd,ilg):
        """Plot BruntVaisalla parameter in the model""" 

        # load x GRID
        grd = self.xzn0
	
        # load DATA to plot
        plt1 = self.nsq[1]
        plt2 = self.nsq_version2[1]
		
        plt3 = np.interp(grd[1],grd[0],self.nsq[0])
        plt4 = np.interp(grd[1],grd[0],self.nsq_version2[0])
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # set plot boundaries   
        to_plot = [plt1,plt2,plt3,plt4]		
        self.set_plt_axis(LAXIS,xbl,xbr,ybu,ybd,to_plot)	
		
        # plot DATA 
        plt.title('Brunt-Vaisalla frequency')
        plt.plot(grd[1],plt1,color='r',label = r'N$^2$')
#        plt.plot(grd[1],plt2,color='b',linestyle='--',label = r'N$^2$ version 2')
        plt.plot(grd[1],plt3,color='g',label = r'N$^2$')
#        plt.plot(grd[1],plt4,color='b',linestyle='--',label = r'N$^2$ version 2')
		
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"N$^2$"

        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=ilg,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
#        plt.savefig('RESULTS/'+self.data_prefix+'mean_BruntVaisalla.png')
	
	