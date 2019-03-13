import numpy as np
import matplotlib.pyplot as plt

# class for plot axis limitation

class ALIMIT:

    def idx_bndry(self,xbl,xbr):
    # calculate indices of grid boundaries 
        xzn0 = np.asarray(self.xzn0)
        xlm = np.abs(xzn0-xbl)
        xrm = np.abs(xzn0-xbr)
        idxl = int(np.where(xlm==xlm.min())[0][0])
        idxr = int(np.where(xrm==xrm.min())[0][0])	
        return idxl,idxr
	
	
    def set_plt_axis(self,LAXIS,xbl,xbr,ybu,ybd,to_plot):

        # calculate INDICES for grid boundaries 
        if LAXIS == 1:
            idxl, idxr = self.idx_bndry(xbl,xbr)		
				
        number_of_curves = len(to_plot)
        #print(number_of_curves)
		
        if (number_of_curves == 1):
		
            # limit x/y axis
            if LAXIS == 0: 
                plt.axis([self.xzn0[0],self.xzn0[-1],np.min(to_plot[0][0:-1]),np.max(to_plot[0][0:-1])])
            if LAXIS == 1:
                plt.axis([xbl,xbr,np.min(to_plot[0][idxl:idxr]),np.max(to_plot[0][idxl:idxr])])
				
        if (number_of_curves == 2): 
            # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1])])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])
            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr])])
                plt.axis([xbl,xbr,minx,maxx])				
				
        if (number_of_curves == 3): 
            # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),np.min(to_plot[2][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),np.max(to_plot[2][0:-1])])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])
            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),np.min(to_plot[2][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),np.max(to_plot[2][idxl:idxr])])
                plt.axis([xbl,xbr,minx,maxx])
				
        if (number_of_curves == 4):						

            # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),np.min(to_plot[2][0:-1]),np.min(to_plot[3][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1])])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])
            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),np.min(to_plot[2][idxl:idxr]),np.min(to_plot[3][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),np.max(to_plot[2][idxl:idxr]),np.max(to_plot[3][idxl:idxr])])
                plt.axis([xbl,xbr,minx,maxx])		
						
        if (number_of_curves == 5):
		
            # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),np.min(to_plot[2][0:-1]),np.min(to_plot[3][0:-1]),np.min(to_plot[4][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1]),np.max(to_plot[4][0:-1])])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])	

            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),np.min(to_plot[2][idxl:idxr]),np.min(to_plot[3][idxl:idxr]),np.min(to_plot[4][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),np.max(to_plot[2][idxl:idxr]),np.max(to_plot[3][idxl:idxr]),np.max(to_plot[4][idxl:idxr])])
                plt.axis([xbl,xbr,minx,maxx])

        if (number_of_curves == 6):
				
		    # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),np.min(to_plot[2][0:-1]),np.min(to_plot[3][0:-1]),np.min(to_plot[4][0:-1]),np.min(to_plot[5][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1]),np.max(to_plot[4][0:-1]),np.max(to_plot[5][0:-1])])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])			
            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),np.min(to_plot[2][idxl:idxr]),\
			    np.min(to_plot[3][idxl:idxr]),np.min(to_plot[4][idxl:idxr]),np.min(to_plot[5][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),np.max(to_plot[2][idxl:idxr]),\
			    np.max(to_plot[3][idxl:idxr]),np.max(to_plot[4][idxl:idxr]),np.max(to_plot[5][idxl:idxr])])
                plt.axis([xbl,xbr,minx,maxx])

		
        if (number_of_curves == 7):		
		
            # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),np.min(to_plot[2][0:-1]),np.min(to_plot[3][0:-1]),np.min(to_plot[4][0:-1]),np.min(to_plot[5][0:-1]),np.min(to_plot[6][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1]),np.min(to_plot[4][0:-1]),np.min(to_plot[5][0:-1]),np.max(to_plot[6][0:-1])])
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])	

            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),np.min(to_plot[2][idxl:idxr]),np.min(to_plot[3][idxl:idxr]),np.min(to_plot[4][idxl:idxr]),np.min(to_plot[5][idxl:idxr]),np.min(to_plot[6][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),np.max(to_plot[2][idxl:idxr]),np.max(to_plot[3][idxl:idxr]),np.max(to_plot[4][idxl:idxr]),np.max(to_plot[5][idxl:idxr]),np.max(to_plot[6][idxl:idxr])])
                plt.axis([xbl,xbr,minx,maxx])
            
        if (number_of_curves == 8):	

            # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),np.min(to_plot[2][0:-1]),\
			    np.min(to_plot[3][0:-1]),np.min(to_plot[4][0:-1]),np.min(to_plot[5][0:-1]),np.min(to_plot[6][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1]),\
			    np.max(to_plot[4][0:-1]),np.max(to_plot[5][0:-1]),np.max(to_plot[6][0:-1]),np.max(to_plot[7][0:-1])])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])
				
            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),np.min(to_plot[2][idxl:idxr]),\
			    np.min(to_plot[3][idxl:idxr]),np.min(to_plot[4][idxl:idxr]),np.min(to_plot[5][idxl:idxr]),np.min(to_plot[6][idxl:idxr]),np.min(to_plot[7][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),np.max(to_plot[2][idxl:idxr]),\
			    np.max(to_plot[3][idxl:idxr]),np.max(to_plot[4][idxl:idxr]),np.max(to_plot[5][idxl:idxr]),np.max(to_plot[6][idxl:idxr]),np.max(to_plot[7][idxl:idxr])])
                plt.axis([xbl,xbr,minx,maxx])
				
        if (number_of_curves == 9):

            # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),\
	                           np.min(to_plot[2][0:-1]),np.min(to_plot[3][0:-1]),\
	                           np.min(to_plot[4][0:-1]),np.min(to_plot[5][0:-1]),\
	                           np.min(to_plot[6][0:-1]),np.min(to_plot[7][0:-1]),\
	                           np.min(to_plot[8][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),\
	                           np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1]),\
	                           np.max(to_plot[4][0:-1]),np.max(to_plot[5][0:-1]),\
	                           np.max(to_plot[6][0:-1]),np.max(to_plot[7][0:-1]),\
	                           np.max(to_plot[8][0:-1])])
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])

            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),\
	                           np.min(to_plot[2][idxl:idxr]),np.min(to_plot[3][idxl:idxr]),\
	                           np.min(to_plot[4][idxl:idxr]),np.min(to_plot[5][idxl:idxr]),\
	                           np.min(to_plot[6][idxl:idxr]),np.min(to_plot[7][idxl:idxr]),\
	                           np.min(to_plot[8][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),\
	                           np.max(to_plot[2][idxl:idxr]),np.max(to_plot[3][idxl:idxr]),\
	                           np.max(to_plot[4][idxl:idxr]),np.max(to_plot[5][idxl:idxr]),\
	                           np.max(to_plot[6][idxl:idxr]),np.max(to_plot[7][idxl:idxr]),\
	                           np.max(to_plot[8][idxl:idxr])])
                plt.axis([xbl,xbr,minx,maxx])

        if (number_of_curves == 10):
        # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),\
                               np.min(to_plot[2][0:-1]),np.min(to_plot[3][0:-1]),\
                               np.min(to_plot[4][0:-1]),np.min(to_plot[5][0:-1]),\
                               np.min(to_plot[6][0:-1]),np.min(to_plot[7][0:-1]),\
                               np.min(to_plot[8][0:-1]),np.min(to_plot[9][0:-1]),\
                               ])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),\
                               np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1]),\
                               np.max(to_plot[4][0:-1]),np.max(to_plot[5][0:-1]),\
                               np.max(to_plot[6][0:-1]),np.max(to_plot[7][0:-1]),\
                               np.max(to_plot[8][0:-1]),np.max(to_plot[9][0:-1]),\
                               ])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])
            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),\
                               np.min(to_plot[2][idxl:idxr]),np.min(to_plot[3][idxl:idxr]),\
                               np.min(to_plot[4][idxl:idxr]),np.min(to_plot[5][idxl:idxr]),\
                               np.min(to_plot[6][idxl:idxr]),np.min(to_plot[7][idxl:idxr]),\
                               np.min(to_plot[8][idxl:idxr]),np.min(to_plot[9][idxl:idxr]),\
                               ])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),\
                               np.max(to_plot[2][idxl:idxr]),np.max(to_plot[3][idxl:idxr]),\
                               np.max(to_plot[4][idxl:idxr]),np.max(to_plot[5][idxl:idxr]),\
                               np.max(to_plot[6][idxl:idxr]),np.max(to_plot[7][idxl:idxr]),\
                               np.max(to_plot[8][idxl:idxr]),np.max(to_plot[9][idxl:idxr]),\
                               ])			
                plt.axis([xbl,xbr,minx,maxx])				
				
        if (number_of_curves == 11):
        # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),\
                               np.min(to_plot[2][0:-1]),np.min(to_plot[3][0:-1]),\
                               np.min(to_plot[4][0:-1]),np.min(to_plot[5][0:-1]),\
                               np.min(to_plot[6][0:-1]),np.min(to_plot[7][0:-1]),\
                               np.min(to_plot[8][0:-1]),np.min(to_plot[9][0:-1]),\
                               np.min(to_plot[10][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),\
                               np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1]),\
                               np.max(to_plot[4][0:-1]),np.max(to_plot[5][0:-1]),\
                               np.max(to_plot[6][0:-1]),np.max(to_plot[7][0:-1]),\
                               np.max(to_plot[8][0:-1]),np.max(to_plot[9][0:-1]),\
                               np.max(to_plot[10][0:-1])])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])
            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),\
                               np.min(to_plot[2][idxl:idxr]),np.min(to_plot[3][idxl:idxr]),\
                               np.min(to_plot[4][idxl:idxr]),np.min(to_plot[5][idxl:idxr]),\
                               np.min(to_plot[6][idxl:idxr]),np.min(to_plot[7][idxl:idxr]),\
                               np.min(to_plot[8][idxl:idxr]),np.min(to_plot[9][idxl:idxr]),\
                               np.min(to_plot[10][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),\
                               np.max(to_plot[2][idxl:idxr]),np.max(to_plot[3][idxl:idxr]),\
                               np.max(to_plot[4][idxl:idxr]),np.max(to_plot[5][idxl:idxr]),\
                               np.max(to_plot[6][idxl:idxr]),np.max(to_plot[7][idxl:idxr]),\
                               np.max(to_plot[8][idxl:idxr]),np.max(to_plot[9][idxl:idxr]),\
                               np.max(to_plot[10][idxl:idxr])])			
                plt.axis([xbl,xbr,minx,maxx])

        if (number_of_curves == 12):
        # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),\
                               np.min(to_plot[2][0:-1]),np.min(to_plot[3][0:-1]),\
                               np.min(to_plot[4][0:-1]),np.min(to_plot[5][0:-1]),\
                               np.min(to_plot[6][0:-1]),np.min(to_plot[7][0:-1]),\
                               np.min(to_plot[8][0:-1]),np.min(to_plot[9][0:-1]),\
                               np.min(to_plot[10][0:-1]),np.min(to_plot[11][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),\
                               np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1]),\
                               np.max(to_plot[4][0:-1]),np.max(to_plot[5][0:-1]),\
                               np.max(to_plot[6][0:-1]),np.max(to_plot[7][0:-1]),\
                               np.max(to_plot[8][0:-1]),np.max(to_plot[9][0:-1]),\
                               np.max(to_plot[10][0:-1]),np.max(to_plot[11][0:-1])])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])
            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),\
                               np.min(to_plot[2][idxl:idxr]),np.min(to_plot[3][idxl:idxr]),\
                               np.min(to_plot[4][idxl:idxr]),np.min(to_plot[5][idxl:idxr]),\
                               np.min(to_plot[6][idxl:idxr]),np.min(to_plot[7][idxl:idxr]),\
                               np.min(to_plot[8][idxl:idxr]),np.min(to_plot[9][idxl:idxr]),\
                               np.min(to_plot[10][idxl:idxr]),np.min(to_plot[11][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),\
                               np.max(to_plot[2][idxl:idxr]),np.max(to_plot[3][idxl:idxr]),\
                               np.max(to_plot[4][idxl:idxr]),np.max(to_plot[5][idxl:idxr]),\
                               np.max(to_plot[6][idxl:idxr]),np.max(to_plot[7][idxl:idxr]),\
                               np.max(to_plot[8][idxl:idxr]),np.max(to_plot[9][idxl:idxr]),\
                               np.max(to_plot[10][idxl:idxr]),np.max(to_plot[11][idxl:idxr])])			
                plt.axis([xbl,xbr,minx,maxx])
				

        if (number_of_curves == 13):
        # limit x/y axis by global min/max from all terms
            if LAXIS == 0:
                minx = np.min([np.min(to_plot[0][0:-1]),np.min(to_plot[1][0:-1]),\
                               np.min(to_plot[2][0:-1]),np.min(to_plot[3][0:-1]),\
                               np.min(to_plot[4][0:-1]),np.min(to_plot[5][0:-1]),\
                               np.min(to_plot[6][0:-1]),np.min(to_plot[7][0:-1]),\
                               np.min(to_plot[8][0:-1]),np.min(to_plot[9][0:-1]),\
                               np.min(to_plot[10][0:-1]),np.min(to_plot[11][0:-1]),\
							   np.min(to_plot[12][0:-1])])
                maxx = np.max([np.max(to_plot[0][0:-1]),np.max(to_plot[1][0:-1]),\
                               np.max(to_plot[2][0:-1]),np.max(to_plot[3][0:-1]),\
                               np.max(to_plot[4][0:-1]),np.max(to_plot[5][0:-1]),\
                               np.max(to_plot[6][0:-1]),np.max(to_plot[7][0:-1]),\
                               np.max(to_plot[8][0:-1]),np.max(to_plot[9][0:-1]),\
                               np.max(to_plot[10][0:-1]),np.max(to_plot[11][0:-1]),\
                               np.max(to_plot[12][0:-1])])			
                plt.axis([self.xzn0[0],self.xzn0[-1],minx,maxx])
            if LAXIS == 1:
                minx = np.min([np.min(to_plot[0][idxl:idxr]),np.min(to_plot[1][idxl:idxr]),\
                               np.min(to_plot[2][idxl:idxr]),np.min(to_plot[3][idxl:idxr]),\
                               np.min(to_plot[4][idxl:idxr]),np.min(to_plot[5][idxl:idxr]),\
                               np.min(to_plot[6][idxl:idxr]),np.min(to_plot[7][idxl:idxr]),\
                               np.min(to_plot[8][idxl:idxr]),np.min(to_plot[9][idxl:idxr]),\
                               np.min(to_plot[10][idxl:idxr]),np.min(to_plot[11][idxl:idxr]),\
                               np.min(to_plot[12][idxl:idxr])])
                maxx = np.max([np.max(to_plot[0][idxl:idxr]),np.max(to_plot[1][idxl:idxr]),\
                               np.max(to_plot[2][idxl:idxr]),np.max(to_plot[3][idxl:idxr]),\
                               np.max(to_plot[4][idxl:idxr]),np.max(to_plot[5][idxl:idxr]),\
                               np.max(to_plot[6][idxl:idxr]),np.max(to_plot[7][idxl:idxr]),\
                               np.max(to_plot[8][idxl:idxr]),np.max(to_plot[9][idxl:idxr]),\
                               np.max(to_plot[10][idxl:idxr]),np.max(to_plot[11][idxl:idxr]),\
                               np.max(to_plot[12][idxl:idxr])])			
                plt.axis([xbl,xbr,minx,maxx])
			
        if LAXIS == 2:
            plt.axis([xbl,xbr,ybd,ybu])
				