import re # python regular expressions
import sys

class FourierReadParams:

    def __init__(self,filename):

        file=open(filename,'r')
        next(file) # skip header line
        next(file) # skip header line

        input=[]
        for line in file:
            # parse out values from square brackets 
            prsvalue = re.search(r'\[(.*)\]', line).group(1)
            input.append(prsvalue)
        file.close()
        
        self.input = input

				
    def getForProp(self,param): 
        
        # choose only lists identified by param        
        match = [s for s in self.input if param in s]       
        datafile  = match[0].split(",")[2]		
        prefix    = match[1].split(",")[2]
        endian    = match[2].split(",")[2]
        precision = match[3].split(",")[2]
        ig        = match[4].split(",")[2]
        laxis     = match[5].split(",")[2]		
        lhc       = match[6].split(",")[2]
        
        return {'datafile':datafile,'prefix':prefix,'endian':endian,'precision':precision,'ig': ig,'laxis':laxis,'lhc':lhc}				

    def getForEqs(self,param): 

        # choose only lists identified by param        
        match = [s for s in self.input if param in s]
        #print(param,match)
        match_split = match[0].split(",")
        #equation = match_split[0]
        plotMee = match_split[1]
        xbl = float(match_split[2])
        xbr = float(match_split[3])
        ybu = float(match_split[4])
        ybd = float(match_split[5])		
        ilg = int(match_split[6])		
		
        return {'plotMee':plotMee,'xbl':xbl,'xbr':xbr,'ybu':ybu,'ybd':ybd,'ilg':ilg}

    def getNetwork(self):
        match = [s for s in self.input if 'network' in s]
        match_split = match[0].split(",")
        return match_split        
	
    def getInuc(self,network,element):
        inuc_tmp = int(network.index(element))
        if inuc_tmp < 10:
            inuc = '000'+str(inuc_tmp)
        if inuc_tmp >= 10 and inuc_tmp < 100:
            inuc = '00'+str(inuc_tmp)			
        if inuc_tmp >= 100 and inuc_tmp < 1000:
            inuc = '0'+str(inuc_tmp)		
        return inuc		
