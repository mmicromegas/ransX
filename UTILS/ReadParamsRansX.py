import re # python regular expressions

class ReadParamsRansX:

    def __init__(self,filename):

        file=open(filename,'r')
        next(file) # skip header line
        next(file) # skip header line

        input=[]
        for line in file:
            prsvalue = re.search(r'\[(.*)\]', line).group(1) # parse out values from square brackets
            input.append(prsvalue)
        file.close()
				
        self.input = input

				
    def getForProp(self,param): 
	
        match = [s for s in self.input if param in s] # choose only lists identified by param
        eht_data = match[0].split(",")[2]
        prefix   = match[1].split(",")[2]	
        ig       = int(match[2].split(",")[2])
        intc     = int(match[3].split(",")[2])
        laxis    = int(match[4].split(",")[2])
        xbl      = float(match[5].split(",")[2])		
        xbr      = float(match[6].split(",")[2])	
				
        return {'eht_data':eht_data,'prefix':prefix,'ig':ig,'intc':intc,'laxis':laxis,'xbl':xbl,'xbr':xbr}				

    def getForEqs(self,param): 
	
        match = [s for s in self.input if param in s] # choose only lists identified by param
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

    def getForEqsBar(self,param): 
	
        match = [s for s in self.input if param in s] # choose only lists identified by param
        match_split = match[0].split(",")
        #equation = match_split[0]
        plotMee = match_split[1]
        xbl = float(match_split[2])
        xbr = float(match_split[3])		
        ybu = float(match_split[4])
        ybd = float(match_split[5])
		
        return {'plotMee':plotMee,'xbl':xbl,'xbr':xbr,'ybu':ybu,'ybd':ybd}

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