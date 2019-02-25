import re # python regular expressions

class ReadParamsTseries:

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

				
    def getForTseries(self,param): 	
	
        match   = [s for s in self.input if param in s] # choose only lists identified by param
        datadir = match[0].split(",")[2]
        endianness = match[1].split(",")[2]
        precision = match[2].split(",")[2]		
        dataout = match[3].split(",")[2]	
        trange_beg  = float(match[4].split(",")[2])
        trange_end  = float(match[4].split(",")[3])
        tavg        = float(match[5].split(",")[2])	
				
        return {'datadir':datadir,'dataout':dataout,'trange_beg':trange_beg,'trange_end':trange_end,'tavg':tavg,'endianness':endianness,'precision':precision}					