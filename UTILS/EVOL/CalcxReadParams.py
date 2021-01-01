import re  # python regular expressions
import sys


class CalcxReadParams:

    def __init__(self, filename):

        file = open(filename, 'r')
        next(file)  # skip header line
        next(file)  # skip header line

        input = []
        for line in file:
            # parse out values from square brackets 
            prsvalue = re.search(r'\[(.*)\]', line).group(1)
            input.append(prsvalue)
        file.close()

        self.input = input

    def getForProp(self, param):

        match = [s for s in self.input if param in s]  # choose only lists identified by param
        eht_data = match[0].split(",")[2]
        dataout = match[1].split(",")[2]
        plabel = match[2].split(",")[2]
        ig = int(match[3].split(",")[2])
        nsdim = int(match[4].split(",")[2])
        ieos = int(match[5].split(",")[2])
        laxis = int(match[6].split(",")[2])
        xbl = float(match[7].split(",")[2])
        xbr = float(match[8].split(",")[2])

        return {'eht_data': eht_data, 'dataout': dataout, 'plabel': plabel,
                    'ig': ig, 'nsdim': nsdim, 'ieos': ieos, 'laxis': laxis, 'xbl': xbl, 'xbr': xbr}

    def getNetwork(self):
        match = [s for s in self.input if 'network' in s]
        match_split = match[0].split(",")
        return match_split

    def getInuc(self, network, element):
        inuc_tmp = int(network.index(element))
        if inuc_tmp < 10:
            inuc = '000' + str(inuc_tmp)
        if inuc_tmp >= 10 and inuc_tmp < 100:
            inuc = '00' + str(inuc_tmp)
        if inuc_tmp >= 100 and inuc_tmp < 1000:
            inuc = '0' + str(inuc_tmp)
        return inuc
