import re  # python regular expressions


class ReadParamsRansX:

    def __init__(self, filename):

        ffile = open(filename, 'r')
        next(ffile)  # skip header line
        next(ffile)  # skip header line

        iinput = []
        for line in ffile:
            prsvalue = re.search(r'\[(.*)\]', line).group(1)  # parse out values from square brackets
            iinput.append(prsvalue)
        ffile.close()

        self.iinput = iinput

    def getForProp(self, param):

        match = [s for s in self.iinput if param in s]  # choose only lists identified by param
        eht_data = match[0].split(",")[2]
        plabel = match[1].split(",")[2]
        prefix = match[2].split(",")[2]
        ig = int(match[3].split(",")[2])
        ieos = int(match[4].split(",")[2])
        intc = int(match[5].split(",")[2])
        laxis = int(match[6].split(",")[2])
        fext = match[7].split(",")[2]
        xbl = float(match[8].split(",")[2])
        xbr = float(match[9].split(",")[2])

        return {'eht_data': eht_data, 'plabel': plabel, 'prefix': prefix, 'ig': ig, 'ieos': ieos, 'intc': intc, 'laxis': laxis,
                'fext': fext, 'xbl': xbl, 'xbr': xbr}

    def getForEqs(self, param):

        match = [s for s in self.iinput if param in s]  # choose only lists identified by param
        # print(param,match)
        match_split = match[0].split(",")
        # equation = match_split[0]
        plotMee = match_split[1]
        xbl = float(match_split[2])
        xbr = float(match_split[3])
        ybu = float(match_split[4])
        ybd = float(match_split[5])
        ilg = int(match_split[6])

        return {'plotMee': plotMee, 'xbl': xbl, 'xbr': xbr, 'ybu': ybu, 'ybd': ybd, 'ilg': ilg}

    def getForEqsBar(self, param):

        match = [s for s in self.iinput if param in s]  # choose only lists identified by param
        match_split = match[0].split(",")
        # equation = match_split[0]
        plotMee = match_split[1]
        xbl = float(match_split[2])
        xbr = float(match_split[3])
        ybu = float(match_split[4])
        ybd = float(match_split[5])

        return {'plotMee': plotMee, 'xbl': xbl, 'xbr': xbr, 'ybu': ybu, 'ybd': ybd}

    def getNetwork(self):
        match = [s for s in self.iinput if 'network' in s]
        match_split = match[0].split(",")
        return match_split

    def getInuc(self, network, element):
        inuc = 0
        inuc_tmp = int(network.index(element))
        if inuc_tmp < 10:
            inuc = '000' + str(inuc_tmp)
        if 10 <= inuc_tmp < 100:
            inuc = '00' + str(inuc_tmp)
        if 100 <= inuc_tmp < 1000:
            inuc = '0' + str(inuc_tmp)
        return inuc
