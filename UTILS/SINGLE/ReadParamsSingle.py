import re  # python regular expressions


class ReadParamsSingle:

    def __init__(self, filename):
        file = open(filename, 'r')
        next(file)  # skip header line
        next(file)  # skip header line

        input = []
        for line in file:
            prsvalue = re.search(r'\[(.*)\]', line).group(1)  # parse out values from square brackets
            input.append(prsvalue)
        file.close()

        self.input = input

    def getForSingle(self, param):
        match = [s for s in self.input if param in s]  # choose only lists identified by param
        datafile = match[0].split(",")[2]
        endianness = match[1].split(",")[2]
        precision = match[2].split(",")[2]
        xbl = float(match[3].split(",")[2])
        xbr = float(match[3].split(",")[3])
        toplot = match[4].split(",")[2:]

        return {'datafile': datafile, 'xbl': xbl, 'xbr': xbr, 'q': toplot, 'endianness': endianness,
                'precision': precision}
