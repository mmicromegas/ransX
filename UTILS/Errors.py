# class for error messages #


class Errors:

    def __init__(self):
        pass

    def errorGeometry(self, ig):
        return " Geometry ig = " + str(ig) + " not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ..."

    def errorOutOfBoundary(self):
        return " Imposed boundary limit in param.evol exceeds the grid limits. EXITING ..."

    def errorAveragedSnapshots(self):
        return " Zero time-averaged snapshots. Adjust your trange and averaging window. EXITING ..."

    def errorOutputFileExtension(self, fext):
        return " Chosen Output File Extension is " + str(fext) + ". Only png and eps are supported for now. EXITING ..."

    def errorProject(self, plabel):
        return " Project Label Not Recognized. Only oburn and ccp Are Supported For Now. EXITING ..."

    def errorProjectSpecific(self):
        return " Project Specific Properties Not Set Yet. EXITING ..."

    def errorEos(self,ieos):
        return "EOS " + str(ieos) + "not implemented. EXITING ..."
