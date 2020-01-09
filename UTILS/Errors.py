# class for error messages

class Errors:

    def __init__(self):
        pass

    def errorGeometry(self, ig):
        return " Geometry ig = " + str(ig) + " not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ..."

    def errorOutOfBoundary(self):
        return " imposed boundary limit in param.evol exceeds the grid limits. EXITING ..."
