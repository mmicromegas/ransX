# class for error messages

class Errors:

    def __init__(self):
        pass

    def errorGeometry(self,ig):
        return " Geometry ig = " + str(ig) + " not defined, use ig = 1 for CARTESIAN, ig = 2 for SPHERICAL, EXITING ..."
