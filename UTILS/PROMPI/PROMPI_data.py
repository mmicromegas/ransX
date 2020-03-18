import numpy as np
import sys


class PROMPI_ransdat:

    def __init__(self, filename, endianness, precision):

        #       find first occurence of dd due to header info stored either on 4 or 8 lines, computer arch dependent

        lookup = "dd"
        iterate = True
        with open(filename.replace("ransdat", "ranshead")) as f:
            for num, line in enumerate(f, 1):
                if (iterate) and (lookup in line):
                    # print('found at line:', num)
                    num_dd = num
                    iterate = False

        fhead = open(filename.replace("ransdat", "ranshead"), 'r')

        header_line1 = fhead.readline().split()
        header_line2 = fhead.readline().split()
        header_line3 = fhead.readline().split()
        header_line4 = fhead.readline().split()

        if num_dd == 9:
            header_line5 = fhead.readline().split()
            header_line6 = fhead.readline().split()
            header_line7 = fhead.readline().split()
            header_line8 = fhead.readline().split()

        self.nstep = int(header_line1[0])
        self.rans_tstart = float(header_line1[1])
        self.rans_tend = float(header_line1[2])
        self.rans_tavg = float(header_line1[3])

        self.qqx = int(header_line2[0])
        self.qqy = int(header_line2[1])
        self.qqz = int(header_line2[2])
        self.nnuc = int(header_line2[3])
        self.nrans = int(header_line2[4])
        ndims = [4, self.nrans, self.qqx]

        self.ransl = []
        for line in range(self.nrans):
            line = fhead.readline().strip()
            self.ransl.append(line)
            # print(self.nrans,line)

        #        for inuc in range(nspec):
        #            self.ransl = [field.replace(str(inuc+1),str(xnuc[inuc])) for field in self.ransl]
        #            self.ransl = [field.replace("0","") for field in self.ransl]

        #        print(self.ransl)

        self.xznl = []
        self.xzn0 = []
        self.xznr = []
        self.yznl = []
        self.yzn0 = []
        self.yznr = []
        self.zznl = []
        self.zzn0 = []
        self.zznr = []

        for line in range(self.qqx):
            line = fhead.readline().strip()
            self.xznl.append(float(line[8:24].strip()))
            self.xzn0.append(float(line[23:38].strip()))
            self.xznr.append(float(line[39:54].strip()))

        for line in range(self.qqy):
            line = fhead.readline().strip()
            self.yznl.append(float(line[8:22].strip()))
            self.yzn0.append(float(line[23:38].strip()))
            self.yznr.append(float(line[39:54].strip()))

        for line in range(self.qqz):
            line = fhead.readline().strip()
            self.zznl.append(float(line[8:23].strip()))
            self.zzn0.append(float(line[24:38].strip()))
            self.zznr.append(float(line[39:54].strip()))

        frans = open(filename, 'rb')

        # irecl_float = 4

        # recl = np.fromfile(frans, count=(512))
        # frans.seek(4)

        # print(recl)

        # frans.close()
        # fhead.close()
        # sys.exit()

        if (endianness == 'little_endian') and (precision == 'single'):
            self.data = np.fromfile(frans, dtype='<f4', count=ndims[0] * ndims[1] * ndims[2])
        elif (endianness == 'little_endian') and (precision == 'double'):
            self.data = np.fromfile(frans, dtype='<f8', count=ndims[0] * ndims[1] * ndims[2])
        elif (endianness == 'big_endian') and (precision == 'single'):
            self.data = np.fromfile(frans, dtype='>f4', count=ndims[0] * ndims[1] * ndims[2])
        elif (endianness == 'big_endian') and (precision == 'double'):
            self.data = np.fromfile(frans, dtype='>f8', count=ndims[0] * ndims[1] * ndims[2])
        else:
            print("PROMPI_data.py: Wrong endianness or precision. Check param.tseries")
            sys.exit()

        # reshape
        self.data = np.reshape(self.data, (ndims[0], ndims[1], ndims[2]), order='F')

        #        self.data = np.fromfile(frans)
        #        self.data = np.reshape(self.data,(ndims[0],ndims[1],ndims[2]),order='F')
        #        self.data = np.reshape(self.data,(ndims[0],ndims[1],ndims[2]),order='C')

        self.ransd = {}

        nx = {'nx': self.qqx}
        self.ransd.update(nx)

        ny = {'ny': self.qqy}
        self.ransd.update(ny)

        nz = {'nz': self.qqz}
        self.ransd.update(nz)

        xzn0 = {'xzn0': self.xzn0}
        self.ransd.update(xzn0)

        xznl = {'xznl': self.xznl}
        self.ransd.update(xznl)

        xznr = {'xznr': self.xznr}
        self.ransd.update(xznr)

        yzn0 = {'yzn0': self.yzn0}
        self.ransd.update(yzn0)

        yznl = {'yznl': self.yznl}
        self.ransd.update(yznl)

        yznr = {'yznr': self.yznr}
        self.ransd.update(yznr)

        zzn0 = {'zzn0': self.zzn0}
        self.ransd.update(zzn0)

        zznl = {'zznl': self.zznl}
        self.ransd.update(zznl)

        zznr = {'zznr': self.zznr}
        self.ransd.update(zznr)

        rans_tstart = {'rans_tstart': self.rans_tstart}
        self.ransd.update(rans_tstart)

        i = 0
        #        print(self.ransl)

        for s in self.ransl:
            field = {str(s): self.data[2, i, :]}
            self.ransd.update(field)
            i += 1

        frans.close()
        fhead.close()

    def rans_header(self):
        return self.rans_tstart, self.rans_tend, self.rans_tavg

    def rans(self):
        return self.ransd

    def rans_list(self):
        return self.ransl

    def ransdict(self):
        print self.eh.keys()


class PROMPI_bindata:

    def __init__(self, filename, ldat):

        #       find first occurence of dd due to header info stored either on 4 or 8 lines, computer arch dependent

        lookup = "density"
        iterate = True
        with open(filename.replace("bindata", "header")) as f:
            for num, line in enumerate(f, 1):
                if (iterate) and (lookup in line):
                    # print('found at line:', num)
                    num_density = num
                    iterate = False

        fhead = open(filename.replace("bindata", "header"), 'r')

        header_line1 = fhead.readline().split()
        header_line2 = fhead.readline().split()
        header_line3 = fhead.readline().split()
        header_line4 = fhead.readline().split()

        if num_density == 9:
            header_line5 = fhead.readline().split()
            header_line6 = fhead.readline().split()
            header_line7 = fhead.readline().split()
            header_line8 = fhead.readline().split()
            # header_line9 = fhead.readline().split()

        self.nstep = int(header_line1[0])
        self.time = float(header_line1[1])

        self.qqx = int(header_line2[0])
        self.qqy = int(header_line2[1])
        self.qqz = int(header_line2[2])
        self.nnuc = int(header_line2[3])
        self.nvar = int(header_line2[4])

        ndims = [self.qqx, self.qqy, self.qqz]

        self.varl = []
        for line in range(self.nvar):
            line = fhead.readline().strip()
            self.varl.append(line)
            # print(line)

        # self.interior_mass = float(fhead.readline())

        xznl = []
        xzn0 = []
        xznr = []
        yznl = []
        yzn0 = []
        yznr = []
        zznl = []
        zzn0 = []
        zznr = []

        # parse grid  (todo: get rid of the hard-coded values
        # perhaps readline format?

        il_l = 8
        il_r = 22

        i0_l = 23
        i0_r = 38

        ir_l = 39
        ir_r = 54

        for line in range(self.qqx):
            line = fhead.readline().strip()
            xznl.append(float(line[il_l:il_r].strip()))
            xzn0.append(float(line[i0_l:i0_r].strip()))
            xznr.append(float(line[ir_l:ir_r].strip()))

        for line in range(self.qqy):
            line = fhead.readline().strip()
            yznl.append(float(line[il_l:il_r].strip()))
            yzn0.append(float(line[i0_l:i0_r].strip()))
            yznr.append(float(line[ir_l:ir_r].strip()))

        for line in range(self.qqz):
            line = fhead.readline().strip()
            zznl.append(float(line[il_l:il_r].strip()))
            zzn0.append(float(line[i0_l:i0_r].strip()))
            zznr.append(float(line[ir_l:ir_r].strip()))

        self.datadict = {}

        for dat in ldat:
            ivar = self.varl.index(dat)
            irecl = self.qqx * self.qqy * self.qqz
            nbyte = irecl * 4
            dstart = int(ivar * nbyte)

            # print(ivar,irecl,nbyte,dstart,self.qqx,self.qqy,self.qqz)

            fblock = open(filename, 'rb')

            # offset read pointer (argument offset is a byte count)         
            fblock.seek(dstart)

            #       https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html#arrays-dtypes-constructing

            #       '<f' little-endian single-precision float
            #       '>f' little-endian single-precision float

            #       >>> dt = np.dtype('b')  # byte, native byte order
            #       >>> dt = np.dtype('>H') # big-endian unsigned short
            #       >>> dt = np.dtype('<f') # little-endian single-precision float
            #       >>> dt = np.dtype('d')  # double-precision floating-point number

            self.data = np.fromfile(fblock, dtype='<f4', count=irecl)

            self.data = np.reshape(self.data, (self.qqx, self.qqy, self.qqz), order='F')
            #           print(self.data)

            fblock.close()

            self.datadict.update({dat: self.data})

            self.datadict.update({'xznl': xznl})
            self.datadict.update({'xzn0': xzn0})
            self.datadict.update({'xznr': xznr})

            self.datadict.update({'yznl': yznl})
            self.datadict.update({'yzn0': yzn0})
            self.datadict.update({'yznr': yznr})

            self.datadict.update({'zznl': zznl})
            self.datadict.update({'zzn0': zzn0})
            self.datadict.update({'zznr': zznr})

            self.datadict.update({'qqx': self.qqx})
            self.datadict.update({'qqy': self.qqy})
            self.datadict.update({'qqz': self.qqz})
            self.datadict.update({'time': self.time})

            self.xznl = xznl
            self.xzn0 = xzn0
            self.xznr = xznr

            self.yznl = yznl
            self.yzn0 = yzn0
            self.yznr = yznr

            self.zznl = zznl
            self.zzn0 = zzn0
            self.zznr = zznr

    def datadict(self):
        return self.datadict

    def eh(self):
        for i in range(self.qqx):
            eh = np.sum(self.data[i][:][:]) / (self.qqy * self.qqz)
        return eh

    def test(self):
        return self.data

    def grid(self):
        return {'nx': self.qqx, 'ny': self.qqy, 'nz': self.qqz,
                'xznl': self.xznl,'xzn0': self.xzn0, 'xznr': self.xznr,
                'yznl': self.yznl,'yzn0': self.yzn0,'yznr': self.yznr,
                'zznl': self.zznl, 'zzn0': self.zzn0, 'zznr': self.zznr}

#        OBSOLETE CODE		

#        tdata = open(ftycho,'r')

#        t_line1 = tdata.readline().split()
#        nspec = int(t_line1[1])

#        xnuc = []
#        for i in range(nspec):
#            xnuc.append(tdata.readline().split()[0])    

#        tdata.close()
