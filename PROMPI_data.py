import numpy as np

class PROMPI_ransdat:

    def __init__(self,filename):

#        tdata = open(ftycho,'r')

#        t_line1 = tdata.readline().split()
#        nspec = int(t_line1[1])

#        xnuc = []
#        for i in range(nspec):
#            xnuc.append(tdata.readline().split()[0])    
        
#        tdata.close()
        
        fhead = open(filename.replace("ransdat","ranshead"),'r') 

        header_line1 = fhead.readline().split()
        header_line2 = fhead.readline().split()
        header_line3 = fhead.readline().split()
        header_line4 = fhead.readline().split()

#       Cyril's output + 4 lines		
        header_line5 = fhead.readline().split()
        header_line6 = fhead.readline().split()
        header_line7 = fhead.readline().split()
        header_line8 = fhead.readline().split()
		
        self.nstep       = int(header_line1[0])
        self.rans_tstart = float(header_line1[1])
        self.rans_tend   = float(header_line1[2])
        self.rans_tavg   = float(header_line1[3])
		
        self.qqx    = int(header_line2[0])
        self.qqy    = int(header_line2[1])
        self.qqz    = int(header_line2[2])
        self.nnuc   = int(header_line2[3])
        self.nrans  = int(header_line2[4])
        ndims = [4,self.nrans,self.qqx]
		

        self.ransl = []		
        for line in range(self.nrans):
            line = fhead.readline().strip()
            self.ransl.append(line)
#            print(self.nrans,line)
			
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
            self.xznl.append(float(line[8:22].strip()))
            self.xzn0.append(float(line[23:38].strip()))
            self.xznr.append(float(line[39:54].strip()))
			
        for line in range(self.qqy):
            line = fhead.readline().strip()        
            self.yznl.append(float(line[8:22].strip()))
            self.yzn0.append(float(line[23:38].strip()))
            self.yznr.append(float(line[39:54].strip()))	

        for line in range(self.qqz):
            line = fhead.readline().strip()
            self.zznl.append(float(line[8:22].strip()))
            self.zzn0.append(float(line[23:38].strip()))
            self.zznr.append(float(line[39:54].strip()))	

        frans = open(filename,'rb')
        self.data = np.fromfile(frans)		
#        self.data = np.fromfile(frans,dtype='>f',count=ndims[0]*ndims[1]*ndims[2])
        self.data = np.reshape(self.data,(ndims[0],ndims[1],ndims[2]),order='F')	

        self.ransd = {}
		
        self.ransd = {"xzn0" : self.xzn0}
		
        i = 0
#        print(self.ransl)
		
        for s in self.ransl:
            field = {str(s) : self.data[2,i,:]}
            self.ransd.update(field)
            i += 1
		
        frans.close()
        fhead.close()		
 
    def rans_header(self):
        return self.rans_tstart,self.rans_tend,self.rans_tavg
 
    def rans(self):	
        return self.ransd
		
    def rans_list(self):
        return self.ransl
		
    def rans_qqx(self):
        return self.qqx

    def rans_qqy(self):
        return self.qqy

    def rans_qqz(self):
        return self.qqz		

    def rans_xznl(self):
        return self.xznl

    def rans_xznr(self):
        return self.xznr		
		
    def ransdict(self):
        print self.eh.keys()
		
    def sterad(self):
        pass
    
class PROMPI_blockdat:

    def __init__(self,filename,dat):

        fhead = open(filename.replace("blockdat","blockhead"),'r')
#        fhead = open(filename.replace("bindata","header"),'r')        

        header_line1 = fhead.readline().split()
        header_line2 = fhead.readline().split()
        header_line3 = fhead.readline().split()
        header_line4 = fhead.readline().split()

        self.nstep  = int(header_line1[0])
        self.time   = float(header_line1[1])
        
        self.qqx    = int(header_line2[0])
        self.qqy    = int(header_line2[1])
        self.qqz    = int(header_line2[2])
        self.nnuc   = int(header_line2[3])
        self.nvar   = int(header_line2[4])

#        header_line5 = fhead.readline().split()
#        header_line6 = fhead.readline().split()
#        header_line7 = fhead.readline().split()
#        header_line8 = fhead.readline().split()

        
        
        ndims = [self.qqx,self.qqy,self.qqz]

        self.varl = []		
        for line in range(self.nvar):
            line = fhead.readline().strip()
            self.varl.append(line)

        self.interior_mass = float(fhead.readline())
            
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
        

        ivar   = self.varl.index(dat)
        irecl  = self.qqx*self.qqy*self.qqz
        nbyte  = irecl*4
        dstart = int(ivar*nbyte)

        print(ivar,irecl,nbyte,dstart,self.qqx,self.qqy,self.qqz)
        
        fblock = open(filename,'rb')

        # offset read pointer (argument offset is a byte count)         
        fblock.seek(dstart) 

# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html#arrays-dtypes-constructing
        
#       '<f' little-endian single-precision float
#       '>f' little-endian single-precision float

# >>> dt = np.dtype('b')  # byte, native byte order
# >>> dt = np.dtype('>H') # big-endian unsigned short
# >>> dt = np.dtype('<f') # little-endian single-precision float
# >>> dt = np.dtype('d')  # double-precision floating-point number

#        self.data = np.fromfile(fblock,dtype='<f4',count=irecl)
        self.data = np.fromfile(fblock,dtype='<f4',count=192)
#        self.data = np.reshape(self.data,(self.qqx,self.qqy,self.qqz),order='F')	
#        print(self.data)
        
        fblock.close()

    def dt(self):
        return self.data
        
    def eh(self):
        for i in range(self.qqx):
            eh = np.sum(self.data[i][:][:])/(self.qqy*self.qqz)
        return eh 

    def test(self):
        return self.data
