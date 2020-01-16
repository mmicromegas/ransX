# ----- Parameter file for rans(eXtreme) and OBURN ---------- #
[]
## Input-Data Directory ...................... ## [fourier,datadir,C:\Users\mmocak\Desktop\GITDEV\ransX\DATA\BINDATA\ccp_two_layers\]
## Input-Data Files ...........................## [fourier,datafiles,ccptwo.res64cubed.fixedmu.opto3.01014.bindata,ccptwo.res128cubed.fixedmu.opto3.01014.bindata,ccptwo.res256cubed.cosma.01014.bindata]
## Filename Prefix For Plots ................. ## [fourier,prefix,ccptwo_fourier_res]
## Input Data Endianness ..................... ## [fourier,endian,little_endian]
## Input Data Floating Point Precision ....... ## [fourier,precision,double]
## Geometry; ig = 1 Cartes, ig = 2 Spheric ... ## [fourier,ig,1]
## Limit Axis ................................ ## [fourier,laxis,2] 
## Location of Horizontal Cut For Fourier .... ## [fourier,lhc,6.0e8]
## Nuclear network ........................... ## [network,fluid1,fluid2]
[]
## Turbulent Kinetic Fourier Energy spectrum . ## [fstke,False,1.,2.e2,2.e14,1.e7,0]
## ux Fourier "Energy" spectrum .............. ## [fsux,False,1.,2.e2,1.e14,1.e7,0]
## uy Fourier "Energy" spectrum .............. ## [fsuy,False,1.,2.e2,1.e14,1.e7,0]
## uz Fourier "Energy" spectrum .............. ## [fsuz,False,1.,2.e2,1.e14,1.e7,0]
## Density Fourier "Energy" spectrum ......... ## [fsdd,False,1.,200.,1.e6,1.e0,0]
## Pressure Fourier "Energy" spectrum ........ ## [fspp,False,1.,200.,1.e40,1.e34,0]
## Temperature Fourier "Energy" spectrum ..... ## [fstt,False,1.,200.,1.e13,1.e6,0]
## Total Energy Fourier "Energy" spectrum .... ## [fset,False,1.,200.,1.e29,1.e21,0]
## Fluid1rons Fourier "Energy" spectrum ...... ## [fs_fluid1,True,1.,200.,1.e7,1.e1,0]
## Fluid2rons Fourier "Energy" spectrum ...... ## [fs_fluid2,True,1.,200.,1.e7,1.e1,0]