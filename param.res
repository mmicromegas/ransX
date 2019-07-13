# ----- Parameter file for rans(eXtreme) and OBURN ---------- #
[]
## Input Data Folder ....................................... ## [prop,eht_data,DATA/TSERIES/]
## Resolution Study Input Files List ....................... ## [prop,eht_res,tseries_ransout_lrezoburn.npy,tseries_ransout_mlrezoburn.npy]
## Filename Prefix For Plots ............................... ## [prop,prefix,oblrez_resstudy_]
## Geomergy; ig = 1 Cartesian, ig = 2 Spherical ............ ## [prop,ig,2]
## Central Time Index ...................................... ## [prop,intc,4]
## Limit Axis .............................................. ## [prop,laxis,2] 
## X-axis Left boundary for properties ..................... ## [prop,xbl,4.e8]
## X-axis Right boundary for properties .................... ## [prop,xbr,9.8e8]
## Nuclear network ......................................... ## [network,neut,prot,he4,c12,o16,ne20,na23,mg24,si28,p31,s32,s34,cl35,ar36,ar38,k39,ca40,ca42,ti44,ti46,cr48,cr50,fe52,fe54,ni56] 
[]
## Brunt-Vaisalla frequency ................................ ## [nsq,True,3.7e8,9.8e8,65.,-5.,0]
## Turbulent Kinetic Energy Stratification ................. ## [tkie,False,3.7e8,9.8e8,5.e13,0.,0]
## Internal Energy Flux Stratification ..................... ## [eintflx,False,3.7e8,9.8e8,3.e27,-1.e27,0]
## Entropy Flux Stratification ............................. ## [entrflx,False,3.7e8,9.8e8,4.e17,-2.e18,0]
## Pressure X Flux Stratification .......................... ## [pressxflx,False,3.7e8,9.8e8,-7.e25,7.e25,0]
## Temperature Flux Stratification ......................... ## [tempflx,False,3.7e8,9.8e8,4.e12,-1.e12,0]
## Enthalpy Flux Stratification ............................ ## [enthflx,False,3.7e8,9.8e8,3.e27,-1.e27,0]
## Turbulent Mass Flux Stratification ...................... ## [tmsflx,False,3.7e8,9.8e8,2.e9,-4.e9,0]
[]
## Neutrons stratification ................................. ## [xrho_neut,False,3.7e8,9.8e8,4.e-11,0.,0] 
## Neutrons flux X stratification .......................... ## [xflxx_neut,False,3.7e8,9.8e8,1.5e-6,-1.e-8,0]
[]
## Protons stratification .................................. ## [xrho_prot,False,3.7e8,9.8e8,1.5e-4,0.,0]
## Protons flux X stratification ........................... ## [xflxx_prot,False,3.7e8,9.8e8,1.e1,-0.2e1,0]
[]
## he4 stratification ...................................... ## [xrho_he4,False,3.7e8,9.8e8,7.e-3,0.,0]
## he4 flux X stratification ............................... ## [xflxx_he4,False,3.7e8,9.8e8,1.e3,-2.e3,0]
[]
## c12 rho stratification .................................. ## [xrho_c12,False,3.7e8,9.8e8,6.e1,0.,0]
## c12 flux X stratification ............................... ## [xflxx_c12,False,3.7e8,9.8e8,5.e6,-1.e6,0]
[]
## o16 rho stratification .................................. ## [xrho_o16,False,3.7e8,9.8e8,8.e5,0.,0]
## o16 flux X stratification ............................... ## [xflxx_o16,False,3.7e8,9.8e8,0.2e10,-6.e10,0]
[]
## ne20 rho stratification ................................. ## [xrho_ne20,False,3.7e8,9.8e8,1.e-3,1.e-6,0]
## ne20 flux X stratification .............................. ## [xflxx_ne20,False,3.7e8,9.8e8,1.e8,-1.e9,0]
[]
## na23 rho stratification ................................. ## [xrho_na23,False,3.7e8,9.8e8,1.4e3,0.,0]
## na23 flux X stratification .............................. ## [xflxx_na23,False,3.7e8,9.8e8,0.2e6,-1.e6,0]
[]
## mg24 rho stratification ................................. ## [xrho_mg24,False,3.7e8,9.8e8,6.e4,0.,0]
## mg24 flux X stratification .............................. ## [xflxx_mg24,False,3.7e8,9.8e8,0.1e9,-1.2e10,0]
[]
## si28 rho stratification ................................. ## [xrho_si28,False,3.7e8,9.8e8,2.2e6,0.,0]
## si28 flux X stratification .............................. ## [xflxx_si28,False,3.7e8,9.8e8,4.e10,-0.2e10,0]
[]
## p31 rho stratification .................................. ## [xrho_p31,False,3.7e8,9.8e8,1.6e3,0.,0]
## p31 flux X stratification ............................... ## [xflxx_p31,False,3.7e8,9.8e8,5.e8,-1.e9,0]
[]
## s32 rho stratification .................................. ## [xrho_s32,False,3.7e8,9.8e8,1.4e6,0.,0]
## s32 flux X stratification ............................... ## [xflxx_s32,False,3.7e8,9.8e8,2.5e10,-5.e9,0]
[]
## s34 rho stratification .................................. ## [xrho_s34,False,3.7e8,9.8e8,7.e4,0.,0]
## s34 flux X stratification ............................... ## [xflxx_s34,False,3.7e8,9.8e8,1.5e9,-0.5e9,0]
[]
## cl35 rho stratification ................................. ## [xrho_cl35,False,3.7e8,9.8e8,2.5e3,0.,0]
## cl35 flux X stratification .............................. ## [xflxx_cl35,False,3.7e8,9.8e8,2.e8,-4.e8,0]
[]
## ar36 rho stratification ................................. ## [xrho_ar36,False,3.7e8,9.8e8,2.e5,0.,0]
## ar36 flux X stratification .............................. ## [xflxx_ar36,False,3.7e8,9.8e8,3.e9,-1.e8,0]
[]
## ar38 rho stratification ................................. ## [xrho_ar38,False,3.7e8,9.8e8,1.4e5,0.,0]
## ar38 flux X stratification .............................. ## [xflxx_ar38,False,3.7e8,9.8e8,2.e9,-0.6e9,0]
[]
## k39 rho stratification .................................. ## [xrho_k39,False,3.7e8,9.8e8,6.e3,0.,0]
## k39 flux X stratification ............................... ## [xflxx_k39,False,3.7e8,9.8e8,2.e8,-3.e8,0]
[]
## ca40 rho stratification ................................. ## [xrho_ca40,False,3.7e8,9.8e8,2.e5,0.,0]
## ca40 flux X stratification .............................. ## [xflxx_ca40,False,3.7e8,9.8e8,1.5e9,-1.e8,0]
[]
## ca42 rho stratification ................................. ## [xrho_ca42,False,3.7e8,9.8e8,3.e3,0.,0]
## ca42 flux X stratification .............................. ## [xflxx_ca42,False,3.7e8,9.8e8,3.e7,-2.e6,0]
[]
## ti44 rho stratification ................................. ## [xrho_ti44,False,3.7e8,9.8e8,1.8e1,0.,0]
## ti44 flux X stratification .............................. ## [xflxx_ti44,False,3.7e8,9.8e8,3.e5,-0.1e5,0]
[]
## ti46 rho stratification ................................. ## [xrho_ti46,False,3.7e8,9.8e8,3.e3,0.,0]
## ti46 flux X stratification .............................. ## [xflxx_ti46,False,3.7e8,9.8e8,1.e7,-1.e6,0]
[]
## cr48 rho stratification ................................. ## [xrho_cr48,False,3.7e8,9.8e8,1.8,0.,0]
## cr48 flux X stratification .............................. ## [xflxx_cr48,False,3.7e8,9.8e8,12.e3,-1.e3,0]
[]
## cr50 rho stratification ................................. ## [xrho_cr50,False,3.7e8,9.8e8,1.e3,0.,0]
## cr50 flux X stratification .............................. ## [xflxx_cr50,False,3.7e8,9.8e8,3.e6,-0.2e6,0]
[]
## fe52 rho stratification ................................. ## [xrho_fe52,False,3.7e8,9.8e8,5.e-2,0.,0]
## fe52 flux X stratification .............................. ## [xflxx_fe52,False,3.7e8,9.8e8,1.5e2,-2.e1,0]
[]
## fe54 rho stratification ................................. ## [xrho_fe54,False,3.7e8,9.8e8,1.8e2,0.,0]
## fe54 flux X stratification .............................. ## [xflxx_fe54,False,3.7e8,9.8e8,1.2e6,-0.5e5,0]
[]
## ni56 rho stratification ................................. ## [xrho_ni56,False,3.7e8,9.8e8,1.2e-3,0.,0]
## ni56 flux X stratification .............................. ## [xflxx_ni56,False,3.7e8,9.8e8,7.,-1.,0]