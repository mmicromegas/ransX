!     
!     
!     
      subroutine eosrad(inputz,ieos,imu,dd,pp,eint,tt,enx,                  &
     &     cv,cp,gc,ge,errorkey)
!
!
!     GAMMA LAW EOS w/ or w/o RADIATION PRESSURE
!     EOS MODES:
!     (note: ieos = 1 : ideal gas w/ radiation pressure)
!     (note: ieos = 4 : ideal gas only, NO radiation pressure)
!
!
!     if inputz==1: take  DENS, COMPOSITION, ENERGY
!     updates TEMP, PRES, GAMMAC, GAMMAE
!
!     if inputz==2: takes DENS, TEMP, COMPOSITION
!     updates ENERGY, PRES, GAMMAC, GAMMAE
!
!     if inputz==3: takes TEMP, PRESS, COMPOSITION
!     returns DENS, ENERGY, GAMMAC, GAMMAE
!
      implicit none      
      include 'dimen.inc'
      include 'constants.inc'
      include 'mpif.h'
      include 'mpicom.inc'
!
      integer*4 inputz      
      integer*4 errorkey, ieos
!     INTERM. RESULT STORAGE VARS
      real*8 imu, dd, pp, eint, tt, gc, ge, cv, cp
      real*8 pgas, prad, enx
      real*8 const1, const2, const3
!     N-R Variables
      real*8 nr1, nr2, nrx, nry, nrdydx, nrtest
      integer*4 nrcnt
!
      errorkey = 0
      !
!     inputz=1:given (energy, density), update (temp, press)
!     [NOTE: uses N-R to solve for temp]
      if(inputz.eq.1) then
!
         if(ieos.eq.1) then
            nr1 = 1.5d0*gascon*dd*imu/arad
            nr2 = (eint*dd)/arad
            nrx = min(max(1.d0, tt), 1.d11)
            nry = nrx**4.d0 + nr1*nrx - nr2
            nrtest = 1.d0
            nrcnt = 1
            do while(abs(nrtest).gt.1.d-3)
               nrdydx = 4*nrx**3.0 + nr1
               nrx    = nrx - nry/nrdydx
               nrx = min(max(1.d0, nrx), 1d13)
               nry = nrx**4.d0 + nr1*nrx - nr2
               nrtest = -nry/nrdydx/nrx
               nrcnt = nrcnt +1
               if(nrcnt.gt.100) then
                  errorkey = 2
                  goto 666
               endif
            enddo
            tt = nrx
            pgas = imu*gascon*dd*tt
            prad = 1.d0/3.d0*arad*tt**4.d0
            pp = pgas + prad
         endif
         if(ieos.eq.4) then
            tt = eint/(1.5d0*gascon*imu)
            pgas = imu*gascon*dd*tt
            prad = 0.d0
            pp = pgas + prad
            enx = (pp/dd+eint)/tt
         endif  
!
!
!
!     inputz=2:given (temp, density), update (energy,pressure)
      elseif(inputz.eq.2) then
         pgas = gascon*imu*dd*tt
         prad = 0.d0
         if(ieos.eq.1) then
            prad = arad/3.d0*tt**4.d0
         endif
         pp   = prad + pgas
         eint = (1.5d0*pgas + 3.d0*prad)/dd
         enx = (pp/dd+eint)/tt
!
!
!
!     inputz=3: given (temp,press), return (density,energy)
      elseif(inputz.eq.3) then
!
         dd   = pp/(gascon*imu*tt)
         ! print*,'eosrad(dd): ',dd
         ! dd   = (pp - arad/3.d0*tt**4.d0)/(gascon*imu*tt)
         pgas =  gascon*imu*dd*tt
         prad = 0.d0
         if(ieos.eq.1) then
            dd   = (pp - arad/3.d0*tt**4.d0)/(gascon*imu*tt)
            prad =  arad/3.d0*tt**4.d0
         endif
         pp = pgas+prad
         eint =  (1.5d0*pgas + 3.d0*prad)/dd
         enx = (pp/dd+eint)/tt
      endif
!
!
!     CALCULATE GAMMAS
      const1    = pgas/pp
      const2    = 4.d0/const1-3.d0
      const3    = (4.d0 - 3.d0*const1)/                                 &
     &     (16d0 - 1.5d0*const1*const1-12.d0*const1)
!
      gc = 1.d0/(1.d0/const1 - const2*const3)
      ge = pp/(eint*dd) + 1.d0
!
      cv = 1.5d0*gascon*imu*(8.d0-7.d0*const1)/const1
      cp = gascon*imu*(1.5d0 +                                          &
     &     (16.d0 - 12.d0*const1 - 3*const1**2.d0)/const1**2.d0)
!
!
!     SUCCESS/RETURN
      return
!
!
!
!     ERROR MESSAGES
 666  continue
      print*,'ERROR(eosrad.f): NR NON-CONVERGENCE'
      return
!
!
      end
