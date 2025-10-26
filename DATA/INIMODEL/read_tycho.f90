!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!     READ IN TYCHO MODEL DATA
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!     ------------------
!     Input model zoning
!     ------------------
!     1. rr(i) is right edge of zone i
!     *2. mm(i) is interior mass up to radius rr(i+1) ***NOTE INDEX DISCREPANCY!!!!
!     3. vr(i) is velocity of mass zone interface, at radius rr(i)
!     4. tt(i) is the temperature of zone between rr(i) and rr(i-1)
!     5. dd(i) is the density of zone between rr(i) and rr(i-1)
!     6. pp(i) is the pressure of zone between rr(i) and rr(i-1)
!     7. dm(i) is the total mass in zone between rr(i) and rr(i-1)
!     8. cv(i) is the convective velocity in zone between rr(i) and rr(i-1)
!     9. ll(i) is the luminosity at zone interface, at radius rr(i)
!     10. xyz(i,jj) is the jjth composition in zone between rr(i) and rr(i-1)
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
      subroutine read_tycho(nzones,rrl,rr,rrr,tt,dd,mm,cv,hf,xyz)
!
      implicit none
      include 'dimen.inc'
      include 'bdimen.inc'
      include 'intgrs.inc'
      include 'float.inc'
      include 'files.inc'
      include 'burnf.inc'
      include 'comp.inc'
      include 'heos.inc'
      include 'constants.inc'
      include 'mpif.h'
      include 'mpicom.inc'
!
      integer*4 i, ii, j, jj, nlines, nspecies, nzones
      integer*4 nriter
      real*8 tt(nzones),dd(nzones)
      real*8 pp(nzones),dm(nzones),hf(nzones),ll(nzones)
      real*8 mm(nzones),cv(nzones),vr(nzones),xyz(nzones,qn)
      real*8 rrp(nzones),mmp(nzones),llp(nzones)
      real*8 rrl(nzones), rrr(nzones), rr(nzones)
      real*8 tempdat,junk(10)
      real*8 xxn(qn)
      real*8 drhodr,dpdr,dlnpdr,dlnrhodr,nsqr,grav
      real*8 lsqr,cssqr,delv,ptot,pptest,pptestmax
      real*8 X,Y,Z,imu,abar,zbar,ye,ye2,snu
      real*8 fgrav, fpress, rr1,rr0,tscross
!
      real*8 ssp, hp, ssd, hd, hp2, ssp2
      real*8 deltt, delpp, delrr, dppdrr
      real*8 nabs, nabad
      real*8 cv1, cp1, en1, pp1, cs1, delta1
      real*8 cv2, cp2, en2, pp2, cs2, delta2
      real*8 cv15, cp15, en15, pp15, cs15, delta15, tt15, dd15
      real*8 xxxtest, xxxtest2
      real*8 Lnuc, delm, mass
      real*8 eox, Lox, fox
!
      integer*4 n,m, nsubcycle
      real*8 eps, dtc, dth, dtnew
      real*8 rcout, rcin, dcout, tcout
      real*8 encout, encin, ennew, delenv
      real*8 deltt1, deltt2
      real*8 dentdr
!
      integer stat(MPI_STATUS_SIZE)
      character*50 fn_name
      logical fn_exist
!
      pi = 4.d0*datan(1.d0)
!
      fn_name = "data/imodel.tycho"
      inquire(file=fn_name,EXIST=fn_exist)
      if(fn_exist) then
         if(myid.eq.0) print*,'MSG(read_tycho): opening model file:', fn_name
      else
         print*,'ERR(read_tycho): no model file: ', fn_name
         call MPI_FINALIZE(ierr)
         stop 'ERR(read_tycho): no model file'
      endif


!     READ IN MODEL DATA

      open(ntlocal,file=fn_name)
      read(ntlocal,*) nlines, nspecies
      if(myid.eq.0) then
         write(6,*) 'MSG(read_tycho): read network definition:'
         write(6,*) 'MSG(read_tycho): nlines,nspecies=',nlines,nspecies
         write(6,*) 'MSG(read_tycho): i xnucid  zz  nn'
      endif
      do i=1,nspecies
9901     format(a5,2i4)
9902     format(' MSG(read_tycho):',i4,a5,2i4)
         read(ntlocal,9901) xnucid(i),xnuczz(i),xnucnn(i)
         xnucaa(i) = xnucnn(i) + xnuczz(i)
         if(myid.eq.0) write(6,9902) i,xnucid(i),xnuczz(i),xnucnn(i)
      enddo

      ncomp = qn
      if(myid.eq.0) print*,'MSG(read_tycho): qn, ncomp, nspecies =', qn, ncomp, nspecies

      if(ncomp.ne.qn) then
         if(myid.eq.0) print*,'ERR(read_tycho): ncomp!=qn: ',ncomp,qn
         call MPI_FINALIZE(ierr)
         stop'ERR(read_tycho): ncomp != qn.'
      endif

      if(nlines.gt.nzones) then
         call MPI_FINALIZE(ierr)
         stop'ERR(read_tycho): nlines > nzones.'
      endif

      if(myid.eq.0) print*,'MSG(read_tycho): begin loop over model: nlines =', nlines


!     LOOP OVER MODEL ZONES

      do i=1,nlines
         read(ntlocal,*) ii, mm(i),rr(i),vr(i),tt(i),dd(i),          &
              pp(i),dm(i),cv(i),ll(i),(xyz(i,jj),jj=1,qn)

         if(i.eq.1) dm(i) = 0.d0

!     ...normalize abundances...
         xxxtest = 0.d0
         do j=1,qn
            xyz(i,j) = xyz(i,j)*dble(xnucaa(j))
            xxxtest = xxxtest + xyz(i,j)
         enddo
         do j=1,qn
            xyz(i,j) = xyz(i,j)/xxxtest
         enddo

!     ...calc heat flux and sv->dd...
         if(rr(i).gt.0.d0) then
            hf(i) = ll(i)/(4.d0*pi*rr(i)**2.d0)
         endif
         if(dd(i).gt.0.d0) then
            dd(i) = 1.d0/dmax1(dd(i),1.d-20)
         endif
      enddo
      if(myid.eq.0) print*,'MSG(read_tycho): finished loop over model zones'
      close(ntlocal)



!     MANAGE ZONES: shift index:
!     rrl: left interface of zone i
!     rrr: right interface of zone i
!     rr : center of zone i
      mm(1) = 0.d0
      do i=1,nlines-1
         rrl(i) = rr(i)
         rrr(i) = rr(i+1)
         rr(i)  = (rrr(i)+rrl(i))/2.d0
         tt(i) = tt(i+1)
         dd(i) = dd(i+1)
         cv(i) = cv(i+1)
         hf(i) = hf(i+1)
         dm(i) = dm(i+1)
         pp(i) = pp(i+1)
         do ii=1,qn
            xyz(i,ii) = xyz(i+1,ii)
         enddo
         if(i.gt.1) mm(i+1) = mm(i)+dm(i)
      enddo

!     COPY LAST ZONE (nzones-nlines) TIMES TO FILL ARRAY
      do i=nlines,nzones
         rrl(i) = rrl(i-1)
         rr(i)  = rr(i-1)
         rrr(i) = rrr(i-1)
         tt(i) = tt(i-1)
         dd(i) = dd(i-1)
         mm(i) = mm(i-1)
         cv(i) = cv(i-1)
         hf(i) = hf(i-1)
         pp(i) = pp(i-1)
         do ii=1,qn
            xyz(i,ii) = xyz(i-1,ii)
         enddo
      enddo



!     ADJUST ENTROPY PROFILES

      if(1.eq.0) then        !ADJUST TEMP GRADIENT IN *1ST* CONVECTION ZONE
         rcin  = 4.345d8
         rcout = 7.120d8
         do i=1,nlines
            if(rr(i).ge.rcout) then
               ii = i
               goto 666
            endif
         enddo
666      continue
         tcout = tt(ii)
         dcout = dd(ii)
         do j=1,25
            xxxinput(j) = xyz(ii,j)
         enddo
         call composition(X,Y,Z,imu,abar,zbar)
         ye = zbar/abar
         call Heos(1,tt(ii),dd(ii),abar,zbar,                        &
              eix,pp1,encout,cv1,cp1,cs1,                            &
              gam1, gam2, gam3, delta1, keyerr)
!     (LOOP OVER ALL ZONES: MAKE CONVECTION ZONE ISENTROPIC)
         do i=1,nlines
            if(rr(i).ge.rcin.and.rr(i).le.rcout) then
               do j=1,25
                  xxxinput(j) = xyz(i,j)
               enddo
               call composition(X,Y,Z,imu,abar,zbar)
               ye = zbar/abar
               call Heos(1,tt(i),dd(i),abar,zbar,                    &
                    eix,pp1,en1,cv1,cp1,cs1,                         &
                    gam1, gam2, gam3, delta1, keyerr)
               deltt1  = tt(i)*(encout - en1)/cv1
               tt(i)   = tt(i)   + deltt1
            endif
         enddo
      endif

      if(1.eq.0) then        !ADJUST ENTROPY PROFILE BENEATH CONVECTION ZONE
         rcin  = 3.0d8
         rcout = 4.345d8
!     get entropy at inner edge of convection zone
         do i=1,nlines
            if(rr(i).ge.rcin) then
               ii = i
               goto 667
            endif
         enddo
667      continue
         do j=1,qn
            xxxinput(j) = xyz(ii,j)
         enddo
         call composition(X,Y,Z,imu,abar,zbar)
         call Heos(1,tt(ii),dd(ii),abar,zbar,                        &
              eix,pp1,encin,cv1,cp1,cs1,                             &
              gam1,gam2,gam3,delta1,keyerr)
!     get entropy at outter edge of convection zone
         do i=1,nlines
            if(rr(i).ge.rcout) then
               ii = i
               goto 668
            endif
         enddo
668      continue
         do j=1,qn
            xxxinput(j) = xyz(ii,j)
         enddo
         call composition(X,Y,Z,imu,abar,zbar)
         call Heos(1,tt(ii),dd(ii),abar,zbar,                        &
              &           eix,pp1,encout,cv1,cp1,cs1,                            &
              &           gam1,gam2,gam3,delta1,keyerr)
         delenv = (encout - encin)/(rcout - rcin)
         do i=1,nlines
            if(rr(i).ge.rcin.and.rr(i).le.rcout) then
               ennew = encin + delenv*(rr(i)-rcin)
!     get new (temp,dens) given (new entropy, old press)
            endif
         enddo
      endif


!     CHECK INITIAL MODEL CONSISTENCY W/TIMMES EOS.

      if(myid.eq.0.and.ixnuc.ne.3) then

         print*,'MSG(read_tycho): checking EOS consistency'
         pptestmax = -1.d100
         do i=1,nlines
            if(tt(i).gt.1.d8) then
               xxxtest = 0.d0
               do j=1,qn
                  xxxinput(j) = xyz(i,j)
                  xxxtest = xxxtest + xxxinput(j)
               enddo
               call composition(X,Y,Z,imu,abar,zbar)
               call Heos(1,tt(i),dd(i),abar,zbar,                          &
                    eix,prx,enx,cvx,cpx,csx,                               &
                    gam1, gam2, gam3, delta, keyerr)
               pptest = dabs(1.d0 - pp(i)/prx)
               if(pptest.gt.5.d-2) then
                  print*,'ERR(read_tycho): EOS Error: press mismatch.'
                  !                  call MPI_FINALIZE(ierr)
!                  stop'ERR(read_tycho): EOS Error: press mismatch.'
               endif
               pp(i) = prx
               if(pptest.gt.pptestmax) pptestmax = pptest
            endif
         enddo

         print*,'MSG(read_tycho): max % deviation from EOS: pptestmax = ', pptestmax*1.d2
      endif


      if(myid.eq.0) print*,'END(read_tycho):'
      if(myid.eq.0) print*,''



!     SUCCESS
      return
!
    end subroutine read_tycho