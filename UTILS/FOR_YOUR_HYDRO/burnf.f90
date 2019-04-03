!------------------------------------------------------------
!     ADVANCE REACTION NETWORK
!------------------------------------------------------------
subroutine burnf(bmode)
!------------------------------------------------------------
!     *** energy generation, composition evolution ***
!
!     written: D. Arnett (26 Nov 1999)
!     spring cleaning: C. Meakin (6 May 2004).......
!     I restructed the entire code, moving network 
!     setup/solution to the burnzone(...) subroutine.
!
!     bmode = 0: diagnostic mode (do not update abundances 
!                or internal energy)
!                 
!     bmode = 1: normal operations....
!
!     
!------------------------------------------------------------

      implicit none
      include 'dimen.inc'
      include 'bdimen.inc'
      include 'burnf.inc'
      include 'vnew.inc'
      include 'float.inc'
      include 'intgrs.inc'
      include 'mesh.inc'
      include 'mpif.h'
      include 'mpicom.inc'
!     
      integer*4 i,j,k,n,m
      integer*4 bmode
      integer*4 idebug
      integer*4 nsubcycle
      integer*4 kmax, jmax
      real*8 eps(qq), snu(qq), xxn(qn,qq), xold(qn,qq)
      real*8 dd(qq), tt(qq), eei(qq), eek(qq)
      real*8 dth,dtnew
      real*8 xxxtest
      real*8 radius(qx), rminburn, rmaxburn

      
      rminburn = 0.35d9 ! problem specific limits
      rmaxburn = 1.0d9
      
      
      idebug = 0     
      jmax = ny
      kmax = nz


      !     ---LOOP OVER (j,k) ZONES, ADVANCING (i)-ROWS---
      do k=1,kmax
         do j=1,jmax     
            
            
            ! SETUP THERMO (rho,tmp) AND ABUNDANCES (xxn) FOR (i)-ROW
            do i=1,nx
               radius(i) = dsqrt(xzn(i)**2.d0 + yzn(j)**2.d0 + zzn(k)**2.d0)
               !if(radius(i).ge.rminburn.and.radius(i).le.rmaxburn) then
                  
                  dd (i) = densty(i,j,k)
                  tt (i) = temp  (i,j,k)
                  eek (i) = 0.5d00*(velx(i,j,k)**2 + vely(i,j,k)**2  &
                       + velz(i,j,k)**2)
                  eei (i) = (energy(i,j,k)-eek(i))*dd(i)
                  xxxtest = 0.d0
                  do n=1,qn
                     m = xnucmap(n)
                     xxn(m,i)=xnuc(i,j,k,n)
                     xold(m,i)=xxn(m,i)
                     xxxtest = xxxtest + xxn(m,i)
                  enddo                  
                  if(dabs(xxxtest-1.d0).gt.0.2d0) then
                     print*,'ERR(burnf):checksum error:',xxxtest
                  endif

               !endif
               enddo
            
            !     TIME ADVANCE NETWORK
            do i=1,nx               
               if(radius(i).ge.rminburn.and.radius(i).le.rmaxburn) then
                  
                  dth = dt         !no subcycling               
                  call burnzone(tt(i),dd(i),xxn(1,i),eps(i),snu(i), &
                       nsubcycle,dth,dtnew)
                  if(nsubcycle.gt.1) goto 666
               else
                  eps(i) = 0.d0
                  snu(i) = 0.d0
               end if
            enddo
            
            !..   UPDATE ENERGY SRC/SNK, INTERNAL ENERGY & ABUNDANCES (MOLE NO.)
            !     COPY UPDATES TO MAIN HYDRO ARRAYS (energy,temp,xnuc).
            
            do i=1,nx
               if(radius(i).ge.rminburn.and.radius(i).le.rmaxburn) then

                  enuc(i,j,k,1) = eps(i)/dt
                  enuc(i,j,k,2) = snu(i)/dt
                  if(bmode.ne.0) then
                     eei  (i)      = eei(i) + (snu(i)+eps(i))*dd(i)
                     energy(i,j,k) = eei(i)/dd(i) + eek(i)
                     do n=1,nuc
                        m = xnucmap(n)
                        xnuc(i,j,k,n) = xxn(m,i)
                        xnucdot(i,j,k,n) = (xxn(m,i) - xold(m,i))/dt
                     enddo
                  endif

               endif
            enddo

            ! END LOOP OVER (j,k) ZONES
         enddo
      enddo

      ! MAKE THERMO CONSISTENT OVER ENTIRE GRID
      if(bmode.ne.0) then
         call eos3d(1)
      endif


      !SUCCESS
      return

      
!     FATAL ERRORS
 666  write(*,*) 'ERR(burnf): burnzone attempted to subcycle.'
      call MPI_FINALIZE(ierr)
      stop 'ERR(burnf): burnzone attempted to subcycle.'



    end subroutine burnf
