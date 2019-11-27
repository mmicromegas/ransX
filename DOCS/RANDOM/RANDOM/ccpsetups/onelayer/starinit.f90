!     
!     THIS SUBROUTINE MAPS A 1D STELLAR 
!     PROFILE ONTO THE MULTI-D GRID.
!     
      subroutine starinit   
!     
      implicit none
      include 'dimen.inc'
      include 'float.inc'
      include 'intgrs.inc'
      include 'vnew.inc'
      include 'mesh.inc'
      include 'grd.inc'
      include 'comp.inc'
      include 'gravity.inc'
      include 'diffusion.inc'
      include 'core.inc'
      include 'heos.inc'
      include 'constants.inc'
      include 'files.inc'
      include 'mpif.h'
      include 'mpicom.inc'
!     
      integer*4 i,ii,j,k
      real*8 vphi, vtheta, vmag
      integer*4 nzones, errorkey
      parameter( nzones = 5001 )   
      real*8 tt(nzones), dd(nzones), hf(nzones)
      real*8 pp(nzones), gg(nzones), mmu(nzones)      
      real*8 mm(nzones), cv(nzones), xyz(nzones, qn)
      real*8 XX, YY, ZZ, imu, cp, gmc, gme, eint
      real*8 kypert, kzpert, fpert, xcut, cs2
      real*8 rglob(qqq), rglobl(qqq), rglobr(qqq)
      real*8 ddglobal(qqq), ttglobal(qqq), ppglobal(qqq)
      real*8 xyzglobal(qqq,qn), cvglobal(qqq), mmglobal(qqq)
      real*8 muglobal(qqq), ggglobal(qqq)      
      real*8 rrr(nzones),rrl(nzones), rr(nzones)
      real*8 ddd, ttt, fpress, fgrav, delrr
      real*8 apress, agrav
      real*8 mmm,bbb,delr,hf0,hf1,xa,xb
      real*8 delx,abar,zbar
      integer*8 nglobal, indx, iseed
      real*8 deltar,dppdr
      real*8 pphse(nx)

      logical l_stabil
      real*8 rdiffok
      integer iter_max      

!     -----------------------------------------------
!     SETUP/READ IN HIGH RESOLUTION 1D MODEL PROFILES
!     -----------------------------------------------


      if(isinit.eq.5) then
         if(myid.eq.0) print*,'MSG(starinit): call read_ccp_single  isinit = ', isinit
         call read_ccp_one_layer(nzones,rrl,rr,rrr,dd,pp,tt,mmu,xyz)
      endif

      
!     --------------------------------
!     SETUP 1D GLOBAL GRID
!     --------------------------------
      if(igeomx.eq.1) then
         nglobal = nyglobal
         do j=1,nglobal
            rglob (j) = gyzn (j)
            rglobr(j) = gyznr(j)
            rglobl(j) = gyznl(j)
         enddo
      else
         nglobal = nxglobal
         do i=1,nglobal
            rglob (i) = gxzn (i)
            rglobr(i) = gxznr(i)
            rglobl(i) = gxznl(i)
         enddo
      endif

!     ---------------------------------
!     INTERPOLATE MODEL ONTO HYDRO GRID
!     ---------------------------------

      if(myid.eq.0) then
         print*,'MSG(starinit): interpolate model onto grid.'
         print*,'MSG(starinit): rrmin,rrmax= ',rr(1),rr(nzones)
      endif
      call ppmlinterp(nglobal,rglobl,rglob,rglobr,ppglobal,nzones,rr,pp)
      call ppmlinterp(nglobal,rglobl,rglob,rglobr,ddglobal,nzones,rr,dd)
      call ppmlinterp(nglobal,rglobl,rglob,rglobr,ttglobal,nzones,rr,tt)      
      call ppmlinterp(nglobal,rglobl,rglob,rglobr,muglobal,nzones,rr,mmu)
      do i=1,qn
         call ppmlinterp(nglobal,rglobl,rglob,rglobr,xyzglobal(1,i),    &
     &        nzones,rr,xyz(1,i))
      enddo
      
      if(myid.eq.0) print*,'MSG(starinit): done interpolating model.'
            
!     -----------------------------------------
!      MAP MODEL TO LOCAL ARRAYS
!     -----------------------------------------
      do k=1,nz
         do j=1,ny
            do i=1,nx
               indx=i+coords(1)*qx
               densty(i,j,k) = ddglobal(indx)
               press(i,j,k)  = ppglobal(indx)
               temp(i,j,k) = ttglobal(indx)
               mu(i,j,k) = muglobal(indx)
!
               do ii=1,ncomp
                  xnuc(i,j,k,ii) = xyzglobal(indx,ii)
               enddo
               do ii=(ncomp+1),qn
                  xnuc(i,j,k,ii) = 0.d0
               enddo
               velx(i,j,k) = 0.d0
               vely(i,j,k) = 0.d0
               velz(i,j,k) = 0.d0               
               enuc(i,j,k,1) = 0.d0
               enuc(i,j,k,2) = 0.d0
            enddo
         enddo
      enddo
 
      call eos3d(3)            
      
!     improve HSE

      l_stabil = .false.

!     input to the iteration

      rdiffok  = 5.d-5 
      iter_max = 50

      dt = dtini
      call tstep
      
      if (l_stabil) &
           call stabil(rdiffok,iter_max)
      

!     SYNC PROCS AND RETURN
      call MPI_BARRIER(commcart, ierr)

      if(myid.eq.0) print*,'END(starinit):'
      if(myid.eq.0) print*,''
      

!     SUCCESS
      return
      
    end subroutine starinit
