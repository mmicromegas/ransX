!
      subroutine stabil(rdiffok,iter_max)
!
!     SUBROUTINE ATTEMPTS TO PUT INITIAL MODEL TO HSE
        
      implicit none
!
      include 'dimen.inc'
      include 'float.inc'
      include 'intgrs.inc'
      include 'vnew.inc'
      include 'mesh.inc'
      include 'comp.inc'
      include 'gravity.inc'
      include 'constants.inc'
      include 'mpif.h'
      include 'mpicom.inc'

      logical l_iterate
      integer iter_max,nit,i,j,k,nn,ii
      integer idebug
      
      real*8 pp_hse(nx),pp_corr(nx),pp_upd(nx)
      real*8 gg(nx),pp(nx),dd(nx)
      
      real*8 xxxtest,tmp,scr1
      real*8 rdiffcum,rdiffsum,rdiffok,rdiffcum_norm 

      idebug = 1
      
      do i=1,nx
         pp(i) = press(i,1,1)
      enddo


      ! update grav for initial storage
      call gravty(1,1)
      
      pi = 4.0*atan(1.0) ! 3.1415926535898d0

      l_iterate = .true.
      nit = 0
      
      do while ( l_iterate )

         nit = nit + 1

         ! get advected density  
         call hydrox(1)

         ! get innermass
         !call mass_shells_density
         
         do i=1,nx
            dd(i) = densty(i,1,1)            
            gg(i) = grav(i+4)
         enddo

         ! initial condition for HSE integration
         pp_hse(nx) = pp(nx)

         do i=nx-1,1,-1
            pp_hse(i) = pp_hse(i+1) - gg(i)*dd(i)*(xznr(i)-xznl(i))           
         enddo
         
         ! update pressure
         do i=nx,1,-1
            pp_upd(i) = pp_hse(i)
         enddo
                  
         rdiffcum = 0.d0
         rdiffsum = 0.d0

         ! check the deviation from the hydrostatic equilibrium
         do i = 1,nx
            scr1 = (press(i,1,1)-pp_hse(i))/pp_hse(i)
            rdiffcum=rdiffcum+abs(scr1)
         end do 
         rdiffcum_norm = rdiffcum/nx

         if (idebug == 1) then
            print*,'DBG(starinit) Stabilization: &
                 iteration, norm. cummulative HSE rel. diff:',nit,rdiffcum_norm
         end if
            
         do k=1,nz
            do j=1,ny
               do i=1,nx
                  press(i,j,k) = pp_upd(i)
                  densty(i,j,k) = dd(i)
               enddo
            enddo
         enddo

!        complete thermodynamical state

         call eos3d ( 3 ) 
         
         do i=1,nx
            pp(i) = pp_upd(i)
         enddo

!     ...normalize mass fraction...
         do k=1,nz
            do j=1,ny
               do i=1,nx    
                  xxxtest = 0.d0
                  do nn=1,ncomp
                     xxxtest = xxxtest + xnuc(i,j,k,nn)
                  enddo
                  do nn=1,ncomp
                     xnuc(i,j,k,nn) = xnuc(i,j,k,nn)/xxxtest
                  enddo
               enddo
            enddo
         enddo

         ! update dt
         call tstep
         
!        print progress in iteration
 
         if( rdiffcum_norm.le.rdiffok ) then
            l_iterate = .false.
            write(*,101) '[HYDRO_STABIL]  STABILE ITERATION:  ',nit
            write(*,102) '[HYDRO_STABIL]  NORMALIZED DEVIATION &
                 FROM HYDROSTATIC EQUILIBRIUM: ',rdiffcum_norm 
         else
            l_iterate = .true.
         end if

         if ( nit.ge.iter_max ) then
            l_iterate = .false.
            write(*,103) '[HYDRO_STABIL] MAX ITERATION REACHED:  ',nit
            write(*,102) '[HYDRO_STABIL]  NORMALIZED &
                 DEVIATION FROM HYDROSTATIC EQUILIBRIUM: ',rdiffcum_norm
         end if
               
      enddo



101   format(1x,a36,i4)
102   format(1x,a56,e15.6e2)
103   format(1x,a39,i4)

      
!     SUCCESS
      return
      
    end subroutine stabil


      
