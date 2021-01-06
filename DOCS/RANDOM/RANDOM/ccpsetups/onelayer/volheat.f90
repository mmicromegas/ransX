!     NUCLEAR ENERGY GENERATION ROUTINE
!     
      subroutine volheat(imode)
!-------------------------------------------------------
!     SIMPLE NE-BURNING NETWORK
!     ------------------------------
!     This is a simple fuel+ash implementation of the 
!     nuclear energy generation for Ne burning.
!     
!-------------------------------------------------------
      implicit none
      include 'dimen.inc'
      include 'vnew.inc'
      include 'intgrs.inc'
      include 'float.inc'
      include 'constants.inc'
      include 'grd.inc'
      include 'mesh.inc'
      include 'mpif.h'
      include 'mpicom.inc'
!
      integer*4 :: imode
      integer*4 i,j,k,l   
      real*8 dd,tt,ei,ek
      real*8 dtb
      real*8 fcool(qx)
      real*8 y

      pi = 4.d0*datan(1.d0)
      
!     SETUP TIME-STEP
!     ---------------
      dtb = dt
!      

      do i=1,nx
         y = xzn(i)/onelu
         if((y.gt.1.875d0).and.(y.le.2.0d0)) then
            fcool(i) = -sin(8.d0*pi*(2.d0-y))
         else
            fcool(i) = 0.d0
         endif
      enddo

      
!     BEGIN LOOP OVER ZONES
!     ---------------------
      do k=1,qz
         do j=1,qy
!
            !     setup thermodynamic arrays
            do i=1,qx
               dd = densty(i,j,k)
               tt = temp(i,j,k)
               ek    = velx(i,j,k)**2.d0+vely(i,j,k)**2.d0+             &
     &              velx(i,j,k)**2.d0
               ei = energy(i,j,k)-ek
!
               enuc(i,j,k,1) = (1.d0/onetu)*(oneeu/onemu)*(0.0001556706d0*(fcool(i)/(dd/onedu)))
               enuc(i,j,k,2) = 0.d0 !neutrino emission term
!
               !     update energy and composition: put back into main arrays
               if(imode.eq.1) then !update internal energy
                  energy(i,j,k)   = energy(i,j,k) + (enuc(i,j,k,1) + enuc(i,j,k,2))*dtb
               endif

               !..
            enddo
         enddo                  !j-loop
      enddo                     !k-loop
!
!..   
!..   update thermodynamics across grid
      if(imode.eq.1) then
         call eos3d(1)
      endif
!
!...  
!     SUCCESS
      return
!...  
!...  
      end
