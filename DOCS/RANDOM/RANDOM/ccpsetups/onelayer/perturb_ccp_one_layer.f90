!
!
!   
!     perturb initial multidimensional model
!   
!
!
subroutine perturb_ccp_one_layer

  implicit none
  include 'dimen.inc'
  include 'float.inc'
  include 'constants.inc'  
  include 'intgrs.inc'
  include 'physcs.inc'
  include 'vnew.inc'
  include 'mesh.inc'
  include 'mpif.h'
  include 'mpicom.inc'

  integer*4 i, j, k     , ii
  real*8 vsound, phi, mag, r8rand
  real*8 fcool(qx)
  real*8 prtb,rho0,y,z

  if(ilog.eq.1) then
     print*,'MSG(perturb_ccp_one_layer): density/temp perturb_ccp_one_layer in convection zone'
  endif

  pi = 4.d0*datan(1.d0)

  do i=1,nx
     y = xzn(i)/onelu
     if((y.gt.1.875d0).and.(y.le.2.0d0)) then
        fcool(i) = -sin(8.d0*pi*(2.d0-y))
     else
        fcool(i) = 0.d0
     endif
  enddo
  
  rho0 = 0.d0
  prtb = 0.d0
  
  ! LOOP OVER ALL ZONES
  do k = 1, qz
     do j = 1, qy
        do  i = 1, qx
           y = yzn(j)/onelu
           z = zzn(k)/onelu
           rho0 = densty(1,j,k)
           prtb = &
                0.00005d0*rho0*fcool(i)*((sin(3.d0*pi*y))+cos(pi*y))*(sin(3.d0*pi*z)-cos(pi*z))
           densty(i,j,k) = densty(i,j,k) + prtb
        enddo
     enddo
  enddo
  
  return

  
end subroutine perturb_ccp_one_layer

