!
!
!   
!     perturb initial multidimensional model
!   
!
!
subroutine perturb_ccp_two_layers

  implicit none
  include 'dimen.inc'
  include 'float.inc'
  include 'intgrs.inc'
  include 'physcs.inc'
  include 'vnew.inc'
  include 'mesh.inc'
  include 'constants.inc'
  include 'mpif.h'
  include 'mpicom.inc'

  integer*4 i, j, k     , ii
  real*8 vsound, phi, mag, r8rand
  real*8 x,y,z,prtb,rho0
  real*8 fheat(nx)
  
  if(ilog.eq.1) then
     print*,'MSG(perturb_ccp_two_layers): density/temp perturb_ccp_two_layersations in convection zone'
  endif

  pi = 4.d0*datan(1.d0)

  do i=1,nx
     x = xzn(i)/onelu
     if((x.ge.1.d0).and.(x.le.1.125d0)) then
        fheat(i) = sin(8.d0*pi*(x-1.d0))
     else
        fheat(i) = 0.d0
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
                0.00005d0*rho0*fheat(i)*((sin(3.d0*pi*y))+cos(pi*y))*(sin(3.d0*pi*z)-cos(pi*z))
           densty(i,j,k) = densty(i,j,k) + prtb
        enddo
     enddo
  enddo

  
  
  return

  
end subroutine perturb_ccp_two_layers

