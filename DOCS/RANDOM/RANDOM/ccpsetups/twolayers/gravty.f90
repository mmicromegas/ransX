!     
!     SETUP GRAVITATIONAL FORCE ARRAY FOR HYDRO
!     
!     THE igrav FLAG IN THE INPUT FILE SETS THE GRAVITY
!     MODE:  igrav=(0:none, 1:planar, 2:shellular)
!     
subroutine gravty (j, k)
  
  implicit none
  include 'dimen.inc'
  include 'float.inc'
  include 'intgrs.inc'
  include 'mesh.inc'
  include 'grd.inc'
  include 'constants.inc'
  include 'gravity.inc'
  include 'vnew.inc'
  include 'core.inc'
  include 'files.inc'
  include 'mpif.h'
  include 'mpicom.inc'
  !include 'basestate.inc'
  
  integer*4 i,j,k, ii
  real*8 :: y,fgy,five_fourth
  
  pi = 4.d0*datan(1.d0)
  
  ! gravity profile 
  do ii=1,nzn8 
     grav(ii) = 0.d0
  enddo
 
  five_fourth = 5.d0/4.d0
  
  if(xyzswp.eq.xyzgrav) then
     if (igrav.eq.6) then
        do ii=1,nzn8
           y = x(ii)/onelu 
           grav(ii) = g0/(y**five_fourth)
        enddo
     endif
!     
     if (igrav.eq.7) then
        do ii=1,nzn8
           y = x(ii)/onelu
           if (y.lt.1.0d0) fgy = 0.d0
           if (y.gt.3.0d0) fgy = 0.d0
           !
           if ((y.ge.1.0625d0).and.(y.le.2.9375d0)) &
                fgy = 1.d0
           if ((y.ge.1.d0).and.(y.lt.1.0625d0)) &
                fgy = 0.5d0*(1.d0+sin(16.d0*pi*(y-1.03125d0)))
           if ((y.gt.2.9375d0).and.(y.le.3.0d0)) &
                fgy = 0.5d0*(1.d0-sin(16.d0*pi*(y-2.96875d0)))
           grav(ii) = fgy*g0/(y**five_fourth)
        enddo
     endif
  endif

  
  return

end subroutine gravty
