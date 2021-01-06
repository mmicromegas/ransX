!
! rans averaging 
!
!
module mod_rans_data
  
  implicit none

  integer*4 :: nrans, irans, rans_nnuc
  integer*4 :: nransX
  real*8, allocatable :: sterad(:,:), fsterad(:,:)
  real*8, allocatable :: havg(:,:,:)
  real*8 :: rans_tstart, rans_tend, rans_tavg, steradtot   
  character(len=24), allocatable :: ransname(:)
  integer*4, allocatable :: iransnuc(:)
  
  contains 
    
    subroutine rans_data_init
     
      implicit none
      include 'dimen.inc' 
      
      rans_nnuc = qn ! rans_nnuc = qn
      nrans     = 332 + 25*rans_nnuc

      allocate(havg(4,nrans,qqx))
      allocate(ransname(nrans))
      
    end subroutine rans_data_init
    
  
end module mod_rans_data
