!
!
!
!
subroutine sterad_init
  
  use mod_rans_data, only: sterad, steradtot, fsterad
  
  implicit none
  include 'dimen.inc'
  include 'intgrs.inc'
  include 'mesh.inc'
  include 'mpif.h'
  include 'mpicom.inc'
  
  integer*4 i, j, k, jj, kk
  real*8 cosyl, cosyr, dphi, dcosy
  logical, save :: FirstCall = .true.
  
  
  steradtot = 0.d0
  
  ! allocate memory for fractional zone area arrays

  if(FirstCall) then
     if(xyzgrav.eq.1) then 
        if(ilog.eq.1) print*,'MSG(sterad_init): allocate sterad(:,:), fsterad(:,:)'
        allocate(sterad(qqy,qqz),fsterad(qy,qz))
        FirstCall = .false.
     else
        call MPI_FINALIZE(ierr)
        stop 'ERR(sterad_init): xyzgrav.ne.1 not implemented'
     endif
  endif
  
  if(igeomx.eq.2.and.igeomy.eq.4.and.igeomz.eq.5) then !spherical geometry: y-z angular shells
     do k=1,qqz
        do j=1,qqy
           cosyl = cos(gyznl(j))
           cosyr = cos(gyznr(j))
           dcosy = cosyl - cosyr
           dphi = gzznr(k) - gzznl(k)
           sterad(j,k) = dcosy*dphi
           steradtot   = steradtot + sterad(j,k)
        enddo
     enddo
     do k=1,qz
        kk = coords(3)*qz + k
        do j=1,qy
           jj = coords(2)*qy + j
           fsterad(j,k) = sterad(jj,kk)/steradtot
        enddo
     enddo

  else if(igeomx.eq.0.and.igeomy.eq.0.and.igeomz.eq.0) then !planar geometry: y-z planes
     do k=1,qqz
        do j=1,qqy
           sterad(j,k)  = 1.d0/dble(qqy)/dble(qqz)
        enddo
     enddo
     steradtot = 1.d0
     do k=1,qz
        do j=1,qy
           fsterad(j,k) = 1.d0/dble(qqy)/dble(qqz)
        enddo
     enddo
  endif

end subroutine sterad_init
