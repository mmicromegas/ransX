subroutine write_rans_header(fn_ranshead)
  
  use mod_mpicom_data
  use mod_tdiff_data
  use mod_rans_data
  implicit none
  include 'dimen.inc'
  include 'mesh.inc'
  include 'float.inc'
  include 'intgrs.inc'
  include 'files.inc'
  include 'mpif.h'
  include 'mpicom.inc'
  
  character*(50), intent(in) :: fn_ranshead
  
  integer*4 i, j, k, n, m
  real*8 havg_sum           (4,nrans,qx )
  real*8 havg_sum_tot_local (4,nrans,qqx)
  real*8 havg_sum_tot_global(4,nrans,qqx)
  integer*4 position(3)
  
9902 format('xzn ',I4.4,3E15.8)
9903 format('yzn ',I4.4,3E15.8)
9904 format('zzn ',I4.4,3E15.8)

  if(myid.eq.0) then
     open(ntlocal,file=fn_ranshead,status='unknown')
     write(ntlocal,'(I9.9,3e15.8)') nstep, rans_tstart, rans_tend, rans_tavg
     write(ntlocal,*) qqx,qqy,qqz,qn,nrans,rans_nnuc
     write(ntlocal,*) bndmnx,bndmxx,bndmny,bndmxy,bndmnz,bndmxz
     write(ntlocal,*) igeomx, igeomy, igeomz,mstates,isinit,idamp,xdamp,itdiff,xtdiff, &
          ftdiff,igrav,xyzgrav,iopac,ieos,ixnuc,iburn
     do i=1,nrans
        write(ntlocal,*) ransname(i)
     enddo
     do i=1,nxglobal
        write(ntlocal,9902) i,gxznl(i),gxzn(i),gxznr(i)
     enddo
     do i=1,nyglobal
        write(ntlocal,9903) i,gyznl(i),gyzn(i),gyznr(i)
     enddo
     do i=1,nzglobal
        write(ntlocal,9904) i,gzznl(i),gzzn(i),gzznr(i)
     enddo
     close(ntlocal)
  endif
  
  call MPI_BARRIER(commcart, ierr)
  
  return
  
end subroutine write_rans_header
