subroutine write_rans_data(fn_ransdat)
  use mod_mpicom_data
  use mod_rans_data
  implicit none
  include 'dimen.inc'
  include 'intgrs.inc'
  include 'files.inc'
  include 'mpif.h'
  include 'mpicom.inc'
  
  character*(50), intent(in) :: fn_ransdat
  integer*4 root
  integer*4 i, j, k, ii, n, m
  integer*4 irecl
  real*8 havg_sum           (4,nrans,qx )
  real*8 havg_sum_tot_local (4,nrans,qqx)
  real*8 havg_sum_tot_global(4,nrans,qqx)
  
  irecl = 2*irecl_float*(nrans*4*qqx) ! (double precision array of size nrans x 4 x nxglobal)
  open(ntlocal, file=fn_ransdat, form='unformatted',access='direct',recl = irecl)
  
  call MPI_ALLREDUCE(havg(1,1,1), havg_sum, 4*qx*nrans, &
       MPI_DOUBLE_PRECISION, MPI_SUM, commyz, ierr)
  
  if(myid_commx00.ne.MPI_UNDEFINED) then
     
     root = 0
     
     do j=1,4
        do k=1,nrans
           do i=1,qqx
              havg_sum_tot_local (j,k,i) = 0.d0
              havg_sum_tot_global(j,k,i) = 0.d0
           enddo
           do i=1,qx
              ii = qx*coords(1) + i
              havg_sum_tot_local(j,k,ii) = havg_sum(j,k,i)
           enddo
        enddo
     enddo
     
     if(ntiles(1).gt.1) then
        call MPI_REDUCE(havg_sum_tot_local, havg_sum_tot_global, qqx*4*nrans, &
             MPI_DOUBLE_PRECISION, MPI_SUM, root, commx00, ierr)
     else
        do j=1,4
           do k=1,nrans
              do i=1,qx
                 havg_sum_tot_global(j,k,i) = havg_sum_tot_local(j,k,i)
              enddo
           enddo
        enddo
     endif
     
     ! normalize time average by averaging time
     
     do k=1,nrans
        do i=1,qqx
           if(rans_tavg.gt.1.d-10) then
              havg_sum_tot_global(3,k,i) = havg_sum_tot_global(3,k,i)/rans_tavg
           else
              havg_sum_tot_global(3,k,i) = havg_sum_tot_global(2,k,i) !use instantaneous in this case
           endif
        enddo
     enddo

     ! write to disk
     
     if(myid_commx00.eq.root) then
        write(ntlocal,rec=1) havg_sum_tot_global
     endif
  endif
  
  close(ntlocal)
  
  call MPI_BARRIER(commcart, ierr)
  
  return
  
end subroutine write_rans_data
