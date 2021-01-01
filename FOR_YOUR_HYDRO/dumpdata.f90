subroutine dumpdata(dumpmode)

  !  dumpmode = 0: dump vars set with the dumpdat array.
  !  dumpmode = 1: create restart file (defunct)

  use mod_mpicom_data
  use mod_tdiff_data
  !use mod_rans_data
  implicit none
  include 'dimen.inc'
  include 'char.inc'
  include 'intgrs.inc'
  include 'files.inc'
  include 'mpif.h'
  include 'mpicom.inc'

  integer*4, intent(in) :: dumpmode

  integer*4 i, j, k, n, m
  integer*4 ndata, idat(qdat + 10)
  character*(5)  ndumpsuffix
  character*(50) fn_vtk
  character*(80) fn_vtk_particles
  character*(50) fn_dxhead, fn_rdxhead
  character*(50) fn_ranshead, fn_ransdat
  character*(50) fn_header, fn_bindata
  character*(50) fn_blockhead, fn_blockdat
  character*(50) fn_rheader,fn_rbindata
  character*(8) :: varname(50)
  logical fexist
  integer stat(MPI_STATUS_SIZE)
  integer*4 position(3)
  integer*4 ndump, rndump, dumpdat(qdat)

  save ndump
  save rndump
  data ndump/1/
  data rndump/1/


  ! LOG THIS CALL

  if(dumpmode.eq.0) then
     if(myid.eq.0) write(6,*) 'MSG(dumpdata): STANDARD, ndump=',ndump
     if(myid.eq.0.or.ilog.eq.1) write(nthlog,*) 'MSG(dumpdata): STANDARD, ndump=',ndump
  endif

  if(dumpmode.eq.1) then
     if(ilog.eq.1) then
        print*,'ERR(dumpdata): dumpmode.eq.1 defunct, return without write'
     endif
     return
  endif


  ! NOTE: VAR NAMES ARE SHORTER THAN 8 CHAR LONG
  varname(1)  = 'density'
  varname(2)  = 'velx'
  varname(3)  = 'vely'
  varname(4)  = 'velz'
  varname(5)  = 'energy'
  varname(6)  = 'press'
  varname(7)  = 'temp'
  varname(8)  = 'gam1'
  varname(9)  = 'gam2'
  varname(10) = 'enuc1'
  varname(11) = 'enuc2'
  varname(12) = 'kdiff'
  varname(13) = 'fradx'
  do i=1,qn
     write(varname(13+i),'(I4.4)') i !composition names
  enddo

  ! ZERO DUMPDAT FLAG

  do i=1,qdat
     dumpdat(i) = 0         !ZERO VALUES
  enddo

  dumpdat(1)  = 1        !dens
  do i=2,(1+nsdim)
     dumpdat(i) = 1      !velocities
  enddo
  dumpdat(5)  = 1        !energy
  dumpdat(6)  = 1        !press
  dumpdat(7)  = 1        !temp
  dumpdat(8)  = 1        !gam1
  dumpdat(9)  = 1        !gam2
  dumpdat(10) = 1        !enuc1
  dumpdat(11) = 1        !enuc2
  dumpdat(12) = 0        !kdiff
  dumpdat(13) = 0        !fradx
  if(itdiff.gt.0) then
     dumpdat(12) = 1
     dumpdat(13) = 1
  endif
  do i=1,qn
     dumpdat(13+i) = 1      !composition variables
  enddo


  ! COUNT DATA TO BE DUMPED, SET MAPPING INDEX "IDAT"

  ndata = 0
  do i=1,(13+qn)            !COUNT NUMBER OF DATA
     if(dumpdat(i).eq.1) then
        ndata = ndata + 1
        idat(ndata) = i
     endif
  enddo


  ! CHECK FILE INDEX

  if(irstrt.ne.0) then
     ndump   = irstrt + 1
     irstrt = 0
  endif


  ! CREATE FILENAMES

101 continue
9901 format(i5.5)

  write(ndumpsuffix,9901) ndump

  fn_vtk = '.'//ndumpsuffix//'.vtk'
  call concat(runname, fn_vtk, fn_vtk)

  fn_dxhead = '.'//ndumpsuffix//'.dxhead'
  call concat(runname,fn_dxhead,fn_dxhead)

  fn_ranshead = '.'//ndumpsuffix//'.ranshead'
  call concat(runname,fn_ranshead,fn_ranshead)

  fn_ransdat = '.'//ndumpsuffix//'.ransdat'
  call concat(runname,fn_ransdat,fn_ransdat)

  fn_bindata = '.'//ndumpsuffix//'.bindata'
  call concat(runname,fn_bindata,fn_bindata)

  fn_blockdat = '.'//ndumpsuffix//'.blockdat'
  call concat(runname,fn_blockdat,fn_blockdat)

  fn_header = '.'//ndumpsuffix//'.header'
  call concat(runname,fn_header,fn_header)

  fn_blockhead = '.'//ndumpsuffix//'.blockhead'
  call concat(runname,fn_blockhead,fn_blockhead)

  fn_vtk_particles = '.particles.'//ndumpsuffix//'.vtk'
  call concat(runname,fn_vtk_particles,fn_vtk_particles)

  inquire(file=fn_bindata,exist=fexist)
  inquire(file=fn_blockdat,exist=fexist)
  if(fexist) then
     ndump=ndump+1
     goto 101
  endif

  call MPI_BARRIER(commcart,ierr)  

  !call write_vtk_file(fn_vtk, varname, idat, ndata)
  !call write_binary_block_files(fn_blockhead, fn_blockdat, varname, idat, ndata)

  !Reinstate bindata files, blockdata is buggy, and bindata is ~identical? SWC 2/4/19.
  call write_bindata_header(fn_header, varname, idat, ndata)
  call write_binary_bindata(fn_bindata, idat, ndata)

  call write_rans_files(fn_ranshead, fn_ransdat)

  ndump = ndump + 1

  return

end subroutine dumpdata
