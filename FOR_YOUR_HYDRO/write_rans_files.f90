subroutine write_rans_files(fn_ranshead, fn_ransdat)

  use mod_rans_data, only: irans
  implicit none
  character*(50), intent(in) :: fn_ransdat, fn_ranshead

  if(irans.gt.0) then
     call write_rans_header(fn_ranshead)
     call write_rans_data(fn_ransdat)
  endif
  
end subroutine write_rans_files

