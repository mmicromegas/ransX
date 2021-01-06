!                                 
!     READ IN TWO LAYERS CONVECTION MODEL (CODE COMPARISON PROJECT)
!

subroutine read_ccp_two_layers(nzones,rrl,rr,rrr,ddi,ppi,tti,mui,xyzi)
!     
  implicit none
  include 'dimen.inc'
  include 'bdimen.inc'
  include 'intgrs.inc'
  include 'float.inc'
  include 'files.inc'
  include 'burnf.inc'
  include 'comp.inc'
  include 'heos.inc'
  include 'constants.inc'
  include 'mpif.h'
  include 'mpicom.inc'
!     
  integer*4 i, ii, j, jj, nlines, nzones

  real*8 rri(nzones), yi(nzones),tti(nzones)
  real*8 ppi(nzones), ddi(nzones)
  real*8 ggi(nzones), mui(nzones)
  real*8 xyzi(nzones,qn),fv(nzones)

  real*8 ggic(nzones),ppic(nzones)
  
  real*8 rrp(nzones), rrl(nzones)
  real*8 rrr(nzones), rr(nzones)
  
  integer stat(MPI_STATUS_SIZE)
  real*8 yyy, rtop, rbot, ccpy, width, deltay, y1, y3, ynlines
  real*8 gzero,deltar,dppdr,dri,py1,five_fourth
      
  character*50 fn_name
  logical fkeyerr
!     
  pi = 4.d0*datan(1.d0)
 
  fn_name = "data/setup-two-layers.in"

  call open_check(fn_name,ntlocal,fkeyerr)
  
  read(ntlocal,*) nlines
  
  if(ilog.eq.1) then 
     write(6,*) 'MSG(read_ccp_two_layers): nlines=',nlines
  endif
  
  if(ilog.eq.1) print*,'MSG(read_ccp_two_layers): begin loop over model: nlines =', nlines
  
  ! init output arrays
  do i=1,nzones
     rri(i) = 0.d0
     ggi(i) = 0.d0    
     ddi(i) = 0.d0
     ppi(i) = 0.d0
     tti(i) = 0.d0
  enddo

  !pi=4.d0*datan(1.d0) ! calculate pi  

  ncomp = qn
  
  ! read model lines  
  do i=1,nlines
     read(ntlocal,*) yi(i), ggi(i), fv(i), ddi(i), ppi(i), tti(i)
  enddo
  
  ! according to code comparison project specs (updated 7/March/2020)

  !onelu = 4.d8             ! length unit 4 Mm
  !onepu = 4.644481d23      ! pressure unit dyne
  !onedu = 1.820940d6       ! density unit g/cm3
  !onetu = 0.7920256        ! time unit in seconds
  !onemu = 1.165402d32      ! mass unit in grams
  !onettu = 3.401423d9    ! temperature unit in Kelvins
  !onegu = onelu/(onetu**2) ! gravitational acceleration unit
  !mu0 = 1.848              ! mean molecular weight of fluid_1
  !mu1 = 1.802              ! mean molecular weight of fluid_2
  !oneluu = 3.752995d49    ! luminosity unit erg/s
  !oneeu  = 2.972468d49     ! energy unit erg
  
  ! convert to physical units
  do i=1,nlines
     rri(i) = onelu*yi(i)
     ggi(i) = onegu*ggi(i)    
     ddi(i) = onedu*ddi(i)
     ppi(i) = onepu*ppi(i)
     tti(i) = onettu*tti(i)
  enddo

  
  ! calculate composition profiles for fluid_1 and fluid_2
  do i=1,nlines
     ! get bottom layer
     if ((yi(i).ge.yi(1)).and.(yi(i).lt.(1.9375d0))) then
        xyzi(i,1) = 1.d0
        xyzi(i,2) = 0.d0
        mui(i) = mu0 
     endif
     ! get transition layer
     if ((yi(i).ge.(1.9375d0)).and.(yi(i).le.(2.0625d0))) then
        xyzi(i,2) = fv(i)/((1.d0-fv(i))*(mu0/mu1)+fv(i))
        xyzi(i,1) = 1.d0-xyzi(i,2)
        mui(i) = 1.d0/((xyzi(i,1)/mu0)+(xyzi(i,2)/mu1))
     endif
     ! get top layer
     if ((yi(i).gt.(2.0625d0)).and.(yi(i).le.(yi(nlines)))) then
        xyzi(i,1) = 0.d0
        xyzi(i,2) = 1.d0
        mui(i) = mu1
     endif
  enddo  


!     MANAGE ZONES: shift index:
!     rrl: left interface of zone i
!     rrr: right interface of zone i
!     rr : center of zone i

     
      do i=1,nlines-1
         dri = (rri(i+1)-rri(i))/2.d0
         rrl(i) = rri(i)-dri
         rrr(i) = rri(i)+dri     
         rr(i)  = rri(i)     
         ppi(i) = ppi(i)
         ddi(i) = ddi(i)
         tti(i) = tti(i)        
         mui(i) = mui(i)
      enddo



!     COPY LAST ZONE (nzones-nlines) TIMES TO FILL ARRAY
      do i=nlines,nzones
         rrl(i) = rrl(i-1)
         rr(i)  = rr(i-1)
         rrr(i) = rrr(i-1)
         ppi(i) = ppi(i-1)
         ddi(i) = ddi(i-1)
         tti(i) = tti(i-1)
         mui(i) = mui(i-1)
         do ii=1,qn
            xyzi(i,ii) = xyzi(i-1,ii)
         enddo
      enddo
  
  print*,'MSG(read_ccp_two_layers): pp, rho:  =', ppi(1),ppi(nlines),ddi(1),ddi(nlines)
  
  if(ilog.eq.1) print*,'END(read_ccp_two_layers):'
  
  return
  
end subroutine read_ccp_two_layers

