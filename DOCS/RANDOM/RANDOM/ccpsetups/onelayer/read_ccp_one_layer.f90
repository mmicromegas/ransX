!                                 
!     READ IN ONE LAYER CONVECTION MODEL (CODE COMPARISON PROJECT)
!

subroutine read_ccp_one_layer(nzones,rrl,rr,rrr,ddi,ppi,tti,mui,xyzi)
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

  real*8 rri(nzones),yi(nzones),tti(nzones)
  real*8 ppi(nzones), ddi(nzones)
  real*8 ggi(nzones), mui(nzones)
  real*8 xyzi(nzones,qn)

  real*8 ggic(nzones),ppic(nzones)
  
  real*8 rrp(nzones), rrl(nzones)
  real*8 rrr(nzones), rr(nzones)
  
  integer stat(MPI_STATUS_SIZE)
  real*8 fv, yyy, rtop, rbot, ccpy, width, deltay, y1, y3, ynlines
  real*8 gzero,deltar,dppdr,py1
  real*8 g0,five_fourth,y,fgy
      
  character*50 fn_name
  logical fkeyerr
!     
  pi = 4.d0*datan(1.d0)
!         
  fn_name = "data/setup_one_layer.in"
  
  call open_check(fn_name,ntlocal,fkeyerr)
  
  read(ntlocal,*) nlines
  
  if(ilog.eq.1) then 
     write(6,*) 'MSG(read_ccp_one_layer): nlines=',nlines
  endif
  
  if(ilog.eq.1) print*,'MSG(read_ccp_one_layer): begin loop over model: nlines =', nlines
  
  ! init output arrays
  do i=1,nzones
     rri(i) = 0.d0
     ggi(i) = 0.d0    
     ddi(i) = 0.d0
     ppi(i) = 0.d0
     ggic(i) = 0.d0
     ppic(i) = 0.d0
  enddo

  !pi=4.d0*datan(1.d0) ! calculate pi  

  ncomp = qn
  
  ! read model lines  
  do i=1,nlines
     read(ntlocal,*) yi(i), ggi(i), ddi(i), ppi(i), tti(i)
  enddo
  
  ! according to code comparison project specs > moved to constants.inc
  
  !onelu = 4.e8             ! length unit 4 Mm 
  !onepu = 4.607893d23      ! pressure unit dyne
  !onedu = 1.82094d6        ! density unit g/cm3
  !onetu = 0.7951638        ! time unit in seconds
  !onemu = 1.165402d32      ! mass unit in grams
  !onettu = 3.401423e9      ! temperature unit in Kelvins
  !onegu = onelu/(onetu**2) ! gravitational acceleration unit   
  !mu0 = 1.848              ! mean molecular weight of fluid_1
  !mu1 = 1.802              ! mean molecular weight of fluid_2
  !oneluu =  3.708735d49    ! luminosity unit erg/s
  !oneeu  = 2.949052d49     ! energy unit erg
  
  ! convert to physical units
  do i=1,nlines
     rri(i) = onelu*yi(i)
     ggi(i) = onegu*ggi(i)    
     ddi(i) = onedu*ddi(i)
     ppi(i) = onepu*ppi(i)
     tti(i) = onettu*tti(i)
  enddo

  do i=1,nlines
     xyzi(i,1) = 1.d0
     mui(i) = 1.d0/((xyzi(i,1)/mu0))     
  enddo
 

!     MANAGE ZONES: shift index:
!     rrl: left interface of zone i
!     rrr: right interface of zone i
!     rr : center of zone i
      do i=1,nlines-1
         rrl(i) = rri(i)
         rrr(i) = rri(i+1)     
         rr(i)  = (rrr(i)+rrl(i))/2.d0     
         ppi(i) = ppi(i+1)
         ddi(i) = ddi(i+1)
         tti(i) = tti(i+1)
         mui(i) = mui(i+1)
         do ii=1,qn
            xyzi(i,ii) = xyzi(i+1,ii)
         enddo
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

  
  !do i=1,nlines
  !   print*,i,rri(i),mui(i),xyzi(i,1),xyzi(i,2),ggi(i)
  !enddo

  !stop
  
  print*,'MSG(read_ccp_one_layer): pp, rho:  =', ppi(1),ppi(nlines),ddi(1),ddi(nlines)
  
  if(ilog.eq.1) print*,'END(read_ccp_one_layer):'
  
  return
  
end subroutine read_ccp_one_layer

