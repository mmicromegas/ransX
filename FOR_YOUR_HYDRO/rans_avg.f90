!
!
!
!
subroutine rans_avg(imode)
  
  ! imode = 0 : initialize
  ! imode = 1 : update
  
  use mod_rans_data
  !use mod_tdiff_data
  implicit none
  include 'dimen.inc'
  include 'constants.inc'
  include 'intgrs.inc'
  include 'float.inc'
  include 'gravity.inc'
  include 'mesh.inc'
  include 'vnew.inc'
  include 'comp.inc'
  include 'mpif.h'
  include 'mpicom.inc'
  
  integer*4 imode
  integer*4 i, j, k, n, ii, jj, kk
  integer*4 ifield
  !
  real*8 fsteradjk, coty, siny
  real*8 abar(qx), zbar(qx), gam1(qx), gam2(qx), gam3(qx)
  real*8 gac(qx), gae(qx)
  real*8 kdf(qx),frx(qx),fry(qx),frz(qx)
  real*8 ek(qx), ei(qx), et(qx), hh(qx), sv(qx)
  real*8 dd(qx), pp(qx), ss(qx), tt(qx), xn(qx,rans_nnuc)
  real*8 ux(qx), uy(qx), uz(qx), en1(qx), en2(qx)
  real*8 chd(qx),cht(qx),chm(qx),cs(qx),ps(qx)
  real*8 dppy(qx), dppz(qx), gg(qx), ggr(qx)
  real*8 dddy(qx), dddz(qx),dtty(qx),dttz(qx)
  real*8 mfx(qx), cpd(qx), cvd(qx)
  real*8 ggravity(qx)  
  ! FOR SPATIAL DERIVATIVES
  real*8 dx, dy, dz
  real*8 ux_yp(qx), ux_ym(qx), ux_zp(qx), ux_zm(qx)
  real*8 uy_yp(qx), uy_ym(qx), uy_zp(qx), uy_zm(qx)
  real*8 uz_yp(qx), uz_ym(qx), uz_zp(qx), uz_zm(qx)
  real*8 pp_yp(qx), pp_ym(qx), pp_zp(qx), pp_zm(qx)
  real*8 dd_yp(qx), dd_ym(qx), dd_zp(qx), dd_zm(qx)
  real*8 tt_yp(qx), tt_ym(qx), tt_zp(qx), tt_zm(qx)
  real*8 dxx(qx), dxy(qx), dxz(qx)
  real*8 dyx(qx), dyy(qx), dyz(qx)
  real*8 dzx(qx), dzy(qx), dzz(qx)
  real*8 uxx(qx), uxy(qx), uxz(qx)
  real*8 uyx(qx), uyy(qx), uyz(qx)
  real*8 uzx(qx), uzy(qx), uzz(qx)
  real*8 sxy(qx), sxz(qx), syz(qx)
  real*8 divu(qx), divux(qx), divuy(qx), divuz(qx)
  real*8 curlx(qx), curly(qx), curlz(qx)
  real*8 enst(qx)
  real*8 gradxei(qx), gradxpp(qx),gradxdd(qx),gradxtt(qx)
  ! SHARED NEIGHBOR DATA
  real*8 ux_n1p(qy,qz), ux_n1m(qy,qz)
  real*8 ux_n2p(qx,qz), ux_n2m(qx,qz)
  real*8 ux_n3p(qx,qy), ux_n3m(qx,qy)
  real*8 uy_n1p(qy,qz), uy_n1m(qy,qz)
  real*8 uy_n2p(qx,qz), uy_n2m(qx,qz)
  real*8 uy_n3p(qx,qy), uy_n3m(qx,qy)
  real*8 uz_n1p(qy,qz), uz_n1m(qy,qz)
  real*8 uz_n2p(qx,qz), uz_n2m(qx,qz)
  real*8 uz_n3p(qx,qy), uz_n3m(qx,qy)
  real*8 ei_n1p(qy,qz), ei_n1m(qy,qz)
  real*8 pp_n1p(qy,qz), pp_n1m(qy,qz)
  real*8 pp_n2p(qy,qz), pp_n2m(qy,qz)
  real*8 pp_n3p(qy,qz), pp_n3m(qy,qz)
  real*8 dd_n1p(qy,qz), dd_n1m(qy,qz)
  real*8 dd_n2p(qy,qz), dd_n2m(qy,qz)
  real*8 dd_n3p(qy,qz), dd_n3m(qy,qz) 
  real*8 tt_n1p(qy,qz), tt_n1m(qy,qz)
  real*8 tt_n2p(qy,qz), tt_n2m(qy,qz)
  real*8 tt_n3p(qy,qz), tt_n3m(qy,qz) 
  ! TEMP STORAGE
  real*8 ekin(qx,qy,qz), eint(qx,qy,qz)
  ! composition variable names
  ! character(len=24) :: xvarname(6)
  character(len=4)  :: xidchar
    
  ! Composition variable names
  INTEGER, PARAMETER :: ix = 25
  character(len=24) xvarname(ix)

  ! TEMP STORAGE
  real*8 enth(qx,qy,qz)
  real*8 xx(rans_nnuc)
  
  ! Xdot terms  
  real*8 xdn(qx,rans_nnuc)  

  !
  real*8 ddabar,ddabarsq,ddabarzbar,sum_xdn_o_an
  real*8 sum_znxdn_o_an,uxddabarsq,uxddabar,uxddabarzbar
  
  integer*4 nsubcycle
  real*8    eps,snu,deltat,dtnew
  real*8 five_fourth,fgy,y
  
  five_fourth = 5.d0/4.d0
  
  ! only implemented with inhomogeneous (gravity) in x-direction
  if(igrav.ne.0.and.xyzgrav.ne.1) then
     call MPI_FINALIZE(ierr)
     stop 'ERR(rans_avg): igrav.ne.0.and.xyzgrav.ne.1 not implemented'
  endif
  
  ! initialize fields for new segment of time averaging (between data dumps)
  
  if(imode.eq.0) then
     rans_tstart = time
     rans_tavg   = 0.d0
     do n=1,nrans
        do i=1,qx
           havg(1,n,i) = 0.d0
           havg(2,n,i) = 0.d0
           havg(3,n,i) = 0.d0
        enddo
     enddo
  else if(imode.eq.1) then
     do n=1,nrans
        do i=1,qx
           havg(1,n,i) = havg(2,n,i)
           havg(2,n,i) = 0.d0
        enddo
     enddo
  endif
    
  ! share neighbor data for velocity
  
  call shareplane_ifaces(velx,ux_n1p,ux_n1m)
  call shareplane_jfaces(velx,ux_n2p,ux_n2m)
  call shareplane_kfaces(velx,ux_n3p,ux_n3m)
  
  call shareplane_ifaces(vely,uy_n1p,uy_n1m)
  call shareplane_jfaces(vely,uy_n2p,uy_n2m)
  call shareplane_kfaces(vely,uy_n3p,uy_n3m)
  
  call shareplane_ifaces(velz,uz_n1p,uz_n1m)
  call shareplane_jfaces(velz,uz_n2p,uz_n2m)
  call shareplane_kfaces(velz,uz_n3p,uz_n3m)
  
  do k=1,qz
     do j=1,qy
        do i=1,qx
           ekin(i,j,k) = (velx(i,j,k)*velx(i,j,k) + & 
                vely(i,j,k)*vely(i,j,k) + velz(i,j,k)*velz(i,j,k))*0.5d0
           eint(i,j,k) = energy(i,j,k) - ekin(i,j,k)
           enth(i,j,k) = press(i,j,k) + press(i,j,k)/densty(i,j,k)
        enddo
     enddo
  enddo
  
  call shareplane_ifaces(eint,ei_n1p,ei_n1m)
  
  call shareplane_ifaces(press,pp_n1p,pp_n1m)  
  call shareplane_jfaces(press,pp_n2p,pp_n2m)
  call shareplane_kfaces(press,pp_n3p,pp_n3m)
  
  call shareplane_ifaces(densty,dd_n1p,dd_n1m)
  call shareplane_jfaces(densty,dd_n2p,dd_n2m)
  call shareplane_kfaces(densty,dd_n3p,dd_n3m)

  call shareplane_ifaces(temp,tt_n1p,tt_n1m)
  call shareplane_jfaces(temp,tt_n2p,tt_n2m)
  call shareplane_kfaces(temp,tt_n3p,tt_n3m)


  ! loop over all zones for averaging
  do k=1,qz
     do j=1,qy
        ! mesh descriptors
        fsteradjk = fsterad(j,k)
        if(igeomx.eq.2) then
           siny      = sin(yzn(j))
           coty      = 1.d0/tan(yzn(j))
        else
           siny      = 1.d0
           coty      = 1.d0 
        endif
        
        do i=1,qx
           dd(i)   = densty(i,j,k)      ! density (g cm-3) 
           sv(i)   = 1.d0/dd(i)         ! specific volume (cm-3 g)
           ux(i)   = velx(i,j,k)        ! x velocity (cm s-1) 
           uy(i)   = vely(i,j,k)        ! y velocity (cm s-1)
           uz(i)   = velz(i,j,k)        ! z velocity (cm s-1)
           ek(i)   = ekin(i,j,k)        ! specific kinetic energy (erg g-1)
           et(i)   = energy(i,j,k)      ! specific total energy (erg g-1)
           ei(i)   = et(i)-ek(i)        ! specific internal energy (erg g-1)
           pp(i)   = press(i,j,k)       ! pressure (erg cm-3)
           hh(i)   = ei(i)+pp(i)/dd(i)  ! specific enthalpy (erg g-1)
           tt(i)   = temp(i,j,k)        ! temperature (K)
           ss(i)   = entropy(i,j,k)     ! specific entropy (erg g-1 K-1) 
           cpd(i)  = cpdat(i,j,k)       ! heat capacity cp -(dQ/dT)_p (Cox & Guili p.221)
           cvd(i)  = cvdat(i,j,k)       ! heat capacity cv -(dQ/dT)_v (Cox & Guili p.221)
           en1(i)  = enuc(i,j,k,1)      ! nuclear energy production (erg g-1 s-1)
           en2(i)  = enuc(i,j,k,2)      ! neutrino losses (erg g-1 s-1)
           abar(i) = azbar(i,j,k,1)     ! mean number of nucleons per isotope
           zbar(i) = azbar(i,j,k,2)     ! mean charge per isotope
           gam1(i) = gamma1(i,j,k)      ! gamma1 -(d ln P/d ln rho)_ad (Cox & Guili p.224) 
           gam2(i) = gamma2(i,j,k)      ! gamma2 (Cox & Guilli p.224) 
           gam3(i) = gamma3(i,j,k)      ! gamma3 (Cox & Guilli p.224)
           gac(i)  = gammac(i,j,k)
           gae(i)  = gammae(i,j,k)
           chd(i)  = chid(i,j,k)        ! chi_rho (d ln P/d ln rho)_T,mu (Cox & Guili p.224,369)
           cht(i)  = chit(i,j,k)        ! chi_T (d ln P/d ln T)_rho,mu (Cox & Guili p.224,369)
           chm(i)  = chim(i,j,k)        ! chi_mu (d ln P/d ln mu)_rho,T (Cox & Guili p.224,369)
           cs(i)   = snd(i,j,k)         ! speed of sound (cm s-1) 
           ps(i)   = psi(i,j,k)         ! degeneracy parameter (Cox & Guili)
           !kdf(i)  = kdiff(i,j,k)       ! thermal conductivity
           !frx(i)  = fradx(i,j,k) ! x flux due to thermal transport 
           !fry(i)  = frady(i,j,k) ! y flux due to thermal transport 
           !frz(i)  = fradz(i,j,k) ! z flux due to thermal transport 
        enddo
        
        do i=1,qx
           do n=1,rans_nnuc
              xn(i,n) = xnuc(i,j,k,n)
              xdn(i,n) = xnucdot(i,j,k,n)
           enddo
        enddo

        ! velocity coordinate gradient tensor: dij
        
        ! angular derivatives of velocity
        if(k.eq.1) then
           do i=1,qx
              ux_zp(i) = velx   (i,j,k+1)
              ux_zm(i) = ux_n3m (i,j    )
              uy_zp(i) = vely   (i,j,k+1)
              uy_zm(i) = uy_n3m (i,j    )
              uz_zp(i) = velz   (i,j,k+1)
              uz_zm(i) = uz_n3m (i,j    )
           enddo
        else if(k.eq.qz) then
           do i=1,qx
              ux_zp(i) = ux_n3p (i,j    )
              ux_zm(i) = velx   (i,j,k-1)
              uy_zp(i) = uy_n3p (i,j)
              uy_zm(i) = vely   (i,j,k-1)
              uz_zp(i) = uz_n3p (i,j    )
              uz_zm(i) = velz   (i,j,k-1)
           enddo
        else
           do i=1,qx
              ux_zp(i) = velx   (i,j,k+1)
              ux_zm(i) = velx   (i,j,k-1)
              uy_zp(i) = vely   (i,j,k+1)
              uy_zm(i) = vely   (i,j,k-1)
              uz_zp(i) = velz   (i,j,k+1)
              uz_zm(i) = velz   (i,j,k-1)
           enddo
        endif        
        if(j.eq.1) then
           do i=1,qx
              ux_yp(i) = velx   (i,j+1,k)
              ux_ym(i) = ux_n2m (i,k    )
              uy_yp(i) = vely   (i,j+1,k)
              uy_ym(i) = uy_n2m (i,k    )
              uz_yp(i) = velz   (i,j+1,k)
              uz_ym(i) = uz_n2m (i,k    )
           enddo
        else if(j.eq.qy) then
           do i=1,qx
              ux_yp(i) = ux_n2p (i,k    )
              ux_ym(i) = velx   (i,j-1,k)
              uy_yp(i) = uy_n2p (i,k    )
              uy_ym(i) = vely   (i,j-1,k)
              uz_yp(i) = uz_n2p (i,k    )
              uz_ym(i) = velz   (i,j-1,k)
           enddo
        else
           do i=1,qx
              ux_yp(i) = velx   (i,j+1,k)
              ux_ym(i) = velx   (i,j-1,k)
              uy_yp(i) = vely   (i,j+1,k)
              uy_ym(i) = vely   (i,j-1,k)
              uz_yp(i) = velz   (i,j+1,k)
              uz_ym(i) = velz   (i,j-1,k)
           enddo
        endif

        jj     = coords(2)*qy + j
        kk     = coords(3)*qz + k
        dy = (gyznr(jj) - gyznl(jj))
        dz = (gzznr(jj) - gzznl(jj))
        if (nsdim.eq.2) dz = 1.d60 ! hack for 2D
        do i=1,qx
           dxy(i) = 0.5d0*(ux_yp(i) - ux_ym(i))/dy
           dyy(i) = 0.5d0*(uy_yp(i) - uy_ym(i))/dy
           dzy(i) = 0.5d0*(uz_yp(i) - uz_ym(i))/dy
           dxz(i) = 0.5d0*(ux_zp(i) - ux_zm(i))/dz
           dyz(i) = 0.5d0*(uy_zp(i) - uy_zm(i))/dz
           dzz(i) = 0.5d0*(uz_zp(i) - uz_zm(i))/dz
        enddo

        do i=2,qx-1
           ii         = coords(1)*qx + i
           dx         = (gxznr(ii) - gxznl(ii))
           dxx(i)     = 0.5d0*(ux(i+1) - ux(i-1))/dx
           dyx(i)     = 0.5d0*(uy(i+1) - uy(i-1))/dx
           dzx(i)     = 0.5d0*(uz(i+1) - uz(i-1))/dx
           gradxei(i) = 0.5d0*(ei(i+1) - ei(i-1))/dx
           gradxpp(i) = 0.5d0*(pp(i+1) - pp(i-1))/dx
           gradxdd(i) = 0.5d0*(dd(i+1) - dd(i-1))/dx
           gradxtt(i) = 0.5d0*(tt(i+1) - tt(i-1))/dx
        enddo
        i      = 1
        ii     = coords(1)*qx + i
        dx = (gxznr(ii) - gxznl(ii))
        if(bndmnx.eq.6) then
           dxx(i)     = 0.5d0*(ux(i+1) - ux_n1m(j,k))/dx
           dyx(i)     = 0.5d0*(uy(i+1) - uy_n1m(j,k))/dx
           dzx(i)     = 0.5d0*(uz(i+1) - uz_n1m(j,k))/dx
           gradxei(i) = 0.5d0*(ei(i+1) - ei_n1m(j,k))/dx
           gradxpp(i) = 0.5d0*(pp(i+1) - pp_n1m(j,k))/dx           
           gradxdd(i) = 0.5d0*(dd(i+1) - dd_n1m(j,k))/dx
           gradxtt(i) = 0.5d0*(tt(i+1) - tt_n1m(j,k))/dx
        else
           dxx(i)     = 0.d0
           dyx(i)     = 0.d0
           dzx(i)     = 0.d0
           gradxei(i) = 0.d0
           gradxpp(i) = 0.d0
           gradxdd(i) = 0.d0
           gradxtt(i) = 0.d0
        endif
        i          = qx
        ii         = coords(1)*qx + i
        dx     = (gxznr(ii) - gxznl(ii))
        if(bndmxx.eq.6) then
           dxx(i)     = 0.5d0*(ux_n1p(j,k) - ux(i-1))/dx
           dyx(i)     = 0.5d0*(uy_n1p(j,k) - uy(i-1))/dx
           dzx(i)     = 0.5d0*(uz_n1p(j,k) - uz(i-1))/dx
           gradxei(i) = 0.5d0*(ei_n1p(j,k) - ei(i-1))/dx
           gradxpp(i) = 0.5d0*(pp_n1p(j,k) - pp(i-1))/dx           
           gradxdd(i) = 0.5d0*(dd_n1p(j,k) - dd(i-1))/dx
           gradxtt(i) = 0.5d0*(tt_n1p(j,k) - tt(i-1))/dx
        else
           dxx(i)     = 0.d0
           dyx(i)     = 0.d0
           dzx(i)     = 0.d0
           gradxei(i) = 0.d0
           gradxpp(i) = 0.d0
           gradxdd(i) = 0.d0
           gradxtt(i) = 0.d0
        endif

        if(j.eq.1) then
           do i=1,qx
              pp_yp(i) = press   (i,j+1,k)
              pp_ym(i) = pp_n2m (i,k    )
           enddo
        else if(j.eq.qy) then
           do i=1,qx
              pp_yp(i) = pp_n2p (i,k    )
              pp_ym(i) = press   (i,j-1,k)
           enddo
        else
           do i=1,qx
              pp_yp(i) = press   (i,j+1,k)
              pp_ym(i) = press   (i,j-1,k)
           enddo
        endif
        
        do i=1,qx
           dppy(i) = 0.5d0*(pp_yp(i) - pp_ym(i))/dy
        enddo
        
        if(k.eq.1) then
           do i=1,qx
              pp_zp(i) = press  (i,j,k+1)
              pp_zm(i) = pp_n3m (i,j    )
           enddo
        else if(k.eq.qz) then
           do i=1,qx
              pp_zp(i) = pp_n3p (i,j    )
              pp_zm(i) = press  (i,j,k-1)
           enddo
        else
           do i=1,qx
              pp_zp(i) = press (i,j,k+1)
              pp_zm(i) = press (i,j,k-1)
           enddo
        endif                

        do i=1,qx
           dppz(i) = 0.5d0*(pp_zp(i) - pp_zm(i))/dz
        enddo

!       density gradient

        if(j.eq.1) then
           do i=1,qx
              dd_yp(i) = densty   (i,j+1,k)
              dd_ym(i) = dd_n2m (i,k    )
           enddo
        else if(j.eq.qy) then
           do i=1,qx
              dd_yp(i) = dd_n2p (i,k    )
              dd_ym(i) = densty   (i,j-1,k)
           enddo
        else
           do i=1,qx
              dd_yp(i) = densty   (i,j+1,k)
              dd_ym(i) = densty  (i,j-1,k)
           enddo
        endif

        do i=1,qx
           dddy(i) = 0.5d0*(dd_yp(i) - dd_ym(i))/dy
        enddo

        if(k.eq.1) then
           do i=1,qx
              dd_zp(i) = densty  (i,j,k+1)
              dd_zm(i) = dd_n3m (i,j    )
           enddo
        else if(k.eq.qz) then
           do i=1,qx
              dd_zp(i) = dd_n3p (i,j    )
              dd_zm(i) = densty  (i,j,k-1)
           enddo
        else
           do i=1,qx
              dd_zp(i) = densty (i,j,k+1)
              dd_zm(i) = densty (i,j,k-1)
           enddo
        endif

        do i=1,qx
           dddz(i) = 0.5d0*(dd_zp(i) - dd_zm(i))/dz
        enddo


!       temperature gradient


        if(j.eq.1) then
           do i=1,qx
              tt_yp(i) = temp   (i,j+1,k)
              tt_ym(i) = tt_n2m (i,k    )
           enddo
        else if(j.eq.qy) then
           do i=1,qx
              tt_yp(i) = tt_n2p (i,k    )
              tt_ym(i) = temp   (i,j-1,k)
           enddo
        else
           do i=1,qx
              tt_yp(i) = temp   (i,j+1,k)
              tt_ym(i) = temp  (i,j-1,k)
           enddo
        endif

        do i=1,qx
           dtty(i) = 0.5d0*(tt_yp(i) - tt_ym(i))/dy
        enddo

        if(k.eq.1) then
           do i=1,qx
              tt_zp(i) = temp  (i,j,k+1)
              tt_zm(i) = tt_n3m (i,j    )
           enddo
        else if(k.eq.qz) then
           do i=1,qx
              tt_zp(i) = tt_n3p (i,j    )
              tt_zm(i) = temp  (i,j,k-1)
           enddo
        else
           do i=1,qx
              tt_zp(i) = temp (i,j,k+1)
              tt_zm(i) = temp (i,j,k-1)
           enddo
        endif

        do i=1,qx
           dttz(i) = 0.5d0*(tt_zp(i) - tt_zm(i))/dz
        enddo
        
        ! --- velocity gradient uij, stress, div, curl, enstrophy
        !
        ! NOTE: These derivatives are geometry dependent  
        !
        if(igeomx.eq.2) then
           do i=1,qx
              uxx(i) = dxx(i)
              uxy(i) = (dxy(i) - uy(i))/xzn(i)
              uxz(i) = (dxz(i)/sin(yzn(j)) - uz(i))/xzn(i)
              
              uyx(i) = dyx(i)
              uyy(i) = (dyy(i) + ux(i))/xzn(i)
              uyz(i) = (dyz(i)/sin(yzn(j)) - uz(i)/tan(yzn(j)))/xzn(i)
              
              uzx(i) = dzx(i)
              uzy(i) = dzy(i)/xzn(i)
              uzz(i) = (dzz(i) + ux(i)*sin(yzn(j)) + uy(i)*cos(yzn(j)))/(xzn(i)*sin(yzn(j)))
              
              sxy(i) = (uxy(i) + uyx(i))*0.5d0
              sxz(i) = (uxz(i) + uzx(i))*0.5d0
              syz(i) = (uyz(i) + uzy(i))*0.5d0
              
              divu (i) = uxx(i) + uyy(i) + uzz(i)
              divux(i) = uxx(i) + 2.d0*ux(i)/xzn(i)
              divuy(i) = uxy(i)/xzn(i) + (uy(i)*cos(yzn(i)))/(xzn(i)*sin(yzn(i)))
              divuz(i) = uxz(i)/(xzn(i)*sin(yzn(i)))
      
              curlx(i) = uzy(i) - uyz(i)
              curly(i) = uxz(i) - uzx(i)
              curlz(i) = uyx(i) - uxy(i)
              enst (i) = curlx(i)*curlx(i) + curly(i)*curly(i) + curlz(i)*curlz(i)
           enddo
        else if(igeomx.eq.0) then
           do i=1,qx
              uxx(i) = dxx(i)
              uxy(i) = dxy(i)
              uxz(i) = dxz(i)
              
              uyx(i) = dyx(i)
              uyy(i) = dyy(i)
              uyz(i) = dyz(i)
              
              uzx(i) = dzx(i)
              uzy(i) = dzy(i)
              uzz(i) = dzz(i)
              
              sxy(i) = (uxy(i) + uyx(i))*0.5d0
              sxz(i) = (uxz(i) + uzx(i))*0.5d0
              syz(i) = (uyz(i) + uzy(i))*0.5d0
 
              divu (i) = uxx(i) + uyy(i) + uzz(i)
              divux(i) = uxx(i)
              divuy(i) = uyy(i)
              divuz(i) = uzz(i)
              
              curlx(i) = uzy(i) - uyz(i)
              curly(i) = uxz(i) - uzx(i)
              curlz(i) = uyx(i) - uxy(i)
              enst (i) = curlx(i)*curlx(i) + curly(i)*curly(i) + curlz(i)*curlz(i)
           enddo
        else
           call MPI_FINALIZE(ierr)
           stop 'ERR(rans_avg): igeomx.ne.{1||0} not implemented'
        endif

        ! update current horizontally averaged fields in havg(2,:,:)
        
        ! dd  (1)
        ifield = 1
        if(imode.eq.0) ransname(ifield) = 'dd'
        do i=1,qx 
           havg(2,ifield,i) = havg(2,ifield,i)  + & 
                dd(i)*fsteradjk
        enddo
        ! ddsq (2)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddsq'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + & 
                dd(i)*dd(i)*fsteradjk
        enddo
        ! tt (3)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'tt'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + & 
                tt(i)*fsteradjk
        enddo
        ! ttsq (4)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttsq'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                tt(i)*tt(i)*fsteradjk
        enddo
        ! pp (5)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'pp'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*fsteradjk
        enddo
        ! ppsq (6)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppsq'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*pp(i)*fsteradjk
        enddo
        ! ux (7)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ux(i)*fsteradjk
        enddo
        ! uy (8)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uy(i)*fsteradjk
        enddo
        ! uz (9)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uz(i)*fsteradjk
        enddo
        ! uxx (10)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uxx(i)*fsteradjk
        enddo
        ! uxy (11)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uxy(i)*fsteradjk
        enddo
        ! uxz (12)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uxz(i)*fsteradjk
        enddo
        ! uyx (13)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uyx(i)*fsteradjk
        enddo
        ! uyy (14)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uyy(i)*fsteradjk
        enddo
        ! uyz (15)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uyz(i)*fsteradjk
        enddo
        ! uzx (16)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uzx(i)*fsteradjk
        enddo
        ! uzy (17)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uzy(i)*fsteradjk
        enddo
        ! uzz (18)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uzz(i)*fsteradjk
        enddo
        ! uxuxx (19)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxuxx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ux(i)*uxx(i)*fsteradjk
        enddo
        ! uyuxy (20)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyuxy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uy(i)*uxy(i)*fsteradjk
        enddo
        ! uzuxz (21)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzuxz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uz(i)*uxz(i)*fsteradjk
        enddo
        ! enst (22)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'enst'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                enst(i)*fsteradjk
        enddo
        ! uxuy (23)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxuy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ux(i)*uy(i)*fsteradjk
        enddo
        ! uxuz (24)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxuz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ux(i)*uz(i)*fsteradjk
        enddo
        ! uyuz (25)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyuz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uy(i)*uz(i)*fsteradjk
        enddo
        ! uxux (26)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ux(i)*ux(i)*fsteradjk
        enddo
        ! uyuy (27)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyuy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uy(i)*uy(i)*fsteradjk
        enddo
        ! uzuz (28)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzuz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uz(i)*uz(i)*fsteradjk
        enddo
        ! ddux (29)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*ux(i)*fsteradjk
        enddo
        ! dduy (30)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*uy(i)*fsteradjk
        enddo
        ! dduz (31)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*fsteradjk
        enddo
        ! dduxux (32)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! dduyuy (33)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! dduzuz (34)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! dduxuy (35)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*uy(i)*fsteradjk
        enddo
        ! dduxuz (36)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*uz(i)*fsteradjk
        enddo
        ! dduyuz (37)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduyuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*uz(i)*fsteradjk
        enddo
        ! ddddux (38)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddddux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*dd(i)*ux(i)*fsteradjk
        enddo
        ! dduxuxux (39)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! dduxuyuy (40)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! dduxuzuz (41)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! dduxuyuz (42)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuyuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*uy(i)*uz(i)*fsteradjk
        enddo
        ! dduyuzuz (43)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduyuzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! dduzuzuz (44)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzuzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! dduzuyuy (45)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzuyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! dduzuxux (46)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzuxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! dduxcoty (47)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxcoty'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*ux(i)*coty*fsteradjk
        enddo
        ! dduycoty (48)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduycoty'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*uy(i)*coty*fsteradjk
        enddo
        ! dduzcoty (49)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzcoty'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*coty*fsteradjk
        enddo
        ! dduxuxcoty (50)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuxcoty'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*ux(i)*coty*fsteradjk
        enddo
        ! dduyuycoty (51)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduyuycoty'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*uy(i)*coty*fsteradjk
        enddo
        ! dduxuycoty (52)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuycoty'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*uy(i)*coty*fsteradjk
        enddo
        ! dduxuzcoty (53)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuzcoty'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*uz(i)*coty*fsteradjk
        enddo
        ! dduzuzcoty (54)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduzuzcoty'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*uz(i)*coty*fsteradjk
        enddo
        ! dduzuycoty (55)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduzuycoty'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*uy(i)*coty*fsteradjk
        enddo
        ! dduzuzuycoty (56)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzuzuycoty'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*uz(i)*uy(i)*coty*fsteradjk
        enddo
        ! et (57)
        ifield = ifield +1
        if(imode.eq.0) ransname(ifield) = 'et'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                et(i)*fsteradjk
        enddo
        ! ddet (58)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddet'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*et(i)*fsteradjk
        enddo
        ! etux (59)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'etux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                et(i)*ux(i)*fsteradjk
        enddo
        ! ddetux (60)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddetux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*et(i)*ux(i)*fsteradjk
        enddo        
        ! ek (61)
        ifield = ifield +1
        if(imode.eq.0) ransname(ifield) = 'ek'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ek(i)*fsteradjk
        enddo
        ! ddek (62)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddek'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ek(i)*fsteradjk
        enddo
        ! ekux (63)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ekux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ek(i)*ux(i)*fsteradjk
        enddo
        ! ddekux (64)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddekux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ek(i)*ux(i)*fsteradjk
        enddo
        ! enuc1   (65)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'enuc1'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                en1(i)*fsteradjk
        enddo
        ! enuc2 (66)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'enuc2'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                en2(i)*fsteradjk
        enddo
        ! ddenuc1 (67)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddenuc1'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*en1(i)*fsteradjk
        enddo
        ! ddenuc2 (68)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddenuc2'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*en2(i)*fsteradjk
        enddo
        ! hh (69)
        ifield = ifield +1
        if(imode.eq.0) ransname(ifield) = 'hh'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                hh(i)*fsteradjk
        enddo
        ! ddhh (70)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddhh'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*hh(i)*fsteradjk
        enddo
        ! ddhhhh (71)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddhhhh'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*hh(i)*hh(i)*fsteradjk
        enddo
        ! ddhhhhux (72)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddhhhhux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*hh(i)*hh(i)*ux(i)*fsteradjk
        enddo
        ! hhux (73)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'hhux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                hh(i)*ux(i)*fsteradjk
        enddo
        ! ddhhux (74)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddhhux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*hh(i)*ux(i)*fsteradjk
        enddo
        ! ddhhuy (75)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddhhuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*hh(i)*uy(i)*fsteradjk
        enddo
        ! ddhhuz (76)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddhhuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*hh(i)*uz(i)*fsteradjk
        enddo
        ! ddhhuxux (77)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddhhuxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*hh(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! ddhhuyuy (78)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddhhuyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*hh(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! ddhhuzuz (79)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddhhuzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*hh(i)*uz(i)*uz(i)*fsteradjk
        enddo 
        ! hhgradxpp (80)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'hhgradxpp'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                hh(i)*gradxpp(i)*fsteradjk
        enddo
        ! hhppdivu (81)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'hhppdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                hh(i)*pp(i)*divu(i)*fsteradjk
        enddo 
        ! ddddenuc1 (82)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'hhddenuc1'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                hh(i)*dd(i)*en1(i)*fsteradjk
        enddo
        ! hhddenuc2 (83)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'hhddenuc2'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                hh(i)*dd(i)*en2(i)*fsteradjk
        enddo 
        ! ei (84)
        ifield = ifield +1
        if(imode.eq.0) ransname(ifield) = 'ei'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*fsteradjk
        enddo
        ! ddei (85)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddei'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*fsteradjk
        enddo
        ! ddeiei (86)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiei'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*ei(i)*fsteradjk
        enddo
        ! ddeiux (87)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*ux(i)*fsteradjk
        enddo       
        ! ddeieiux (88)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeieiux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*ei(i)*ux(i)*fsteradjk
        enddo        
        ! ddeiuxux (89)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*ux(i)*ux(i)*fsteradjk
        enddo         
        ! ddeiuyuy (90)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! ddeiuzuz (91)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*uz(i)*uz(i)*fsteradjk
        enddo  
        ! eigradxpp (92)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eigradxpp'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*gradxpp(i)*fsteradjk
        enddo
        ! eiddenuc1 (93)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eiddenuc1'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*dd(i)*en1(i)*fsteradjk
        enddo
        ! eiddenuc2 (94)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eiddenuc2'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*dd(i)*en2(i)*fsteradjk
        enddo 
        ! eippdivu (95)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eippdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*pp(i)*divu(i)*fsteradjk
        enddo 
        ! eidivu (96)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eidivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*divu(i)*fsteradjk
        enddo 
        ! eipp (97)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eipp'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*pp(i)*fsteradjk
        enddo         
        ! eiux (98)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eiux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*ux(i)*fsteradjk
        enddo
        ! eiuy (99)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eiuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*uy(i)*fsteradjk
        enddo
        ! eiuz (100)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eiuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*uz(i)*fsteradjk
        enddo
        ! ddeiux (101)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*ux(i)*fsteradjk
        enddo
        ! ddeiuy (102)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*uy(i)*fsteradjk
        enddo
        ! ddeiuz (103)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*uz(i)*fsteradjk
        enddo
        ! gradxei (104)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'gradxei'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                gradxei(i)*fsteradjk
        enddo
        ! ppgradxei (105)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ppgradxei'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                pp(i)*gradxei(i)*fsteradjk
        enddo
        ! ddeiuxux (106)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! ddeiuxuy (107)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuxuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*ux(i)*uy(i)*fsteradjk
        enddo
        ! ddeiuxuz (108)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuxuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*ux(i)*uz(i)*fsteradjk
        enddo
        ! ddeiuyuy (109)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! ddeiuyuz (110)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuyuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*uy(i)*uz(i)*fsteradjk
        enddo
        ! ddeiuzuz (111)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddeiuzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ei(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! dduxenuc1 (112)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduxenuc1'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*en1(i)*fsteradjk
        enddo 
        ! dduxenuc2 (113)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduxenuc2'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*en2(i)*fsteradjk
        enddo
        ! dduyenuc1 (114)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduyenuc1'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*en1(i)*fsteradjk
        enddo 
        ! dduyenuc2 (115)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduyenuc2'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*en2(i)*fsteradjk
        enddo
        ! dduzenuc1 (116)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduzenuc1'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*en1(i)*fsteradjk
        enddo 
        ! dduzenuc2 (117)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduzenuc2'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*en2(i)*fsteradjk
        enddo  
        ! ss (118)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ss'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ss(i)*fsteradjk
        enddo
        ! sssq (119)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'sssq'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ss(i)*ss(i)*fsteradjk
        enddo
        ! ddss (120)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddss'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ss(i)*fsteradjk
        enddo
        ! ddsssq (121)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddsssq'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ss(i)*ss(i)*fsteradjk
        enddo
        ! ssux (122)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ssux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ss(i)*ux(i)*fsteradjk
        enddo
        ! ddssux (123)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddssux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ss(i)*ux(i)*fsteradjk
        enddo
        ! ddssuy (124)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddssuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ss(i)*uy(i)*fsteradjk
        enddo
        ! ddssuz (125)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddssuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ss(i)*uz(i)*fsteradjk
        enddo
        ! ddssssux (126)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddssssux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ss(i)*ss(i)*ux(i)*fsteradjk
        enddo
        ! ddssuxux (127)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddssuxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ss(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! ddssuyuy (128)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddssuyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ss(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! ddssuzuz (129)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddssuzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ss(i)*uz(i)*uz(i)*fsteradjk
        enddo       
        ! ddenuc1_o_tt (130)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddenuc1_o_tt'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (dd(i)*en1(i)/tt(i))*fsteradjk
        enddo
        ! ddenuc2_o_tt (131)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddenuc2_o_tt'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (dd(i)*en2(i)/tt(i))*fsteradjk
        enddo
        ! ddssenuc1_o_tt (132)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddssenuc1_o_tt'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (dd(i)*ss(i)*en1(i)/tt(i))*fsteradjk
        enddo
        ! ddssenuc2_o_tt (133)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddssenuc2_o_tt'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (dd(i)*ss(i)*en2(i)/tt(i))*fsteradjk
        enddo
        ! dduxenuc1_o_tt (134)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduxenuc1_o_tt'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (dd(i)*ux(i)*en1(i)/tt(i))*fsteradjk
        enddo
        ! dduxenuc2_o_tt (135)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduxenuc2_o_tt'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (dd(i)*ux(i)*en2(i)/tt(i))*fsteradjk
        enddo
        ! ssgradxpp (136)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ssgradxpp'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
               ss(i)*gradxpp(i)*fsteradjk
        enddo        
        ! sv (137)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'sv'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                sv(i)*fsteradjk
        enddo
        ! svux (138)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'svux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                sv(i)*ux(i)*fsteradjk
        enddo
        ! svdivu (139)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'svdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                sv(i)*divu(i)*fsteradjk
        enddo
        ! svgradxpp (140)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'svgradxpp'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
               sv(i)*gradxpp(i)*fsteradjk
        enddo
        ! svdduyuy (141)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'svdduyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
               sv(i)*dd(i)*uy(i)*uy(i)*fsteradjk
        enddo        
        ! svdddduyuy (142)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'svdddduyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
               sv(i)*dd(i)*dd(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! svdduzuz (143)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'svdduzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
               sv(i)*dd(i)*uz(i)*uz(i)*fsteradjk
        enddo        
        ! svdddduzuz (144)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'svdddduzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
               sv(i)*dd(i)*dd(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! divu (145)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'divu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                divu(i)*fsteradjk
        enddo
        ! dddivu (146)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dddivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*divu(i)*fsteradjk
        enddo
        ! dddddivu (147)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dddddivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*dd(i)*divu(i)*fsteradjk
        enddo
        ! divux (148)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'divux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                divux(i)*fsteradjk
        enddo        
        ! divuy (149)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'divuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                divuy(i)*fsteradjk
        enddo
        ! divuz (150)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'divuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                divuz(i)*fsteradjk
        enddo
        ! dddivux (151)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dddivux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*divux(i)*fsteradjk
        enddo        
        ! dddivuy (152)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dddivuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*divuy(i)*fsteradjk
        enddo
        ! dddivuz (153)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dddivuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*divuz(i)*fsteradjk
        enddo
        ! uxdivu (154)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uxdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ux(i)*divu(i)*fsteradjk
        enddo
        ! uydivu (155)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uydivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uy(i)*divu(i)*fsteradjk
        enddo
        ! uzdivu (156)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uzdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uz(i)*divu(i)*fsteradjk
        enddo
        ! uxdivux (157)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uxdivux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ux(i)*divux(i)*fsteradjk
        enddo
        ! uxdivuy (158)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uxdivuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ux(i)*divuy(i)*fsteradjk
        enddo
        ! uxdivuz (159)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uxdivuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ux(i)*divuz(i)*fsteradjk
        enddo
        ! uydivux (160)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uydivux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uy(i)*divux(i)*fsteradjk
        enddo
        ! uydivuy (161)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uydivuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uy(i)*divuy(i)*fsteradjk
        enddo
        ! uydivuz (162)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uydivuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uy(i)*divuz(i)*fsteradjk
        enddo        
        ! uzdivux (163)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uzdivux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uz(i)*divux(i)*fsteradjk
        enddo
        ! uzdivuy (164)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uzdivuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uz(i)*divuy(i)*fsteradjk
        enddo
        ! uzdivuz (165)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uzdivuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uz(i)*divuz(i)*fsteradjk
        enddo
        ! ppdivux (166)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ppdivux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                pp(i)*divux(i)*fsteradjk
        enddo
        ! ppdivuy (167)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ppdivuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                pp(i)*divuy(i)*fsteradjk
        enddo
        ! ppdivuz (168)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ppdivuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                pp(i)*divuz(i)*fsteradjk
        enddo
        ! ppdivu (169)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ppdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                pp(i)*divu(i)*fsteradjk
        enddo
        ! uxppdivu (170)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uxppdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ux(i)*pp(i)*divu(i)*fsteradjk
        enddo
        ! uyppdivu (171)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uyppdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uy(i)*pp(i)*divu(i)*fsteradjk
        enddo
        ! uzppdivu (172)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uzppdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uz(i)*pp(i)*divu(i)*fsteradjk
        enddo        
        ! gamma1 (173)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gamma1'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gam1(i)*fsteradjk
        enddo
        ! gamma2 (174)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gamma2'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gam2(i)*fsteradjk
        enddo
        ! gamma3 (175)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gamma3'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gam3(i)*fsteradjk
        enddo
        ! gamma3ddenuc1 (176)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gamma3ddenuc1'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gam3(i)*dd(i)*en1(i)*fsteradjk
        enddo
        ! gamma3ddenuc2 (177)        
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gamma3ddenuc2'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gam3(i)*dd(i)*en2(i)*fsteradjk
        enddo
        ! gamma1ppdivu (178)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gamma1ppdivu'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gam1(i)*pp(i)*divu(i)*fsteradjk
        enddo
        ! gamma1pp (179)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gamma1pp'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gam1(i)*pp(i)*fsteradjk
        enddo
        ! gamma1divu (180)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gamma1divu'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gam1(i)*divu(i)*fsteradjk
        enddo
        ! ddttsq (181)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddttsq'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*tt(i)*tt(i)*fsteradjk
        enddo
        ! ddtt (182)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddtt'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*tt(i)*fsteradjk
        enddo        
        ! ttux (183)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                tt(i)*ux(i)*fsteradjk
        enddo
        ! ttuy (184)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttuy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                tt(i)*uy(i)*fsteradjk
        enddo
        ! ttuz (185)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttuz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                tt(i)*uz(i)*fsteradjk
        enddo
        ! ddttux (186)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddttux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*tt(i)*ux(i)*fsteradjk
        enddo
        ! ddttuy (187)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddttuy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*tt(i)*uy(i)*fsteradjk
        enddo
        ! ddttuz (188)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddttuz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*tt(i)*uz(i)*fsteradjk
        enddo
        ! ttdivu (189)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttdivu'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                tt(i)*divu(i)*fsteradjk
        enddo
        ! enuc1_o_cv (190)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'enuc1_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (en1(i)/cvd(i))*fsteradjk
        enddo
        ! enuc2_o_cv (191)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'enuc2_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (en2(i)/cvd(i))*fsteradjk
        enddo
        ! ddenuc1_o_cv (192)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddenuc1_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (dd(i)*en1(i)/cvd(i))*fsteradjk
        enddo
        ! ddenuc2_o_cv (193)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddenuc2_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (dd(i)*en2(i)/cvd(i))*fsteradjk
        enddo
        ! ddttuxux (194)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddttuxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*tt(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! ddttuyuy (195)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddttuyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*tt(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! ddttuzuz (196)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddttuzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*tt(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! ttuxux (197)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttuxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
               tt(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! ttuyuy (198)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttuyuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                tt(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! ttuzuz (199)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttuzuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                tt(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! ttttux (200)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttttux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                tt(i)*tt(i)*ux(i)*fsteradjk
        enddo
        ! ttttdivu (201)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttttdivu'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                tt(i)*tt(i)*divu(i)*fsteradjk
        enddo
        ! ttdivu (202)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttdivu'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                tt(i)*divu(i)*fsteradjk
        enddo
        ! ttddenuc1_o_cv (203)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttddenuc1_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (tt(i)*dd(i)*en1(i)/cvd(i))*fsteradjk
        enddo
        ! ttddenuc2_o_cv (204)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttddenuc2_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (tt(i)*dd(i)*en2(i)/cvd(i))*fsteradjk
        enddo
        ! ttenuc1_o_cv (205)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttenuc1_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (tt(i)*en1(i)/cvd(i))*fsteradjk
        enddo
        ! ttenuc2_o_cv (206)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttenuc2_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (tt(i)*en2(i)/cvd(i))*fsteradjk
        enddo
        ! ttux (207)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                tt(i)*ux(i)*fsteradjk
        enddo    
        ! ttgradxpp_o_dd (208)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttgradxpp_o_dd'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (tt(i)*gradxpp(i)/dd(i))*fsteradjk
        enddo    
        ! gradxpp_o_dd (209)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gradxpp_o_dd'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (gradxpp(i)/dd(i))*fsteradjk
        enddo  
        ! ppgradxpp_o_dd (210)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppgradxpp_o_dd'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (pp(i)*gradxpp(i)/dd(i))*fsteradjk
        enddo
        ! gradypp_o_dd (211)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gradypp_o_dd'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (dppy(i)/dd(i))*fsteradjk
        enddo 
        ! ppgradypp_o_dd (212)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppgradypp_o_dd'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (pp(i)*dppy(i)/dd(i))*fsteradjk
        enddo
        ! gradzpp_o_ddsiny (213)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gradzpp_o_ddsiny'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (dppz(i)/(dd(i)*siny))*fsteradjk
        enddo 
        ! ppgradzpp_o_ddsiny (214)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppgradzpp_o_ddsiny'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (pp(i)*dppz(i)/(dd(i)*siny))*fsteradjk
        enddo
        ! ppgradzpp_o_siny (215)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppgradzpp_o_siny'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (pp(i)*dppz(i)/siny)*fsteradjk
        enddo
        ! gradzpp_o_siny (216)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gradzpp_o_siny'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                (dppz(i)/siny)*fsteradjk
        enddo
        ! uxttdivu (217)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxttdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ux(i)*tt(i)*divu(i)*fsteradjk
        enddo 
        ! ttdivu (218)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ttdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                tt(i)*divu(i)*fsteradjk
        enddo 
        ! uxenuc1_o_cv (219)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxenuc1_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (ux(i)*en1(i)/cvd(i))*fsteradjk
        enddo
        ! uxenuc2_o_cv (220)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxenuc2_o_cv'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                (ux(i)*en2(i)/cvd(i))*fsteradjk
        enddo
        ! ppux (221)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                pp(i)*ux(i)*fsteradjk
        enddo
        ! ppuy (222)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                pp(i)*uy(i)*fsteradjk
        enddo
        ! ppuz (223)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                pp(i)*uz(i)*fsteradjk
        enddo        
        ! ppppux (224)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppppux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*pp(i)*ux(i)*fsteradjk
        enddo
        ! ddppux (225)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddppux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*pp(i)*ux(i)*fsteradjk
        enddo
        ! ddppuy (226)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddppuy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*pp(i)*uy(i)*fsteradjk
        enddo
        ! ddppuz (227)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddppuz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*pp(i)*uz(i)*fsteradjk
        enddo
        ! ppuxux (228)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuxux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! ppuyuy (229)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuyuy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*uy(i)*uy(i)*fsteradjk
        enddo
        ! ppuzuz (230)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuzuz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! ppuyux (231)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuyux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*uy(i)*ux(i)*fsteradjk
        enddo
        ! ppuzux (232)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuzux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*uz(i)*ux(i)*fsteradjk
        enddo
        ! ppuzuy (233)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuzuy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*uz(i)*uy(i)*fsteradjk
        enddo
        ! ppuzuzcoty (234)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuzuzcoty'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*uz(i)*uz(i)*coty*fsteradjk
        enddo
        ! ppuzuycoty (235)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppuzuycoty'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*uz(i)*uy(i)*coty*fsteradjk
        enddo        
        ! uzuzcoty (236)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzuzcoty'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uz(i)*uz(i)*coty*fsteradjk
        enddo
        ! uzuycoty (237)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzuycoty'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uz(i)*uy(i)*coty*fsteradjk
        enddo  
        ! ppddenuc1 (238)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppddenuc1'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*dd(i)*en1(i)*fsteradjk
        enddo
        ! ppddenuc2 (239)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppddenuc2'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*dd(i)*en2(i)*fsteradjk
        enddo
        ! ppenuc1 (240)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppenuc1'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*en1(i)*fsteradjk
        enddo
        ! ppenuc2 (241)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppenuc2'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*en2(i)*fsteradjk
        enddo
        ! ppppdivu (242)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ppppdivu'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                pp(i)*pp(i)*divu(i)*fsteradjk
        enddo        
        ! gradypp (243)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gradypp'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dppy(i)*fsteradjk
        enddo        
        ! abar (244)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'abar'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                abar(i)*fsteradjk
        enddo
        ! zbar (245)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'zbar'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                zbar(i)*fsteradjk
        enddo
        ! ddabar (246)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabar'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*abar(i)*fsteradjk
        enddo
        ! ddzbar (247)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddzbar'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*zbar(i)*fsteradjk
        enddo
        ! abarsq (248)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'abarsq'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                abar(i)*abar(i)*fsteradjk
        enddo
        ! zbarsq (249)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'zbarsq'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                zbar(i)*zbar(i)*fsteradjk
        enddo
        ! ddabarsq_sum_xdn_o_an (250)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabarsq_sum_xdn_o_an'
        do i=1,qx
           sum_xdn_o_an = 0.d0
           ddabarsq = dd(i)*abar(i)*abar(i)
           do n=1,rans_nnuc
              sum_xdn_o_an = sum_xdn_o_an + xdn(i,n)/xnucaa(n)
           enddo
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ddabarsq*sum_xdn_o_an*fsteradjk
        enddo
        ! uxddabarsq_sum_xdn_o_an (251)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxddabarsq_sum_xdn_o_an'
        do i=1,qx
           sum_xdn_o_an = 0.d0
           uxddabarsq = ux(i)*dd(i)*abar(i)*abar(i)
           do n=1,rans_nnuc
              sum_xdn_o_an = sum_xdn_o_an + xdn(i,n)/xnucaa(n)
           enddo
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uxddabarsq*sum_xdn_o_an*fsteradjk
        enddo
        ! ddabazbar_sum_xdn_o_an (252)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabazbar_sum_xdn_o_an'
        do i=1,qx
           sum_xdn_o_an = 0.d0
           ddabarzbar = dd(i)*abar(i)*zbar(i)
           do n=1,rans_nnuc
              sum_xdn_o_an = sum_xdn_o_an + xdn(i,n)/xnucaa(n)
           enddo
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ddabarzbar*sum_xdn_o_an*fsteradjk
        enddo
        ! ddabar_sum_znxdn_o_an (253)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabar_sum_znxdn_o_an'
        do i=1,qx
           sum_znxdn_o_an = 0.d0
           ddabar = dd(i)*abar(i)
           do n=1,rans_nnuc
              sum_znxdn_o_an = sum_znxdn_o_an + xnuczz(n)*xdn(i,n)/xnucaa(n)
           enddo
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ddabar*sum_znxdn_o_an*fsteradjk
        enddo
        ! uxddabazbar_sum_xdn_o_an (254)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxddabazbar_sum_xdn_o_an'
        do i=1,qx
           sum_xdn_o_an = 0.d0
           uxddabarzbar = ux(i)*dd(i)*abar(i)*zbar(i)
           do n=1,rans_nnuc
              sum_xdn_o_an = sum_xdn_o_an + xdn(i,n)/xnucaa(n)
           enddo
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uxddabarzbar*sum_xdn_o_an*fsteradjk
        enddo
        ! uxddabar_sum_znxdn_o_an (255)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxddabar_sum_znxdn_o_an'
        do i=1,qx
           sum_znxdn_o_an = 0.d0
           uxddabar = dd(i)*abar(i)
           do n=1,rans_nnuc
              sum_znxdn_o_an = sum_znxdn_o_an + xnuczz(n)*xdn(i,n)/xnucaa(n)
           enddo
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uxddabar*sum_znxdn_o_an*fsteradjk
        enddo
        ! abargradxpp (256)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'abargradxpp'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                abar(i)*gradxpp(i)*fsteradjk
        enddo        
        ! ddabarux (257)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabarux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*abar(i)*ux(i)*fsteradjk
        enddo
        ! ddabaruy (258)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabaruy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*abar(i)*uy(i)*fsteradjk
        enddo        
        ! ddabaruz (259)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabaruz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*abar(i)*uz(i)*fsteradjk
        enddo
        ! ddabaruxux (260)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabaruxux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*abar(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! ddabaruyuy (261)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabaruyuy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*abar(i)*uy(i)*uy(i)*fsteradjk
        enddo        
        ! ddabaruzuz (262)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddabaruzuz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*abar(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! zbargradxpp (263)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'zbargradxpp'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                zbar(i)*gradxpp(i)*fsteradjk
        enddo        
        ! ddzbarux (264)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddzbarux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*zbar(i)*ux(i)*fsteradjk
        enddo
        ! ddzbaruy (265)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddzbaruy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*zbar(i)*uy(i)*fsteradjk
        enddo        
        ! ddzbaruz (266)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddzbaruz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*zbar(i)*ux(i)*fsteradjk
        enddo
        ! ddzbaruxux (267)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddzbaruxux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*zbar(i)*ux(i)*ux(i)*fsteradjk
        enddo
        ! ddzbaruyuy (268)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddzbaruyuy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*zbar(i)*uy(i)*uy(i)*fsteradjk
        enddo        
        ! ddzbaruzuz (269)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddzbaruzuz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*zbar(i)*uz(i)*uz(i)*fsteradjk
        enddo
        ! cp (270)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'cp'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                cpd(i)*fsteradjk
        enddo
        ! ddcp (271)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddcp'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*cpd(i)*fsteradjk
        enddo        
        ! cv (272)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'cv'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                cvd(i)*fsteradjk
        enddo
        ! ddcv (273)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddcv'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*cvd(i)*fsteradjk
        enddo        
        ! chid (274)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'chid'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                chd(i)*fsteradjk
        enddo
        ! chit (275)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'chit'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                cht(i)*fsteradjk
        enddo
        ! chim (276)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'chim'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                chm(i)*fsteradjk
        enddo        
        ! sound speed (277)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'sound'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                cs(i)*fsteradjk
        enddo
        ! psi (278) ! degeneracy parameter
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'psi'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ps(i)*fsteradjk
        enddo
        
        ! NOTE: These depend on geometry and gravity solver

        ! mm (279)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'mm'
        if((igrav.eq.1).or.(igrav.eq.2)) then
           do i=1,qx
              havg(2,ifield,i) = havg(2,ifield,i) + &
                   mass(i)*fsteradjk
           enddo
        else 
           do i=1,qx
              havg(2,ifield,i) = havg(2,ifield,i) + 0.d0
           enddo
        endif

        ! get gravitational acceleration        
        if((igrav.eq.1).or.(igrav.eq.2)) then
           do i=1,qx
              ggravity(i) = -bigG*mass(i)/xznl(i)**2.d0               
           enddo
        else
           do i=1,qx
              ggravity(i) = 0.d0
           enddo
        endif

        if(igrav.eq.6) then
           do i=1,qx
              y = xzn(i)/onelu 
              ggravity(i) =  g0/(y**five_fourth) 
           enddo
        endif

        if(igrav.eq.7) then
           do i=1,qx
              y = xzn(i)/onelu
              if ((y.ge.1.0625d0).and.(y.le.2.9375d0)) &
                   fgy = 1.d0
              if ((y.ge.1.d0).and.(y.lt.1.0625d0)) &
                   fgy = 0.5d0*(1.d0+sin(16.d0*pi*(y-1.03125d0)))
              if ((y.gt.2.9375d0).and.(y.le.3.0d0)) &
                   fgy = 0.5d0*(1.d0-sin(16.d0*pi*(y-2.96875d0)))
              ggravity(i) = fgy*g0/(y**five_fourth)
           enddo
        endif
        
           
        ! gg (280)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gg'
        if((igrav.eq.1).or.(igrav.eq.2).or.(igrav.eq.6).or.(igrav.eq.7)) then
           do i=1,qx
              havg(2,ifield,i) = havg(2,ifield,i) +  &
                   ggravity(i)*fsteradjk
              gg(i) = havg(2,ifield,i)
           enddo
        else if(igrav.eq.5) then !Keele problem specific (keeleshell) 
           do i=1,qx
              havg(2,ifield,i) = havg(2,ifield,i)  &
                   - 1.5d17/xzn(i)*fsteradjk
              gg(i) = havg(2,ifield,i)
           enddo
        else
           do i=1,qx
              havg(2,ifield,i) = havg(2,ifield,i) + 0.d0
              gg(i) = havg(2,ifield,i)
           enddo
        endif
        
        ! ddgg (281)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ddgg'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ggravity(i)*fsteradjk
        enddo
        ! uxddgg (282)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'uxddgg'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ux(i)*dd(i)*ggravity(i)*fsteradjk
        enddo
        ! hhddgg (283)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'hhddgg'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                hh(i)*dd(i)*ggravity(i)*fsteradjk
        enddo
        ! ssddgg (284)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'ssddgg'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ss(i)*dd(i)*ggravity(i)*fsteradjk
        enddo
        ! abarddgg (285)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'abarddgg'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                abar(i)*dd(i)*ggravity(i)*fsteradjk
        enddo
        ! zbarddgg (286)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'zbarddgg'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                zbar(i)*dd(i)*ggravity(i)*fsteradjk
        enddo
        ! eiddgg (287)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'eiddgg'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ei(i)*dd(i)*ggravity(i)*fsteradjk
        enddo
        ! dduxdivu (288)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduxdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*divu(i)*fsteradjk
        enddo
        ! dduydivu (289)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduydivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*divu(i)*fsteradjk
        enddo
        ! dduzdivu (290)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduzdivu'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*divu(i)*fsteradjk
        enddo
        ! dduxdivux (291)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduxdivux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*divux(i)*fsteradjk
        enddo
        ! dduxdivuy (292)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduxdivuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*divuy(i)*fsteradjk
        enddo
        ! dduxdivuz (293)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduxdivuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*ux(i)*divuz(i)*fsteradjk
        enddo
        ! dduydivux (294)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduydivux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*divux(i)*fsteradjk
        enddo
        ! dduydivuy (295)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduydivuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*divuy(i)*fsteradjk
        enddo
        ! dduydivuz (296)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduydivuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uy(i)*divuz(i)*fsteradjk
        enddo
        ! dduzdivux (297)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduzdivux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*divux(i)*fsteradjk
        enddo
        ! dduzdivuy (298)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduzdivuy'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*divuy(i)*fsteradjk
        enddo
        ! dduzdivuz (299)
        ifield = ifield+1
        if(imode.eq.0) ransname(ifield) = 'dduzdivuz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uz(i)*divuz(i)*fsteradjk
        enddo
        ! uxuyy (300)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxuyy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ux(i)*uyy(i)*fsteradjk
        enddo
        ! uxuzz (301)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxuzz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ux(i)*uzz(i)*fsteradjk
        enddo
        ! uyuxx (302)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyuxx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uy(i)*uxx(i)*fsteradjk
        enddo
        ! uyuyy (303)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyuyy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uy(i)*uyy(i)*fsteradjk
        enddo
        ! uyuzz (304)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyuzz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uy(i)*uzz(i)*fsteradjk
        enddo
        ! uzuxx (305)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzuxx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uz(i)*uxx(i)*fsteradjk
        enddo
        ! uzuyy (306)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzuyy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uz(i)*uyy(i)*fsteradjk
        enddo
        ! uzuzz (307)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzuzz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                uz(i)*uzz(i)*fsteradjk
        enddo
        ! dduxuxx (308)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuxx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*ux(i)*uxx(i)*fsteradjk
        enddo
        ! dduxuyy (309)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuyy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                 dd(i)*ux(i)*uyy(i)*fsteradjk
        enddo
        ! dduxuzz (310)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxuzz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                 dd(i)*ux(i)*uzz(i)*fsteradjk
        enddo
        ! dduyuxx (311)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduyuxx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                 dd(i)*uy(i)*uxx(i)*fsteradjk
        enddo
        ! dduyuyy (312)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduyuyy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                 dd(i)*uy(i)*uyy(i)*fsteradjk
        enddo
        ! dduyuzz (313)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduyuzz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                 dd(i)*uy(i)*uzz(i)*fsteradjk
        enddo
        ! dduzuxx (314)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzuxx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                 dd(i)*uz(i)*uxx(i)*fsteradjk
        enddo
        ! dduzuyy (315)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzuyy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                 dd(i)*uz(i)*uyy(i)*fsteradjk
        enddo
        ! dduzuzz (316)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzuzz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                 dd(i)*uz(i)*uzz(i)*fsteradjk
        enddo
        ! dduxx (317)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*uxx(i)*fsteradjk
        enddo
        ! dduyy (318)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduyy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*uyy(i)*fsteradjk
        enddo
        ! dduzz (319)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzz'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                dd(i)*uzz(i)*fsteradjk
        enddo

        ! gammac (320)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gammac'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gac(i)*fsteradjk
        enddo

        ! gammae (321)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'gammae'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                gae(i)*fsteradjk
        enddo        

        ! uxuxux (322)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uxuxux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                ux(i)*ux(i)*ux(i)*fsteradjk
        enddo

        ! uyuyux (323)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uyuyux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uy(i)*uy(i)*ux(i)*fsteradjk
        enddo

        ! uzuzux (324)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'uzuzux'
        do i=1,qx
           havg(2,ifield,i) =  havg(2,ifield,i) + &
                uz(i)*uz(i)*ux(i)*fsteradjk
        enddo

        ! alphac (325)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'alphac'
        if (igrav.eq.7) then ! identify ccptwo
          do i=1,qx
             havg(2,ifield,i) =  havg(2,ifield,i) + &
                 (-(mu0*pp(i)/(gascon*tt(i)*xn(i,1)*xn(i,1))) - &
                  (mu1*pp(i)/(gascon*tt(i)*xn(i,2)*xn(i,2))))*fsteradjk 
          enddo
        else 
          do i=1,qx
             havg(2,ifield,i) =  0.d0
          enddo
        endif
      

        ! ddttux (326)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'ddttux'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*tt(i)*ux(i)*fsteradjk
        enddo


        ! dduxttx (327)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduxttx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*ux(i)*gradxtt(i)*fsteradjk
        enddo


        ! dduytty (328)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduytty'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*uy(i)*dtty(i)*fsteradjk
        enddo

        ! dduzttz (329)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'dduzttz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                dd(i)*uz(i)*dttz(i)*fsteradjk
        enddo


        ! eiuxddx (330)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'eiuxddx'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ei(i)*ux(i)*gradxdd(i)*fsteradjk
        enddo


        ! eiuyddy (331)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'eiuyddy'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
                ei(i)*uy(i)*dddy(i)*fsteradjk
        enddo

        ! eiuzddz (332)
        ifield = ifield + 1
        if(imode.eq.0) ransname(ifield) = 'eiuzddz'
        do i=1,qx
           havg(2,ifield,i) = havg(2,ifield,i) + &
               ei(i)*uz(i)*dddz(i)*fsteradjk
        enddo

        ! kdiff (xxx)
        !ifield = ifield + 1
        !if(imode.eq.0) ransname(ifield) = 'kdiff'
        !do i=1,qx
        !   havg(2,ifield,i) = havg(2,ifield,i) + &
        !        kdf(i)*fsteradjk
        !enddo        

        ! fradx (xxx)
        !ifield = ifield + 1
        !if(imode.eq.0) ransname(ifield) = 'fradx'
        !do i=1,qx
        !   havg(2,ifield,i) = havg(2,ifield,i) + &
        !        frx(i)*fsteradjk
        !enddo        

        ! frady (xxx)
        !ifield = ifield + 1
        !if(imode.eq.0) ransname(ifield) = 'frady'
        !do i=1,qx
        !   havg(2,ifield,i) = havg(2,ifield,i) + &
        !        fry(i)*fsteradjk
        !enddo

        ! fradz (xxx)
        !ifield = ifield + 1
        !if(imode.eq.0) ransname(ifield) = 'fradz'
        !do i=1,qx
        !   havg(2,ifield,i) = havg(2,ifield,i) + &
        !        frz(i)*fsteradjk
        !enddo   
        
      
        ! NOTE: These are composition-field dependent (depends on ixnuc)
        ! These are variable length: # composition fields = rans_nnuc * 16

        do n=1,rans_nnuc                                      
           if(imode.eq.0) then ! construct varnames
              9901 format(i0.4)
              write(xidchar,9901) n
              xvarname(1)  = 'x'  //xidchar             ! X
              xvarname(2)  = 'x'  //xidchar//'sq'       ! X*X
              xvarname(3)  = 'ddx'//xidchar             ! dd*X 
              xvarname(4)  = 'ddx'//xidchar//'sq'       ! dd*X*X
              xvarname(5)  = 'x'  //xidchar//'ux'       !  X*ux
              xvarname(6)  = 'ddx'//xidchar//'ux'       !  dd*X*ux
              xvarname(7)  = 'ddx'//xidchar//'uy'       ! dd*X*uy
              xvarname(8)  = 'ddx'//xidchar//'uz'       ! dd*X*uz
              xvarname(9)  = 'ddx'//xidchar//'dot'      ! dd*Xdot
              xvarname(10) = 'ddx'//xidchar//'dotux'    ! dd*Xdot*ux
              xvarname(11) = 'ddx'//xidchar//'dotuy'    ! dd*Xdot*uy
              xvarname(12) = 'ddx'//xidchar//'dotuz'    ! dd*Xdot*uz              
              xvarname(13) = 'ddx'//xidchar//'squx'     ! dd*X*X*ux
              xvarname(14) = 'ddx'//xidchar//'uxux'     ! dd*X*ux*ux
              xvarname(15) = 'ddx'//xidchar//'uyuy'     ! dd*X*uy*uy
              xvarname(16) = 'ddx'//xidchar//'uzuz'     ! dd*X*uz*uz
              xvarname(17) = 'ddx'//xidchar//'uxuy'     ! dd*X*ux*uy
              xvarname(18) = 'ddx'//xidchar//'uxuz'     ! dd*X*ux*uz
              xvarname(19) = 'ddx'//xidchar//'uzuycoty' ! dd*X*uz*uy*coty              
              xvarname(20) = 'ddx'//xidchar//'uzuzcoty' ! dd*X*uz*uz*coty              
              xvarname(21) = 'x'  //xidchar//'gradxpp'  ! X*gradxpp
              xvarname(22) = 'x'  //xidchar//'gradypp'  ! X*gradypp
              xvarname(23) = 'x'  //xidchar//'gradzpp_o_siny'    ! X*gradzpp_o_siny
              xvarname(24) = 'ddx'//xidchar//'x'//xidchar//'dot' ! dd*X*Xdot
              xvarname(25) = 'x'  //xidchar//'ddgg'     ! X*dd*gg
              !xvarname(26)  = 'alphac'//xidchar          ! alphac
           endif

           ! xn
           ifield = ifield+1                  
           if(imode.eq.0) then 
              ransname(ifield) = xvarname(1)
           endif
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   xn(i,n)*fsteradjk                      
           enddo
           ! xnsq 
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(2)    
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   xn(i,n)*xn(i,n)*fsteradjk              
           enddo
           ! ddxn 
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(3)    
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*fsteradjk                
           enddo
           ! ddxnsq 
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(4)    
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*xn(i,n)*fsteradjk        
           enddo
           ! xnux
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(5)    
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   xn(i,n)*ux(i)*fsteradjk                
           enddo
           ! ddxnux 
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(6)    
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*ux(i)*fsteradjk          
           enddo
           ! ddxnuy 
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(7)    
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*uy(i)*fsteradjk          
           enddo
           ! ddxnuz 
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(8)    
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*uz(i)*fsteradjk          
           enddo
           ! ddxndot
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(9)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xdn(i,n)*fsteradjk                      
           enddo	
           ! ddxndotux
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(10)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xdn(i,n)*ux(i)*fsteradjk                      
           enddo
           ! ddxndotuy
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(11)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xdn(i,n)*uy(i)*fsteradjk                      
           enddo
           ! ddxndotuz
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(12)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xdn(i,n)*uz(i)*fsteradjk                      
           enddo
           ! ddxnsqux
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(13)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*xn(i,n)*ux(i)*fsteradjk                      
           enddo
           ! ddxnuxux
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(14)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*ux(i)*ux(i)*fsteradjk                      
           enddo
           ! ddxnuyuy
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(15)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*uy(i)*uy(i)*fsteradjk                      
           enddo           
           ! ddxnuzuz
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(16)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*uz(i)*uz(i)*fsteradjk                      
           enddo
           ! ddxnuxuy
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(17)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*ux(i)*uy(i)*fsteradjk                      
           enddo
           ! ddxnuxuz
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(18)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*ux(i)*uz(i)*fsteradjk                      
           enddo
           ! ddxnuzuycoty
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(19)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*uz(i)*uy(i)*coty*fsteradjk                      
           enddo
          ! ddxnuzuzcoty
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(20)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*uz(i)*uz(i)*coty*fsteradjk                      
           enddo                      
           ! xngradxpp
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(21)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   xn(i,n)*gradxpp(i)*fsteradjk                      
           enddo
           ! xngradypp
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(22)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   xn(i,n)*dppy(i)*fsteradjk                      
           enddo
           ! xngradzpp_o_siny
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(23)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   (xn(i,n)*dppz(i)/siny)*fsteradjk                      
           enddo
           ! ddxnxdot
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(24)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   dd(i)*xn(i,n)*xdn(i,n)*fsteradjk                      
           enddo
           ! xnddgr
           ifield = ifield+1                          
           if(imode.eq.0) ransname(ifield) = xvarname(25)  		   
           do i=1,qx                                      
              havg(2,ifield,i) =  havg(2,ifield,i) + &    
                   xn(i,n)*dd(i)*ggravity(i)*fsteradjk                      
           enddo
           ! alphacx
           !ifield = ifield+1
           !if(imode.eq.0) ransname(ifield) = xvarname(26)

           !if (igrav.eq.7) then ! identify ccptwo
           ! do i=1,qx
           !  if (n.eq.1) havg(2,ifield,i) = havg(2,ifield,i) - (mu0*pp(i)/(gascon*tt(i)*xn(i,n)*xn(i,n)))*fsteradjk
           !  if (n.eq.2) havg(2,ifield,i) = havg(2,ifield,i) - (mu1*pp(i)/(gascon*tt(i)*xn(i,n)*xn(i,n)))*fsteradjk
           ! enddo
           !else
           !    havg(2,ifield,i) =  0.d0
           !endif

        enddo
       
        ! check on field count
        if(ifield.ne.nrans) then
           write(*,*) ifield,nrans
           if(myid.eq.0) print*,'ERR(rans_avg): field count, ifield.ne.rans'
           call MPI_FINALIZE(ierr)
           stop 'ERR(rans_avg): field count, ifield.ne.nrans'
        endif

     enddo
  enddo

 
  
  ! update running average: havg(3,:,:)
  
  if(imode.eq.1) then 
     rans_tavg = rans_tavg + dt
     rans_tend = time
     do n=1,nrans
        do i=1,qx 
           havg(3,n,i) = havg(3,n,i) + &
                (havg(2,n,i) + havg(1,n,i))*0.5d0*dt
        enddo
     enddo
  else if(imode.eq.0) then
     do n=1,nrans
        do i=1,qx
           havg(4,n,i) = havg(2,n,i) !store first instance in this averaging interval
        enddo
     enddo
  endif

  return

  
end subroutine rans_avg



