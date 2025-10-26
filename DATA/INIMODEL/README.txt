PROMPI/setups/oburn/3d/src/INIT
/read_tycho.f90

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!
!     READ IN TYCHO MODEL DATA
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!     ------------------
!     Input model zoning
!     ------------------
!     1. rr(i) is right edge of zone i
!     *2. mm(i) is interior mass up to radius rr(i+1) ***NOTE INDEX DISCREPANCY!!!!
!     3. vr(i) is velocity of mass zone interface, at radius rr(i)
!     4. tt(i) is the temperature of zone between rr(i) and rr(i-1)
!     5. dd(i) is the density of zone between rr(i) and rr(i-1)
!     6. pp(i) is the pressure of zone between rr(i) and rr(i-1)
!     7. dm(i) is the total mass in zone between rr(i) and rr(i-1)
!     8. cv(i) is the convective velocity in zone between rr(i) and rr(i-1)
!     9. ll(i) is the luminosity at zone interface, at radius rr(i)
!     10. xyz(i,jj) is the jjth composition in zone between rr(i) and rr(i-1)
!
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!