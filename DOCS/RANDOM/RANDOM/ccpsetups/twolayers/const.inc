!..include file physcs.inc
!..MUST follow include dimen.inc
!..
!..physical constants, 
!..gravity, ficticous forces, energy gain, loss, mass element
!..
!
!
!
      real*8          gascon, arad, gammalaw, clight
      real*8          bigG
!
!      parameter ( gascon    = 8.3147515d7        )
      parameter ( gascon    = 8.3144629d7        )
!      parameter ( gammalaw  = 1.66666666667d0    )
      parameter ( gammalaw = 1.4d0             )
      parameter ( arad      = 7.56489d-15        )	
      parameter ( clight    = 3.0d10             )
      parameter ( bigG      = 6.673D-8           )
!
      real*8          grav, fict, dedt, s
!
      common /physcs/ grav(qq), fict(qq), dedt(qq), s(qq)
