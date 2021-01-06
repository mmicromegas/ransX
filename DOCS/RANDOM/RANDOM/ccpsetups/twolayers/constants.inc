!..   include file: constants.inc
!..   
!..   PHYSICAL CONSTANTS
!
      real*8   gascon, arad, gammalaw
      real*8   clight, bigG, kboltz, echarge, hplanck
      real*8   stefanb
      real*8   MeV, KeV
      real*8   mh, mel, mhe, Lsun, Msun, Avogadro
!..
!..
!      parameter ( gascon    = 8.3147515d7        )
      parameter ( gascon    = 8.3144629d7        )
!      parameter ( gascon    = 1.d0        )
      parameter ( gammalaw  = 1.66666666667d0    )
!      parameter ( gammalaw  = 1.4d0              )
      parameter ( arad      = 7.56489d-15        )
      parameter ( stefanb   = 5.67d-5            )
      parameter ( clight    = 2.99792458d10      )
      parameter ( bigG      = 6.673d-8           )
!      parameter ( bigG      = 1.0d0              )
      parameter ( kboltz    = 1.38e-16           )
      parameter ( hplanck   = 6.626176d-27       ) 
      parameter ( KeV       = 1.602d-9           )
      parameter ( MeV       = 1.602d-6           )
      parameter ( mh        = 1.6726d-24         )
      parameter ( mhe       = 6.6442d-24         )
      parameter ( mel       = 9.109534e-28       )
      parameter ( echarge   = 1.60203d-20        )
      parameter ( Lsun      = 3.826e33           )
      parameter ( Msun      = 1.98892d33         )
      parameter ( Avogadro  = 6.022137d23        )


!     CODE COMPARISON PROJECT

      real*8 onelu,onepu,onedu,onetu,onemu,onegu,mu0,mu1,oneluu,oneeu
      real*8 onettu,g0

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
      !g0 = -1.414870d0*onegu 


      parameter ( onelu = 4.d8         )
      parameter ( onepu = 4.644481d23  )
      parameter ( onedu = 1.820940d6   )
      parameter ( onetu = 0.7920256    )
      parameter ( onemu = 1.165402d32  )
      parameter ( onettu = 3.401423d9  )
      parameter ( onegu = onelu/(onetu**2) )
      parameter ( mu0   = 1.848        )
      parameter ( mu1   = 1.802        )
      parameter ( oneluu = 3.752995d49 )
      parameter ( oneeu = 2.972468d49  )
      parameter ( g0 = -1.414870d0*onegu )


