!----------------------------------------------------------------------
!     UNIX/MPI Version: Casey A. Meakin:        ( 15 April     2003 )
!     Linux version:    W. David Arnett:        ( 30 September 1999 )
!     Originally developed by: 
!     The code was developed from CRAY version  ( 14 May       1990 )
!     which was written by
!     B. A. Fryxell,   University of Arizona,  Tucson,    USA
!     E. Mueller,      MPI f. Astrophysik,     Garching,  FRG
!     PROMETHEUS  originated from a direct eulerian PPM-code as
!     described in Colella and Woodward (JCP, 54 (1984), 174).
!----------------------------------------------------------------------
!     P R O M E T H E U S
!     -------------------
!     PRO gram  for
!     M ulti-dimensional
!     E ulerian
!     T hermonuclear 
!     H ydrodynamics  with 
!     E xplicit 
!     U p-wind
!     S econd-order  differencing
!     
!----------------------------------------------------------------------
!     
!     SEE BOTTOM OF FILE FOR MORE DEVELOPMENT NOTES AND BUG FIXES
!     
!----------------------------------------------------------------------

program prompi
  
  use mod_rans_data, only: irans, rans_data_init
  use mod_tdiff_data
  use mod_basestate_data
  implicit none
  
  include 'dimen.inc'
  include 'vnew.inc'
  include 'vold.inc'
  include 'mesh.inc'
  include 'float.inc'
  include 'intgrs.inc'
  include 'grd.inc'
  include 'char.inc'
  include 'cmovy.inc'
  include 'ceos.inc'
  include 'bndry.inc'
  include 'files.inc'
  include 'timing.inc'
  include 'physcs.inc'
  include 'diffusion.inc'
  include 'gravity.inc'
  
  include 'mpicom.inc'
  include 'mpif.h'
  
  logical tobe
  integer*4 irandom, idebug, i
      

  pi = 4.0*atan(1.0) ! 3.1415926535898d0

  
  !     RUNTIME FLAGS

  imultilog   = 0
  idebug      = 0
  mpi_used    = 1
  nuc         = qn


  !     INIT MPI
  call MPI_INIT(ierr)    !Initialize MPI Libraries
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid_w, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs_w, ierr)
  myid = myid_w
      
  if(myid.eq.0.or.imultilog.eq.1) ilog = 1
  
  print*,'MSG(prompi): Welcome to PROMPI, nprocs_w =',nprocs_w
  !if(nprocs_w==1) stop
  
  !     INIT TIMERS

  call zero_timers(0)
  tinit(1)  = MPI_WTIME()
  ttotal(1) = MPI_WTIME()             
  timewc    = 0.d0
  ntcycle   = 0
  

  !     READ IN PARAMETER FILE
  
  call MPI_BARRIER(MPI_COMM_WORLD,ierr)
  !call input(irandom)
  call input_nml !namelist version of run-time parameter input

      
  !     DO SOME PARAMETER CHECKS

  if(igrav.eq.0) mstates = 0

!      if(iburn.ne.0.and.bdim.ne.nsdim) then
!         print*,'ERR(prompi): bdim.ne.nsdim: incorrect bndry.inc'
!         call MPI_FINALIZE(ierr)
!         stop 'ERR(prompi): bdim.ne.nsdim: incorrect bndry.inc'
!      endif


!     SETUP MPI COMM. TOPOLOGY
  call MPI_BARRIER(MPI_COMM_WORLD,ierr)
  call mstart
  


!     OPEN HYDRO LOG(s) FILE
      
      write(myidsuffix, 9901) myid
9901  format(i5.5)
      call getchardate(chardate)
      fname = '.logfile.'//myidsuffix//'.'
      call concat(runname,fname,fname)
      call concat(fname,chardate,fname)
      if(myid.eq.0) print*,'MSG(prompi): creating logfile: ', fname
      if(ilog.eq.1) then
         open(unit=nthlog,file=fname,status='unknown')
      endif

      
!     LOG PROBLEM SETUP

      if(myid.eq.0) then
         write(6,*) 'MSG(prompi): Problem Dimensions'
         write(6,*) 'MSG(prompi): nx, ny, nz = ', nx, ny, nz
         write(6,*) 'MSG(prompi): qx, qy, qz = ', qx, qy, qz
         write(6,*) 'MSG(prompi): qn         = ', qn
         write(6,*) 'MSG(prompi): ntiles     = ', ntiles
      endif
      if(ilog.eq.1) then 
         write(nthlog,*) 'MSG(prompi): Problem Dimensions'
         write(nthlog,*) 'MSG(prompi): nx, ny, nz = ', nx, ny, nz
         write(nthlog,*) 'MSG(prompi): qx, qy, qz = ', qx, qy, qz
         write(nthlog,*) 'MSG(prompi): qn         = ', qn
         write(nthlog,*) 'MSG(prompi): ntiles     = ', ntiles
      endif


      ! init nuclear reaction network
      
      if(iburn.eq.2) then 
         if(myid.eq.0) print*,'MSG(prompi): call burfinit : init burn data.'
         call burnfinit
         if(isinit.gt.0) then 
            if(myid.eq.0) print*,'MSG(prompi): call starinit_burn : init star related burn data.'
            call starinit_burn
         endif
      endif

      if(iburn.eq.4) then
         if(ilog.eq.1) print*,'MSG(prompi): call burnfinit_mesa'
         ierr = 0
         call burnfinit_mesa(ilog,ierr)
      endif



      ! setup data and grid for new run
      
      if(irstrt.eq.0) then

         if(ilog.eq.1) print*,'MSG(prompi): CREATING NEW MODEL.'
         
         if(ilog.eq.1) print*,'MSG(prompi): call grid'
         call grid
         
         if(igrav.ne.0) then
            if(ilog.eq.1) print*,'MSG(prompi): calling sterad_init'
            call sterad_init
         endif
         
         if(ilog.eq.1) print*,'MSG(prompi): Initialize Problem'
         if(ilog.eq.1) print*,'MSG(prompi): isinit = ', isinit
         
         if(isinit.eq.0) then                                 
            if(ilog.eq.1) print*,'MSG(prompi): calling problem_setup'    
            call problem_setup                                
         endif
         
         if(isinit.ge.1.and.isinit.le.2) then
            if(ilog.eq.1) print*,'MSG(prompi): call starinit'
            call starinit
            if(ilog.eq.1) print*,'MSG(prompi): call perturb'
            call perturb
         endif
         
         if(isinit.eq.3) then
            if(ilog.eq.1) print*,'MSG(prompi): call basestate_init'
            call basestate_init(0.d0,2.d0*xmax)
            if(ilog.eq.1) print*,'MSG(prompi): call imodel_init'
            call imodel_init
            !if(myid.eq.0) print*,'MSG(prompi): call perturb'
            !call perturb
         endif
         
         if(isinit.eq.4) then !Keele U. project: plane-parallel HSE integrator
            if(xyzgrav.eq.1) then
               call basestate_init(xmin,xmax)
            else
               call MPI_FINALIZE(ierr)
               stop 'ERR(prompi): xyzgrav.ne.1 not implemented for isinit.eq.4'
            endif
            if(ilog.eq.1) print*,'MSG(prompi): call init_keele'
            call init_keele
            if(ilog.eq.1) print*,'MSG(prompi): call perturb'
            call perturb
         endif
         
         if(ilog.eq.1) print*,'MSG(prompi): calling eos3d(2)'
         call eos3d(2)          !RUN EOS OVER FIELDS
         
         if(myid.eq.0) then 
            print*,'MSG(prompi): setup initial timestepping, cycle counting variables.'
         endif
         time   = 0.d0
         nstep  = 0
         nbegin = 1         
         dt       = dtini
         dtglobal = dtini
         dtlocal  = dtini
      endif
      
      

      ! setup data and grid for restart run
      
      if(irstrt.gt.0) then
         
         if(myid.eq.0) print*,'MSG(prompi):RESTARTING FROM DATA DUMP.'
         if(myid.eq.0) print*,'MSG(prompi): call grid'
         call grid
         
         if(igrav.ne.0) then
            if(ilog.eq.1) print*,'MSG(prompi): calling sterad_init'
            call sterad_init
         endif

         if(ilog.eq.1) print*,'MSG(prompi): isinit = ', isinit

         if(isinit.ge.1.and.isinit.le.2) then 
            if(myid.eq.0) print*,'MSG(prompi): call starinit'
            call starinit          !SETUP OTHER CONSTANTS/BOUNDARY ZONES
         endif
         
         if(isinit.eq.3) then
            if(igrav.eq.3) then 
               if(ilog.eq.1) print*,'MSG(prompi): call basestate_init'
               call basestate_init
               if(ilog.eq.1) print*,'MSG(prompi): call imodel_init'
               call imodel_init
            endif
         endif
         
         if(myid.eq.0) print*,'MSG(prompi): calling restart.'
         call restart(irstrt,'./data')
         
         if(irestart_eos.eq.2) then
            if(ilog.eq.1) print*,'MSG(prompi): calling eos3d(2) after restart'   
            call eos3d(2)
         else if(irestart_eos.eq.1) then
            if(ilog.eq.1) print*,'MSG(prompi): calling eos3d(1) after restart'
            call eos3d(2)
         endif
         
         !call rg_init
         
         nstep  = 0
         nbegin = 1
         dt       = dtini
         dtglobal = dtini
         dtlocal  = dtini
         
      endif
      
      ! zero grid motion velocities (pure eulerian grid implementation)
      call zero_grdvel


      ! init burn

      if(iburn.eq.2) then                                                      
         dt = 1.d0                                                             
         if(ilog.eq.1) print*,'MSG(prompi): call burnf(0), dt = ', dt          
         call burnf(0)                                                         
         dt = dtini
      endif
      if(iburn.eq.3) then                                                              
         !if(ilog.eq.1) print*,'MSG(prompi): call massheat(0): init enuc profiles'     
         !call massheat(0)                                                             
         if(ilog.eq.1) print*,'MSG(prompi): call volheat(0): init enuc profiles'       
         call volheat(0)                                                               
      endif
      if(iburn.eq.4) then                                                      
         dt = 1.d0                                                             
         if(ilog.eq.1) print*,'DBG(prompi): call burnf_mesa(0), dt = ', dt     
         call burnf_mesa(0)                                                    
         dt = dtini
      endif
      
      
      ! init opacities and radiative diffusion fluxes
      if(itdiff.gt.0) then
         if(ilog.eq.1) print*,'MSG(prompi): call tdiff_data_init'
         call tdiff_data_init
         if(ilog.eq.1) print*,'MSG(prompi): call opac3d(1)'
         call opac3d(1) !update radiative conduction array
         if(ilog.eq.1) print*,'MSG(prompi): call tdiff(0)'
         call tdiff(0) !imode=0: update fluxes but NOT internal energy
      endif
      
      
      ! init rans fields
      if(isinit.eq.3) then
         if(idebug.eq.1.and.myid.eq.0) & 
              print*,'DEBUG(prompi): to call enclosed_mass_basestate.'
         call enclosed_mass_basestate
      endif
      if(irans.ne.0) then
         if(ilog.eq.1) print*,'MSG(prompi): call hydrox(0)'
         call hydrox  (0)
         if(ilog.eq.1) print*,'MSG(prompi): call rans_data_init'
         call rans_data_init
         if(ilog.eq.1) print*,'MSG(prompi): call rans_avg(0)'
         call rans_avg(0)
      endif
      

      ! DUMP INITIAL MODEL
      if(myid.eq.0) print*,'MSG(prompi):call dumpdata(0)'
      call dumpdata(0)
      
      
      !print*,'DBG(prompi): debug stop'
      !call MPI_FINALIZE(ierr)
      !stop 0
      
      ! test basestate mapping
      !tmap(1) = MPI_WTIME()
      !      call map_grid_to_basestate
      if(isinit.eq.3) then 
         if(ilog.eq.1) print*,'MSG(prompi): isinit==3, to call enclose_mass_basestate'
         call enclosed_mass_basestate
      endif
      !tmap(2) = MPI_WTIME() - tmap(1)
      

      !debugging output
      if(ilog.eq.1.and.isinit.eq.3) then
9902     format(10e17.8)
         if(ilog.eq.1) then
            open(ntlocal,file='map_to_basestate.txt',status='unknown')
            write(ntlocal,*) basepoints
            write(ntlocal,*) &
                 'mmbase rlbase rrbase ddbase'
            do i=1,basepoints
               write(ntlocal,9902) mmbase(i),rlbase(i), rrbase(i),ddbase(i)
            enddo
            close(ntlocal)
         endif
      endif
      


      

!     SYNC MPI AND START HYDRO LOOP 

      call MPI_BARRIER(commcart, ierr)
      tinit(2) =  MPI_WTIME()
      if(myid.eq.0) then
         print*,'MSG(prompi): BEGIN MAIN HYDRO CYCLE'
         print*,'MSG(prompi): tinit = ', tinit(2) - tinit(1)
         print*,''
      endif
      if(ilog.eq.1) then
         write(nthlog,*) 'MSG(prompi): BEGIN MAIN HYDRO CYCLE'
         write(nthlog,*) 'MSG(prompi): tinit = ', tinit(2) - tinit(1)
         write(nthlog,*) ''
      endif


!     SETUP IO SWITCH COUNTERS

      nout1 = 0
      tout1 = 0.d00
      nrst  = 0
      trst  = 0.d00


      !- debug stop
      
      !if(ilog.eq.1) then
      !   print*,'DBG(prompi): stop before hydro cycling'
      !endif
      !call MPI_FINALIZE(ierr)
      !stop 'DBG(prompi): stop before hydro cycling'


      do ncycl = nbegin, nend, 2
                                    
!     BEGIN TIMING CYCLES
         
         if(ntcycle.eq.0) then
            call zero_timers(1)
            tcycle(1) = MPI_WTIME()
         endif
         
!     PERFORM FIRST HALF OF CYCLE: 1st TIME STEP

!         if(nstep.eq.1) call grdvel


!     GRAVITY

         tgravity(1) = MPI_WTIME()
         if(igrav.eq.3) then
            if(idebug.eq.1.and.ilog.eq.1) &
                 print*,'DEBUG(prompi): to call enclosed_mass_basestate.'
            !call enclosed_mass_basestate
         endif
         if(igrav.eq.2) then
            call mass_shells_density
         endif
         tgravity(2) = tgravity(2) + (MPI_WTIME()-tgravity(1))
         
!     HYDRO SWEEPS 
         
         thydro(1) = MPI_WTIME()
         call hydro_ppm_xyz(1,1)
         thydro(2) = thydro(2) + (MPI_WTIME()-thydro(1))            

!     RADIATIVE DIFFUSION
         
         ttdiff(1) = MPI_WTIME()
         if(itdiff.eq.1) then 
            call opac3d(1) !update krad array
            call tdiff(0)
            if(idebug.eq.1.and.myid.eq.0) print*,'DEBUG:called diffusion'
         endif
         ttdiff(2) = ttdiff(2) + (MPI_WTIME()-ttdiff(1))
         
!     NUCLEAR BURNING OPTIONS
         
         tburn(1) = MPI_WTIME()
         if(iburn.eq.1) then 
            if(idebug.eq.1.and.myid.eq.0)print*,'DEBUG(prompi): to call msburn.'
            call msburn
         endif
         if(iburn.eq.2) then
            if(idebug.eq.1.and.myid.eq.0) print*,'DEBUG(prompi): to call burnf(1).'
            call burnf(1)
         endif
         if(iburn.eq.3) then
            !if(idebug.eq.1.and.ilog.eq.1) print*,'DEBGU(prompi): to call massheat(1)'
            !call massheat(1)
            if(idebug.eq.1.and.ilog.eq.1) print*,'DEBUG(prompi): to call volheat(1)'
            call volheat(1)
         endif
         tburn(2) = tburn(2) + (MPI_WTIME()-tburn(1))
         
!     VELOCITY DAMPING

         !DEBUGINGDEBUGINGDEBUGIN
!         if(nout1.ge.nout.or.tout1.ge.tout) then !DEBUGGING OUTPUT!!
!            call dumpdata(0)
!         endif
         
         if(idamp.ne.0) then
            if(idebug.eq.1.and.myid.eq.0) print*,'DEBUG(prompi): to call damping.'
            call damping
         endif

         
!     RANS AVERAGING
         trans(1) = MPI_WTIME()
         if(irans.ne.0) then
            if(idebug.eq.1.and.myid.eq.0) print*,'DEBUG(prompi): to call rans_avg(1).'
            call rans_avg(1)
         endif
         trans(2) = trans(2) + (MPI_WTIME() - trans(1))


         tio(1) = MPI_WTIME()
         
!     DATA DUMPS
         if(nout1.ge.nout.or.tout1.ge.tout.or.twallout1.ge.twallout) then                               
            nout1 = 0
            tout1 = 0.d0
            twallout1 = 0.d0
            call dumpdata(0)                                                   
            if(irans.ne.0) call rans_avg(0)
         endif
         
!     RESTART FILES                                                           
         if (nrstrt.gt.0.and.nrst.ge.nrstrt.or.trst.ge.trstrt) then            
            nrst = 0                                                           
            trst = 0.d0                                                        
!            call dumpdata(1)                                                  
         endif
                                                                               
!     UPDATE COUNTERS                                                          
         timewc = MPI_WTIME() - ttotal(1)
         time   = time  + dt                                                   
         nstep  = ncycl                                                        
         nout1  = nout1 + 1                                                    
         tout1  = tout1 + dt                                                   
         twallout1 = twallout1 + dt
         nrst   = nrst  + 1                                                    
         trst   = trst  + dt                                                   
            
         tio(2) = tio(2) + (MPI_WTIME() - tio(1))

                                                                   
!     CHECK STOPPING CRITERIA                                                  
         if (time .gt. tmax .or. timewc .ge. twallmax)  go to 200
         

!     PERFORM SECOND HALF OF CYLE: 2nd TIME STEP


!        MASS_SHELLS (SPHERICAL GRAVITY)

         tgravity(1) = MPI_WTIME()
         if(igrav.eq.3) then
            if(idebug.eq.1.and.myid.eq.0) & 
                 print*,'DEBUG(prompi): to call enclosed_mass_basestate'
            !call enclosed_mass_basestate
         endif
         if(igrav.eq.2) then
            !call mass_shells_density
         endif
         tgravity(2) = tgravity(2) + (MPI_WTIME()-tgravity(1))
         
         
!       HYDRO SWEEPS

         thydro(1) = MPI_WTIME()
         call hydro_ppm_xyz(-1,1)
         thydro(2) = thydro(2) + (MPI_WTIME()-thydro(1))


!      THERMAL DIFFUSION

         ttdiff(1) = MPI_WTIME()
         if(itdiff.eq.1) then 
            !call diffusion
            call opac3d(1) ! update radiative conductivity
            call tdiff(0)
            if(idebug.eq.1.and.myid.eq.0) print*,'DEBUG: diffusion(2) done'
         endif
         ttdiff(2) = ttdiff(2) + (MPI_WTIME()-ttdiff(1))

         
!      NUCLEAR BURNING OPTIONS

         tburn(1) = MPI_WTIME()
         if(iburn.eq.1) then 
            if(idebug.eq.1.and.myid.eq.0)print*,'DEBUG(prompi):to call msburn.'
            call msburn
         endif
         if(iburn.eq.2) then
            if(idebug.eq.1.and.myid.eq.0) print*,'DEBUG: to call burnf(1)'
            call burnf(1)
         endif
         if(iburn.eq.3) then
            !if(idebug.eq.1.and.ilog.eq.1) print*,'DEBUG(prompi): to call massheat(1)'
            !call massheat(1)
            if(idebug.eq.1.and.ilog.eq.1) print*,'DEBUG(prompi): to call volheat(1)'
            call volheat(1)
         endif

         tburn(2) = tburn(2) + (MPI_WTIME() - tburn(1))
         
!        VELOCITY DAMPING
         
         !DEBUGGINGDEBUGDEBUGDEBUG
!         if(nout1.ge.nout.or.tout1.ge.tout) then !DEBUGGING DEBUGGING
!            call dumpdata(0)
!         endif
         if(idamp.ne.0) then
            if(idebug.eq.1.and.myid.eq.0) print*,'DEBUG(prompi): to call damping.'
            call damping
         endif
      
         
         
!     RANS AVERAGING

         trans(1) = MPI_WTIME()
         if(irans.ne.0) then
            if(idebug.eq.1.and.myid.eq.0) print*,'DEBUG(prompi): to call rans_avg(1)'
            call rans_avg(1)
         endif
         trans(2) = trans(2)+ (MPI_WTIME() - trans(1))
         
         tio(1) = MPI_WTIME()
         
!     DATA DUMPS                                       
         
         if(nout1.ge.nout.or.tout1.ge.tout.or.twallout1.ge.twallout) then       
            nout1 = 0                                  
            tout1 = 0.d0                               
            twallout1 = 0.d0
            call dumpdata(0)                           
            if(irans.ne.0) call rans_avg(0)
         endif
         

!     UPDATE COUNTERS                                  
         timewc = MPI_WTIME() - ttotal(1)
         time   = time   + dt                          
         nstep  = ncycl  + 1                           
         nout1  = nout1  + 1                           
         tout1  = tout1  + dt                          
         twallout1 = twallout1 + dt
         nrst   = nrst   + 1                           
         trst   = trst   + dt                          
         
         tio(2) = tio(2) + (MPI_WTIME()-tio(1))        


!        CHECK STOPPING CRITERIA
         if (time .gt. tmax .or. timewc .ge. twallmax)  go to 200


!        NEW TIMESTEP
         
         ttstep(1) = MPI_WTIME()
         if(idebug.eq.1.and.myid.eq.0) print*,'DEBUG: to call tstep'
         call tstep
         ttstep(2) = ttstep(2) + (MPI_WTIME()-ttstep(1))         

!       TIMING / CYCLE COUNTING
 
         tio(1) = MPI_WTIME()
         
         tcycle(2) = MPI_WTIME()
         ttotal(2) = tcycle(2)
         ntcycle   = ntcycle + 1
         
         tio(2) = tio(2) + (MPI_WTIME()-tio(1))

!       STDOUT REPORT

         if (mod(nstep,itstp).eq.0.or.ncycl.eq.nbegin) call report
         
        
      enddo
!     --------------------
!     || END HYDRO LOOP ||
!     --------------------



200   continue
!     -------------------------
!     ||      END OF RUN     ||
!     -------------------------
      
      if(timewc .ge. twallmax) then 
         if(ilog.eq.1) print*,'MSG(prompi): timewc .ge. timewallmax'
         call dumpdata(0)
      endif
      if(time .ge. tmax) then 
         if(ilog.eq.1) print*,'MSG(prompi): time .ge. tmax'
         call dumpdata(0)
      endif

      ! CLOSE LOG FILES
      
      if(ilog.eq.1) close(nthlog)      
      
      ! SYNC MPI BEFORE FINALIZE
      call MPI_BARRIER(commcart,ierr)
      if(ilog.eq.1) then
         print*,'MSG(PROMPI): Normal termination in prompi'
         print*,'MSG(PROMPI): call MPI_FINALIZE(ierr)'
      endif
      call MPI_FINALIZE(ierr)
 
      open (unit=777,file='run_complete_file')
      write (777,*) "successful run"
      close(777)
      !stop 'MSG(prompi): END OF SUCCESSFUL RUN'
      
    end program prompi




!-------------------------------------------------------------------
!     
!     DEVELOPMENT NOTES AND BUG FIXES
!     
!-------------------------------------------------------------------
!     It is capable of treating the following additional physics:
!     
!     (a)  a general equation of state by a Riemann solver for real
!     gases according to Colella and Glaz (JCP, 59 (1985), 264).
!     The equation of state uses a linear combination of
!     (1) radiation, (2) ideal ion gas, (3) Arnett's tabular
!     eos for e-e+ with hashed lookup
!     
!     (b)  a nuclear reaction network; the coupling of network and
!     hydro was done as proposed by Mueller (A&A, 162 (1986),103).
!     This has been modified to use Arnett's alpha chain
!-------------------------------------------------------------------
!
!     Meakin's devel log:
!     
!     1. parallelized code using MPI, April 2003.
!     2. updated artificial viscoity, April 2003.
!     3. implicit and explicit temperature diffusion solvers, May 2003.
!     (cndiff, and ediff: see those subroutines for documentation)
!     4. parallel diffusion solver replaced with diffusion.f and shareplane3d.f
!     Dec. 2003.
!     5. code broken into directory structure based on phsyics, Jan. 2004.
!     
!..   revisions prior to 9-30-99:
!     it can read Arnett's 1D initial models.
!     it has spherically symmetric Newtonian gravity.
!     it can have a finite inner radius and mass.
!-----------------------------------------------------------------------
!     This version contains a number of bug fixes (BAF 3/17/90)
!     1.  The calculation of dloga in subroutine states was corrected
!     to make the value symmetric about a reflecting boundary
!     2.  The calculation of scrch1 in subroutine monot was changed
!     to remove an asymmetry about a reflecting boundary
!     3.  The values of uav and urel at a reflecting boundary were
!     forced to be zero at the end of subroutine riemann
!     4.  Calls to eos3d were added after sweeping in each direction
!     Additional bug fixes (EWM 4/14/90)
!     1.  In subroutine bndry loop 416 the array xn(9-1,n) had to
!     be changed into xn(9-i,n).
!     2.  In subroutine bndry loop 546 the array xnin(i-nzn) had to
!     be changed into xn(i-nzn,n).
!     3.  In subroutine hydrow in loop 212 the array scrch1 was not
!     correctly defined for 2-d problems and xyzswp=1.
!     4.  In subroutine flaten in loops 20 (last statement only) and
!     25 (arguments of the amax-function) the array shockd had to
!     be changed into shockd1 .
!-----------------------------------------------------------------------
!     Moving grid added 4/19/90 by BAF
!     Bug fixes concerning grid velocity  (EWM 6/18/90)
!     1.  In subroutine hydrow the coordinates of the zone centers
!     where not updated, if the grid was moved. This is now done
!     in loop 15.
!=======================================================================
