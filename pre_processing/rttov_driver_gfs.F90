!-------------------------------------------------------------------------------
! Name: rttov_driver_gfs.F90
!
! Purpose:
! Initialise and run RTTOV on the profiles contained in preproc_prtm and output
! the results into three NetCDF files (LWRTM, SWRTM, PRTM).
! This version is for atmosphere data from NOAA's GFS
!
! Description and Algorithm details:
! 1)  Select the appropriate coefficient file for the sensor.
! 2)  Initialise options structure.
! 3)  Write out details of long and shortwave channels.
! 4)  Build the profiles structure from preproc_prtm. This currently neglects
!     surfacetype and the details required for the addsolar option.
! 5)  Write the profiles structure to the PRTM output file.
! 6)  Loop of long and shortwave.
! 7)     Set up coefficients.
! 8)     Allocate channel and emissivity arrays.
! 9)     Read RTTOV emissivity atlas. (Consider inputting emissivity.)
! 10)    Allocate radiance and transmission structures.
! 11)    Perform RTTOV direct calculation
! 12)    Reformat RTTOV output into a multidimensional array and perform
!        calculations to determine required above/below cloud fields.
!        If shortwave, apply airmass correction.
! 13)    Write results to appropriate output file.
! 14)    Deallocate arrays and structures.
!
! Arguments:
! Name           Type   In/Out/Both Description
! ------------------------------------------------------------------------------
! coef_path      string in   Folder containing RTTOV coefficient files
! emiss_path     string in   Folder containing MODIS monthly average emissivity
! sensor         string in   Name of sensor.
! platform       string in   Name of satellite platform.
! preproc_dims   struct both Summary of preprocessing grid definitions
! preproc_geoloc struct in   Summary of preprocessing lat/lon
! preproc_geo    struct in   Summary of preprocessing geometry
! preproc_prtm   struct both Summary of profiles and surface fields
! netcdf_info    struct both Summary of NCDF file properties.
! channel_info   struct in   Structure summarising the channels to be processed
! year           sint   in   Year of observation (4 digit)
! month          sint   in   Month of year (1-12)
! day            siny   in   Day of month (1-31)
! verbose        logic  in   T: print status information; F: don't
!
! History:
! 2017/02/05, SP: Initial version
! 2017/02/25, SP: Update to RTTOV v12.1 (EKWork)
!
! $Id$
!
! Bugs:
! - Emissivity is read in elsewhere in the code but not utilised here.
! - BRDF not yet implemented here, so RTTOV internal calculation used.
! - Possible issue with conversion from layers to levels.
!-------------------------------------------------------------------------------

module rttov_driver_gfs_m

implicit none

contains

subroutine rttov_driver_gfs(coef_path,emiss_path,sensor,platform,preproc_dims, &
     preproc_geoloc,preproc_geo,preproc_prtm,preproc_surf,netcdf_info, &
     channel_info,year,month,day,use_modis_emis,verbose)

   use channel_structures_m
   use netcdf_output_m
   use orac_ncdf_m
   use preproc_constants_m
   use preproc_structures_m

   ! rttov_const contains useful RTTOV constants
   use rttov_const, only: &
        errorstatus_success, errorstatus_fatal, &
        zenmax, zenmaxv9

   ! rttov_types contains definitions of all RTTOV data types
   use rttov_types, only: &
       rttov_options,     &
       rttov_coefs,       &
       rttov_chanprof,    &
       rttov_profile,     &
       rttov_emissivity,  &
       rttov_reflectance, &
       rttov_transmission,&
       rttov_radiance,    &
       rttov_radiance2,   &
       rttov_traj

  use mod_rttov_emis_atlas, only : &
        rttov_emis_atlas_data, &
        atlas_type_ir, atlas_type_mw

   ! jpim, jprb and jplm are the RTTOV integer, real and logical KINDs
   use parkind1, only: jpim, jprb, jplm

   implicit none

#include "rttov_alloc_prof.interface"
#include "rttov_read_coefs.interface"
#include "rttov_setup_emis_atlas.interface"
#include "rttov_get_emis.interface"
#include "rttov_alloc_rad.interface"
#include "rttov_alloc_transmission.interface"
#include "rttov_alloc_traj.interface"
#include "rttov_direct.interface"
#include "rttov_deallocate_emis_atlas.interface"
#include "rttov_dealloc_coefs.interface"

   ! Arguments
   character(len=path_length),     intent(in)    :: coef_path
   character(len=path_length),     intent(in)    :: emiss_path
   character(len=sensor_length),   intent(in)    :: sensor
   character(len=platform_length), intent(in)    :: platform
   type(preproc_dims_t),           intent(in)    :: preproc_dims
   type(preproc_geoloc_t),         intent(in)    :: preproc_geoloc
   type(preproc_geo_t),            intent(in)    :: preproc_geo
   type(preproc_prtm_t),           intent(inout) :: preproc_prtm
   type(preproc_surf_t),           intent(in)    :: preproc_surf
   type(netcdf_output_info_t),     intent(inout) :: netcdf_info
   type(channel_info_t),           intent(in)    :: channel_info
   integer(kind=sint),             intent(in)    :: year, month, day
   logical,                        intent(in)    :: use_modis_emis
   logical,                        intent(in)    :: verbose

   ! RTTOV in/outputs
   type(rttov_options)                  :: opts
   type(rttov_coefs)                    :: coefs
   type(rttov_emis_atlas_data)          :: emis_atlas
   type(rttov_chanprof),    allocatable :: chanprof(:)
   type(rttov_profile),     allocatable :: profiles(:)
   logical(kind=jplm),      allocatable :: calcemis(:)
   type(rttov_emissivity),  allocatable :: emissivity(:)
   real(kind=jprb),         allocatable :: emis_data(:)
   type(rttov_transmission)             :: transmission
   type(rttov_radiance)                 :: radiance
   type(rttov_radiance2)                :: radiance2
   type(rttov_traj)                     :: traj

   ! RTTOV variables
   integer(kind=jpim)                   :: stat
   integer(kind=jpim)                   :: nprof, nevals, imonth
   integer(kind=jpim)                   :: nlevels, nlayers
   integer(kind=jpim),      allocatable :: input_chan(:)
   logical                              :: write_rttov

   ! Loop variables
   integer(kind=lint)                   :: i, j
   integer(kind=lint)                   :: i_, j_
   integer(kind=lint)                   :: i_coef
   integer(kind=lint)                   :: count, nchan
   integer(kind=lint)                   :: idim, jdim

   ! Coefficient file selection
   character(len=file_length)           :: coef_file
   character(len=path_length)           :: coef_full_path

   ! Scratch variables
   integer(kind=lint),      allocatable :: dummy_lint_1dveca(:)
   integer(kind=lint),      allocatable :: dummy_lint_1dvecb(:)
   real(kind=sreal),        allocatable :: dummy_sreal_1dveca(:)

   ! Useful aliases
   integer,                 parameter   :: ALLOC=1, DEALLOC=0

   ! View variables
   integer(kind=sint)                   :: cview
   integer,                 allocatable :: chan_pos(:)
   real                                 :: p_0, sec_vza, lambda, tau_ray_0, &
                                           tau_ray_p


   if (verbose) write(*,*) '<<<<<<<<<<<<<<< Entering rttov_driver()'

   if (verbose) write(*,*) 'coef_path: ', trim(coef_path)
   if (verbose) write(*,*) 'emiss_path: ', trim(emiss_path)
   if (verbose) write(*,*) 'sensor: ', trim(sensor)
   if (verbose) write(*,*) 'platform: ', trim(platform)
   if (verbose) write(*,*) 'Date: ', year, month, day
#ifdef NEW_RTTOV
   if (verbose) write(*,*) 'Using new RTTOV version (>11.2)'
#endif


   ! Determine coefficient filename (Vis/IR distinction made later)
   select case (trim(sensor))
   case('ATSR2')
      coef_file = 'rtcoef_ers_2_atsr.dat'
   case('AATSR')
      coef_file = 'rtcoef_envisat_1_atsr.dat'
   case('AHI')
      if (trim(platform) == 'Himawari') then
         coef_file = 'rtcoef_himawari_8_ahi.dat'
      else
         write(*,*) 'ERROR: rttov_driver(): Invalid HIMAWARI platform: ', &
                    trim(platform)
         stop error_stop_code
      end if
   case('AVHRR')
      if (index(platform,'noaa') >= 1) then
         if(platform(5:5) == '1') then
            coef_file = 'rtcoef_noaa_'//platform(5:6)//'_avhrr.dat'
          else
            coef_file = 'rtcoef_noaa_'//platform(5:5)//'_avhrr.dat'
          end if
      else if (index(platform,'metop') >= 1) then
         coef_file = 'rtcoef_metop_'//platform(6:7)//'_avhrr.dat'
      else
         write(*,*) 'ERROR: rttov_driver(): Invalid AVHRR platform: ', &
                    trim(platform)
         stop error_stop_code
      end if
   case('MODIS')
      if (trim(platform) == 'TERRA') then
         coef_file = 'rtcoef_eos_1_modis.dat'
      else if (trim(platform) == 'AQUA') then
         coef_file = 'rtcoef_eos_2_modis.dat'
      else
         write(*,*) 'ERROR: rttov_driver(): Invalid MODIS platform: ', &
                    trim(platform)
         stop error_stop_code
      end if
   case('SEVIRI')
      if (trim(platform) == 'MSG1') then
         coef_file = 'rtcoef_msg_1_seviri.dat'
      else if (trim(platform) == 'MSG2') then
         coef_file = 'rtcoef_msg_2_seviri.dat'
      else if (trim(platform) == 'MSG3') then
         coef_file = 'rtcoef_msg_3_seviri.dat'
      else if (trim(platform) == 'MSG4') then
         coef_file = 'rtcoef_msg_4_seviri.dat'
      else
         write(*,*) 'ERROR: rttov_driver(): Invalid SEVIRI platform: ', &
                    trim(platform)
         stop error_stop_code
      end if
   case('SLSTR')
      if (trim(platform) == 'Sentinel-3') then
         coef_file = 'rtcoef_sentinel3_1_slstr.dat'
      else
         write(*,*) 'ERROR: rttov_driver(): Invalid SLSTR platform: ', &
                    trim(platform)
         stop error_stop_code
      end if
   case('VIIRS')
      if (trim(platform) == 'SuomiNPP') then
         coef_file = 'rtcoef_jpss_0_viirs.dat'
      else
         write(*,*) 'ERROR: rttov_driver(): Invalid VIIRS platform: ', &
                    trim(platform)
         stop error_stop_code
      end if
   case default
      write(*,*) 'ERROR: rttov_driver(): Invalid sensor: ', trim(sensor)
      stop error_stop_code
   end select

   if (verbose) write(*,*) 'RTTOV coef file: ', trim(coef_file)


   ! Initialise options structure (leaving default settings be)
   opts % interpolation % addinterp = .true.  ! Interpolate input profile
   ! Removed as occassionally returns negative ozone at 0.005 hPa
    opts % interpolation % reg_limit_extrap = .true.  ! Extrapolate to 0.5 Pa
   opts % config % do_checkinput = .false. ! necessary due to negative
     ! extrapolated values; from RTTOV 11 homepage: turns off RTTOV's internal
     ! checking for unphysical profile values and values outside the
     ! regression limits (NB by doing this the extrapolated values outside
     ! the regression limits will be reset to the limits: it will not result
     ! in unphysical extrapolated profile values being used)
   opts % rt_all % use_q2m   = .false. ! Do not use surface humidity
   opts % rt_all % addrefrac = .true.  ! Include refraction in path calc
   opts % rt_ir % addsolar   = .false. ! Do not include reflected solar
   opts % rt_ir % ozone_data = .true.  ! Include ozone profile
   opts % config % verbose   = .false. ! Display only fatal error messages

!   opts%config%do_checkinput=.true.

   if (verbose) write(*,*) 'Write static information to the output files'

   ! Write LW channel information
   if (channel_info%nchannels_lw /= 0) then
      allocate(dummy_lint_1dveca(channel_info%nchannels_lw))
      allocate(dummy_lint_1dvecb(channel_info%nchannels_lw))
      allocate(dummy_sreal_1dveca(channel_info%nchannels_lw))
      count=0
      do i=1,channel_info%nchannels_total
         if (channel_info%channel_lw_flag(i) == 1) then
            count = count + 1
            dummy_lint_1dveca(count)  = i
            dummy_lint_1dvecb(count)  = channel_info%channel_ids_instr(i)
            dummy_sreal_1dveca(count) = channel_info%channel_wl_abs(i)
         end if
      end do

      call nc_write_array(netcdf_info%ncid_lwrtm, 'lw_channel_abs_ids', &
              netcdf_info%vid_lw_channel_abs_ids, dummy_lint_1dveca, &
              1, 1, channel_info%nchannels_lw)
      call nc_write_array(netcdf_info%ncid_lwrtm, 'lw_channel_instr_ids', &
              netcdf_info%vid_lw_channel_instr_ids, dummy_lint_1dvecb, &
              1, 1, channel_info%nchannels_lw)
      call nc_write_array(netcdf_info%ncid_lwrtm, 'lw_channel_wvl', &
              netcdf_info%vid_lw_channel_wvl, dummy_sreal_1dveca, &
              1, 1, channel_info%nchannels_lw)

      deallocate(dummy_lint_1dveca)
      deallocate(dummy_lint_1dvecb)
      deallocate(dummy_sreal_1dveca)
   end if


   ! Write SW channel information
   if (channel_info%nchannels_sw /= 0) then
      allocate(dummy_lint_1dveca(channel_info%nchannels_sw))
      allocate(dummy_lint_1dvecb(channel_info%nchannels_sw))
      allocate(dummy_sreal_1dveca(channel_info%nchannels_sw))
      count=0
      do i=1,channel_info%nchannels_total
         if (channel_info%channel_sw_flag(i) == 1) then
            count = count + 1
            dummy_lint_1dveca(count)  = i
            dummy_lint_1dvecb(count)  = channel_info%channel_ids_instr(i)
            dummy_sreal_1dveca(count) = channel_info%channel_wl_abs(i)
         end if
      end do

      call nc_write_array(netcdf_info%ncid_swrtm, 'sw_channel_abs_ids', &
              netcdf_info%vid_sw_channel_abs_ids, dummy_lint_1dveca, &
              1, 1, channel_info%nchannels_sw)
      call nc_write_array(netcdf_info%ncid_swrtm, 'sw_channel_instr_ids', &
              netcdf_info%vid_sw_channel_instr_ids, dummy_lint_1dvecb, &
              1, 1, channel_info%nchannels_sw)
      call nc_write_array(netcdf_info%ncid_swrtm, 'sw_channel_wvl', &
              netcdf_info%vid_sw_channel_wvl, dummy_sreal_1dveca, &
              1, 1, channel_info%nchannels_sw)

      deallocate(dummy_lint_1dveca)
      deallocate(dummy_lint_1dvecb)
!     deallocate(dummy_sreal_1dveca)
   end if


   ! Allocate input profile structures (coefs struct not required as addclouds
   ! and addaerosl not set)
   if (verbose) write(*,*) 'Allocate profile structure'

   nprof   = preproc_dims%xdim * preproc_dims%ydim
   nlayers = preproc_dims%kdim - 1
   nlevels = preproc_dims%kdim
   allocate(profiles(nprof))
   call rttov_alloc_prof(stat, nprof, profiles, nlevels, opts, &
        ALLOC, init=.true._jplm)
   if (stat /= errorstatus_success)  then
      write(*,*) 'ERROR: rttov_alloc_prof(), errorstatus = ', stat
      stop error_stop_code
   end if

   profiles%id = 'standard'

   ! Copy preprocessing grid data into RTTOV profile structure
   ! Create a lowest layer from the surface properties
   count = 0
   do jdim=preproc_dims%min_lat,preproc_dims%max_lat
      do idim=preproc_dims%min_lon,preproc_dims%max_lon
         count = count + 1

         ! If using RTTOV version 11.3 or greater then
         ! set gas units to 1, specifying gas input in kg/kg
         ! (2 = ppmv moist air, 1 = kg/kg, 0 = old method, -1 = ppmv dry air)
#ifdef NEW_RTTOV
         profiles(count)%gas_units = 1
#endif
         ! Profile information
         profiles(count)%p(:) = preproc_prtm%pressure(idim,jdim,:)
         profiles(count)%t(:) = preproc_prtm%temperature(idim,jdim,:)
         ! convert from kg/kg to ppmv by multiplying with dry air molecule
         ! weight(28.9644)/molecule weight of gas (e.g. o3  47.9982)*1.0E6
#ifdef NEW_RTTOV
         profiles(count)%q(:) = &
              preproc_prtm%spec_hum(idim,jdim,:)
         profiles(count)%o3(:) = &
              preproc_prtm%ozone(idim,jdim,:)
#else
         profiles(count)%q(:) = &
              preproc_prtm%spec_hum(idim,jdim,:) * q_mixratio_to_ppmv
         profiles(count)%o3(:) = &
              preproc_prtm%ozone(idim,jdim,:) * o3_mixratio_to_ppmv
#endif
         ! Surface information
         profiles(count)%s2m%p = exp(preproc_prtm%lnsp(idim,jdim)) * pa2hpa
         profiles(count)%s2m%t = preproc_prtm%temp2(idim,jdim)
         profiles(count)%s2m%u = preproc_prtm%u10(idim,jdim)
         profiles(count)%s2m%v = preproc_prtm%v10(idim,jdim)
         profiles(count)%s2m%wfetc = 100000.0
         profiles(count)%p(nlevels) = profiles(count)%s2m%p
!         profiles(count)%t(nlevels) = preproc_prtm%skin_temp(idim,jdim)

         ! These features currently disabled and so do not need to be input
         profiles(count)%cfraction = 0.
!        profiles(count)%ctp   = profiles(count)%p(profiles(count)%nlayers)
!        profiles(count)%s2m%q = profiles(count)%q(profiles(count)%nlayers)
!        profiles(count)%s2m%o = profiles(count)%o3(profiles(count)%nlayers)

         profiles(count)%skin%t = preproc_prtm%skin_temp(idim,jdim)
         ! Force land emissivity from the atlas. !!CONSIDER REVISION!!
         profiles(count)%skin%surftype  = 0
         profiles(count)%skin%watertype = 1

         profiles(count)%date(1) = year
         profiles(count)%date(2) = month
         profiles(count)%date(3) = day
         profiles(count)%elevation = 0. ! One day, we will do something here
         profiles(count)%latitude  = preproc_geoloc%latitude(jdim)
         ! Manual may say this is 0-360, but src/emsi_atlas/mod_iratlas.F90
         ! line 790 disagrees
         profiles(count)%longitude = preproc_geoloc%longitude(idim)
         ! Use poor man's approach to snow fraction
         if (preproc_prtm%snow_albedo(idim,jdim) > 0.) &
              profiles(count)%skin%snow_fraction = 1.

         ! Write profiles structure to PRTM file (array operations needed to
         ! recast structure in form nc_write_array recognises)
         i_ = idim - preproc_dims%min_lon + 1
         j_ = jdim - preproc_dims%min_lat + 1
         call nc_write_array(netcdf_info%ncid_prtm, 'lon_rtm', &
              netcdf_info%vid_lon_pw, &
              (/profiles(count)%longitude/), &
              1, i_, 1)
         call nc_write_array(netcdf_info%ncid_prtm, 'lat_rtm', &
              netcdf_info%vid_lat_pw, &
              (/profiles(count)%latitude/), &
              1, j_, 1)
         call nc_write_array(netcdf_info%ncid_prtm, 'pprofile_rtm', &
              netcdf_info%vid_pprofile_lev_pw, &
              reshape(profiles(count)%p, (/nlevels,1,1/)), &
              1, 1, nlevels, 1, i_, 1, 1, j_, 1)
         call nc_write_array(netcdf_info%ncid_prtm, 'tprofile_rtm', &
              netcdf_info%vid_tprofile_lev_pw, &
              reshape(profiles(count)%t, (/nlevels,1,1/)), &
              1, 1, nlevels, 1, i_, 1, 1, j_, 1)
         call nc_write_array(netcdf_info%ncid_prtm, 'hprofile_rtm', &
              netcdf_info%vid_hprofile_lev_pw, &
              reshape(preproc_prtm%phi_lev(idim, jdim,:), &
              (/nlevels,1,1/)), 1, 1, nlevels, 1, i_, 1, 1, j_, 1)
         call nc_write_array(netcdf_info%ncid_prtm, 'qprofile_rtm', &
              netcdf_info%vid_qprofile_lev_pw, &
              reshape(profiles(count)%q, (/nlevels,1,1/)), &
              1, 1, nlevels, 1, i_, 1, 1, j_, 1)
         call nc_write_array(netcdf_info%ncid_prtm, 'o3profile_rtm', &
              netcdf_info%vid_o3profile_lev_pw, &
              reshape(profiles(count)%o3, (/nlevels,1,1/)), &
              1, 1, nlevels, 1, i_, 1, 1, j_, 1)
      end do
   end do

   ! Write fields not in profiles structure
!  call nc_write_array(netcdf_info%ncid_prtm, 'lsf_rtm', &
!       netcdf_info%vid_lsf_pw, &
!       preproc_prtm%land_sea_mask, &
!       1, 1, preproc_dims%xdim, 1, 1, preproc_dims%ydim)

   ! Do RTTOV calculations for long and shortwave in turn
   if (verbose) write(*,*) 'Do RTTOV calculations'

   ! Loop over view geometries
   do cview=1,channel_info%nviews
      if (verbose) write(*,*) ' - Calculating for viewing geometry number', cview

      count = 0
      do jdim=preproc_dims%min_lat,preproc_dims%max_lat
         do idim=preproc_dims%min_lon,preproc_dims%max_lon
            count = count + 1
            profiles(count)%zenangle = preproc_geo%satza(idim,jdim,cview)
         end do
      end do

      do i_coef=1,2
         ! Set factors that differ between long and shortwave
         if (i_coef == 1) then
            ! Longwave
            nchan = 0

            ! Loop to determine how many LW channels exist with a given view
            do i_=1,channel_info%nchannels_lw
               if (channel_info%lw_view_ids(i_) == cview) nchan = nchan + 1
            end do

            if (nchan == 0) cycle

            allocate(input_chan(nchan))
            allocate(chan_pos(nchan))

            j_ = 1
            do i_=1,channel_info%nchannels_lw
               if (channel_info%lw_view_ids(i_) == cview) then
                  chan_pos(j_) = i_
                  input_chan(j_) = channel_info%channel_ids_rttov_coef_lw(i_)
                  j_ = j_ + 1
               end if
            end do

            ! This assumes the recommended structure of the RTTOV coef library
            coef_full_path = trim(adjustl(coef_path))//'/rttov7pred54L/'// &
                 trim(adjustl(coef_file))
         else
            ! Shortwave
            nchan = 0

            ! Loop to determine how many SW channels exist with a given view
            do i_=1,channel_info%nchannels_sw
               if (channel_info%sw_view_ids(i_) == cview) nchan = nchan + 1
            end do

            if (nchan == 0) cycle

            allocate(input_chan(nchan))
            allocate(chan_pos(nchan))

            j_ = 1
            do i_=1,channel_info%nchannels_sw
               if (channel_info%sw_view_ids(i_) == cview) then
                  chan_pos(j_) = i_
                  input_chan(j_) = channel_info%channel_ids_rttov_coef_sw(i_)
                  j_ = j_ + 1
               end if
            end do

            coef_full_path = trim(adjustl(coef_path))//'/rttov9pred54L/'// &
                 trim(adjustl(coef_file))
         end if

         if (verbose) write(*,*) 'Read coefficients'
         call rttov_read_coefs(stat, coefs, opts, form_coef='formatted', &
              channels=input_chan, file_coef=coef_full_path)
         if (stat /= errorstatus_success) then
            write(*,*) 'ERROR: rttov_read_coefs(), errorstatus = ', stat
            stop error_stop_code
         end if

         ! Force all SW channels to be processed
         if (i_coef == 2) coefs%coef%ss_val_chn = 1

         if (verbose) write(*,*) 'Allocate channel and emissivity arrays'
         allocate(chanprof(nchan))
         allocate(emissivity(nchan))
         allocate(emis_data(nchan))
         allocate(calcemis(nchan))

         chanprof%prof = 1
         do j=1,nchan
            chanprof(j)%chan = j
         end do

         if (verbose) write(*,*) 'Allocate RTTOV structures'
         call rttov_alloc_rad(stat, nchan, radiance, nlevels, ALLOC, radiance2, &
              init=.true._jplm)
         if (stat /= errorstatus_success) then
            write(*,*) 'ERROR: rttov_alloc_rad(), errorstatus = ', stat
            stop error_stop_code
         end if
         call rttov_alloc_transmission(stat, transmission, nlevels, nchan, &
              ALLOC, init=.true._jplm)
         if (stat /= errorstatus_success) then
            write(*,*) 'ERROR: rttov_alloc_transmission(), errorstatus = ', stat
            stop error_stop_code
         end if
         call rttov_alloc_traj(stat, 1, nchan, opts, nlevels, coefs, &
              ALLOC, traj=traj)
         if (stat /= errorstatus_success) then
            write(*,*) 'ERROR: rttov_alloc_traj(), errorstatus = ', stat
            stop error_stop_code
         end if

         if (verbose) write(*,*) 'Fetch emissivity atlas'
         imonth=month
         call rttov_setup_emis_atlas(stat, opts, imonth, atlas_type_ir, emis_atlas, &
              coefs=coefs, path=emiss_path)
         if (stat /= errorstatus_success) then
            write(*,*) 'ERROR: rttov_setup_emis_atlas(), errorstatus = ', stat
            stop error_stop_code
         end if

         ! Loop over profiles (as the conditions for processing LW and SW are
         ! different, we can't just pass the whole array)
         count = 0
         if (verbose) write(*,*) 'Run RTTOV'
         do jdim=preproc_dims%min_lat,preproc_dims%max_lat
            do idim=preproc_dims%min_lon,preproc_dims%max_lon
               count = count + 1

               ! Process points that contain information and satisfy the zenith
               ! angle restrictions of the coefficient file
               if ((i_coef == 1 .and. &
                    preproc_dims%counter_lw(idim,jdim,cview) > 0 .and. &
                    profiles(count)%zenangle <= zenmax) .or. &
                   (i_coef == 2 .and. &
                    preproc_dims%counter_sw(idim,jdim,cview) > 0 .and. &
                    profiles(count)%zenangle <= zenmaxv9)) then

                  if (i_coef == 1) then
                     ! Fetch emissivity from atlas
                     call rttov_get_emis(stat, opts, chanprof, &
                          profiles(count:count), coefs, emis_atlas,emis_data)
                     if (stat /= errorstatus_success) then
                        write(*,*) 'ERROR: rttov_get_emis(), errorstatus = ', &
                             stat
                        stop error_stop_code
                     end if
                     emissivity%emis_in = emis_data

                     ! Fetch emissivity from the MODIS CIMSS emissivity product
                     if (use_modis_emis) then
                        where (preproc_surf%emissivity(idim,jdim,:) /= &
                             sreal_fill_value)
                           emissivity%emis_in = &
                                preproc_surf%emissivity(idim,jdim,:)
                        end where
                     end if

                     calcemis = emissivity%emis_in <= dither
                  end if

                  ! Call RTTOV for this profile
                  call rttov_direct(stat, chanprof, opts, &
                       profiles(count:count), coefs, transmission, radiance, &
                       radiance2, calcemis, emissivity, traj=traj)
                  if (stat /= errorstatus_success) then
                     write(*,*) 'ERROR: rttov_direct(), errorstatus = ', stat
                     stop error_stop_code
                  end if

                  write_rttov = .true.
               else
                  write_rttov = .false.
               end if

               ! Remove the Rayleigh component from the RTTOV tranmittances.
               ! (Private communication from Philip Watts.)
               if (i_coef == 2) then
                  p_0 = 1013.

                  sec_vza = 1. / cos(profiles(count)%zenangle * d2r)

                  do i_ = 1, nchan
                     ! Rayleigh optical thickness for the atmosphere down to 1013
                     ! hPa (Hansen and Travis, 1974)
                     lambda = dummy_sreal_1dveca(i_)
                     tau_ray_0 = .008569 * lambda**(-4) * &
                          (1. + .0113 * lambda**(-2) + .00013 * lambda**(-4))

                     do j_ = 1, nlevels
                        ! Pressure and path dependent Rayleigh optical thickness
                        tau_ray_p = tau_ray_0 * profiles(count)%p(j_) / p_0 * &
                             sec_vza

                        ! Corrected level transmittances
                        transmission%tau_levels(j_, i_) = &
                             transmission%tau_levels(j_, i_) / exp(-tau_ray_p)
                     end do

                     ! Corrected total transmittances
                     transmission%tau_total(i_) = &
                          transmission%tau_total(i_) / exp(-tau_ray_p)
                  end do
               end if

               ! Reformat and write output to NCDF files
               if (i_coef == 1) then
                  do i_=1,nchan
                     call write_ir_rttov(netcdf_info, &
                          idim-preproc_dims%min_lon+1, &
                          jdim-preproc_dims%min_lat+1, &
                          profiles(count)%nlevels, emissivity, transmission, &
                          radiance, radiance2, write_rttov, chan_pos(i_), i_)
                  end do
               else
                  do i_=1,nchan
                     call write_solar_rttov(netcdf_info, coefs, &
                          idim-preproc_dims%min_lon+1, &
                          jdim-preproc_dims%min_lat+1, &
                          profiles(count)%nlevels, profiles(count)%zenangle, &
                          transmission, write_rttov, chan_pos(i_), i_)
                  end do
               end if
            end do
         end do


         if (verbose) write(*,*) 'Deallocate structures'

         call rttov_deallocate_emis_atlas(emis_atlas)
         call rttov_alloc_traj(stat, 1, nchan, opts, nlevels, coefs, DEALLOC, &
              traj)
         call rttov_alloc_transmission(stat, transmission, nlevels, nevals, &
              DEALLOC)
         call rttov_alloc_rad(stat, nevals, radiance, nlevels, DEALLOC, &
              radiance2)
         call rttov_dealloc_coefs(stat, coefs)

         deallocate(input_chan)
         deallocate(chanprof)
         deallocate(emissivity)
         deallocate(emis_data)
         deallocate(calcemis)
         deallocate(chan_pos)
      end do !coef loop
   end do  !view loop

   if (channel_info%nchannels_sw /= 0) then
      deallocate(dummy_sreal_1dveca)
   end if
   call rttov_alloc_prof(stat, nprof, profiles, nlevels, opts, DEALLOC)
   deallocate(profiles)

   if (verbose) write(*,*) '>>>>>>>>>>>>>>> Leaving rttov_driver()'

end subroutine rttov_driver_gfs


#include "write_ir_rttov.F90"
#include "write_solar_rttov.F90"


end module rttov_driver_gfs_m