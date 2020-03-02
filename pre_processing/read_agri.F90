!-------------------------------------------------------------------------------
! Name: read_agri.F90
!
! Purpose:
! Module for FY4 AGRI I/O routines.
! To run the preprocessor with AGRI use:
! AGRI as the sensor name (1st line of driver)
! Any 4km file as the l1b filename (2nd line)
! TThe same file as the geo filename (3rd line)
!!
! History:
! 2018/08/14, SP: Initial version.
!
! Bugs:
! Only tested with full disk AGRI images, not China regional coverage..
!-------------------------------------------------------------------------------

module read_agri_m

   implicit none

   private

   public :: read_agri_dimensions, &
             read_agri_data

contains

!-----------------------------------------------------------------------------
! Name: read_agri_dimensions
!
! Purpose:
!
! Description and Algorithm details:
!
! Arguments:
! Name           Type    In/Out/Both Description
! fname          string  in   Full path to one AGRI file
! n_across_track lint    out  Number columns in the AGRI image
! n_along_track  lint    out  Number lines   in the AGRI image
! startx         lint    both First column desired by the caller
! endx           lint    both First line desired by the caller
! starty         lint    both Last column desired by the caller
! endy           lint    both Last line desired by the caller
! verbose        logical in   If true then print verbose information.
!
! Note: startx,endx,starty,endy currently ignored.
! It will always process the full scene. This will be fixed.
!-----------------------------------------------------------------------------
subroutine read_agri_dimensions(fname, n_across_track, n_along_track, &
                                startx, endx, starty, endy, verbose)

    use iso_c_binding
    use orac_ncdf_m
    use preproc_constants_m

    implicit none

    character(path_length), intent(in)    :: fname
    integer(lint),          intent(out)   :: n_across_track, n_along_track
    integer(lint),          intent(inout) :: startx, endx, starty, endy
    logical,                intent(in)    :: verbose

    integer :: fid, ierr

    if (verbose) write(*,*) '<<<<<<<<<<<<<<< read_agri_dimensions()'

    ! Open the file.
    call nc_open(fid, fname, ierr)

    startx = 1
    starty = 1

    endy = nc_dim_length(fid,'lat',verbose)
    endx = nc_dim_length(fid,'lon',verbose)

    n_across_track = endx
    n_along_track = endy

    if (verbose) write(*,*) '>>>>>>>>>>>>>>> read_agri_dimensions()'

end subroutine read_agri_dimensions


subroutine parse_time(intime, year, mon, day, hour, minu)
!-----------------------------------------------------------------------------
! Name: parse_time
!
! Purpose:
! To parse the start and end times from AGRI
!
! Description and Algorithm details:
!
! Arguments:
! Name                Type    In/Out/Both Description
! intime              string  in   A 12 character timestamp
! year                int     out  The year extracted from timestamp
! mon                 int     out  The month extracted from timestamp
! day                 int     out  The day extracted from timestamp
! hour                int     out  The hour extracted from timestamp
! minu                int     out  The minute extracted from timestamp
!-----------------------------------------------------------------------------

    use preproc_constants_m

    implicit none

    character(len=12), intent(in) :: intime
    integer(kind=sint), intent(out):: year
    integer(kind=sint), intent(out):: mon
    integer(kind=sint), intent(out):: day
    integer(kind=sint), intent(out):: hour
    integer(kind=sint), intent(out):: minu

    character(len=4) :: cyear
    character(len=2) :: cmon, cday, chour, cminu

    cyear = trim(adjustl(intime(1:4)))
    cmon = trim(adjustl(intime(5:6)))
    cday = trim(adjustl(intime(7:8)))
    chour = trim(adjustl(intime(9:10)))
    cminu = trim(adjustl(intime(11:12)))

    read(cyear(1:len_trim(cyear)), '(I4)') year
    read(cmon(1:len_trim(cmon)), '(I2)') mon
    read(cday(1:len_trim(cday)), '(I2)') day
    read(chour(1:len_trim(chour)), '(I2)') hour
    read(cminu(1:len_trim(cminu)), '(I2)') minu

end subroutine parse_time


subroutine compute_time(ncid, imager_time, ny)
!-----------------------------------------------------------------------------
! Name: compute_time
!
! Purpose:
! To compute the time array
!
! Description and Algorithm details:
!
! Arguments:
! Name                Type    In/Out/Both Description
! ncid                int     in   The ID of the netCDF file open for reading
! imager_time         struct  out  The output structure
! ny                  int     in   The size of the y dimension of the image
!-----------------------------------------------------------------------------

    use preproc_constants_m
    use imager_structures_m
    use orac_ncdf_m
    use calender_m
    use netcdf

    implicit none

    integer,             intent(in)  :: ncid
    type(imager_time_t), intent(out) :: imager_time
    integer,             intent(in)  :: ny

    ! Time stuff
    character(len=12)  :: start_time
    character(len=12)  :: end_time
    integer(kind=sint) :: st_yr, st_mn, st_dy, st_hr, st_mi
    integer(kind=sint) :: en_yr, en_mn, en_dy, en_hr, en_mi
    real(kind=dreal)   :: jd1, jd2, dfrac1, dfrac2, slo

    ! netCDF stuff
    integer :: ierr

    integer :: j

    ierr = nf90_get_att(ncid, NF90_GLOBAL, 'start_time', start_time)
    if (ierr.ne.NF90_NOERR) then
      write(*,*) 'ERROR: read_agri_data(), ', trim(nf90_strerror(ierr)), &
           ', name: start_time'
      stop -1
    end if
    ierr = nf90_get_att(ncid, NF90_GLOBAL, 'end_time', end_time)
    if (ierr.ne.NF90_NOERR) then
      write(*,*) 'ERROR: read_agri_data(), ', trim(nf90_strerror(ierr)), &
           ', name: end_time'
      stop -1
    end if

    call parse_time(start_time, st_yr, st_mn, st_dy, st_hr, st_mi)
    call parse_time(end_time, en_yr, en_mn, en_dy, en_hr, en_mi)
    call GREG2JD(st_yr, st_mn, st_dy, jd1)
    call GREG2JD(en_yr, en_mn, en_dy, jd2)

    ! Add on a fraction to account for the start / end times
    dfrac1 = (float(st_hr)/24.0) + (float(st_mi)/(24.0*60.0))
    dfrac2 = (float(en_hr)/24.0) + (float(en_mi)/(24.0*60.0))
    jd1 = jd1 + dfrac1
    jd2 = jd2 + dfrac2

    ! Compute linear regression slope
    slo = (jd2-jd1)/ny

    ! Put correct julian date into each location in the time array
    do j=1,ny
      imager_time%time(:,j) = jd1+(slo*float(j))
    end do

end subroutine compute_time


subroutine agri_retr_anc(ncid, imager_angles, imager_geolocation)
!-----------------------------------------------------------------------------
! Name: agri_retr_anc
!
! Purpose:
! To retrieve geodata and angles data for AGRI
!
! Description and Algorithm details:
!
! Arguments:
! Name                Type    In/Out/Both Description
! ncid                int     in   The ID of the netCDF file open for reading
! imager_angles       struct  out  Holds angles data
! imager_geolocation  struct  out  Holds geolocation data
!-----------------------------------------------------------------------------

    use preproc_constants_m
    use imager_structures_m
    use orac_ncdf_m
    use calender_m
    use netcdf

    implicit none

    integer,                    intent(in)  :: ncid
    type(imager_angles_t),      intent(out) :: imager_angles
    type(imager_geolocation_t), intent(out) :: imager_geolocation

    ! Time stuff
    character(len=12)  :: start_time
    character(len=12)  :: end_time
    integer(kind=sint) :: st_yr, st_mn, st_dy, st_hr, st_mi
    integer(kind=sint) :: en_yr, en_mn, en_dy, en_hr, en_mi
    real(kind=dreal)   :: jd1, jd2, dfrac1, dfrac2, slo

    ! netCDF stuff
    integer :: ierr, did

    integer :: j

    ! Get latitude
    ierr=nf90_inq_varid(ncid, 'latitude', did)
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error opening dataset latitude'
        stop error_stop_code
    end if
    ierr = nf90_get_var(ncid, did, imager_geolocation%latitude)
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error reading dataset latitude', ierr
        print*, trim(nf90_strerror(ierr))
        stop error_stop_code
    end if

    ! Get longitude
    ierr=nf90_inq_varid(ncid, 'longitude', did)
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error opening dataset longitude'
        stop error_stop_code
    end if
    ierr = nf90_get_var(ncid, did, imager_geolocation%longitude)
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error reading dataset longitude', ierr
        print*, trim(nf90_strerror(ierr))
        stop error_stop_code
    end if

    ! Get SZA
    ierr=nf90_inq_varid(ncid, 'SZA', did)
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error opening dataset SZA'
        stop error_stop_code
    end if
    ierr = nf90_get_var(ncid, did, imager_angles%solzen(:,:,1))
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error reading dataset SZA', ierr
        print*, trim(nf90_strerror(ierr))
        stop error_stop_code
    end if

    ! Get SAA
    ierr=nf90_inq_varid(ncid, 'SAA', did)
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error opening dataset SAA'
        stop error_stop_code
    end if
    ierr = nf90_get_var(ncid, did, imager_angles%solazi(:,:,1))
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error reading dataset SAA', ierr
        print*, trim(nf90_strerror(ierr))
        stop error_stop_code
    end if

    ! Get VZA
    ierr=nf90_inq_varid(ncid, 'VZA', did)
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error opening dataset VZA'
        stop error_stop_code
    end if
    ierr = nf90_get_var(ncid, did, imager_angles%satzen(:,:,1))
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error reading dataset VZA', ierr
        print*, trim(nf90_strerror(ierr))
        stop error_stop_code
    end if

    ! Get VAA
    ierr=nf90_inq_varid(ncid, 'VAA', did)
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error opening dataset VAA'
        stop error_stop_code
    end if
    ierr = nf90_get_var(ncid, did, imager_angles%satazi(:,:,1))
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_anc(): Error reading dataset VAA', ierr
        print*, trim(nf90_strerror(ierr))
        stop error_stop_code
    end if

    imager_angles%solzen(:,:,1) = abs(imager_angles%solzen(:,:,1))
    imager_angles%satzen(:,:,1) = abs(imager_angles%satzen(:,:,1))

    ! Check units to remove anything that's out-of-range.
    where(imager_geolocation%latitude(:,:)  .gt. 100) &
        imager_geolocation%latitude(:,:) = sreal_fill_value
    where(imager_geolocation%latitude(:,:)  .lt. -100) &
        imager_geolocation%latitude(:,:) = sreal_fill_value
    where(imager_geolocation%longitude(:,:) .gt. 200) &
        imager_geolocation%longitude(:,:) = sreal_fill_value
    where(imager_geolocation%longitude(:,:) .lt. -200) &
        imager_geolocation%longitude(:,:) = sreal_fill_value
    where(imager_angles%solazi(:,:,1)       .gt. 900) &
        imager_angles%solazi(:,:,1) = sreal_fill_value
    where(imager_angles%solzen(:,:,1)       .gt. 900) &
        imager_angles%solzen(:,:,1) = sreal_fill_value
    where(imager_angles%satzen(:,:,1)       .gt. 900) &
        imager_angles%satzen(:,:,1) = sreal_fill_value
    where(imager_angles%satazi(:,:,1)       .gt. 900) &
        imager_angles%satazi(:,:,1) = sreal_fill_value

    ! Rescale zens + azis into correct format

    where(imager_angles%solazi(:,:,1) .ne. sreal_fill_value .and. &
         imager_angles%satazi(:,:,1) .ne. sreal_fill_value)
       imager_angles%relazi(:,:,1) = abs(imager_angles%satazi(:,:,1) - &
            imager_angles%solazi(:,:,1))

        where (imager_angles%relazi(:,:,1) .gt. 180.)
            imager_angles%relazi(:,:,1) = 360. - imager_angles%relazi(:,:,1)
        end where

        imager_angles%relazi(:,:,1) = 180. - imager_angles%relazi(:,:,1)

        imager_angles%solazi(:,:,1) = imager_angles%solazi(:,:,1) + 180.
        where (imager_angles%solazi(:,:,1) .gt. 360.)
            imager_angles%solazi(:,:,1) = imager_angles%solazi(:,:,1) - 360.
        end where
        imager_angles%satazi(:,:,1) = imager_angles%satazi(:,:,1) + 180.
        where (imager_angles%satazi(:,:,1) .gt. 360.)
            imager_angles%satazi(:,:,1) = imager_angles%satazi(:,:,1) - 360.
        end where
    end where

end subroutine agri_retr_anc


subroutine agri_retr_band(ncid, band, iband, solband, imager_measurements)
!-----------------------------------------------------------------------------
! Name: agri_retr_band
!
! Purpose:
! To retrieve one band of AGRI data from a netCDF file
!
! Description and Algorithm details:
!
! Arguments:
! Name                Type    In/Out/Both Description
! ncid                int     in   The ID of the netCDF file open for reading
! band                str     in   The in-file band variable name
! iband               int     in   The band location in the output struct
! solband             int     in   Switch for solar bands
! imager_measurements struct  out  The struct storing the actual data
!-----------------------------------------------------------------------------

    use preproc_constants_m
    use imager_structures_m
    use orac_ncdf_m
    use calender_m
    use netcdf

    implicit none

    integer,                     intent(in)  :: ncid
    character(len=3),            intent(in)  :: band
    integer,                     intent(in)  :: iband
    integer,                     intent(in)  :: solband
    type(imager_measurements_t), intent(out) :: imager_measurements

    ! netCDF stuff
    integer :: ierr, did

    ! Get band
    ierr=nf90_inq_varid(ncid, band, did)
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_band(): Error opening dataset', band
        stop error_stop_code
    end if
    ierr = nf90_get_var(ncid, did, imager_measurements%data(:,:,iband))
    if (ierr.ne.NF90_NOERR) then
        print*, 'ERROR: agri_retr_band(): Error reading dataset', band, ierr
        print*, trim(nf90_strerror(ierr))
        stop error_stop_code
    end if

    ! If it's a solar band then we have to divide by 100 as Satpy refl is in range 0->100
    if (solband .eq. 1) then
        imager_measurements%data(:,:,iband) = imager_measurements%data(:,:,iband) / 100
        ! Check units to remove anything that's out-of-range for solar bands
        where(imager_measurements%data(:,:,iband)  .gt. 2) &
            imager_measurements%data(:,:,iband) = sreal_fill_value
        where(imager_measurements%data(:,:,iband)  .lt. -2) &
            imager_measurements%data(:,:,iband) = sreal_fill_value
    else
        ! Check units to remove anything that's out-of-range for thermal bands
        where(imager_measurements%data(:,:,iband)  .gt. 600) &
            imager_measurements%data(:,:,iband) = sreal_fill_value
        where(imager_measurements%data(:,:,iband)  .lt. 10) &
            imager_measurements%data(:,:,iband) = sreal_fill_value
    end if
end subroutine agri_retr_band


!-----------------------------------------------------------------------------
! Name: read_agri
!
! Purpose:
! To read the requested AGRI data from netcdf-format files.
!
! Description and Algorithm details:
!
! Arguments:
! Name                Type    In/Out/Both Description
! infile              string  in   Full path to any AGRI datafile
! geofile             string  in   Full path to the same file
! imager_geolocation  struct  both Members within are populated
! imager_measurements struct  both Members within are populated
! imager_angles       struct  both Members within are populated
! imager_flags        struct  both Members within are populated
! imager_time         struct  both Members within are populated
! channel_info        struct  both Members within are populated
! verbose             logical in   If true then print verbose information.
!-----------------------------------------------------------------------------
subroutine read_agri_data(infile,imager_geolocation, imager_measurements, &
                          imager_angles, imager_time, channel_info, &
                          global_atts, verbose)

    use iso_c_binding
    use orac_ncdf_m
    use netcdf
    use calender_m
    use channel_structures_m
    use global_attributes_m
    use imager_structures_m
    use preproc_constants_m
    use system_utils_m

    implicit none

    character(len=path_length),  intent(in)    :: infile
    type(imager_geolocation_t),  intent(inout) :: imager_geolocation
    type(imager_measurements_t), intent(inout) :: imager_measurements
    type(imager_angles_t),       intent(inout) :: imager_angles
    type(imager_time_t),         intent(inout) :: imager_time
    type(channel_info_t),        intent(in)    :: channel_info
    type(global_attributes_t),   intent(inout) :: global_atts
    logical,                     intent(in)    :: verbose

    integer(c_int)              :: n_bands
    integer(c_int), allocatable :: band_ids(:)
    integer(c_int), allocatable :: band_units(:)
    integer                     :: startx, nx
    integer                     :: starty, ny

    ! netCDF stuff
    integer                     :: ierr, ncid

    ! Various
    integer                     :: i
    character(len=3)            :: cur_band

    if (verbose) write(*,*) '<<<<<<<<<<<<<<< Entering read_agri_data()'

    ! Figure out the channels to process
    n_bands = channel_info%nchannels_total
    allocate(band_ids(n_bands))
    band_ids = channel_info%channel_ids_instr
    allocate(band_units(n_bands))

    call nc_open(ncid, infile, ierr)

    startx = imager_geolocation%startx
    nx = imager_geolocation%nx
    starty = imager_geolocation%starty
    ny = imager_geolocation%ny

    ! First we sort out the time data
    call compute_time(ncid, imager_time, ny)

    ! Now we load the ancillary data
    call agri_retr_anc(ncid, imager_angles, imager_geolocation)

    do i = 1, n_bands
        write(cur_band,'(i2.2)') band_ids(i)
        cur_band = 'C'//cur_band
        call agri_retr_band(ncid, cur_band, i, channel_info%channel_sw_flag(i), imager_measurements)
    end do

    if (verbose) write(*,*) '>>>>>>>>>>>>>>> Leaving read_agri_data()'

    end subroutine read_agri_data

end module read_agri_m