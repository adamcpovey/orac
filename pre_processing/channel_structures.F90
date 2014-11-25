!-------------------------------------------------------------------------------
! Name: channel_structures.F90
!
! Purpose:
! Define variables types which hold the channel information
!
! Description and Algorithm details:
! PUT EVERYTHING IN HERE NECESSARY FOR CHANNEL INFORMATION!!
! PLEASE COMMENT SUFFICIENTLY TO CLEARLY EXPLAIN THE ROLE OF THE
! INDIVIDUAL INFORMATION!!
!
! Arguments:
! None
!
! History:
! 2012/06/01, MJ: writes initial version.
! 2012/06/18, GT: Made some changes for dual view indexing
! 2012/08/22, GT: Added nview (number of viewing geometries)
! 2014/10/15, GM: Added map_ids_abs_to_ref_band_land and
!    map_ids_abs_to_ref_band_sea and removed channel_proc_flag.
!
! $Id$
!
! Bugs:
! None known.
!-------------------------------------------------------------------------------

module channel_structures

   use preproc_constants

   implicit none

   type channel_info_s

      !total number of channels to be handled in any way, even if not all are
      !processed. Note that multiple views with the same channel are treated as
      !separate channels by the code. Also, you cannot assume that
      !channels_total=nchannels_sw + nchannels_lw as some channels (3.6 microns
      !for eg) are both solar and thermal.
      !So, for example, all channels and dual view for AATSR would mean:
      ! nchannels_total = 14 (7 wavelengths in 2 views)
      ! nchannels_sw    = 10 (first 5 wavelengths have a solar component)
      ! nchannels_lw    = 6  (last 3 wavelengths have a thermal component)
      integer(kind=lint) :: nchannels_total,nchannels_sw,nchannels_lw
      ! Number of different viewing geometries
      integer(kind=lint) :: nviews

      !channel ids (=numbers):
      !wrt original instrument definition
      !Note that these values may well repeat for multi-view instruments, like
      !AATSR: (/ 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7 /)
      integer(kind=lint), dimension(:), pointer :: channel_ids_instr
      !wrt plain numbering 1,2,..... with regard to increasing wavelength and
      !then view
      integer(kind=lint), dimension(:), pointer :: channel_ids_abs

      !wavelength (in micrometers) array wrt to absolute channel numbering
      ! i.e. For all AATSR channels in both views, this would be
      ! (/ 0.55, 0.67, 0.87, 1.6, 3.7, 11, 12, 0.55, 0.67, 0.87, 1.6, 3.7, 11, 12 /)
      real(kind=sreal), dimension(:), pointer ::  channel_wl_abs

      !arrays containing 0/1 flags to identify to which part (sw/lw) of the
      !spectrum they are assigned. could be used to determine the number of
      !channels used as well.
      integer(kind=lint), dimension(:), pointer :: channel_sw_flag
      integer(kind=lint), dimension(:), pointer :: channel_lw_flag

      !channel number wrt its position in the RTTOV coefficient file
      integer(kind=lint), dimension(:), pointer :: channel_ids_rttov_coef_sw
      integer(kind=lint), dimension(:), pointer :: channel_ids_rttov_coef_lw

      !map the abs channel ids to the ancillary reflectance input bands
      integer(kind=lint), dimension(:), pointer :: map_ids_abs_to_ref_band_land
      integer(kind=lint), dimension(:), pointer :: map_ids_abs_to_ref_band_sea

      !arrays containing the viewing geometry index for each channel
      integer(kind=lint), dimension(:), pointer ::  channel_view_ids

   end type channel_info_s

contains

include "allocate_channel_info.F90"
include "deallocate_channel_info.F90"

end module channel_structures