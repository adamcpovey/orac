!-------------------------------------------------------------------------------
! Name: utils_for_main
!
! Purpose:
!
! History:
! 2015/02/24, Name: Pulled these out of preprocessing_for_orac.F90 to get them
!    module produced interfaces.
!
! $Id$
!-------------------------------------------------------------------------------

module utils_for_main

   implicit none

contains

subroutine handle_parse_error(name)

   use preproc_constants

   implicit none

   character(len=*), intent(in) :: name

   write(*,*) 'ERROR: Error parsing value for: ', trim(name)

   stop error_stop_code

end subroutine handle_parse_error


subroutine parse_required(lun, value, name)

   use parsing
   use preproc_constants

   implicit none

   integer,          intent(in) :: lun
   character(len=*), intent(out):: value
   character(len=*), intent(in) :: name

   character(path_length) :: line

   if (parse_driver(lun, line) /= 0 .or. parse_string(line, value) /= 0) &
      call handle_parse_error(name)

end subroutine parse_required


subroutine parse_optional(label, value, n_channels, channel_ids)

   use parsing
   use preproc_constants

   implicit none

   character(len=*), intent(in)    :: label
   character(len=*), intent(in)    :: value
   integer,          intent(out)   :: n_channels
   integer, pointer, intent(inout) :: channel_ids(:)

   select case (label)
   case('N_CHANNELS')
      if (parse_string(value, n_channels) /= 0) &
         call handle_parse_error(label)
      allocate(channel_ids(n_channels))
   case('CHANNEL_IDS')
      if (n_channels == 0) then
         write(*,*) 'ERROR: must set option n_channels before option channels'
         stop error_stop_code
      endif
      if (parse_string(value, channel_ids) /= 0) &
         call handle_parse_error(label)
   case default
      write(*,*) 'ERROR: Unknown option: ', trim(label)
      stop error_stop_code
   end select

end subroutine parse_optional


integer function parse_logical(string, value) result(status)

   use preproc_constants

   implicit none

   character(len=*), intent(in)  :: string
   logical,          intent(out) :: value

   status = 0

   if (trim(adjustl(string)) .eq. '1' .or.&
       trim(adjustl(string)) .eq. 't' .or. &
       trim(adjustl(string)) .eq. 'true' .or. &
       trim(adjustl(string)) .eq. 'T' .or. &
       trim(adjustl(string)) .eq. 'True') then
        value = .true.
   else if &
      (trim(adjustl(string)) .eq. '0' .or. &
       trim(adjustl(string)) .eq. 'f' .or. &
       trim(adjustl(string)) .eq. 'false' .or. &
       trim(adjustl(string)) .eq. 'F' .or. &
       trim(adjustl(string)) .eq. 'False') then
        value = .false.
   else
        status = -1
   end if

end function parse_logical


function calc_n_chunks(n_segments, segment_starts, segment_ends, &
                       chunk_size) result (n_chunks)

   implicit none

   integer, intent(in) :: n_segments
   integer, intent(in) :: segment_starts(n_segments)
   integer, intent(in) :: segment_ends(n_segments)
   integer, intent(in) :: chunk_size
   integer             :: n_chunks

   integer :: i

   n_chunks = 0

   do i = 1, n_segments
      n_chunks = n_chunks + (segment_ends(i) - segment_starts(i)) / chunk_size + 1
   end do

end function calc_n_chunks


subroutine chunkify(n_segments, segment_starts, segment_ends, &
                    chunk_size, n_chunks, chunk_starts, chunk_ends)

   implicit none

   integer, intent(in)  :: n_segments
   integer, intent(in)  :: segment_starts(n_segments)
   integer, intent(in)  :: segment_ends(n_segments)
   integer, intent(in)  :: chunk_size
   integer, intent(out) :: n_chunks
   integer, intent(out) :: chunk_starts(*)
   integer, intent(out) :: chunk_ends(*)

   integer :: i

   n_chunks = 1

   do i = 1, n_segments
      chunk_starts(n_chunks) = segment_starts(i)

      do while (chunk_starts(n_chunks) + chunk_size .lt. segment_ends(i))
         chunk_ends(n_chunks) = chunk_starts(n_chunks) + chunk_size - 1
         n_chunks = n_chunks + 1
         chunk_starts(n_chunks) = chunk_starts(n_chunks - 1) + chunk_size
      end do

      chunk_ends(n_chunks) = segment_ends(i)

      n_chunks = n_chunks + 1
   end do

   n_chunks = n_chunks - 1

end subroutine chunkify

end module utils_for_main