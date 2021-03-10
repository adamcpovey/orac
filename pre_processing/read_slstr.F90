!-------------------------------------------------------------------------------
! Name: read_slstr.F90
!
! Purpose:
! Module for SLSTR read routines.
!
! History:
! 2016/06/14, SP: First version.
! 2020/05/22, AP: Moved definition of constants here.
!
! Bugs:
! None known.
!-------------------------------------------------------------------------------

module read_slstr_m

   implicit none

contains

! When defined, the offset between the nadir and oblique views is assumed
! to be constant. Otherwise, the two longitude fields are read and compared
! to determine an appropriate offset. The value below is the mode of running
! slstr_get_alignment() on each row of 1000 random SLSTR (A) images. 548 was
! returned in 1e5 cases. 546-551 were returned O(1e3) times, while other values
! in the range 530-570 each appeared < 1e2 times.
#define CONSTANT_OBLIQUE_OFFSET 548

#include "read_slstr_main.F90"
#include "read_slstr_funcs.F90"

end module read_slstr_m
