# Base directory for all libraries and include files taken from the
# environmental variable ORAC_LIBBASE

# Set up libraries and includes
LIBS = -L$(ORAC_LIBBASE)/lib \
       -Wl,-rpath-link=$(ORAC_LIBBASE)/lib \
       -lemosR64 -lemos -lfftw3 \
       -lhdfeos -lGctp \
       -leccodes_f90 -leccodes \
       -lmfhdf -lmfhdf_fortran -lmfhdf_fcstub -lhdf \
       -lnetcdff \
       -lnetcdf \
       -lrttov13_coef_io -lrttov13_emis_atlas -lrttov13_hdf \
       -lrttov13_parallel -lrttov13_main -lrttov13_other \
       -lhdf5 -lhdf5_fortran -lhdf5_hl -lhdf5_hl_fortran \
       -lsz \
       -ljpeg -lm -lz -lstdc++ -lblas -llapack

INC = -I./ \
      -I$(ORAC_LIBBASE)/include \
      -I$(ORAC_LIBBASE)/include/shared \
      -I$(ORAC_LIBBASE)/mod

CINC = -I./ \
       -I$(ORAC_LIBBASE)/include


# Configuration options

# Uncomment if you want to enable OpenMP for RTTOV computations.
INC  += -DINCLUDE_RTTOV_OPENMP

# Uncomment if ATSR support is desired.
LIBS += -lepr_api
INC  += -DINCLUDE_ATSR_SUPPORT
CINC += -DINCLUDE_ATSR_SUPPORT

# Uncomment if Numerical Recipes is available for cubic spline profile
# interpolation and bilinear LUT interpolation.
LIBS += -lnr
INC  += -DINCLUDE_NR

# Uncomment if Fu_Liou support is desired for broadband fluxes.
LIBS += -lEd3Fu_201212
INC  += -DINCLUDE_FU_LIOU_SUPPORT

# Uncomment if Himawari support is desired.
LIBS += -lhimawari_util
INC  += -DINCLUDE_HIMAWARI_SUPPORT

# Uncomment if SEVIRI support is desired.
LIBS += -lseviri_util
INC  += -DINCLUDE_SEVIRI_SUPPORT
