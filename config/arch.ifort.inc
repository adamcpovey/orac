# Directory for object files
OBJS = obj

# Define Fortran 77 compiler
F77 = ifort

# Define Fortran 90 compiler
F90 = ifort

# Define C compiler
CC = gcc

# Define C++ compiler
CXX = g++

# Define C preprocessor
CPP = cpp

# Set Fortran 77 compiler flags
F77FLAGS = -O3 -cpp -g -check bounds

# Set Fortran 90 compiler flags
FFLAGS   = -O3 -cpp -g -check bounds
# Uncomment if OpenMP support is desired
FFLAGS  += -qopenmp
# If you are running out of space on the stack, use this instead of openmp
#FFLAGS  += -heap-arrays

LFLAGS = -lifcore -qopenmp

# Set C compiler  flags
CFLAGS = -O3 -g

AUXFLAGS = -module $(OBJS)

# Set Bison/Flex parser flags
FLEXFLAGS =
BISONFLAGS =
# Uncomment if debugging the parser
#FLEXFLAGS += -d
#BISONFLAGS += -t
