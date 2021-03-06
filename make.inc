###### Local configuration:

CUDA_INSTALL_PATH = /fhgfs/prometheus/buildsets/eb141014/software/CUDA/5.0.35-1
QDP_INSTALL_PATH = 
QDPXX_CXXFLAGS = 
QDPXX_LDFLAGS = 
QDPXX_LIBS = 

CPU_ARCH = x86_64  	  # x86 or x86_64
GPU_ARCH = sm_20	  # sm_11, sm_12, sm_13, sm_20, sm_21, sm_30, or sm_35
OS       = linux	  # linux or osx

PYTHON = /usr	  # python 2.5 or later required for 'make gen'

# compilation options
HOST_DEBUG = no              # compile host debug code
DEVICE_DEBUG = no          # compile device debug code for cuda-gdb 
VERBOSE = no                    # display kernel register useage
DSLASH_PROFILING = no  # multi-gpu dslash profiling
FERMI_DBLE_TEX=yes        # enable double-precision texture reads on Fermi?

BUILD_WILSON_DIRAC = yes       	# build Wilson Dirac operators?
BUILD_CLOVER_DIRAC = no       	# build clover Dirac operators?
BUILD_DOMAIN_WALL_DIRAC = no   	# build domain wall Dirac operators?
BUILD_STAGGERED_DIRAC = no		# build staggered Dirac operators?
BUILD_TWISTED_MASS_DIRAC = yes	# build twisted mass Dirac operators?
BUILD_FATLINK = no             		# build code for computing asqtad fat links?
BUILD_HISQLINK = no			# build code for computing hisq fat links?
BUILD_GAUGE_FORCE = no         	# build code for (1-loop Symanzik) gauge force?
BUILD_FERMION_FORCE = no	      	# build code for asqtad fermion force?
BUILD_HISQ_FORCE = no			# build code for hisq fermion force?

# Multi-GPU options
BUILD_MULTI_GPU = yes  # set to 'yes' to build the multi-GPU code
BUILD_QMP = no              # set to 'yes' to build the QMP multi-GPU code
BUILD_MPI = yes              # set to 'yes' to build the MPI multi-GPU code
OVERLAP_COMMS = yes      # set to 'yes' to overlap comms and compute

# GPUdirect options
GPU_DIRECT = no            # set to 'yes' to allow GPU and NIC to shared pinned buffers

# Packing option
DEVICE_PACK = yes	     # set to 'yes' to enable packing and unpacking on the device

BUILD_QIO = no    # set to 'yes' to build QIO code for binary I/O

USE_QDPJIT = no  # build QDP-JIT support?

#FECC = MPICC 	           # front-end CC
#FECXX = MPICXX            # front-end CXX
#FEF90 =             # front-end F90

MPI_HOME=/fhgfs/prometheus/buildsets/eb141014/software/MVAPICH2/1.9a2-GCC-4.6.3-CUDA
QMP_HOME=
QIO_HOME=

FECC=$(MPI_HOME)/bin/mpicc
FECXX=$(MPI_HOME)/bin/mpicxx


NUMA_AFFINITY=no   # enable NUMA affinity?

######

INC = -I$(CUDA_INSTALL_PATH)/include

ifeq ($(strip $(CPU_ARCH)), x86_64)
  ifeq ($(strip $(OS)), osx)
    LIB = -L$(CUDA_INSTALL_PATH)/lib -lcudart
    NVCCOPT = -m64
  else
    LIB = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
  endif
  COPT += -DPOINTER_SIZE=8
  NVCCOPT += -DPOINTER_SIZE=8
else
  LIB = -L$(CUDA_INSTALL_PATH)/lib -lcudart
  COPT = -malign-double -DPOINTER_SIZE=4
  NVCCOPT = -DPOINTER_SIZE=4
endif

COMP_CAP = $(GPU_ARCH:sm_%=%0)

COPT += -D__COMPUTE_CAPABILITY__=$(COMP_CAP)
NVCCOPT += -D__COMPUTE_CAPABILITY__=$(COMP_CAP)

TESLA_ARCH = $(shell [ $(COMP_CAP) -lt 200 ] && echo true)
ifneq ($(TESLA_ARCH),true)
  NVCCOPT += -ftz=true -prec-div=false -prec-sqrt=false
endif

ifeq ($(strip $(BUILD_MULTI_GPU)), yes)
  COPT += -DMULTI_GPU
  NVCCOPT += -DMULTI_GPU
endif

CC  = $(FECC)
CXX = $(FECXX)
F90 = $(FEF90)

ifeq ($(strip $(BUILD_MPI)), yes)
  MPI_CFLAGS =
  MPI_LDFLAGS =
  MPI_LIBS =
  INC += -DMPI_COMMS $(MPI_CFLAGS) -I$(MPI_HOME)/include
  LIB += $(MPI_LDFLAGS) $(MPI_LIBS)
  FACE_COMMS_OBJS=face_mpi.o comm_mpi.o
else
  FACE_COMMS_OBJS=face_qmp.o
endif

ifeq ($(strip $(BUILD_QMP)), yes)
  QMP_CFLAGS = $(shell $(QMP_HOME)/bin/qmp-config --cflags )
  QMP_LDFLAGS = $(shell $(QMP_HOME)/bin/qmp-config --ldflags )
  QMP_LIBS = $(shell $(QMP_HOME)/bin/qmp-config --libs )
  INC += -DQMP_COMMS $(QMP_CFLAGS)
  LIB += $(QMP_LDFLAGS) $(QMP_LIBS)
  FACE_COMMS_OBJS=face_qmp.o comm_qmp.o
endif 

ifeq ($(strip $(OVERLAP_COMMS)), yes)
  COPT += -DOVERLAP_COMMS
  NVCCOPT += -DOVERLAP_COMMS
endif

ifeq ($(strip $(BUILD_QIO)), yes)
  INC += -DHAVE_QIO -I$(QIO_HOME)/include
  LIB += -L$(QIO_HOME)/lib -lqio -llime
  QIO_UTIL = qio_util.o layout_hyper.o gauge_qio.o
endif

ifeq ($(strip $(BUILD_WILSON_DIRAC)), yes)
  NVCCOPT += -DGPU_WILSON_DIRAC
  COPT += -DGPU_WILSON_DIRAC
  DIRAC_TEST = dslash_test invert_test
endif
ifeq ($(strip $(BUILD_DOMAIN_WALL_DIRAC)), yes)
  NVCCOPT += -DGPU_DOMAIN_WALL_DIRAC
  COPT += -DGPU_DOMAIN_WALL_DIRAC
  DIRAC_TEST = dslash_test invert_test
endif
ifeq ($(strip $(BUILD_STAGGERED_DIRAC)), yes)
  NVCCOPT += -DGPU_STAGGERED_DIRAC
  COPT += -DGPU_STAGGERED_DIRAC
  STAGGERED_DIRAC_TEST=staggered_dslash_test staggered_invert_test
endif
ifeq ($(strip $(BUILD_CLOVER_DIRAC)), yes)
  NVCCOPT += -DGPU_CLOVER_DIRAC -DGPU_WILSON_DIRAC
  COPT += -DGPU_CLOVER_DIRAC -DGPU_WILSON_DIRAC
endif
ifeq ($(strip $(BUILD_TWISTED_MASS_DIRAC)), yes)
  NVCCOPT += -DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC
  COPT += -DGPU_TWISTED_MASS_DIRAC -DGPU_WILSON_DIRAC
endif
ifeq ($(strip $(BUILD_FATLINK)), yes)
  NVCCOPT += -DGPU_FATLINK
  FATLINK_TEST=llfat_test
  COPT += -DGPU_FATLINK
endif
ifeq ($(strip $(BUILD_HISQLINK)), yes)
  ifneq ($(strip $(BUILD_FATLINK)), yes) 
    NVCCOPT += -DGPU_FATLINK
    COPT    += -DGPU_FATLINK
    FATLINK_TEST=llfat_test
  endif
  UNITARIZE_LINK_TEST=unitarize_link_test
  NVCCOPT += -DGPU_UNITARIZE
  COPT    += -DGPU_UNITARIZE
endif
ifeq ($(strip $(BUILD_GAUGE_FORCE)), yes)
  NVCCOPT += -DGPU_GAUGE_FORCE
  COPT += -DGPU_GAUGE_FORCE
  GAUGE_FORCE_TEST=gauge_force_test
endif
ifeq ($(strip $(BUILD_FERMION_FORCE)), yes)
  NVCCOPT += -DGPU_FERMION_FORCE
  COPT += -DGPU_FERMION_FORCE
  FERMION_FORCE_TEST=fermion_force_test
endif
ifeq ($(strip $(BUILD_HISQ_FORCE)), yes)
  NVCCOPT += -DGPU_HISQ_FORCE
  COPT += -DGPU_HISQ_FORCE
  HISQ_PATHS_FORCE_TEST=hisq_paths_force_test
  HISQ_UNITARIZE_FORCE_TEST=hisq_unitarize_force_test
endif
ifeq ($(strip $(HOST_DEBUG)), yes)
  NVCCOPT += -g -DHOST_DEBUG
  COPT += -g -fno-inline -DHOST_DEBUG
endif
ifeq ($(strip $(DEVICE_DEBUG)), yes)
  NVCCOPT += -G
endif
ifeq ($(strip $(VERBOSE)), yes)
  NVCCOPT += --ptxas-options=-v
endif
ifeq ($(strip $(DSLASH_PROFILING)), yes)
  NVCCOPT += -DDSLASH_PROFILING
  COPT +=  -DDSLASH_PROFILING
endif

ifeq ($(strip $(FERMI_DBLE_TEX)), no)
  NVCCOPT += -DFERMI_NO_DBLE_TEX
  COPT += -DFERMI_NO_DBLE_TEX
endif

ifeq ($(strip $(GPU_DIRECT)), yes)
  NVCCOPT += -DGPU_DIRECT
  COPT += -DGPU_DIRECT
endif

ifeq ($(strip $(DEVICE_PACK)), yes)
  NVCCOPT += -DDEVICE_PACK
  COPT += -DDEVICE_PACK
endif

ifeq ($(strip $(OS)), osx)
  NUMA_AFFINITY = no
endif

ifeq ($(strip $(NUMA_AFFINITY)), yes)
  NVCCOPT += -DNUMA_AFFINITY
  COPT += -DNUMA_AFFINITY
  NUMA_AFFINITY_OBJS=numa_affinity.o
endif


### Next conditional is necessary.
### QDPXX_CXXFLAGS contains "-O3".
### We must make sure its not given
### twice to nvcc. It would complain.

ifeq ($(strip $(USE_QDPJIT)), yes)
  NVCCOPT += -DUSE_QDPJIT
  COPT += -DUSE_QDPJIT
  LIB += $(QDPXX_LDFLAGS) $(QDPXX_LIBS)
  INC += -I$(QDP_INSTALL_PATH)/include

  CFLAGS = -Wall -std=c99 $(COPT) $(INC)
  CXXFLAGS = -Wall $(COPT) $(INC) $(QDPXX_CXXFLAGS)
  NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc 
  NVCCFLAGS = $(NVCCOPT) -arch=$(GPU_ARCH) $(INC)
  LDFLAGS = -fPIC $(LIB)
else
  CFLAGS = -Wall -O3 -std=c99 $(COPT) $(INC)
  CXXFLAGS = -Wall -O3 $(COPT) $(INC)
  NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc 
  NVCCFLAGS = -O3 $(NVCCOPT) -arch=$(GPU_ARCH) $(INC)
  LDFLAGS = -fPIC $(LIB)
endif

CXXFLAGS += -DADD_ -fno-strict-aliasing -DMAGMA_WITH_MKL -DMAGMA_SETAFFINITY -DMIN_CUDA_ARCH=200