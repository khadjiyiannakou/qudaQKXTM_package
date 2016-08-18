#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <sys/time.h>

#include <quda.h>
#include <quda_internal.h>
#include <comm_quda.h>
#include <tune_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <color_spinor_field.h>

#ifdef NUMA_AFFINITY
#include <numa_affinity.h>
#endif

#include <cuda.h>
#include <mpi.h>
#include <qudaQKXTM.h>

#ifdef MULTI_GPU
extern void exchange_cpu_sitelink_ex(int* X, int *R, void** sitelink, QudaGaugeFieldOrder cpu_order,
				     QudaPrecision gPrecision, int optflag);
#endif // MULTI_GPU

#ifdef GPU_GAUGE_FORCE
#include <gauge_force_quda.h>
#endif

#define MAX(a,b) ((a)>(b)? (a):(b))
#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

#define spinorSiteSize 24 // real numbers per spinor

#define MAX_GPU_NUM_PER_NODE 16

// define newQudaGaugeParam() and newQudaInvertParam()
#define INIT_PARAM
#include "check_params.h"
#undef INIT_PARAM

// define (static) checkGaugeParam() and checkInvertParam()
#define CHECK_PARAM
#include "check_params.h"
#undef CHECK_PARAM

// define printQudaGaugeParam() and printQudaInvertParam()
#define PRINT_PARAM
#include "check_params.h"
#undef PRINT_PARAM

#include "face_quda.h"

int numa_affinity_enabled = 1;

using namespace quda;

cudaGaugeField *gaugePrecise = NULL;
cudaGaugeField *gaugeSloppy = NULL;
cudaGaugeField *gaugePrecondition = NULL;

// It's important that these alias the above so that constants are set correctly in Dirac::Dirac()
cudaGaugeField *&gaugeFatPrecise = gaugePrecise;
cudaGaugeField *&gaugeFatSloppy = gaugeSloppy;
cudaGaugeField *&gaugeFatPrecondition = gaugePrecondition;

cudaGaugeField *gaugeLongPrecise = NULL;
cudaGaugeField *gaugeLongSloppy = NULL;
cudaGaugeField *gaugeLongPrecondition = NULL;

//cudaCloverField *cloverPrecise = NULL;
//cudaCloverField *cloverSloppy = NULL;
//cudaCloverField *cloverPrecondition = NULL;

/*Dirac *diracPrecise = NULL;
Dirac *diracSloppy = NULL;
Dirac *diracPrecondition = NULL;
*/


cudaDeviceProp deviceProp;
cudaStream_t *streams;

static bool initialized = false;

//!< Profiler for initQuda
TimeProfile profileInit("initQuda");

//!< Profile for loadGaugeQuda / saveGaugeQuda
TimeProfile profileGauge("loadGaugeQuda");

//!< Profile for loadCloverQuda
TimeProfile profileClover("loadCloverQuda");

//!< Profiler for invertQuda
TimeProfile profileInvert("invertQuda");

//!< Profiler for invertMultiShiftQuda
TimeProfile profileMulti("invertMultiShiftQuda");

//!< Profiler for invertMultiShiftMixedQuda
TimeProfile profileMultiMixed("invertMultiShiftMixedQuda");

//!< Profiler for endQuda
TimeProfile profileEnd("endQuda");

int getGpuCount()
{
  int count;
  cudaGetDeviceCount(&count);
  if (count <= 0){
    errorQuda("No devices supporting CUDA");
  }
  if(count > MAX_GPU_NUM_PER_NODE){
    errorQuda("GPU count (%d) is larger than limit\n", count);
  }
  return count;
}


void setVerbosityQuda(QudaVerbosity verbosity, const char prefix[], FILE *outfile)
{
  setVerbosity(verbosity);
  setOutputPrefix(prefix);
  setOutputFile(outfile);
}


void initQuda(int dev)
{
  profileInit[QUDA_PROFILE_TOTAL].Start();

  //static bool initialized = false;
  if (initialized) return;
  initialized = true;

#if defined(GPU_DIRECT) && defined(MULTI_GPU) && (CUDA_VERSION == 4000)
  //check if CUDA_NIC_INTEROP is set to 1 in the enviroment
  // not needed for CUDA >= 4.1
  char* cni_str = getenv("CUDA_NIC_INTEROP");
  if(cni_str == NULL){
    errorQuda("Environment variable CUDA_NIC_INTEROP is not set\n");
  }
  int cni_int = atoi(cni_str);
  if (cni_int != 1){
    errorQuda("Environment variable CUDA_NIC_INTEROP is not set to 1\n");    
  }
#endif

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  
  if(getVerbosity() >= QUDA_SUMMARIZE){
    fprintf(stdout,"Rank %d found %d devices\n",comm_rank(),deviceCount);
  }

  if (deviceCount == 0) {
    errorQuda("No devices supporting CUDA");
  }

  for(int i=0; i<deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
    // if (getVerbosity() >= QUDA_SUMMARIZE) {
    //  printfQuda("Found device %d: %s\n", i, deviceProp.name);
    // }
  }

#ifdef MULTI_GPU
  comm_init();
  if (dev < 0) dev = comm_gpuid();
#else
  if (dev < 0 || dev >= 16) errorQuda("Invalid device number %d", dev);
#endif

  cudaGetDeviceProperties(&deviceProp, dev);
  checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
  if (deviceProp.major < 1) {
    errorQuda("Device %d does not support CUDA", dev);
  }
  
  if (getVerbosity() >= QUDA_SUMMARIZE) {
    fprintf(stdout,"Rank %d using device %d: %s with device memory %f MB\n",comm_rank() , dev, deviceProp.name,deviceProp.totalGlobalMem/(1024.*1024.));
  }
  cudaSetDevice(dev);
  checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode

  comm_barrier(); // for smooth printing


#ifdef NUMA_AFFINITY
  if(numa_affinity_enabled){
    setNumaAffinity(dev);
  }
#endif
  // if the device supports host-mapped memory, then enable this
  if(deviceProp.canMapHostMemory) cudaSetDeviceFlags(cudaDeviceMapHost);
  checkCudaError();

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaGetDeviceProperties(&deviceProp, dev);

  streams = new cudaStream_t[Nstream];
  for (int i=0; i<Nstream; i++) {
    cudaStreamCreate(&streams[i]);
  }
  checkCudaError();
  createDslashEvents();

  initBlas();

  loadTuneCache(getVerbosity());

  profileInit[QUDA_PROFILE_TOTAL].Stop();
}


void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  profileGauge[QUDA_PROFILE_TOTAL].Start();

  if (!initialized) errorQuda("QUDA not initialized");
  if (getVerbosity() == QUDA_DEBUG_VERBOSE) printQudaGaugeParam(param);

  checkGaugeParam(param);

  // Set the specific cpu parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(h_gauge, *param);

  cpuGaugeField cpu(gauge_param);

  profileGauge[QUDA_PROFILE_INIT].Start();  
  // switch the parameters for creating the mirror precise cuda gauge field
  gauge_param.create = QUDA_NULL_FIELD_CREATE;
  gauge_param.precision = param->cuda_prec;
  gauge_param.reconstruct = param->reconstruct;
  gauge_param.pad = param->ga_pad;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION || 
		       gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  cudaGaugeField *precise = new cudaGaugeField(gauge_param);
  profileGauge[QUDA_PROFILE_INIT].Stop();  

  profileGauge[QUDA_PROFILE_H2D].Start();  
  precise->loadCPUField(cpu, QUDA_CPU_FIELD_LOCATION);

  param->gaugeGiB += precise->GBytes();

  // switch the parameters for creating the mirror sloppy cuda gauge field
  gauge_param.precision = param->cuda_prec_sloppy;
  gauge_param.reconstruct = param->reconstruct_sloppy;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION || 
		       gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  cudaGaugeField *sloppy = NULL;
  if (param->cuda_prec != param->cuda_prec_sloppy) {
    sloppy = new cudaGaugeField(gauge_param);
    sloppy->loadCPUField(cpu, QUDA_CPU_FIELD_LOCATION);
    param->gaugeGiB += sloppy->GBytes();
  } else {
    sloppy = precise;
  }

  // switch the parameters for creating the mirror preconditioner cuda gauge field
  gauge_param.precision = param->cuda_prec_precondition;
  gauge_param.reconstruct = param->reconstruct_precondition;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION || 
		       gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  cudaGaugeField *precondition = NULL;
  if (param->cuda_prec_sloppy != param->cuda_prec_precondition) {
    precondition = new cudaGaugeField(gauge_param);
    precondition->loadCPUField(cpu, QUDA_CPU_FIELD_LOCATION);
    param->gaugeGiB += precondition->GBytes();
  } else {
    precondition = sloppy;
  }
  profileGauge[QUDA_PROFILE_H2D].Stop();  
  
  switch (param->type) {
  case QUDA_WILSON_LINKS:
    //if (gaugePrecise) errorQuda("Precise gauge field already allocated");
    gaugePrecise = precise;
    //if (gaugeSloppy) errorQuda("Sloppy gauge field already allocated");
    gaugeSloppy = sloppy;
    //if (gaugePrecondition) errorQuda("Precondition gauge field already allocated");
    gaugePrecondition = precondition;
    break;
  case QUDA_ASQTAD_FAT_LINKS:
    if (gaugeFatPrecise) errorQuda("Precise gauge fat field already allocated");
    gaugeFatPrecise = precise;
    if (gaugeFatSloppy) errorQuda("Sloppy gauge fat field already allocated");
    gaugeFatSloppy = sloppy;
    if (gaugeFatPrecondition) errorQuda("Precondition gauge fat field already allocated");
    gaugeFatPrecondition = precondition;
    break;
  case QUDA_ASQTAD_LONG_LINKS:
    if (gaugeLongPrecise) errorQuda("Precise gauge long field already allocated");
    gaugeLongPrecise = precise;
    if (gaugeLongSloppy) errorQuda("Sloppy gauge long field already allocated");
    gaugeLongSloppy = sloppy;
    if (gaugeLongPrecondition) errorQuda("Precondition gauge long field already allocated");
    gaugeLongPrecondition = precondition;
    break;
  default:
    errorQuda("Invalid gauge type");   
  }

  profileGauge[QUDA_PROFILE_TOTAL].Stop();
}

void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  profileGauge[QUDA_PROFILE_TOTAL].Start();

  if (!initialized) errorQuda("QUDA not initialized");
  checkGaugeParam(param);

  // Set the specific cpu parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(h_gauge, *param);
  cpuGaugeField cpuGauge(gauge_param);
  cudaGaugeField *cudaGauge = NULL;
  switch (param->type) {
  case QUDA_WILSON_LINKS:
    cudaGauge = gaugePrecise;
    break;
  case QUDA_ASQTAD_FAT_LINKS:
    cudaGauge = gaugeFatPrecise;
    break;
  case QUDA_ASQTAD_LONG_LINKS:
    cudaGauge = gaugeLongPrecise;
    break;
  default:
    errorQuda("Invalid gauge type");   
  }

  profileGauge[QUDA_PROFILE_D2H].Start();  
  cudaGauge->saveCPUField(cpuGauge, QUDA_CPU_FIELD_LOCATION);
  profileGauge[QUDA_PROFILE_D2H].Stop();  

  profileGauge[QUDA_PROFILE_TOTAL].Stop();
}



void freeGaugeQuda(void) 
{  
  if (!initialized) errorQuda("QUDA not initialized");
  if (gaugeSloppy != gaugePrecondition && gaugePrecondition) delete gaugePrecondition;
  if (gaugePrecise != gaugeSloppy && gaugeSloppy) delete gaugeSloppy;
  if (gaugePrecise) delete gaugePrecise;

  gaugePrecondition = NULL;
  gaugeSloppy = NULL;
  gaugePrecise = NULL;

  if (gaugeLongSloppy != gaugeLongPrecondition && gaugeLongPrecondition) delete gaugeLongPrecondition;
  if (gaugeLongPrecise != gaugeLongSloppy && gaugeLongSloppy) delete gaugeLongSloppy;
  if (gaugeLongPrecise) delete gaugeLongPrecise;

  gaugeLongPrecondition = NULL;
  gaugeLongSloppy = NULL;
  gaugeLongPrecise = NULL;

  if (gaugeFatSloppy != gaugeFatPrecondition && gaugeFatPrecondition) delete gaugeFatPrecondition;
  if (gaugeFatPrecise != gaugeFatSloppy && gaugeFatSloppy) delete gaugeFatSloppy;
  if (gaugeFatPrecise) delete gaugeFatPrecise;
  
  gaugeFatPrecondition = NULL;
  gaugeFatSloppy = NULL;
  gaugeFatPrecise = NULL;
}




void endQuda(void)
{
  profileEnd[QUDA_PROFILE_TOTAL].Start();

  if (!initialized) return;

  LatticeField::freeBuffer();
  cudaColorSpinorField::freeBuffer();
  cudaColorSpinorField::freeGhostBuffer();
  cpuColorSpinorField::freeGhostBuffer();
  FaceBuffer::flushPinnedCache();
  freeGaugeQuda();


  endBlas();

  if (streams) {
    for (int i=0; i<Nstream; i++) cudaStreamDestroy(streams[i]);
    delete []streams;
    streams = NULL;
  }
  destroyDslashEvents();

  saveTuneCache(getVerbosity());

  // end this CUDA context
  cudaDeviceReset();

  initialized = false;

  profileEnd[QUDA_PROFILE_TOTAL].Stop();

  // print out the profile information of the lifetime of the library
  if (getVerbosity() >= QUDA_SUMMARIZE) {
    profileInit.Print();
    profileGauge.Print();
    profileClover.Print();
    profileInvert.Print();
    profileMulti.Print();
    profileMultiMixed.Print();
    profileEnd.Print();

    printfQuda("\n");
    printPeakMemUsage();
    printfQuda("\n");
  }

  assertAllMemFree();
}


namespace quda {

  void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    double kappa = inv_param->kappa;
    if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
      kappa *= gaugePrecise->Anisotropy();
    }

    switch (inv_param->dslash_type) {
    case QUDA_WILSON_DSLASH:
      diracParam.type = pc ? QUDA_WILSONPC_DIRAC : QUDA_WILSON_DIRAC;
      break;
      //    case QUDA_CLOVER_WILSON_DSLASH:
      // diracParam.type = pc ? QUDA_CLOVERPC_DIRAC : QUDA_CLOVER_DIRAC;
      //break;
      // case QUDA_DOMAIN_WALL_DSLASH:
      //diracParam.type = pc ? QUDA_DOMAIN_WALLPC_DIRAC : QUDA_DOMAIN_WALL_DIRAC;
      //BEGIN NEW :
      // diracParam.Ls = inv_param->Ls;
      //END NEW    
      //break;
      //case QUDA_ASQTAD_DSLASH:
      //diracParam.type = pc ? QUDA_ASQTADPC_DIRAC : QUDA_ASQTAD_DIRAC;
      //break;
    case QUDA_TWISTED_MASS_DSLASH:
      diracParam.type = pc ? QUDA_TWISTED_MASSPC_DIRAC : QUDA_TWISTED_MASS_DIRAC;
      break;
    default:
      errorQuda("Unsupported dslash_type %d", inv_param->dslash_type);
    }

    diracParam.matpcType = inv_param->matpc_type;
    diracParam.dagger = inv_param->dagger;
    diracParam.gauge = gaugePrecise;
    diracParam.fatGauge = gaugeFatPrecise;
    diracParam.longGauge = gaugeLongPrecise;    
    //    diracParam.clover = cloverPrecise;
    diracParam.kappa = kappa;
    diracParam.mass = inv_param->mass;
    diracParam.m5 = inv_param->m5;
    diracParam.mu = inv_param->mu;
    diracParam.verbose = getVerbosity();

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = 1;   // comms are always on
    }
  }


  void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    setDiracParam(diracParam, inv_param, pc);

    diracParam.gauge = gaugeSloppy;
    diracParam.fatGauge = gaugeFatSloppy;
    diracParam.longGauge = gaugeLongSloppy;    
    //    diracParam.clover = cloverSloppy;

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = 1;   // comms are always on
    }

  }

  // The preconditioner currently mimicks the sloppy operator with no comms
  void setDiracPreParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
  {
    setDiracParam(diracParam, inv_param, pc);

    diracParam.gauge = gaugePrecondition;
    diracParam.fatGauge = gaugeFatPrecondition;
    diracParam.longGauge = gaugeLongPrecondition;    
    //    diracParam.clover = cloverPrecondition;

    for (int i=0; i<4; i++) {
      diracParam.commDim[i] = 0; // comms are always off
    }

  }

  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve)
  {
    DiracParam diracParam;
    DiracParam diracSloppyParam;
    DiracParam diracPreParam;
    
    setDiracParam(diracParam, &param, pc_solve);
    setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
    setDiracPreParam(diracPreParam, &param, pc_solve);
    
    d = Dirac::create(diracParam); // create the Dirac operator   
    dSloppy = Dirac::create(diracSloppyParam);
    dPre = Dirac::create(diracPreParam);
  }

  void massRescale(QudaDslashType dslash_type, double &kappa, QudaSolutionType solution_type, 
		   QudaMassNormalization mass_normalization, cudaColorSpinorField &b)
  {   
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Mass rescale: Kappa is: %g\n", kappa);
      printfQuda("Mass rescale: mass normalization: %d\n", mass_normalization);
      double nin = norm2(b);
      printfQuda("Mass rescale: norm of source in = %g\n", nin);
    }
 
    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      if (mass_normalization != QUDA_MASS_NORMALIZATION) {
	errorQuda("Staggered code only supports QUDA_MASS_NORMALIZATION");
      }
      return;
    }

    // multiply the source to compensate for normalization of the Dirac operator, if necessary
    switch (solution_type) {
    case QUDA_MAT_SOLUTION:
      if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	  mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	axCuda(2.0*kappa, b);
      }
      break;
    case QUDA_MATDAG_MAT_SOLUTION:
      if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	  mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	axCuda(4.0*kappa*kappa, b);
      }
      break;
    case QUDA_MATPC_SOLUTION:
      if (mass_normalization == QUDA_MASS_NORMALIZATION) {
	axCuda(4.0*kappa*kappa, b);
      } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	axCuda(2.0*kappa, b);
      }
      break;
    case QUDA_MATPCDAG_MATPC_SOLUTION:
      if (mass_normalization == QUDA_MASS_NORMALIZATION) {
	axCuda(16.0*pow(kappa,4), b);
      } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	axCuda(4.0*kappa*kappa, b);
      }
      break;
    default:
      errorQuda("Solution type %d not supported", solution_type);
    }

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Mass rescale done\n");   
    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Mass rescale: Kappa is: %g\n", kappa);
      printfQuda("Mass rescale: mass normalization: %d\n", mass_normalization);
      double nin = norm2(b);
      printfQuda("Mass rescale: norm of source out = %g\n", nin);
    }

  }

  void massRescaleCoeff(QudaDslashType dslash_type, double &kappa, QudaSolutionType solution_type, 
			QudaMassNormalization mass_normalization, double &coeff)
  {    
    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      if (mass_normalization != QUDA_MASS_NORMALIZATION) {
	errorQuda("Staggered code only supports QUDA_MASS_NORMALIZATION");
      }
      return;
    }

    // multiply the source to compensate for normalization of the Dirac operator, if necessary
    switch (solution_type) {
    case QUDA_MAT_SOLUTION:
      if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	  mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	coeff *= 2.0*kappa;
      }
      break;
    case QUDA_MATDAG_MAT_SOLUTION:
      if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	  mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	coeff *= 4.0*kappa*kappa;
      }
      break;
    case QUDA_MATPC_SOLUTION:
      if (mass_normalization == QUDA_MASS_NORMALIZATION) {
	coeff *= 4.0*kappa*kappa;
      } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	coeff *= 2.0*kappa;
      }
      break;
    case QUDA_MATPCDAG_MATPC_SOLUTION:
      if (mass_normalization == QUDA_MASS_NORMALIZATION) {
	coeff*=16.0*pow(kappa,4);
      } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	coeff*=4.0*kappa*kappa;
      }
      break;
    default:
      errorQuda("Solution type %d not supported", solution_type);
    }

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Mass rescale done\n");   
  }
}

/*void QUDA_DiracField(QUDA_DiracParam *param) {
  
  }*/

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity)
{

  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);
  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  //  if (cloverPrecise == NULL && inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
  //   errorQuda("Clover field not allocated");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  ColorSpinorParam cpuParam(h_in, inv_param->input_location, *inv_param, gaugePrecise->X(), 1);

  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : 
    static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*in_h);
    double gpu = norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    axCuda(gaugePrecise->Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->Dslash(out, in, parity); // apply the operator
  delete dirac; // clean up

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;
  
  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*out_h);
    double gpu = norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}


void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  pushVerbosity(inv_param->verbosity);

  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);
  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  //  if (cloverPrecise == NULL && inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
  //  errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
	     inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, inv_param->input_location, *inv_param, gaugePrecise->X(), pc);
  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*in_h);
    double gpu = norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->M(out, in); // apply the operator
  delete dirac; // clean up

  double kappa = inv_param->kappa;
  if (pc) {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.5/kappa, out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
	inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.5/kappa, out);
    }
  }

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*out_h);
    double gpu = norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}


void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  pushVerbosity(inv_param->verbosity);

  if (inv_param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);
  if (!initialized) errorQuda("QUDA not initialized");
  if (gaugePrecise == NULL) errorQuda("Gauge field not allocated");
  //  if (cloverPrecise == NULL && inv_param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) 
  //  errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(inv_param);

  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
	     inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, inv_param->input_location, *inv_param, gaugePrecise->X(), pc);
  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));  

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);
  
  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*in_h);
    double gpu = norm2(in);
    printfQuda("In CPU %e CUDA %e\n", cpu, gpu);
  }

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  //  double kappa = inv_param->kappa;
  //  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa *= gaugePrecise->anisotropy;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->MdagM(out, in); // apply the operator
  delete dirac; // clean up

  double kappa = inv_param->kappa;
  if (pc) {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(1.0/pow(2.0*kappa,4), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
	inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    }
  }

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;

  if (getVerbosity() >= QUDA_VERBOSE) {
    double cpu = norm2(*out_h);
    double gpu = norm2(out);
    printfQuda("Out CPU %e CUDA %e\n", cpu, gpu);
  }

  delete out_h;
  delete in_h;

  popVerbosity();
}

quda::cudaGaugeField* checkGauge(QudaInvertParam *param) {
  quda::cudaGaugeField *cudaGauge = NULL;
  if (param->dslash_type != QUDA_ASQTAD_DSLASH) {
    if (gaugePrecise == NULL) errorQuda("Precise gauge field doesn't exist");
    if (gaugeSloppy == NULL) errorQuda("Sloppy gauge field doesn't exist");
    if (gaugePrecondition == NULL) errorQuda("Precondition gauge field doesn't exist");
    cudaGauge = gaugePrecise;
  } else {
    if (gaugeFatPrecise == NULL) errorQuda("Precise gauge fat field doesn't exist");
    if (gaugeFatSloppy == NULL) errorQuda("Sloppy gauge fat field doesn't exist");
    if (gaugeFatPrecondition == NULL) errorQuda("Precondition gauge fat field doesn't exist");

    if (gaugeLongPrecise == NULL) errorQuda("Precise gauge long field doesn't exist");
    if (gaugeLongSloppy == NULL) errorQuda("Sloppy gauge long field doesn't exist");
    if (gaugeLongPrecondition == NULL) errorQuda("Precondition gauge long field doesn't exist");
    cudaGauge = gaugeFatPrecise;
  }
  return cudaGauge;
}


void invertQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  profileInvert[QUDA_PROFILE_TOTAL].Start();

  if (!initialized) errorQuda("QUDA not initialized");
  if (param->dslash_type == QUDA_DOMAIN_WALL_DSLASH) setKernelPackT(true);

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

  checkInvertParam(param);

  // It was probably a bad design decision to encode whether the system is even/odd preconditioned (PC) in
  // solve_type and solution_type, rather than in separate members of QudaInvertParam.  We're stuck with it
  // for now, though, so here we factorize everything for convenience.

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) || (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) || (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) || (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(hp_b, param->input_location, *param, X, pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = hp_x;
  ColorSpinorField *h_x = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam); 

  /*
  double *host = (double*) malloc(24*24*24*48*4*3*2*sizeof(double));
  double *ptrEven = host;
  double *ptrOdd = host + (24*24*24*48*4*3*2)/2 ;

  cudaMemcpy(ptrEven,b->Even().V(),24*24*24*48*4*3*sizeof(double) , cudaMemcpyDeviceToHost);
  cudaMemcpy(ptrOdd,b->Odd().V(),24*24*24*48*4*3*sizeof(double) , cudaMemcpyDeviceToHost);

  FILE *ptr_out;
  ptr_out = fopen("see_b.dat","w");
  for(int eo = 0 ; eo < 2 ; eo++)
    for(int mu = 0 ; mu < 4 ; mu++)
      for(int ic = 0 ; ic < 3 ; ic++)
	for(int iv = 0 ; iv < X[0]*X[1]*X[2]*X[3]/2 ; iv++){
	  fprintf(ptr_out,"%d %d %d %d %+e %+e\n",eo,mu,ic,iv,host[eo*4*3*X[0]*X[1]*X[2]*X[3] + mu*3*X[0]*X[1]*X[2]*X[3] + ic*X[0]*X[1]*X[2]*X[3] + iv*2 + 0] , host[eo*4*3*X[0]*X[1]*X[2]*X[3] + mu*3*X[0]*X[1]*X[2]*X[3] + ic*X[0]*X[1]*X[2]*X[3] + iv*2 + 1]);
	}
  comm_exit(-1);
  */


  if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) { // download initial guess
    // initial guess only supported for single-pass solvers
    if ((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) &&
	(param->solve_type == QUDA_DIRECT_SOLVE || param->solve_type == QUDA_DIRECT_PC_SOLVE)) {
      errorQuda("Initial guess not supported for two-pass solver");
    }

    x = new cudaColorSpinorField(*h_x, cudaParam); // solution  
  } else { // zero initial guess
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    x = new cudaColorSpinorField(cudaParam); // solution
  }

  if (param->residual_type == QUDA_HEAVY_QUARK_RESIDUAL && 
      (param->inv_type != QUDA_CG_INVERTER && param->inv_type != QUDA_BICGSTAB_INVERTER) ) {
    errorQuda("Heavy quark residual only supported for CG and BiCGStab");
  }
    
  profileInvert[QUDA_PROFILE_H2D].Stop();

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nh_b = norm2(*h_b);
    double nb = norm2(*b);
    double nh_x = norm2(*h_x);
    double nx = norm2(*x);
    printfQuda("Source: CPU = %g, CUDA copy = %g\n", nh_b, nb);
    printfQuda("Solution: CPU = %g, CUDA copy = %g\n", nh_x, nx);
  }

  setDslashTuning(param->tune, getVerbosity());
  setBlasTuning(param->tune, getVerbosity());

  dirac.prepare(in, out, *x, *b, param->solution_type);
  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = norm2(*in);
    double nout = norm2(*out);
    printfQuda("Prepared source = %g\n", nin);   
    printfQuda("Prepared solution = %g\n", nout);   
  }

  massRescale(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, *in);

  if (getVerbosity() >= QUDA_VERBOSE) {
    double nin = norm2(*in);
    printfQuda("Prepared source post mass rescale = %g\n", nin);   
  }
  
  // solution_type specifies *what* system is to be solved.
  // solve_type specifies *how* the system is to be solved.
  //
  // We have the following four cases (plus preconditioned variants):
  //
  // solution_type    solve_type    Effect
  // -------------    ----------    ------
  // MAT              DIRECT        Solve Ax=b
  // MATDAG_MAT       DIRECT        Solve A^dag y = b, followed by Ax=y
  // MAT              NORMOP        Solve (A^dag A) x = (A^dag b)
  // MATDAG_MAT       NORMOP        Solve (A^dag A) x = b
  //
  // We generally require that the solution_type and solve_type
  // preconditioning match.  As an exception, the unpreconditioned MAT
  // solution_type may be used with any solve_type, including
  // DIRECT_PC and NORMOP_PC.  In these cases, preparation of the
  // preconditioned source and reconstruction of the full solution are
  // taken care of by Dirac::prepare() and Dirac::reconstruct(),
  // respectively.

  if (pc_solution && !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }

  if (!mat_solution && !pc_solution && pc_solve) {
    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
  }

  if (mat_solution && !direct_solve) { // prepare source: b' = A^dag b
    cudaColorSpinorField tmp(*in);
    dirac.Mdag(*in, tmp);
  } else if (!mat_solution && direct_solve) { // perform the first of two solves: A^dag y = b
    DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    Solver *solve = Solver::create(*param, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    copyCuda(*in, *out);
    delete solve;
  }

  if (direct_solve) {
    DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    Solver *solve = Solver::create(*param, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    delete solve;
  } else {
    DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
    Solver *solve = Solver::create(*param, m, mSloppy, mPre, profileInvert);
    (*solve)(*out, *in);
    delete solve;
  }

  if (getVerbosity() >= QUDA_VERBOSE){
   double nx = norm2(*x);
   printfQuda("Solution = %g\n",nx);
  }
  dirac.reconstruct(*x, *b, param->solution_type);
  
  profileInvert[QUDA_PROFILE_D2H].Start();
  *h_x = *x;
  profileInvert[QUDA_PROFILE_D2H].Stop();
  
  if (getVerbosity() >= QUDA_VERBOSE){
    double nx = norm2(*x);
    double nh_x = norm2(*h_x);
    printfQuda("Reconstructed: CUDA solution = %g, CPU copy = %g\n", nx, nh_x);
  }

  delete h_b;
  delete h_x;
  delete b;
  delete x;

  delete d;
  delete dSloppy;
  delete dPre;

  popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());

  profileInvert[QUDA_PROFILE_TOTAL].Stop();
}



void quda::inverter(QKXTM_Propagator &uprop,QKXTM_Propagator &dprop ,void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param, int *sourcePosition)  // 
{

  //before we perform inversions we need to calculate the plaquette and APE smearing

  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();
  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge();

  //  printfQuda("The gaugeField needs\n");
  //qkxTM_gaugeTmp->printInfo();
  double plaq;

  qkxTM_gaugeTmp->packGauge(gaugeAPE);
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, uprop); // use uprop temporary for ape smearing
  
  delete qkxTM_gaugeTmp;

  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);


  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
 if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);

 // for up quark test
 b->changeTwist(QUDA_TWIST_PLUS);
 x->changeTwist(QUDA_TWIST_PLUS);
 b->Even().changeTwist(QUDA_TWIST_PLUS);
 b->Odd().changeTwist(QUDA_TWIST_PLUS);
 x->Even().changeTwist(QUDA_TWIST_PLUS);
 x->Odd().changeTwist(QUDA_TWIST_PLUS);


 // for loop  

 QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();
 QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();

 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));




 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts


   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorGauss->uploadToCuda(*b);
     
   //   double norm_b = norm2(*b);
   // printfQuda("%e\n",norm_b);
   
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 

   cudaColorSpinorField tmp(*in);                // !! check
   dirac.Mdag(*in, tmp);                        // indirect method needs apply of D^+ on source vector

   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   //   massRescale(inv_param->dslash_type, inv_param->kappa, inv_param->solution_type, inv_param->mass_normalization, *out); 
   qkxTM_vectorTmp->downloadFromCuda(*x);
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorGauss->scaleVector(2*inv_param->kappa);
   }
   int mu = ip/3;
   int ic = ip%3;
   uprop.absorbVector(*qkxTM_vectorGauss,mu,ic);

   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
 }

 b->changeTwist(QUDA_TWIST_MINUS);
 x->changeTwist(QUDA_TWIST_MINUS);
 b->Even().changeTwist(QUDA_TWIST_MINUS);
 b->Odd().changeTwist(QUDA_TWIST_MINUS);
 x->Even().changeTwist(QUDA_TWIST_MINUS);
 x->Odd().changeTwist(QUDA_TWIST_MINUS);

 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorGauss->uploadToCuda(*b);
     
   //   double norm_b = norm2(*b);
   // printfQuda("%e\n",norm_b);
   
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 

   cudaColorSpinorField tmp(*in);                // !! check
   dirac.Mdag(*in, tmp);                        // indirect method needs apply of D^+ on source vector

   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   //   massRescale(inv_param->dslash_type, inv_param->kappa, inv_param->solution_type, inv_param->mass_normalization, *out);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);

   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorGauss->scaleVector(2*inv_param->kappa);
   }

   int mu = ip/3;
   int ic = ip%3;
   dprop.absorbVector(*qkxTM_vectorGauss,mu,ic);


   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   
 }

 /// rotate propagator to physical basis
  uprop.rotateToPhysicalBasePlus();
  dprop.rotateToPhysicalBaseMinus();

 free(input_vector);

 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;

 delete d;
 delete dSloppy;
 delete dPre;
 
 freeGaugeQuda();

 // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
 saveTuneCache(getVerbosity());

 profileInvert[QUDA_PROFILE_TOTAL].Stop();
}



void quda::invert_Vector_tmLQCD(QKXTM_Vector &vec,void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param, int *sourcePosition)  // 
{

  //before we perform inversions we need to calculate the plaquette and APE smearing

  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();
  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge();

  QKXTM_Propagator *prop = new QKXTM_Propagator();

  //  printfQuda("The gaugeField needs\n");
  //qkxTM_gaugeTmp->printInfo();
  double plaq;

  qkxTM_gaugeTmp->packGauge(gaugeAPE);
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, *prop); // use uprop temporary for ape smearing
  
  delete qkxTM_gaugeTmp;
  delete prop;

  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);


  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = inv_param->twist_flavor; 
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
 if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);


 // for loop  

 QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();

 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));


 for(int ip = 0 ; ip < 1 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts


   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   qkxTM_vectorTmp->applyGammaTransformation(); // apply transformation on source 

   Gaussian_smearing(vec, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   vec.uploadToCuda(*b);
     
   //   double norm_b = norm2(*b);
   // printfQuda("%e\n",norm_b);
   
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 

   cudaColorSpinorField tmp(*in);                // !! check
   dirac.Mdag(*in, tmp);                        // indirect method needs apply of D^+ on source vector

   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   //   massRescale(inv_param->dslash_type, inv_param->kappa, inv_param->solution_type, inv_param->mass_normalization, *out); 
   qkxTM_vectorTmp->downloadFromCuda(*x);
   Gaussian_smearing(vec, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     vec.scaleVector(2*inv_param->kappa);
   }
   
   vec.applyGammaTransformation(); // apply transformation on solution
   vec.download();
 }


 free(input_vector);

 delete qkxTM_vectorTmp;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;

 delete d;
 delete dSloppy;
 delete dPre;
 
 freeGaugeQuda();

 // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
 saveTuneCache(getVerbosity());

 profileInvert[QUDA_PROFILE_TOTAL].Stop();
}




void quda::ThpTwp(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
		  int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename, char *threep_filename , int Nmom , int momElem[][3])  
{         

  /*
    This function creates threep point function of Proton using fix sink method and twop point function of proton
    Any additional information will be included here
   */

  //before we perform inversions we need to calculate the plaquette and APE smearing

  //  int momSink[0][3];

  //  momSink[0][0] = 0;
  // momSink[0][1] = 0;
  // momSink[0][2] = 1;

  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();    // temporary gauge to pefrorm APE smearing
  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(QKXTM_N2);  // because we want a backup in ram
  
  QKXTM_Propagator *uprop = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing

  double plaq;
  qkxTM_gaugeTmp->packGauge(gaugeAPE);       
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

  qkxTM_gaugeAPE->packGaugeToBackup(gaugeAPE); // backup pure gauge on backup pointer

  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, *uprop); // use *uprop temporary for ape smearing
  qkxTM_gaugeAPE->justDownloadGauge();        // just download smeared gauge from gpu in gpu form
  delete qkxTM_gaugeTmp;                                // dont need it anymore

  // from now on we have on device pointer the smeared gauge in gpu form
  // on primary host pointer we have the smeared gauge in gpu form
  // on backup host pointer we have the unsmeared gauge in gpu form
  // if you want to have smeared gauge on gpu use loadGauge if you want unsmeared gauge on gpu use loadGaugeFromBackup
  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);

  QKXTM_Propagator *dprop = new QKXTM_Propagator();   

  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
 if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);

 // for up quark test


 // for loop  

 QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
 QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));



 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));


   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);

   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark
     
   
   zeroCuda(*x);
   
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   //  cudaColorSpinorField tmp_up(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   
    (*solve)(*out, *in);   

      
   dirac.reconstruct(*x, *b, inv_param->solution_type);

   qkxTM_vectorTmp->downloadFromCuda(*x);
   
   
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   int mu = ip/3;
   int ic = ip%3;
   uprop->absorbVector(*qkxTM_vectorTmp,mu,ic);

      
    delete tmp_up;

   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   

   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);

   qkxTM_vectorGauss->uploadToCuda(*b);               // re-upload source for inversion for down quark
        
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   //cudaColorSpinorField tmp_down(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector

   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);

   qkxTM_vectorTmp->downloadFromCuda(*x);


   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   dprop->absorbVector(*qkxTM_vectorTmp,mu,ic);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   
   
 }



  // from now on start the sequential source      //////////////////////////////////////////////////////// 

 int my_fixSinkTime;
 my_fixSinkTime = fixSinkTime - comm_coords(3) * X[3];        // choose gpus which have this time slice in memory
 

 QKXTM_Propagator3D *uprop3D = new QKXTM_Propagator3D();     // to save device memory because is fix sink we define 3D propagators 
 QKXTM_Propagator3D *dprop3D = new QKXTM_Propagator3D();
 QKXTM_Propagator *seqProp = new QKXTM_Propagator();      // sequential propagator
 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();
 
 
 if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
   uprop3D->absorbTimeSlice(*uprop,my_fixSinkTime);                      // copy only the specific timeslice 
   dprop3D->absorbTimeSlice(*dprop,my_fixSinkTime);                     
 } // the others perform nothing

   
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){
   
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )    qkxTM_vectorTmp->copyPropagator3D(*uprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // perform smearing
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )   uprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);    
   
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) qkxTM_vectorTmp->copyPropagator3D(*dprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) dprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);
   
   } // now the 3D propagators are smeared
 

 // First we create threepoint function for projector type1 

 //+++++++++++++++++++++++++++++++++++++++++++++++++++++++ part 1  type 1 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 //------------------------------------------------------------------------------------------------------------------------------------------

 

 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();
 
  
 // now we must create sequential propagator for part1 (PROTON,part1=upart) , (NEUTRON,part1=dpart)
       
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){

     if(testParticle == QKXTM_PROTON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) seqSourceFixSinkPart1(*qkxTM_vectorTmp,*uprop3D,*dprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE1, testParticle);}
     else if(testParticle == QKXTM_NEUTRON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) seqSourceFixSinkPart1(*qkxTM_vectorTmp,*dprop3D,*uprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE1, testParticle);}
     else{
       errorQuda("Not implement for both particles yet");}

     comm_barrier();

     qkxTM_vectorTmp->conjugate();    // unleash after test
     //     qkxTM_vectorTmp->applyMomentum(momSink[0][0],momSink[0][1],momSink[0][2]);
     qkxTM_vectorTmp->applyGamma5();

     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     
           
     if(testParticle == QKXTM_PROTON){
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);
     }
     else{
       b->changeTwist(QUDA_TWIST_PLUS);
       x->changeTwist(QUDA_TWIST_PLUS);
       b->Even().changeTwist(QUDA_TWIST_PLUS);
       b->Odd().changeTwist(QUDA_TWIST_PLUS);
       x->Even().changeTwist(QUDA_TWIST_PLUS);
       x->Odd().changeTwist(QUDA_TWIST_PLUS);

     }

     // check rescale
     //     qkxTM_vectorGauss->scaleVector(1e+10);
     ////////
     
     qkxTM_vectorGauss->uploadToCuda(*b);           
     zeroCuda(*x);
     dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   
     cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
     dirac.Mdag(*in, *tmp);                        // indirect method needs apply of D^+ on source vector

     (*solve)(*out, *in);   
     dirac.reconstruct(*x, *b, inv_param->solution_type);
     qkxTM_vectorTmp->downloadFromCuda(*x);

     // check rescale
     //  qkxTM_vectorTmp->scaleVector(1e-10);
     ////////

     if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
       qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
     }

     seqProp->absorbVector(*qkxTM_vectorTmp,nu,c2);

     qkxTM_vectorTmp->zero();     
     qkxTM_vectorGauss->zero();
    
     delete tmp;
     printfQuda("Finish Inversion for sequential part 1 type1 quark %d/12 \n",nu*3+c2+1);
     
   }  // for loop to create sequential propagator
  
 
 char filename_threep_part1[257];
 sprintf(filename_threep_part1,"%s_%s",threep_filename,"part1");

 // at this step to calculate derivative operators we need the unsmeared gauge
 qkxTM_gaugeAPE->loadGaugeFromBackup();

 if(testParticle == QKXTM_PROTON){
   fixSinkContractions(*seqProp,*uprop,*qkxTM_gaugeAPE ,QKXTM_TYPE1, filename_threep_part1, Nmom, momElem , testParticle , 1);
   printfQuda("Finish contractions for Proton threepoint part1 type1\n");
 }
 else{
   fixSinkContractions(*seqProp,*dprop,*qkxTM_gaugeAPE ,QKXTM_TYPE1, filename_threep_part1, Nmom, momElem , testParticle , 1);
   printfQuda("Finish contractions for Neutron threepoint part1 type1\n");
 }

 // now we want back the smeared gauge to perfrom gaussian smearing
 qkxTM_gaugeAPE->loadGauge();





 //+++++++++++++++++++++++++++++++++++++++++++++++++++++++ part 2  type 1 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 //------------------------------------------------------------------------------------------------------------------------------------------

 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();

 // now we must create sequential propagator for part2 (PROTON,part2=dpart) , (NEUTRON,part2=upart)
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){


     if(testParticle == QKXTM_PROTON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) seqSourceFixSinkPart2(*qkxTM_vectorTmp,*uprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE1, testParticle);}
     else if(testParticle == QKXTM_NEUTRON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) seqSourceFixSinkPart2(*qkxTM_vectorTmp,*dprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE1, testParticle);}
     else{
       errorQuda("Not implement for both particles yet");}
     
     comm_barrier();

     qkxTM_vectorTmp->conjugate();
     //     qkxTM_vectorTmp->applyMomentum(momSink[0][0],momSink[0][1],momSink[0][2]);
     qkxTM_vectorTmp->applyGamma5();

     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);


     if(testParticle == QKXTM_PROTON){
       b->changeTwist(QUDA_TWIST_PLUS);
       x->changeTwist(QUDA_TWIST_PLUS);
       b->Even().changeTwist(QUDA_TWIST_PLUS);
       b->Odd().changeTwist(QUDA_TWIST_PLUS);
       x->Even().changeTwist(QUDA_TWIST_PLUS);
       x->Odd().changeTwist(QUDA_TWIST_PLUS);
     }
     else{
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);

     }

     // check rescale
     //     qkxTM_vectorGauss->scaleVector(1e+10);
     ////////


     qkxTM_vectorGauss->uploadToCuda(*b);           
     zeroCuda(*x);
     dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 

     cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);

     dirac.Mdag(*in, *tmp);                        // indirect method needs apply of D^+ on source vector

     (*solve)(*out, *in);   
     dirac.reconstruct(*x, *b, inv_param->solution_type);
     qkxTM_vectorTmp->downloadFromCuda(*x);

     // check rescale
     //     qkxTM_vectorTmp->scaleVector(1e-10);
     ////////


     if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
       qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
     }

     seqProp->absorbVector(*qkxTM_vectorTmp,nu,c2);

     qkxTM_vectorTmp->zero();     
     qkxTM_vectorGauss->zero();

     delete tmp;
     printfQuda("Finish Inversion for sequential part 2 type1 quark %d/12 \n",nu*3+c2+1);

   }  // for loop to create sequential propagator

 char filename_threep_part2[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/threep_part2";
 sprintf(filename_threep_part2,"%s_%s",threep_filename,"part2");
 // at this step to calculate derivative operators we need the unsmeared gauge
 qkxTM_gaugeAPE->loadGaugeFromBackup();

 if(testParticle == QKXTM_PROTON){
   fixSinkContractions(*seqProp,*dprop,*qkxTM_gaugeAPE ,QKXTM_TYPE1, filename_threep_part2, Nmom, momElem , testParticle , 2);
   printfQuda("Finish contractions for Proton threepoint part2 type1 \n");
 }
 else{
   fixSinkContractions(*seqProp,*uprop,*qkxTM_gaugeAPE, QKXTM_TYPE1, filename_threep_part2, Nmom, momElem , testParticle , 2);
   printfQuda("Finish contractions for Neutron threepoint part2 type1\n");
 }
 // now we want back the smeared gauge to perfrom gaussian smearing
 qkxTM_gaugeAPE->loadGauge();


 //---------------------------------------       finish type1 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 //+++++++++++++++++++++++++++++++++++++++++++++++++++++++ part 1  type 2 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 //------------------------------------------------------------------------------------------------------------------------------------------
 /*
 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();
 
  
 // now we must create sequential propagator for part1 (PROTON,part1=upart) , (NEUTRON,part1=dpart)

 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){

     if(testParticle == QKXTM_PROTON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )seqSourceFixSinkPart1(*qkxTM_vectorTmp,*uprop3D,*dprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE2, testParticle);}
     else if(testParticle == QKXTM_NEUTRON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )seqSourceFixSinkPart1(*qkxTM_vectorTmp,*dprop3D,*uprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE2, testParticle);}
     else{
       errorQuda("Not implement for both particles yet");}

     comm_barrier();

     qkxTM_vectorTmp->conjugate();    // unleash after test
     qkxTM_vectorTmp->applyGamma5();

     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     
           
     if(testParticle == QKXTM_PROTON){
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);
     }
     else{
       b->changeTwist(QUDA_TWIST_PLUS);
       x->changeTwist(QUDA_TWIST_PLUS);
       b->Even().changeTwist(QUDA_TWIST_PLUS);
       b->Odd().changeTwist(QUDA_TWIST_PLUS);
       x->Even().changeTwist(QUDA_TWIST_PLUS);
       x->Odd().changeTwist(QUDA_TWIST_PLUS);

     }

     qkxTM_vectorGauss->uploadToCuda(*b);           
     zeroCuda(*x);
     dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   
     cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
     dirac.Mdag(*in, *tmp);                        // indirect method needs apply of D^+ on source vector

     (*solve)(*out, *in);   
     dirac.reconstruct(*x, *b, inv_param->solution_type);
     qkxTM_vectorTmp->downloadFromCuda(*x);


     if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
       qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
     }

     seqProp->absorbVector(*qkxTM_vectorTmp,nu,c2);

     qkxTM_vectorTmp->zero();     
     qkxTM_vectorGauss->zero();
    
     delete tmp;
     printfQuda("Finish Inversion for sequential part 1 type2 quark %d/12 \n",nu*3+c2+1);
     
   }  // for loop to create sequential propagator
  
 
 // char filename_threep_part1_type2[257];
 sprintf(filename_threep_part1,"%s_%s",threep_filename,"part1");

 // at this step to calculate derivative operators we need the unsmeared gauge
 qkxTM_gaugeAPE->loadGaugeFromBackup();

 if(testParticle == QKXTM_PROTON){
   fixSinkContractions(*seqProp,*uprop,*qkxTM_gaugeAPE ,QKXTM_TYPE2, filename_threep_part1, Nmom, momElem , testParticle , 1);
   printfQuda("Finish contractions for Proton threepoint part1 type2\n");
 }
 else{
   fixSinkContractions(*seqProp,*dprop,*qkxTM_gaugeAPE ,QKXTM_TYPE2, filename_threep_part1, Nmom, momElem , testParticle , 1);
   printfQuda("Finish contractions for Neutron threepoint part1 type2\n");
 }

 // now we want back the smeared gauge to perfrom gaussian smearing
 qkxTM_gaugeAPE->loadGauge();





 //+++++++++++++++++++++++++++++++++++++++++++++++++++++++ part 2  type 2 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 //------------------------------------------------------------------------------------------------------------------------------------------

 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();

 // now we must create sequential propagator for part2 (PROTON,part2=dpart) , (NEUTRON,part2=upart)
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){


     if(testParticle == QKXTM_PROTON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )seqSourceFixSinkPart2(*qkxTM_vectorTmp,*uprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE2, testParticle);}
     else if(testParticle == QKXTM_NEUTRON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )seqSourceFixSinkPart2(*qkxTM_vectorTmp,*dprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE2, testParticle);}
     else{
       errorQuda("Not implement for both particles yet");}

     comm_barrier();
     qkxTM_vectorTmp->conjugate();
     qkxTM_vectorTmp->applyGamma5();

     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);


     if(testParticle == QKXTM_PROTON){
       b->changeTwist(QUDA_TWIST_PLUS);
       x->changeTwist(QUDA_TWIST_PLUS);
       b->Even().changeTwist(QUDA_TWIST_PLUS);
       b->Odd().changeTwist(QUDA_TWIST_PLUS);
       x->Even().changeTwist(QUDA_TWIST_PLUS);
       x->Odd().changeTwist(QUDA_TWIST_PLUS);
     }
     else{
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);

     }

     qkxTM_vectorGauss->uploadToCuda(*b);           
     zeroCuda(*x);
     dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 

     cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);

     dirac.Mdag(*in, *tmp);                        // indirect method needs apply of D^+ on source vector

     (*solve)(*out, *in);   
     dirac.reconstruct(*x, *b, inv_param->solution_type);
     qkxTM_vectorTmp->downloadFromCuda(*x);


     if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
       qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
     }

     seqProp->absorbVector(*qkxTM_vectorTmp,nu,c2);

     qkxTM_vectorTmp->zero();     
     qkxTM_vectorGauss->zero();

     delete tmp;
     printfQuda("Finish Inversion for sequential part 2 type2 quark %d/12 \n",nu*3+c2+1);

   }  // for loop to create sequential propagator

 // char filename_threep_part2[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/threep_part2";
 sprintf(filename_threep_part2,"%s_%s",threep_filename,"part2");
 // at this step to calculate derivative operators we need the unsmeared gauge
 qkxTM_gaugeAPE->loadGaugeFromBackup();

 if(testParticle == QKXTM_PROTON){
   fixSinkContractions(*seqProp,*dprop,*qkxTM_gaugeAPE ,QKXTM_TYPE2, filename_threep_part2, Nmom, momElem , testParticle , 2);
   printfQuda("Finish contractions for Proton threepoint part2 type2 \n");
 }
 else{
   fixSinkContractions(*seqProp,*uprop,*qkxTM_gaugeAPE, QKXTM_TYPE2, filename_threep_part2, Nmom, momElem , testParticle , 2);
   printfQuda("Finish contractions for Neutron threepoint part2 type2\n");
 }
 // now we want back the smeared gauge to perfrom gaussian smearing
 qkxTM_gaugeAPE->loadGauge();


 //------------------------------------------------------- finish type2 -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */



 delete seqProp;
 delete uprop3D;
 delete dprop3D;
 ////////////////////////////////////////////////
 uprop->rotateToPhysicalBasePlus();
 dprop->rotateToPhysicalBaseMinus();
 
 char filename_out_proton[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_proton.dat";
 char filename_out_neutron[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_neutron.dat";                   
 
  sprintf(filename_out_proton,"%s_%s.dat",twop_filename,"proton");
  sprintf(filename_out_neutron,"%s_%s.dat",twop_filename,"neutron");

 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){

     qkxTM_vectorTmp->copyPropagator(*uprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     uprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

     qkxTM_vectorTmp->copyPropagator(*dprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     dprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

   }

 performContractions(*uprop, *dprop, Nmom, momElem, filename_out_proton, filename_out_neutron);
 
 free(input_vector);
 

 delete uprop;
 delete dprop;
 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;

 delete d;
 delete dSloppy;
 delete dPre;
 
 freeGaugeQuda();

 // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
 saveTuneCache(getVerbosity());

 profileInvert[QUDA_PROFILE_TOTAL].Stop();
}


void quda::ThpTwp_Pion(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
		       int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename, char *threep_filename , int Nmom , int momElem[][3])  
{         


  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();    // temporary gauge to pefrorm APE smearing
  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(QKXTM_N2);  // because we want a backup in ram
  
  QKXTM_Propagator *uprop = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing
  QKXTM_Propagator *dprop = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing

  double plaq;
  qkxTM_gaugeTmp->packGauge(gaugeAPE);       
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

  qkxTM_gaugeAPE->packGaugeToBackup(gaugeAPE); // backup pure gauge on backup pointer

  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, *uprop); // use *uprop temporary for ape smearing
  qkxTM_gaugeAPE->justDownloadGauge();        // just download smeared gauge from gpu in gpu form
  delete qkxTM_gaugeTmp;                                // dont need it anymore

  // from now on we have on device pointer the smeared gauge in gpu form
  // on primary host pointer we have the smeared gauge in gpu form
  // on backup host pointer we have the unsmeared gauge in gpu form
  // if you want to have smeared gauge on gpu use loadGauge if you want unsmeared gauge on gpu use loadGaugeFromBackup
  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);



  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
 if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);

 // for up quark test


 // for loop  

 QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
 QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));


   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);

   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark
     
   
   zeroCuda(*x);
   
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   //  cudaColorSpinorField tmp_up(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   
    (*solve)(*out, *in);   

      
   dirac.reconstruct(*x, *b, inv_param->solution_type);

   qkxTM_vectorTmp->downloadFromCuda(*x);
   
   
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   int mu = ip/3;
   int ic = ip%3;
   uprop->absorbVector(*qkxTM_vectorTmp,mu,ic);

      
    delete tmp_up;

   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   

   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);

   qkxTM_vectorGauss->uploadToCuda(*b);               // re-upload source for inversion for down quark
        
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   //cudaColorSpinorField tmp_down(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector

   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);

   qkxTM_vectorTmp->downloadFromCuda(*x);


   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   dprop->absorbVector(*qkxTM_vectorTmp,mu,ic);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   
   
 }

 /*
 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   int my_src[4];   


   // change to up quark
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);
   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source

   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts
   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   //  cudaColorSpinorField tmp_up(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   
    (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }
   int mu = ip/3;
   int ic = ip%3;
   uprop->absorbVector(*qkxTM_vectorTmp,mu,ic);
   delete tmp_up;
   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   ///////////////////////////////////////////////////////////////////////////////////////////
   // change to down quark
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);

   // find where to put source   

   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   //  cudaColorSpinorField tmp_up(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   
    (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }
   dprop->absorbVector(*qkxTM_vectorTmp,mu,ic);
   delete tmp_down;
   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
 }
 */
 int my_fixSinkTime;
 QKXTM_Propagator *seqProp = new QKXTM_Propagator();      // sequential propagator

  // from now on start the sequential source      //////////////////////////////////////////////////////// 
 my_fixSinkTime = fixSinkTime - comm_coords(3) * X[3];        // choose gpus which have this time slice in memory
 QKXTM_Propagator3D *uprop3D = new QKXTM_Propagator3D();     // to save device memory because is fix sink we define 3D propagators 
 QKXTM_Propagator3D *dprop3D = new QKXTM_Propagator3D();     // to save device memory because is fix sink we define 3D propagators 
 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();
 
 
 if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
   uprop3D->absorbTimeSlice(*uprop,my_fixSinkTime);                      // copy only the specific timeslice 
   dprop3D->absorbTimeSlice(*dprop,my_fixSinkTime);                      // copy only the specific timeslice 
 } // the others perform nothing

 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )    qkxTM_vectorTmp->copyPropagator3D(*uprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // perform smearing
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )   uprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);    

     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )    qkxTM_vectorTmp->copyPropagator3D(*dprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // perform smearing
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )   dprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);    
   } // now the 3D propagators are smeared
 // First we create threepoint function for projector type1 
 //+++++++++++++++++++++++++++++++++++++++++++++++++++++++ pion^+ +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 //------------------------------------------------------------------------------------------------------------------------------------------
 // now we must create sequential propagator for part1 (PROTON,part1=upart) , (NEUTRON,part1=dpart)
 int pxi[7][3];
 pxi[0][0] = 0;
 pxi[0][1] = 0;
 pxi[0][2] = 0;

 pxi[1][0] = 1;
 pxi[1][1] = 0;
 pxi[1][2] = 0;

 pxi[2][0] = -1;
 pxi[2][1] = 0;
 pxi[2][2] = 0;

 pxi[3][0] = 0;
 pxi[3][1] = 1;
 pxi[3][2] = 0;
 
 pxi[4][0] = 0;
 pxi[4][1] = -1;
 pxi[4][2] = 0;
 
 pxi[5][0] = 0;
 pxi[5][1] = 0;
 pxi[5][2] = 1;

 pxi[6][0] = 0;
 pxi[6][1] = 0;
 pxi[6][2] = -1;

 char charPxi[7][257];
 strcpy(charPxi[0],"px0py0pz0");
 strcpy(charPxi[1],"px1py0pz0");
 strcpy(charPxi[2],"pxm1py0pz0");
 strcpy(charPxi[3],"px0py1pz0");
 strcpy(charPxi[4],"px0pym1pz0");
 strcpy(charPxi[5],"px0py0pz1");
 strcpy(charPxi[6],"px0py0pzm1");

 for(int im = 0 ; im < 7 ; im++){
   
   for(int nu = 0 ; nu < 4 ; nu++)
     for(int c2 = 0 ; c2 < 3 ; c2++){
       qkxTM_vectorTmp->zero();
       qkxTM_vectorGauss->zero();     
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) qkxTM_vectorTmp->getVectorProp3D(*uprop3D,my_fixSinkTime,nu,c2);
       qkxTM_vectorTmp->applyGamma5();
       qkxTM_vectorTmp->applyMomentum(pxi[im][0],pxi[im][1],pxi[im][2]);
       Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);
       qkxTM_vectorGauss->uploadToCuda(*b);           
       zeroCuda(*x);
       dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
       cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
       dirac.Mdag(*in, *tmp);                        // indirect method needs apply of D^+ on source vector
       (*solve)(*out, *in);   
       dirac.reconstruct(*x, *b, inv_param->solution_type);
       qkxTM_vectorTmp->downloadFromCuda(*x);
       if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	 qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
       }
       seqProp->absorbVector(*qkxTM_vectorTmp,nu,c2);
       qkxTM_vectorTmp->zero();     
       qkxTM_vectorGauss->zero();
       delete tmp;
       printfQuda("Finish Inversion %d/12 for pion^+ and %s momentum \n",nu*3+c2+1,charPxi[im]);
     }  // for loop to create sequential propagator
   
   char filename_threep_pionPlus[257];
   sprintf(filename_threep_pionPlus,"%s_%s.%s.dat",threep_filename,"pionPlus",charPxi[im]);
   // at this step to calculate derivative operators we need the unsmeared gauge
   //   qkxTM_gaugeAPE->loadGaugeFromBackup();
   fixSinkContractions(*seqProp,*uprop,*qkxTM_gaugeAPE ,QKXTM_TYPE1, filename_threep_pionPlus, Nmom, momElem , testParticle , 1);
   printfQuda("Finish contractions for Pion^+ type1\n");
   
   
   ///////////////////////////////////////// Pion^- ////////////////////////////////
   for(int nu = 0 ; nu < 4 ; nu++)
     for(int c2 = 0 ; c2 < 3 ; c2++){
       qkxTM_vectorTmp->zero();
       qkxTM_vectorGauss->zero();     
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) qkxTM_vectorTmp->getVectorProp3D(*dprop3D,my_fixSinkTime,nu,c2);
       qkxTM_vectorTmp->applyGamma5();
       qkxTM_vectorTmp->applyMomentum(pxi[im][0],pxi[im][1],pxi[im][2]);
       Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
       b->changeTwist(QUDA_TWIST_PLUS);
       x->changeTwist(QUDA_TWIST_PLUS);
       b->Even().changeTwist(QUDA_TWIST_PLUS);
       b->Odd().changeTwist(QUDA_TWIST_PLUS);
       x->Even().changeTwist(QUDA_TWIST_PLUS);
       x->Odd().changeTwist(QUDA_TWIST_PLUS);
       qkxTM_vectorGauss->uploadToCuda(*b);           
       zeroCuda(*x);
       dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
       cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
       dirac.Mdag(*in, *tmp);                        // indirect method needs apply of D^+ on source vector
       (*solve)(*out, *in);   
       dirac.reconstruct(*x, *b, inv_param->solution_type);
       qkxTM_vectorTmp->downloadFromCuda(*x);
       if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	 qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
       }
       seqProp->absorbVector(*qkxTM_vectorTmp,nu,c2);
       qkxTM_vectorTmp->zero();     
       qkxTM_vectorGauss->zero();
       delete tmp;
       printfQuda("Finish Inversion %d/12 for pion^- and %s momentum\n",nu*3+c2+1, charPxi[im]);
     }  // for loop to create sequential propagator
   char filename_threep_pionMinus[257];
   sprintf(filename_threep_pionMinus,"%s_%s.%s.dat",threep_filename,"pionMinus",charPxi[im]);
   // at this step to calculate derivative operators we need the unsmeared gauge
   //   qkxTM_gaugeAPE->loadGaugeFromBackup();
   fixSinkContractions(*seqProp,*dprop,*qkxTM_gaugeAPE ,QKXTM_TYPE1, filename_threep_pionMinus, Nmom, momElem , testParticle , 1);
   printfQuda("Finish contractions for Pion^- type1\n");
   
 }
 //////////////////////////////////////////////////////////////

 // now we want back the smeared gauge to perfrom gaussian smearing
 qkxTM_gaugeAPE->loadGauge();
 delete seqProp;
 delete uprop3D;
 delete dprop3D;
 ////////////////////////////////////////////////

 uprop->rotateToPhysicalBasePlus();
 dprop->rotateToPhysicalBaseMinus();
 
 char filename_out_pionPlus[257];
 char filename_out_pionMinus[257];

 int NmomTwop = 7;
 int (*momTwop)[3];

 momTwop = (int(*)[3]) calloc(7*3,sizeof(int));

 momTwop[0][0] = 0;
 momTwop[0][1] = 0;
 momTwop[0][2] = 0;

  momTwop[1][0] = 1;
  momTwop[1][1] = 0;
  momTwop[1][2] = 0;

  momTwop[2][0] = -1;
  momTwop[2][1] = 0;
  momTwop[2][2] = 0;

  momTwop[3][0] = 0;
  momTwop[3][1] = 1;
  momTwop[3][2] = 0;

  momTwop[4][0] = 0;
  momTwop[4][1] = -1;
  momTwop[4][2] = 0;

  momTwop[5][0] = 0;
  momTwop[5][1] = 0;
  momTwop[5][2] = 1;

  momTwop[6][0] = 0;
  momTwop[6][1] = 0;
  momTwop[6][2] = -1;

 
 sprintf(filename_out_pionPlus,"%s_%s.dat",twop_filename,"pionPlus");
 sprintf(filename_out_pionMinus,"%s_%s.dat",twop_filename,"pionMinus");

 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){

     qkxTM_vectorTmp->copyPropagator(*uprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     uprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

     qkxTM_vectorTmp->copyPropagator(*dprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     dprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

   }

 performContractionsPion(*uprop, NmomTwop, momTwop, filename_out_pionPlus);
 performContractionsPion(*dprop, NmomTwop, momTwop, filename_out_pionMinus);

 
 free(input_vector);
 

 delete uprop;
 delete dprop;
 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;

 delete d;
 delete dSloppy;
 delete dPre;
 
 freeGaugeQuda();

 // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
 saveTuneCache(getVerbosity());

 profileInvert[QUDA_PROFILE_TOTAL].Stop();
}




void quda::ThpTwp_nonLocal(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
			   int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename, char *threep_filename , int Nmom , int momElem[][3], int direction, int NmomSink, int momSink[][3])  
{         

  /*
  int NmomSink=1;
  int momSink[1][3];

  momSink[0][0] = 0;
  momSink[0][1] = 0;
  momSink[0][2] = 2;
  */
  /*
    This function creates threep point function of Proton using fix sink method and twop point function of proton
    Any additional information will be included here
   */

  //before we perform inversions we need to calculate the plaquette and APE smearing

  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();    // temporary gauge to pefrorm APE smearing
  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(QKXTM_N2);  // because we want a backup in ram
  
  QKXTM_Propagator *uprop = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing

  double plaq;
  qkxTM_gaugeTmp->packGauge(gaugeAPE);       
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

  qkxTM_gaugeAPE->packGaugeToBackup(gaugeAPE); // backup pure gauge on backup pointer

  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, *uprop); // use *uprop temporary for ape smearing

  
  // !!!!!!!!!!!!!!!!!!!!!!!!! remove these two after the tests and bring back smearing
  //  qkxTM_gaugeAPE->packGauge(gaugeAPE); // remove it later
  // qkxTM_gaugeAPE->loadGauge(); // remove it later
  

  double zeta = -0.6;
  //  double zeta = 0.45;
  //  double zeta = 0.;
  UxMomentumPhase(*qkxTM_gaugeAPE, momSink[0][0], momSink[0][1], momSink[0][2], zeta);

  qkxTM_gaugeAPE->justDownloadGauge();        // just download smeared gauge from gpu in gpu form
  delete qkxTM_gaugeTmp;                                // dont need it anymore


  // from now on we have on device pointer the smeared gauge in gpu form
  // on primary host pointer we have the smeared gauge in gpu form
  // on backup host pointer we have the unsmeared gauge in gpu form
  // if you want to have smeared gauge on gpu use loadGauge if you want unsmeared gauge on gpu use loadGaugeFromBackup
  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);

  QKXTM_Propagator *dprop = new QKXTM_Propagator();   

  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
 if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);

 // for up quark test


 // for loop  

 QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
 QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));



 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));


   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);

   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark
     
   
   zeroCuda(*x);
   
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   //  cudaColorSpinorField tmp_up(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   
    (*solve)(*out, *in);   

      
   dirac.reconstruct(*x, *b, inv_param->solution_type);

   qkxTM_vectorTmp->downloadFromCuda(*x);
   
   
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   int mu = ip/3;
   int ic = ip%3;
   uprop->absorbVector(*qkxTM_vectorTmp,mu,ic);

      
    delete tmp_up;

   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   

   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);

   qkxTM_vectorGauss->uploadToCuda(*b);               // re-upload source for inversion for down quark
        
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   //cudaColorSpinorField tmp_down(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector

   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);

   qkxTM_vectorTmp->downloadFromCuda(*x);


   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   dprop->absorbVector(*qkxTM_vectorTmp,mu,ic);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   
   
 }



  // from now on start the sequential source      //////////////////////////////////////////////////////// 

 int my_fixSinkTime;
 my_fixSinkTime = fixSinkTime - comm_coords(3) * X[3];        // choose gpus which have this time slice in memory
 

 QKXTM_Propagator3D *uprop3D = new QKXTM_Propagator3D();     // to save device memory because is fix sink we define 3D propagators 
 QKXTM_Propagator3D *dprop3D = new QKXTM_Propagator3D();
 QKXTM_Propagator *seqProp = new QKXTM_Propagator();      // sequential propagator
 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();
 
 
 if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
   uprop3D->absorbTimeSlice(*uprop,my_fixSinkTime);                      // copy only the specific timeslice 
   dprop3D->absorbTimeSlice(*dprop,my_fixSinkTime);                     
 } // the others perform nothing

   
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){
   
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )    qkxTM_vectorTmp->copyPropagator3D(*uprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // perform smearing
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )   uprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);    
   
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) qkxTM_vectorTmp->copyPropagator3D(*dprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) dprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);
   
   } // now the 3D propagators are smeared


 // Create the wilson path for z direction
 QKXTM_Gauge *qkxTM_gauge = new QKXTM_Gauge();
 qkxTM_gauge->packGauge(gaugeAPE);
 qkxTM_gauge->loadGauge();               // now we have the gauge on device                                                        
 double *deviceWilsonPath = createWilsonPath(*qkxTM_gauge,direction);
 double *deviceWilsonPathBwd = createWilsonPathBwd(*qkxTM_gauge, direction);
 delete qkxTM_gauge;
 

 // First we create threepoint function for projector type1 

 //+++++++++++++++++++++++++++++++++++++++++++++++++++++++ part 1  type 1 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 //------------------------------------------------------------------------------------------------------------------------------------------

 

 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();
 
  
 // now we must create sequential propagator for part1 (PROTON,part1=upart) , (NEUTRON,part1=dpart)
       
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){

     if(testParticle == QKXTM_PROTON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) seqSourceFixSinkPart1(*qkxTM_vectorTmp,*uprop3D,*dprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE1, testParticle);}
     else if(testParticle == QKXTM_NEUTRON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) seqSourceFixSinkPart1(*qkxTM_vectorTmp,*dprop3D,*uprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE1, testParticle);}
     else{
       errorQuda("Not implement for both particles yet");}

     comm_barrier();

     qkxTM_vectorTmp->applyMomentum(momSink[0][0],momSink[0][1],momSink[0][2]);
     qkxTM_vectorTmp->conjugate();    // unleash after test
     qkxTM_vectorTmp->applyGamma5();

     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     // Apply momentum to the sequential source

           
     if(testParticle == QKXTM_PROTON){
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);
     }
     else{
       b->changeTwist(QUDA_TWIST_PLUS);
       x->changeTwist(QUDA_TWIST_PLUS);
       b->Even().changeTwist(QUDA_TWIST_PLUS);
       b->Odd().changeTwist(QUDA_TWIST_PLUS);
       x->Even().changeTwist(QUDA_TWIST_PLUS);
       x->Odd().changeTwist(QUDA_TWIST_PLUS);

     }

     qkxTM_vectorGauss->uploadToCuda(*b);           
     zeroCuda(*x);
     dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   
     cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
     dirac.Mdag(*in, *tmp);                        // indirect method needs apply of D^+ on source vector

     (*solve)(*out, *in);   
     dirac.reconstruct(*x, *b, inv_param->solution_type);
     qkxTM_vectorTmp->downloadFromCuda(*x);


     if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
       qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
     }

     seqProp->absorbVector(*qkxTM_vectorTmp,nu,c2);

     qkxTM_vectorTmp->zero();     
     qkxTM_vectorGauss->zero();
    
     delete tmp;
     printfQuda("Finish Inversion for sequential part 1 type1 quark %d/12 \n",nu*3+c2+1);
     
   }  // for loop to create sequential propagator
  
 
 char filename_threep_part1[257];
 sprintf(filename_threep_part1,"%s_%s",threep_filename,"part1");

  qkxTM_gaugeAPE->loadGaugeFromBackup();
 if(testParticle == QKXTM_PROTON){
   fixSinkContractions_nonLocal(*seqProp,*uprop,*qkxTM_gaugeAPE , QKXTM_TYPE1, filename_threep_part1, Nmom, momElem , testParticle , 1, deviceWilsonPath,deviceWilsonPathBwd, direction);
   printfQuda("Finish contractions for Proton threepoint part1 type1\n");
 }
 else{
   fixSinkContractions_nonLocal(*seqProp,*dprop,*qkxTM_gaugeAPE , QKXTM_TYPE1, filename_threep_part1, Nmom, momElem , testParticle , 1, deviceWilsonPath,deviceWilsonPathBwd, direction);
   printfQuda("Finish contractions for Neutron threepoint part1 type1\n");
 }
 
 qkxTM_gaugeAPE->loadGauge();

 //Apply new contraction codes

 
 // at this step to calculate derivative operators we need the unsmeared gauge
 /*
 qkxTM_gaugeAPE->loadGaugeFromBackup();

 if(testParticle == QKXTM_PROTON){
   fixSinkContractions(*seqProp,*uprop,*qkxTM_gaugeAPE ,QKXTM_TYPE1, filename_threep_part1, Nmom, momElem , testParticle , 1);
   printfQuda("Finish contractions for Proton threepoint part1 type1\n");
 }
 else{
   fixSinkContractions(*seqProp,*dprop,*qkxTM_gaugeAPE ,QKXTM_TYPE1, filename_threep_part1, Nmom, momElem , testParticle , 1);
   printfQuda("Finish contractions for Neutron threepoint part1 type1\n");
 }

 // now we want back the smeared gauge to perfrom gaussian smearing
 qkxTM_gaugeAPE->loadGauge();
 */




 //+++++++++++++++++++++++++++++++++++++++++++++++++++++++ part 2  type 1 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 //------------------------------------------------------------------------------------------------------------------------------------------

 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();

 // now we must create sequential propagator for part2 (PROTON,part2=dpart) , (NEUTRON,part2=upart)
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){


     if(testParticle == QKXTM_PROTON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) seqSourceFixSinkPart2(*qkxTM_vectorTmp,*uprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE1, testParticle);}
     else if(testParticle == QKXTM_NEUTRON){
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) seqSourceFixSinkPart2(*qkxTM_vectorTmp,*dprop3D,my_fixSinkTime,nu,c2, QKXTM_TYPE1, testParticle);}
     else{
       errorQuda("Not implement for both particles yet");}
     
     comm_barrier();

     qkxTM_vectorTmp->applyMomentum(momSink[0][0],momSink[0][1],momSink[0][2]);
     qkxTM_vectorTmp->conjugate();
     qkxTM_vectorTmp->applyGamma5();

     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     
     // Apply momentum to the sequential source


     if(testParticle == QKXTM_PROTON){
       b->changeTwist(QUDA_TWIST_PLUS);
       x->changeTwist(QUDA_TWIST_PLUS);
       b->Even().changeTwist(QUDA_TWIST_PLUS);
       b->Odd().changeTwist(QUDA_TWIST_PLUS);
       x->Even().changeTwist(QUDA_TWIST_PLUS);
       x->Odd().changeTwist(QUDA_TWIST_PLUS);
     }
     else{
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);

     }

     qkxTM_vectorGauss->uploadToCuda(*b);           
     zeroCuda(*x);
     dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 

     cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);

     dirac.Mdag(*in, *tmp);                        // indirect method needs apply of D^+ on source vector

     (*solve)(*out, *in);   
     dirac.reconstruct(*x, *b, inv_param->solution_type);
     qkxTM_vectorTmp->downloadFromCuda(*x);


     if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
       qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
     }

     seqProp->absorbVector(*qkxTM_vectorTmp,nu,c2);

     qkxTM_vectorTmp->zero();     
     qkxTM_vectorGauss->zero();

     delete tmp;
     printfQuda("Finish Inversion for sequential part 2 type1 quark %d/12 \n",nu*3+c2+1);

   }  // for loop to create sequential propagator

 char filename_threep_part2[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/threep_part2";
 sprintf(filename_threep_part2,"%s_%s",threep_filename,"part2");

  qkxTM_gaugeAPE->loadGaugeFromBackup();
 if(testParticle == QKXTM_PROTON){
   fixSinkContractions_nonLocal(*seqProp,*dprop,*qkxTM_gaugeAPE , QKXTM_TYPE1, filename_threep_part2, Nmom, momElem , testParticle , 2, deviceWilsonPath, deviceWilsonPathBwd,direction);
   printfQuda("Finish contractions for Proton threepoint part1 type1\n");
 }
 else{
   fixSinkContractions_nonLocal(*seqProp,*uprop,*qkxTM_gaugeAPE , QKXTM_TYPE1, filename_threep_part2, Nmom, momElem , testParticle , 2, deviceWilsonPath, deviceWilsonPathBwd,direction);
   printfQuda("Finish contractions for Neutron threepoint part1 type1\n");
 }
qkxTM_gaugeAPE->loadGauge();
 /*
 
 // at this step to calculate derivative operators we need the unsmeared gauge
 qkxTM_gaugeAPE->loadGaugeFromBackup();

 if(testParticle == QKXTM_PROTON){
   fixSinkContractions(*seqProp,*dprop,*qkxTM_gaugeAPE ,QKXTM_TYPE1, filename_threep_part2, Nmom, momElem , testParticle , 2);
   printfQuda("Finish contractions for Proton threepoint part2 type1 \n");
 }
 else{
   fixSinkContractions(*seqProp,*uprop,*qkxTM_gaugeAPE, QKXTM_TYPE1, filename_threep_part2, Nmom, momElem , testParticle , 2);
   printfQuda("Finish contractions for Neutron threepoint part2 type1\n");
 }
 // now we want back the smeared gauge to perfrom gaussian smearing
 qkxTM_gaugeAPE->loadGauge();
 */

 //---------------------------------------       finish type1 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





 delete seqProp;
 delete uprop3D;
 delete dprop3D;
 ////////////////////////////////////////////////
 uprop->rotateToPhysicalBasePlus();
 dprop->rotateToPhysicalBaseMinus();
 
 char filename_out_proton[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_proton.dat";
 char filename_out_neutron[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_neutron.dat";                   
 
  sprintf(filename_out_proton,"%s_%s.dat",twop_filename,"proton");
  sprintf(filename_out_neutron,"%s_%s.dat",twop_filename,"neutron");

 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){

     qkxTM_vectorTmp->copyPropagator(*uprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     uprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

     qkxTM_vectorTmp->copyPropagator(*dprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     dprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

   }

 performContractions(*uprop, *dprop, NmomSink, momSink, filename_out_proton, filename_out_neutron);
 
 free(input_vector);
 

 delete uprop;
 delete dprop;
 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;

 delete d;
 delete dSloppy;
 delete dPre;
 
 freeGaugeQuda();
 cudaFree(deviceWilsonPath);
 cudaFree(deviceWilsonPathBwd);
 // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
 saveTuneCache(getVerbosity());

 profileInvert[QUDA_PROFILE_TOTAL].Stop();
}


// void quda::ThpTwp_Pion_nonLocal(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
// 			   int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename, char *threep_filename , int Nmom , int momElem[][3], int direction, int NmomSink, int momSink[][3])  
// {         

//   //before we perform inversions we need to calculate the plaquette and APE smearing

//   QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();    // temporary gauge to pefrorm APE smearing
//   QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(QKXTM_N2);  // because we want a backup in ram
  
//   QKXTM_Propagator *uprop = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing

//   double plaq;
//   qkxTM_gaugeTmp->packGauge(gaugeAPE);       
//   qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

//   qkxTM_gaugeAPE->packGaugeToBackup(gaugeAPE); // backup pure gauge on backup pointer

//   plaq = qkxTM_gaugeTmp->calculatePlaq();
//   printfQuda("Plaquette is %e\n",plaq);

//   APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, *uprop); // use *uprop temporary for ape smearing
//   qkxTM_gaugeAPE->justDownloadGauge();        // just download smeared gauge from gpu in gpu form
//   delete qkxTM_gaugeTmp;                                // dont need it anymore

//   // from now on we have on device pointer the smeared gauge in gpu form
//   // on primary host pointer we have the smeared gauge in gpu form
//   // on backup host pointer we have the unsmeared gauge in gpu form
//   // if you want to have smeared gauge on gpu use loadGauge if you want unsmeared gauge on gpu use loadGaugeFromBackup
//   plaq = qkxTM_gaugeAPE->calculatePlaq();
//   printfQuda("Plaquette smeared is %e\n",plaq);

//   //////////////////////////////////////////////////////////
//   // now we need to load the gauge field for inversion 
//   loadGaugeQuda((void*)gauge, gauge_param);
//   ////////////////////////////////////

//   profileInvert[QUDA_PROFILE_TOTAL].Start();
//   if (!initialized) errorQuda("QUDA not initialized");
//   // check the gauge fields have been created
//   cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
//   checkInvertParam(inv_param);


//   bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
//   bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
//   bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
//   bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

//   inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
//   if (!pc_solve) inv_param->spinorGiB *= 2;
//   inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
//   if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
//     inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
//   } else {
//     inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
//   }

//   inv_param->secs = 0;
//   inv_param->gflops = 0;
//   inv_param->iter = 0;

//   Dirac *d = NULL;
//   Dirac *dSloppy = NULL;
//   Dirac *dPre = NULL;

//   // create the dirac operator
//   createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

//   Dirac &dirac = *d;
//   Dirac &diracSloppy = *dSloppy;
//   Dirac &diracPre = *dPre;

//   profileInvert[QUDA_PROFILE_H2D].Start();

//   cudaColorSpinorField *b = NULL;
//   cudaColorSpinorField *x = NULL;
//   cudaColorSpinorField *in = NULL;
//   cudaColorSpinorField *out = NULL;

//   const int *X = cudaGauge->X();
  
//   // create cudaColorSpinorField vectors

//   ColorSpinorParam qkxTMParam; 
//   qkxTMParam.nColor = 3;
//   qkxTMParam.nSpin = 4;
//   qkxTMParam.nDim = 4;
//   for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
//   qkxTMParam.pad = inv_param->sp_pad;
//   qkxTMParam.precision = inv_param->cuda_prec;
//   qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
//   if(!pc_solution){
//     qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
//   }else{
//     qkxTMParam.x[0] /= 2;
//     qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
//   }
  
//   if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
//     errorQuda("qudaQKXTM package supports only dirac order");
  
//   qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
//   qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

//   if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
//     errorQuda("qudaQKXTM package supports only ukqcd");
  
//   qkxTMParam.gammaBasis = inv_param->gamma_basis;
//   qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
//   qkxTMParam.v = NULL;
//   qkxTMParam.norm = NULL;
//   qkxTMParam.verbose = inv_param->verbosity;

//   b = new cudaColorSpinorField(qkxTMParam);
//   x = new cudaColorSpinorField(qkxTMParam);


//   setDslashTuning(inv_param->tune, getVerbosity());
//   setBlasTuning(inv_param->tune, getVerbosity());
  

  

//  if (pc_solution && !pc_solve) {
//    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
//  }
 
//  if (!mat_solution && !pc_solution && pc_solve) {
//    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
//  }
 

//  if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
//  if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

//  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
//  Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);

//  // for up quark test


//  // for loop  

//  QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
//  QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


//  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));



//  for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
//    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));


//    // change to up quark
//    b->changeTwist(QUDA_TWIST_PLUS);
//    x->changeTwist(QUDA_TWIST_PLUS);

//    b->Even().changeTwist(QUDA_TWIST_PLUS);
//    b->Odd().changeTwist(QUDA_TWIST_PLUS);
//    x->Even().changeTwist(QUDA_TWIST_PLUS);
//    x->Odd().changeTwist(QUDA_TWIST_PLUS);

//    // find where to put source
//    int my_src[4];
//    for(int i = 0 ; i < 4 ; i++)
//      my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

//    if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
//      *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
//    qkxTM_vectorTmp->flagsToFalse();                                                 // hack
//    qkxTM_vectorTmp->packVector(input_vector);
//    qkxTM_vectorTmp->loadVector();
//    Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);

//    qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark
     
   
//    zeroCuda(*x);
   
//    dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   
//    cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
//    //  cudaColorSpinorField tmp_up(*in);
//    dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   
//     (*solve)(*out, *in);   

      
//    dirac.reconstruct(*x, *b, inv_param->solution_type);

//    qkxTM_vectorTmp->downloadFromCuda(*x);
   
   
//    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
//      qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
//    }

//    int mu = ip/3;
//    int ic = ip%3;
//    uprop->absorbVector(*qkxTM_vectorTmp,mu,ic);

      
//     delete tmp_up;

//    printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   

   
//  }



//   // from now on start the sequential source      //////////////////////////////////////////////////////// 

//  int my_fixSinkTime;
//  my_fixSinkTime = fixSinkTime - comm_coords(3) * X[3];        // choose gpus which have this time slice in memory
 

//  QKXTM_Propagator3D *uprop3D = new QKXTM_Propagator3D();     // to save device memory because is fix sink we define 3D propagators 
//  QKXTM_Propagator *seqProp = new QKXTM_Propagator();      // sequential propagator
//  qkxTM_vectorTmp->zero();
//  qkxTM_vectorGauss->zero();
 
 
//  if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
//    uprop3D->absorbTimeSlice(*uprop,my_fixSinkTime);                      // copy only the specific timeslice 
//  } // the others perform nothing

   
//  for(int nu = 0 ; nu < 4 ; nu++)
//    for(int c2 = 0 ; c2 < 3 ; c2++){
   
//      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )    qkxTM_vectorTmp->copyPropagator3D(*uprop3D,my_fixSinkTime,nu,c2);
//      comm_barrier();
//      Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // perform smearing
//      comm_barrier();
//      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )   uprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);    
      
//    } // now the 3D propagators are smeared


//  // Create the wilson path for z direction
//  QKXTM_Gauge *qkxTM_gauge = new QKXTM_Gauge();
//  qkxTM_gauge->packGauge(gaugeAPE);
//  qkxTM_gauge->loadGauge();               // now we have the gauge on device                                                        
//  double *deviceWilsonPath = createWilsonPath(*qkxTM_gauge,direction);
//  delete qkxTM_gauge;
 

//  // First we create threepoint function for projector type1 

//  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++ part 1  type 1 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  //------------------------------------------------------------------------------------------------------------------------------------------

 

 
  
//  // now we must create sequential propagator for part1 (PROTON,part1=upart) , (NEUTRON,part1=dpart)
       
//  for(int nu = 0 ; nu < 4 ; nu++)
//    for(int c2 = 0 ; c2 < 3 ; c2++){

//      qkxTM_vectorTmp->zero();
//      qkxTM_vectorGauss->zero();
     
//      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) qkxTM_vectorTmp->getVectorProp3D(*uprop3D,my_fixSinkTime,nu,c2);
//      qkxTM_vectorTmp->applyMomentum(momSink[0][0],momSink[0][1],momSink[0][2]);
//      qkxTM_vectorTmp->applyGamma5();

//      Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     
//      // Apply momentum to the sequential source

           

//        b->changeTwist(QUDA_TWIST_MINUS);
//        x->changeTwist(QUDA_TWIST_MINUS);
//        b->Even().changeTwist(QUDA_TWIST_MINUS);
//        b->Odd().changeTwist(QUDA_TWIST_MINUS);
//        x->Even().changeTwist(QUDA_TWIST_MINUS);
//        x->Odd().changeTwist(QUDA_TWIST_MINUS);

//      qkxTM_vectorGauss->uploadToCuda(*b);           
//      zeroCuda(*x);
//      dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   
//      cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
//      dirac.Mdag(*in, *tmp);                        // indirect method needs apply of D^+ on source vector

//      (*solve)(*out, *in);   
//      dirac.reconstruct(*x, *b, inv_param->solution_type);
//      qkxTM_vectorTmp->downloadFromCuda(*x);


//      if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
//        qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
//      }

//      seqProp->absorbVector(*qkxTM_vectorTmp,nu,c2);

//      qkxTM_vectorTmp->zero();     
//      qkxTM_vectorGauss->zero();
    
//      delete tmp;
//      printfQuda("Finish Inversion for sequential part 1 type1 quark %d/12 \n",nu*3+c2+1);
     
//    }  // for loop to create sequential propagator
  
 
//  char filename_threep_part1[257];
//  sprintf(filename_threep_part1,"%s_%s",threep_filename,"part1");

//   qkxTM_gaugeAPE->loadGaugeFromBackup();

//    fixSinkContractions_nonLocal(*seqProp,*uprop,*qkxTM_gaugeAPE ,QKXTM_TYPE1, filename_threep_part1, Nmom, momElem , testParticle , 1, deviceWilsonPath,direction);
//    printfQuda("Finish contractions for Proton threepoint part1 type1\n");
 
//  qkxTM_gaugeAPE->loadGauge();



//  delete seqProp;
//  delete uprop3D;

//  ////////////////////////////////////////////////
//  uprop->rotateToPhysicalBasePlus();
 
//  char filename_out_pion[257];
 
//   sprintf(filename_out_pion,"%s_%s.dat",twop_filename,"pion");

//  for(int nu = 0 ; nu < 4 ; nu++)
//    for(int c2 = 0 ; c2 < 3 ; c2++){

//      qkxTM_vectorTmp->copyPropagator(*uprop,nu,c2);
//      Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
//      uprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

//    }

//  performContractionsPion(*uprop, NmomSink, momSink, filename_out_pion);
 
//  free(input_vector);
 

//  delete uprop;
//  delete qkxTM_vectorTmp;
//  delete qkxTM_vectorGauss;
//  delete qkxTM_gaugeAPE;
//  delete solve;
//  delete b;
//  delete x;

//  delete d;
//  delete dSloppy;
//  delete dPre;
 
//  freeGaugeQuda();
//  cudaFree(deviceWilsonPath);
//  // popVerbosity();

//   // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
//  saveTuneCache(getVerbosity());

//  profileInvert[QUDA_PROFILE_TOTAL].Stop();
// }



static void createUndilutedNoiseVectorFixSink(double *noiseVec ,const int *X,const int fixTime , gsl_rng *rng){

  for(int iz = 0 ; iz < X[2] ; iz++)
    for(int iy = 0 ; iy < X[1] ; iy++)
      for(int ix = 0 ; ix < X[0] ; ix++)
	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int c1 = 0 ; c1 < 3 ; c1++){
	    
	    int randomNumber = gsl_rng_uniform_int(rng,4); // creates 0,1,2,3 uniformlly distributed
	    int pos = fixTime*X[2]*X[1]*X[0]*4*3*2 + iz*X[1]*X[0]*4*3*2 + iy*X[0]*4*3*2 + ix*4*3*2 + mu*3*2 + c1*2;
	    
	    switch(randomNumber){
	    case 0:
	      noiseVec[pos + 0] = 1.;
	      break;
	    case 1:
	      noiseVec[pos + 0] = -1.;
	      break;
	    case 2:
	      noiseVec[pos + 1] = 1.;
	      break;
	    case 3:
	      noiseVec[pos + 1] = -1.;
	      break;
	    }

	  }
  
}


static void createFulldilutedNoiseVectorFixSink(double *noiseVec ,const int *X,const int fixTime ,int mu,int c1,gsl_rng *rng){ // noise vector must be zero before use this function


  for(int iz = 0 ; iz < X[2] ; iz++)
    for(int iy = 0 ; iy < X[1] ; iy++)
      for(int ix = 0 ; ix < X[0] ; ix++){
	    
	    int randomNumber = gsl_rng_uniform_int(rng,4); // creates 0,1,2,3 uniformlly distributed
	    int pos = fixTime*X[2]*X[1]*X[0]*4*3*2 + iz*X[1]*X[0]*4*3*2 + iy*X[0]*4*3*2 + ix*4*3*2 + mu*3*2 + c1*2;
	    
	    switch(randomNumber){
	    case 0:
	      noiseVec[pos + 0] = 1.;
	      break;
	    case 1:
	      noiseVec[pos + 0] = -1.;
	      break;
	    case 2:
	      noiseVec[pos + 1] = 1.;
	      break;
	    case 3:
	      noiseVec[pos + 1] = -1.;
	      break;
	    }

	  }

}


// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// global variables
//FILE *ptr_file_proton[4] = {NULL,NULL,NULL,NULL};
//FILE *ptr_file_neutron[4] = {NULL,NULL,NULL,NULL};

void quda::ThpTwp_stoch(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
			int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename, 
			char *threep_filename , int Nmom , int momElem[][3], unsigned long int seed , int Nstoch)  
{         

  /*
    This function creates threep point function using stochastic method and twop using standard method
   */

  //before we perform inversions we need to calculate the plaquette and APE smearing

  if( seed == 0 ) errorQuda("Error seed must be different from 0");
  if( Nstoch <= 0) errorQuda("Error Nstoch must be greater than 1");

  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();    // temporary gauge to pefrorm APE smearing
  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(QKXTM_N2);  // because we want a backup in ram
  
  QKXTM_Propagator *uprop = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing

  double plaq;
  qkxTM_gaugeTmp->packGauge(gaugeAPE);       
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

  qkxTM_gaugeAPE->packGaugeToBackup(gaugeAPE); // backup pure gauge on backup pointer

  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, *uprop); // use *uprop temporary for ape smearing
  qkxTM_gaugeAPE->justDownloadGauge();        // just download smeared gauge from gpu in gpu form
  delete qkxTM_gaugeTmp;                                // dont need it anymore

  // from now on we have on device pointer the smeared gauge in gpu form
  // on primary host pointer we have the smeared gauge in gpu form
  // on backup host pointer we have the unsmeared gauge in gpu form
  // if you want to have smeared gauge on gpu use loadGauge if you want unsmeared gauge on gpu use loadGaugeFromBackup
  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);

  QKXTM_Propagator *dprop = new QKXTM_Propagator();   

  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
 if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);




 // for loop  

 QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
 QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

 double startClock,stopClock;

 if(input_vector == NULL) errorQuda("Error allocate input_vector on Host");

 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));


   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();

   startClock = MPI_Wtime();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   stopClock = MPI_Wtime();
   printfQuda("Smearing time is %f minutes\n",(stopClock-startClock)/60.);


   startClock = MPI_Wtime();
   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   //  cudaColorSpinorField tmp_up(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   stopClock = MPI_Wtime();
   printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);
   
   
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   int mu = ip/3;
   int ic = ip%3;
   uprop->absorbVector(*qkxTM_vectorTmp,mu,ic);
   delete tmp_up;

   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);

   qkxTM_vectorGauss->uploadToCuda(*b);               // re-upload source for inversion for down quark
        
   startClock = MPI_Wtime();
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   //cudaColorSpinorField tmp_down(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   stopClock = MPI_Wtime();
   printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);


   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   dprop->absorbVector(*qkxTM_vectorTmp,mu,ic);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   
   
   //TODO: In the future I will include strange quark
 }

 printfQuda("\n Start stochastic estimation of threepoint function\n\n");


 startClock = MPI_Wtime();
 
 int my_fixSinkTime;
 my_fixSinkTime = fixSinkTime - comm_coords(3) * X[3];        // choose gpus which have this time slice in memory
 
 QKXTM_Propagator3D *uprop3D = new QKXTM_Propagator3D();     // to save device memory because is fix sink we define 3D propagators 
 QKXTM_Propagator3D *dprop3D = new QKXTM_Propagator3D();

 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();
 
 if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
   uprop3D->absorbTimeSlice(*uprop,my_fixSinkTime);                      // copy only the specific timeslice 
   dprop3D->absorbTimeSlice(*dprop,my_fixSinkTime);                     
 } // the others perform nothing

   
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){
   
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )    qkxTM_vectorTmp->copyPropagator3D(*uprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // perform smearing
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )   uprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);    
   
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) qkxTM_vectorTmp->copyPropagator3D(*dprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) dprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);
     comm_barrier();
   } // now the 3D propagators are smeared

 uprop3D->broadcast(fixSinkTime); // broadcast the smeared propagators at tsink from the node which has the sink to the other
 dprop3D->broadcast(fixSinkTime);

 unsigned long int ranlux_seed = seed;
 srand(seed);
 for(int i = 0 ; i < comm_rank() ; i++)
   ranlux_seed = rand();
 gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlux);
 gsl_rng_set(rng,ranlux_seed);                 // initialize random number generator
 
  QKXTM_Vector3D *xi_unsmeared3D = new QKXTM_Vector3D();
 // QKXTM_Vector3D *xi_smeared3D = new QKXTM_Vector3D();

 stopClock = MPI_Wtime();
 printfQuda("Preparing time is %f minutes\n",(stopClock-startClock)/60.);

 //////////////////////////////
 ////////////////////////////////////// METHOD 1

 // allocate memory for quantities where you sum on
 // double *mem_oneD = (double*)calloc(8*4*X[3]*Nmom*4*4*2,sizeof(double));
 // double *mem_Local = (double*)calloc(10*X[3]*Nmom*4*4*2,sizeof(double));
 // double *mem_Noether = (double*)calloc(4*X[3]*Nmom*4*4*2,sizeof(double));

 // if(mem_oneD == NULL || 

 for(int istoch = 0 ; istoch < Nstoch ; istoch++)            // start loop over noise vectors
   for(int mu = 0 ; mu < 4 ; mu++)
     for(int c1 = 0 ; c1 < 3 ; c1++){

       startClock = MPI_Wtime();

       memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double)); // important
       qkxTM_vectorTmp->zero();
       qkxTM_vectorGauss->zero();
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) createFulldilutedNoiseVectorFixSink((double*) input_vector, X, my_fixSinkTime ,mu,c1, rng );
       comm_barrier();
       qkxTM_vectorTmp->flagsToFalse();                                                 // hack
       qkxTM_vectorTmp->packVector(input_vector);
       qkxTM_vectorTmp->loadVector();
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) xi_unsmeared3D->absorbTimeSlice(*qkxTM_vectorTmp,my_fixSinkTime);
       xi_unsmeared3D->broadcast(fixSinkTime);
       Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // now on qkxTM_vectorGauss we keep ksi smeared

       stopClock = MPI_Wtime();
       printfQuda("Smearing time is %f minutes\n",(stopClock-startClock)/60.);
       

       // at this step to calculate derivative operators we need the unsmeared gauge
       qkxTM_gaugeAPE->loadGaugeFromBackup();
       
       // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            create Upart          %%%%%%%%%%%%%%%%%%%%%%%%%%%%
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);
       

       startClock = MPI_Wtime();
       
       qkxTM_vectorGauss->uploadToCuda(*b);           
       zeroCuda(*x);
       dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector   
       cudaColorSpinorField *tmp_u = new cudaColorSpinorField(*in);
       dirac.Mdag(*in, *tmp_u);                        // indirect method needs apply of D^+ on source vector
       (*solve)(*out, *in);   
       dirac.reconstruct(*x, *b, inv_param->solution_type);
       qkxTM_vectorTmp->downloadFromCuda(*x);                          // now on qkxTM_vectorTmp we keep phi solution vector
       if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	 qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
       } 
       delete tmp_u;

       stopClock = MPI_Wtime();
       printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);

       char filename_upart[257];
       sprintf(filename_upart,"%s_%s",threep_filename,"upart");

       startClock = MPI_Wtime();
       threepStochUpart(*qkxTM_vectorTmp, *xi_unsmeared3D ,*uprop,*uprop3D ,*dprop3D, *qkxTM_gaugeAPE,fixSinkTime,filename_upart, Nmom, momElem);
       stopClock = MPI_Wtime();
       printfQuda("Contraction time is %f minutes\n",(stopClock-startClock)/60.);

       printfQuda("Finish contractions upart (Nstoch,mu,c1) : (%d,%d,%d)\n",istoch,mu,c1);
       // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            create Dpart          %%%%%%%%%%%%%%%%%%%%%%%%%%%%

       b->changeTwist(QUDA_TWIST_PLUS);
       x->changeTwist(QUDA_TWIST_PLUS);
       b->Even().changeTwist(QUDA_TWIST_PLUS);
       b->Odd().changeTwist(QUDA_TWIST_PLUS);
       x->Even().changeTwist(QUDA_TWIST_PLUS);
       x->Odd().changeTwist(QUDA_TWIST_PLUS);

       startClock = MPI_Wtime();
       qkxTM_vectorGauss->uploadToCuda(*b);           
       zeroCuda(*x);
       dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector   
       cudaColorSpinorField *tmp_d = new cudaColorSpinorField(*in);
       dirac.Mdag(*in, *tmp_d);                        // indirect method needs apply of D^+ on source vector
       (*solve)(*out, *in);   
       dirac.reconstruct(*x, *b, inv_param->solution_type);
       qkxTM_vectorTmp->downloadFromCuda(*x);                          // now on qkxTM_vectorTmp we keep phi solution vector
       if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	 qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
       } 
       delete tmp_d;
       stopClock = MPI_Wtime();
       printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);

       //	 printfQuda("Finish inversion dpart Nstoch : %d\n",istoch+1);
       char filename_dpart[257];
       sprintf(filename_dpart,"%s_%s",threep_filename,"dpart");

       startClock = MPI_Wtime();
       threepStochDpart(*qkxTM_vectorTmp, *xi_unsmeared3D ,*dprop,*uprop3D ,*dprop3D, *qkxTM_gaugeAPE, fixSinkTime, filename_dpart,Nmom, momElem);
       stopClock = MPI_Wtime();
       printfQuda("Contraction time is %f minutes\n",(stopClock-startClock)/60.);

       printfQuda("Finish contractions dpart (Nstoch,mu,c1) : (%d,%d,%d)\n",istoch,mu,c1);
       
       // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            create Spart          %%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
       // now we want back the smeared gauge to perfrom gaussian smearing
       qkxTM_gaugeAPE->loadGauge();
       
       
     } // close loop over stochastic vectors
 


 //////////////////////////////// METHOD2
 /*
 for(int istoch = 0 ; istoch < Nstoch ; istoch++){            // start loop over noise vectors

   startClock = MPI_Wtime();

   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double)); // important
   qkxTM_vectorTmp->zero();
   qkxTM_vectorGauss->zero();
   if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) createUndilutedNoiseVectorFixSink((double*) input_vector, X, my_fixSinkTime , rng );
   comm_barrier();

   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();


   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // now on qkxTM_vectorGauss we keep ksi smeared

   if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) xi_smeared3D->absorbTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime);
   xi_smeared3D->broadcast(fixSinkTime);

   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();


   
   stopClock = MPI_Wtime();
   printfQuda("Smearing time is %f minutes\n",(stopClock-startClock)/60.);
   


   
   // at this step to calculate derivative operators we need the unsmeared gauge
   qkxTM_gaugeAPE->loadGaugeFromBackup();
   
   // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            create Upart          %%%%%%%%%%%%%%%%%%%%%%%%%%%%
   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);
   
   
   startClock = MPI_Wtime();
   
   //   qkxTM_vectorGauss->uploadToCuda(*b);           
   qkxTM_vectorTmp->uploadToCuda(*b);           
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector   
   cudaColorSpinorField *tmp_u = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_u);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorGauss->downloadFromCuda(*x);                          // now on qkxTM_vectorTmp we keep phi solution vector

   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorGauss->scaleVector(2*inv_param->kappa);
   } 
   delete tmp_u;
   
   stopClock = MPI_Wtime();
   printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);
   
   char filename_upart[257];
   sprintf(filename_upart,"%s_%s",threep_filename,"upart");

   startClock = MPI_Wtime();
   threepStochUpart(*qkxTM_vectorGauss, *xi_smeared3D ,*uprop,*uprop3D ,*dprop3D, *qkxTM_gaugeAPE,fixSinkTime,filename_upart, Nmom, momElem);
   stopClock = MPI_Wtime();
   printfQuda("Contraction time is %f minutes\n",(stopClock-startClock)/60.);
   
   printfQuda("Finish contractions upart (Nstoch) : (%d)\n",istoch);
   // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            create Dpart          %%%%%%%%%%%%%%%%%%%%%%%%%%%%

   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);
   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   startClock = MPI_Wtime();
   qkxTM_vectorTmp->uploadToCuda(*b);           
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector   
   cudaColorSpinorField *tmp_d = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_d);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorGauss->downloadFromCuda(*x);                          // now on qkxTM_vectorTmp we keep phi solution vector
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	 qkxTM_vectorGauss->scaleVector(2*inv_param->kappa);
       } 
   delete tmp_d;
   stopClock = MPI_Wtime();
   printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);

   //	 printfQuda("Finish inversion dpart Nstoch : %d\n",istoch+1);
   char filename_dpart[257];
   sprintf(filename_dpart,"%s_%s",threep_filename,"dpart");
   
   startClock = MPI_Wtime();
   threepStochDpart(*qkxTM_vectorGauss, *xi_smeared3D ,*dprop,*uprop3D ,*dprop3D, *qkxTM_gaugeAPE, fixSinkTime, filename_dpart,Nmom, momElem);
   stopClock = MPI_Wtime();
   printfQuda("Contraction time is %f minutes\n",(stopClock-startClock)/60.);
   
   printfQuda("Finish contractions dpart (Nstoch) : (%d)\n",istoch);
   
   // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            create Spart          %%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
       // now we want back the smeared gauge to perfrom gaussian smearing
   qkxTM_gaugeAPE->loadGauge();
       
   
 } // close loop over stochastic vectors
 */


 //////////////////////////////////////////////// twop ///////////////////////

 uprop->rotateToPhysicalBasePlus();
 dprop->rotateToPhysicalBaseMinus();
 
 char filename_out_proton[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_proton.dat";
 char filename_out_neutron[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_neutron.dat";                   
 
  sprintf(filename_out_proton,"%s_%s.dat",twop_filename,"proton");
  sprintf(filename_out_neutron,"%s_%s.dat",twop_filename,"neutron");

 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){

     qkxTM_vectorTmp->copyPropagator(*uprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     uprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

     qkxTM_vectorTmp->copyPropagator(*dprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     dprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

   }

 performContractions(*uprop, *dprop, Nmom, momElem, filename_out_proton, filename_out_neutron);


 delete uprop3D;
 delete dprop3D;
 delete xi_unsmeared3D;

 gsl_rng_free(rng);
 free(input_vector);
 delete uprop;
 delete dprop;
 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;
  
  freeGaugeQuda();

  // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());
  
  profileInvert[QUDA_PROFILE_TOTAL].Stop();
  
  
}

//******************************************************************************************************************************************************
//******************************************************************************************************************************************************
void quda::ThpTwp_stoch_WilsonLinks(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
			int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename, 
			char *threep_filename , int Nmom , int momElem[][3], unsigned long int seed , int Nstoch,int NmomSink,int momSink[][3])  
{         

  /*
    This function creates threep point function using stochastic method and twop using standard method
   */

  //before we perform inversions we need to calculate the plaquette and APE smearing

  if( seed == 0 ) errorQuda("Error seed must be different from 0");
  if( Nstoch <= 0) errorQuda("Error Nstoch must be greater than 1");

  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();    // temporary gauge to pefrorm APE smearing
  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(QKXTM_N2);  // because we want a backup in ram
  
  QKXTM_Propagator *uprop = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing

  double plaq;
  qkxTM_gaugeTmp->packGauge(gaugeAPE);       
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

  qkxTM_gaugeAPE->packGaugeToBackup(gaugeAPE); // backup pure gauge on backup pointer

  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, *uprop); // use *uprop temporary for ape smearing
  qkxTM_gaugeAPE->justDownloadGauge();        // just download smeared gauge from gpu in gpu form
  delete qkxTM_gaugeTmp;                                // dont need it anymore

  // from now on we have on device pointer the smeared gauge in gpu form
  // on primary host pointer we have the smeared gauge in gpu form
  // on backup host pointer we have the unsmeared gauge in gpu form
  // if you want to have smeared gauge on gpu use loadGauge if you want unsmeared gauge on gpu use loadGaugeFromBackup
  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);

  QKXTM_Propagator *dprop = new QKXTM_Propagator();   

  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
 if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);




 // for loop  

 QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
 QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

 double startClock,stopClock;

 if(input_vector == NULL) errorQuda("Error allocate input_vector on Host");

 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));


   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();

   startClock = MPI_Wtime();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   stopClock = MPI_Wtime();
   printfQuda("Smearing time is %f minutes\n",(stopClock-startClock)/60.);


   startClock = MPI_Wtime();
   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   //  cudaColorSpinorField tmp_up(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   stopClock = MPI_Wtime();
   printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);
   
   
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   int mu = ip/3;
   int ic = ip%3;
   uprop->absorbVector(*qkxTM_vectorTmp,mu,ic);
   delete tmp_up;

   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);

   qkxTM_vectorGauss->uploadToCuda(*b);               // re-upload source for inversion for down quark
        
   startClock = MPI_Wtime();
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   //cudaColorSpinorField tmp_down(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   stopClock = MPI_Wtime();
   printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);


   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   dprop->absorbVector(*qkxTM_vectorTmp,mu,ic);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   
   
   //TODO: In the future I will include strange quark
 }

 printfQuda("\n Start stochastic estimation of threepoint function\n\n");


 startClock = MPI_Wtime();
 
 int my_fixSinkTime;
 my_fixSinkTime = fixSinkTime - comm_coords(3) * X[3];        // choose gpus which have this time slice in memory
 
 QKXTM_Propagator3D *uprop3D = new QKXTM_Propagator3D();     // to save device memory because is fix sink we define 3D propagators 
 QKXTM_Propagator3D *dprop3D = new QKXTM_Propagator3D();

 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();
 
 if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
   uprop3D->absorbTimeSlice(*uprop,my_fixSinkTime);                      // copy only the specific timeslice 
   dprop3D->absorbTimeSlice(*dprop,my_fixSinkTime);                     
 } // the others perform nothing

   
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){
   
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )    qkxTM_vectorTmp->copyPropagator3D(*uprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // perform smearing
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )   uprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);    
   
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) qkxTM_vectorTmp->copyPropagator3D(*dprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) dprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);
     comm_barrier();
   } // now the 3D propagators are smeared

 uprop3D->broadcast(fixSinkTime); // broadcast the smeared propagators at tsink from the node which has the sink to the other
 dprop3D->broadcast(fixSinkTime);

 unsigned long int ranlux_seed = seed;
 srand(seed);
 for(int i = 0 ; i < comm_rank() ; i++)
   ranlux_seed = rand();
 gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlux);
 gsl_rng_set(rng,ranlux_seed);                 // initialize random number generator
 
  QKXTM_Vector3D *xi_unsmeared3D = new QKXTM_Vector3D();
 // QKXTM_Vector3D *xi_smeared3D = new QKXTM_Vector3D();

 stopClock = MPI_Wtime();
 printfQuda("Preparing time is %f minutes\n",(stopClock-startClock)/60.);

 //////////////////////////////
 ////////////////////////////////////// METHOD 1

 // allocate memory for quantities where you sum on
 // double *mem_oneD = (double*)calloc(8*4*X[3]*Nmom*4*4*2,sizeof(double));
 // double *mem_Local = (double*)calloc(10*X[3]*Nmom*4*4*2,sizeof(double));
 // double *mem_Noether = (double*)calloc(4*X[3]*Nmom*4*4*2,sizeof(double));

 // if(mem_oneD == NULL || 


 // Create the wilson path for z direction
 QKXTM_Gauge *qkxTM_gauge = new QKXTM_Gauge();
 qkxTM_gauge->packGauge(gaugeAPE);
 qkxTM_gauge->loadGauge();               // now we have the gauge on device                                                        
 double *deviceWilsonPath = createWilsonPath(*qkxTM_gauge);

 delete qkxTM_gauge;

 for(int istoch = 0 ; istoch < Nstoch ; istoch++)            // start loop over noise vectors
   for(int mu = 0 ; mu < 4 ; mu++)
     for(int c1 = 0 ; c1 < 3 ; c1++){

       startClock = MPI_Wtime();

       memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double)); // important
       qkxTM_vectorTmp->zero();
       qkxTM_vectorGauss->zero();
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) createFulldilutedNoiseVectorFixSink((double*) input_vector, X, my_fixSinkTime ,mu,c1, rng );
       //       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) createUndilutedNoiseVectorFixSink((double*) input_vector, X, my_fixSinkTime , rng );

       comm_barrier();
       qkxTM_vectorTmp->flagsToFalse();                                                 // hack
       qkxTM_vectorTmp->packVector(input_vector);
       qkxTM_vectorTmp->loadVector();
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) xi_unsmeared3D->absorbTimeSlice(*qkxTM_vectorTmp,my_fixSinkTime);
       xi_unsmeared3D->broadcast(fixSinkTime);
       Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // now on qkxTM_vectorGauss we keep ksi smeared

       stopClock = MPI_Wtime();
       printfQuda("Smearing time is %f minutes\n",(stopClock-startClock)/60.);
       

       // at this step to calculate derivative operators we need the unsmeared gauge
       //   qkxTM_gaugeAPE->loadGaugeFromBackup();
       
       // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            create Upart          %%%%%%%%%%%%%%%%%%%%%%%%%%%%
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);
       

       startClock = MPI_Wtime();
       
       qkxTM_vectorGauss->uploadToCuda(*b);           
       zeroCuda(*x);
       dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector   
       cudaColorSpinorField *tmp_u = new cudaColorSpinorField(*in);
       dirac.Mdag(*in, *tmp_u);                        // indirect method needs apply of D^+ on source vector
       (*solve)(*out, *in);   
       dirac.reconstruct(*x, *b, inv_param->solution_type);
       qkxTM_vectorTmp->downloadFromCuda(*x);                          // now on qkxTM_vectorTmp we keep phi solution vector
       if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	 qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
       } 
       delete tmp_u;

       stopClock = MPI_Wtime();
       printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);

       char filename_upart[257];
       sprintf(filename_upart,"%s_%s",threep_filename,"upart");

       startClock = MPI_Wtime();
       threepStochUpart_WilsonLinks(*qkxTM_vectorTmp, *xi_unsmeared3D ,*uprop,*uprop3D ,*dprop3D, deviceWilsonPath,fixSinkTime,filename_upart, Nmom, momElem,NmomSink,momSink);
       stopClock = MPI_Wtime();
       printfQuda("Contraction time is %f minutes\n",(stopClock-startClock)/60.);

       printfQuda("Finish contractions upart (Nstoch,mu,c1) : (%d,%d,%d)\n",istoch,mu,c1);
       //printfQuda("Finish contractions upart (Nstoch) : (%d)\n",istoch);
       // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            create Dpart          %%%%%%%%%%%%%%%%%%%%%%%%%%%%

       b->changeTwist(QUDA_TWIST_PLUS);
       x->changeTwist(QUDA_TWIST_PLUS);
       b->Even().changeTwist(QUDA_TWIST_PLUS);
       b->Odd().changeTwist(QUDA_TWIST_PLUS);
       x->Even().changeTwist(QUDA_TWIST_PLUS);
       x->Odd().changeTwist(QUDA_TWIST_PLUS);

       startClock = MPI_Wtime();
       qkxTM_vectorGauss->uploadToCuda(*b);           
       zeroCuda(*x);
       dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector   
       cudaColorSpinorField *tmp_d = new cudaColorSpinorField(*in);
       dirac.Mdag(*in, *tmp_d);                        // indirect method needs apply of D^+ on source vector
       (*solve)(*out, *in);   
       dirac.reconstruct(*x, *b, inv_param->solution_type);
       qkxTM_vectorTmp->downloadFromCuda(*x);                          // now on qkxTM_vectorTmp we keep phi solution vector
       if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	 qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
       } 
       delete tmp_d;
       stopClock = MPI_Wtime();
       printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);

       //	 printfQuda("Finish inversion dpart Nstoch : %d\n",istoch+1);
       char filename_dpart[257];
       sprintf(filename_dpart,"%s_%s",threep_filename,"dpart");

       startClock = MPI_Wtime();
       threepStochDpart_WilsonLinks(*qkxTM_vectorTmp, *xi_unsmeared3D ,*dprop,*uprop3D ,*dprop3D, deviceWilsonPath, fixSinkTime, filename_dpart,Nmom, momElem,NmomSink,momSink);
       stopClock = MPI_Wtime();
       printfQuda("Contraction time is %f minutes\n",(stopClock-startClock)/60.);

       printfQuda("Finish contractions dpart (Nstoch,mu,c1) : (%d,%d,%d)\n",istoch,mu,c1);
       //printfQuda("Finish contractions dpart (Nstoch) : (%d)\n",istoch);
       // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            create Spart          %%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
       // now we want back the smeared gauge to perfrom gaussian smearing
       //qkxTM_gaugeAPE->loadGauge();
       
       
     } // close loop over stochastic vectors
 




 //////////////////////////////////////////////// twop ///////////////////////
 /*
 int NmomSink = 4;
 int momElemSink[4][3];
 momElemSink[0][0] = 0;
 momElemSink[0][1] = 0;
 momElemSink[0][2] = 0;

 momElemSink[1][0] = 0;
 momElemSink[1][1] = 0;
 momElemSink[1][2] = 1;

 momElemSink[2][0] = 0;
 momElemSink[2][1] = 0;
 momElemSink[2][2] = 2;

 momElemSink[3][0] = 0;
 momElemSink[3][1] = 0;
 momElemSink[3][2] = 3;
 */

 uprop->rotateToPhysicalBasePlus();
 dprop->rotateToPhysicalBaseMinus();
 
 char filename_out_proton[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_proton.dat";
 char filename_out_neutron[257];// = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_neutron.dat";                   
 
  sprintf(filename_out_proton,"%s_%s.dat",twop_filename,"proton");
  sprintf(filename_out_neutron,"%s_%s.dat",twop_filename,"neutron");

 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){

     qkxTM_vectorTmp->copyPropagator(*uprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     uprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

     qkxTM_vectorTmp->copyPropagator(*dprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     dprop->absorbVector(*qkxTM_vectorGauss,nu,c2);

   }

 performContractions(*uprop, *dprop, NmomSink, momSink, filename_out_proton, filename_out_neutron);


 delete uprop3D;
 delete dprop3D;
 delete xi_unsmeared3D;

 gsl_rng_free(rng);
 free(input_vector);
 delete uprop;
 delete dprop;
 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;
  
  freeGaugeQuda();
  cudaFree(deviceWilsonPath);
  // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());
  
  profileInvert[QUDA_PROFILE_TOTAL].Stop();
  
  
}

///////////////////////////////////////////
/*
 *
 *
 *
 *
 *//////////////////////////////

//#define CHECK_PION

#ifdef CHECK_PION
extern int G_localL[QUDAQKXTM_DIM];
extern int G_nColor;
extern int G_nSpin;
extern int G_localVolume;
#endif

void quda::ThpTwp_stoch_Pion_WilsonLinks(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
			int *sourcePosition, int fixSinkTime, char *twop_filename, 
			char *threep_filename , int Nmom , int momElem[][3], unsigned long int seed , int Nstoch,int NmomSink,int momSink[][3])  
{         
#ifdef CHECK_PION
  int G_localVolume3D = G_localL[0]*G_localL[1]*G_localL[2];
#endif
  /*
    This function creates threep point function using stochastic method and twop using standard method
   */

  //before we perform inversions we need to calculate the plaquette and APE smearing

  if( seed == 0 ) errorQuda("Error seed must be different from 0");
  if( Nstoch <= 0) errorQuda("Error Nstoch must be greater than 1");

  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();    // temporary gauge to pefrorm APE smearing
  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(QKXTM_N2);  // because we want a backup in ram
  
  QKXTM_Propagator *uprop = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing

  double plaq;
  qkxTM_gaugeTmp->packGauge(gaugeAPE);       
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

  qkxTM_gaugeAPE->packGaugeToBackup(gaugeAPE); // backup pure gauge on backup pointer

  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, *uprop); // use *uprop temporary for ape smearing
  qkxTM_gaugeAPE->justDownloadGauge();        // just download smeared gauge from gpu in gpu form
  delete qkxTM_gaugeTmp;                                // dont need it anymore

  // from now on we have on device pointer the smeared gauge in gpu form
  // on primary host pointer we have the smeared gauge in gpu form
  // on backup host pointer we have the unsmeared gauge in gpu form
  // if you want to have smeared gauge on gpu use loadGauge if you want unsmeared gauge on gpu use loadGaugeFromBackup
  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);


  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
 if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);




 // for loop  

 QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
 QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

 double startClock,stopClock;

 if(input_vector == NULL) errorQuda("Error allocate input_vector on Host");

 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));


   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();

   startClock = MPI_Wtime();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   stopClock = MPI_Wtime();
   printfQuda("Smearing time is %f minutes\n",(stopClock-startClock)/60.);


   startClock = MPI_Wtime();
   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   //  cudaColorSpinorField tmp_up(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   stopClock = MPI_Wtime();
   printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);
   
   
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   int mu = ip/3;
   int ic = ip%3;
   uprop->absorbVector(*qkxTM_vectorTmp,mu,ic);
   delete tmp_up;

   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
 }


 /*
#ifdef CHECK_PION
 uprop->download();
 FILE *ptr_prop;
 ptr_prop=fopen("fwd_prop.dat","w");
 for(int iv = 0 ; iv < G_localVolume ; iv++)
   for(int mu = 0 ; mu < G_nSpin ; mu++)                // always work with format colors inside spins
     for(int nu = 0 ; nu < G_nSpin ; nu++)
       for(int c1 = 0 ; c1 < G_nColor ; c1++)
	 for(int c2 = 0 ; c2 < G_nColor ; c2++){
	     fprintf(ptr_prop,"%+e %+e\n",uprop->H_elem()[iv*G_nSpin*G_nSpin*G_nColor*G_nColor*2 + mu*G_nSpin*G_nColor*G_nColor*2 + nu*G_nColor*G_nColor*2 + c1*G_nColor*2 + c2*2 + 0],uprop->H_elem()[iv*G_nSpin*G_nSpin*G_nColor*G_nColor*2 + mu*G_nSpin*G_nColor*G_nColor*2 + nu*G_nColor*G_nColor*2 + c1*G_nColor*2 + c2*2 + 1] );
	   }
 fclose(ptr_prop);
#endif
 */
 printfQuda("\n Start stochastic estimation of threepoint function\n\n");


 startClock = MPI_Wtime();
 
 int my_fixSinkTime;
 my_fixSinkTime = fixSinkTime - comm_coords(3) * X[3];        // choose gpus which have this time slice in memory
 
 QKXTM_Propagator3D *uprop3D = new QKXTM_Propagator3D();     // to save device memory because is fix sink we define 3D propagators 


 qkxTM_vectorTmp->zero();
 qkxTM_vectorGauss->zero();
 
 if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
   uprop3D->absorbTimeSlice(*uprop,my_fixSinkTime);                      // copy only the specific timeslice 
 } // the others perform nothing

   
 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){
   
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )    qkxTM_vectorTmp->copyPropagator3D(*uprop3D,my_fixSinkTime,nu,c2);
     comm_barrier();
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // perform smearing
     comm_barrier();
     if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) )   uprop3D->absorbVectorTimeSlice(*qkxTM_vectorGauss,my_fixSinkTime,nu,c2);       
     comm_barrier();
   } // now the 3D propagators are smeared

 uprop3D->broadcast(fixSinkTime); // broadcast the smeared propagators at tsink from the node which has the sink to the other

 /*
#ifdef CHECK_PION
 uprop3D->download();
 FILE *ptr_prop3D;
 ptr_prop3D=fopen("fwd_prop3D.dat","w");
 for(int iv = 0 ; iv < G_localVolume3D ; iv++)
   for(int mu = 0 ; mu < G_nSpin ; mu++)                // always work with format colors inside spins
     for(int nu = 0 ; nu < G_nSpin ; nu++)
       for(int c1 = 0 ; c1 < G_nColor ; c1++)
	 for(int c2 = 0 ; c2 < G_nColor ; c2++){
	     fprintf(ptr_prop3D,"%+e %+e\n",uprop3D->H_elem()[iv*G_nSpin*G_nSpin*G_nColor*G_nColor*2 + mu*G_nSpin*G_nColor*G_nColor*2 + nu*G_nColor*G_nColor*2 + c1*G_nColor*2 + c2*2 + 0],uprop3D->H_elem()[iv*G_nSpin*G_nSpin*G_nColor*G_nColor*2 + mu*G_nSpin*G_nColor*G_nColor*2 + nu*G_nColor*G_nColor*2 + c1*G_nColor*2 + c2*2 + 1] );
	   }
 fclose(ptr_prop3D);

#endif
 */
 unsigned long int ranlux_seed = seed;
 srand(seed);
 for(int i = 0 ; i < comm_rank() ; i++)
   ranlux_seed = rand();
 gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlux);
 gsl_rng_set(rng,ranlux_seed);                 // initialize random number generator
 
  QKXTM_Vector3D *xi_unsmeared3D = new QKXTM_Vector3D();
 // QKXTM_Vector3D *xi_smeared3D = new QKXTM_Vector3D();

 stopClock = MPI_Wtime();
 printfQuda("Preparing time is %f minutes\n",(stopClock-startClock)/60.);

 // Create the wilson path for z direction
 QKXTM_Gauge *qkxTM_gauge = new QKXTM_Gauge();
 qkxTM_gauge->packGauge(gaugeAPE);
 qkxTM_gauge->loadGauge();               // now we have the gauge on device                                                        
 double *deviceWilsonPath = createWilsonPath(*qkxTM_gauge);

 delete qkxTM_gauge;

 for(int istoch = 0 ; istoch < Nstoch ; istoch++)
   for(int mu = 0 ; mu < 4 ; mu++)
     for(int c1 = 0 ; c1 < 3 ; c1++){

       startClock = MPI_Wtime();

       memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double)); // important
       qkxTM_vectorTmp->zero();
       qkxTM_vectorGauss->zero();
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) createFulldilutedNoiseVectorFixSink((double*) input_vector, X, my_fixSinkTime ,mu,c1, rng );
       //       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) createUndilutedNoiseVectorFixSink((double*) input_vector, X, my_fixSinkTime , rng );

       comm_barrier();
       qkxTM_vectorTmp->flagsToFalse();                                                 // hack
       qkxTM_vectorTmp->packVector(input_vector);
       qkxTM_vectorTmp->loadVector();
       if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) xi_unsmeared3D->absorbTimeSlice(*qkxTM_vectorTmp,my_fixSinkTime);
       xi_unsmeared3D->broadcast(fixSinkTime);
       Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);  // now on qkxTM_vectorGauss we keep ksi smeared

       stopClock = MPI_Wtime();
       printfQuda("Smearing time is %f minutes\n",(stopClock-startClock)/60.);
       

       // at this step to calculate derivative operators we need the unsmeared gauge
       //   qkxTM_gaugeAPE->loadGaugeFromBackup();
       
       b->changeTwist(QUDA_TWIST_MINUS);
       x->changeTwist(QUDA_TWIST_MINUS);
       b->Even().changeTwist(QUDA_TWIST_MINUS);
       b->Odd().changeTwist(QUDA_TWIST_MINUS);
       x->Even().changeTwist(QUDA_TWIST_MINUS);
       x->Odd().changeTwist(QUDA_TWIST_MINUS);
       

       startClock = MPI_Wtime();
       
       qkxTM_vectorGauss->uploadToCuda(*b);           
       zeroCuda(*x);
       dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector   
       cudaColorSpinorField *tmp_u = new cudaColorSpinorField(*in);
       dirac.Mdag(*in, *tmp_u);                        // indirect method needs apply of D^+ on source vector
       (*solve)(*out, *in);   
       dirac.reconstruct(*x, *b, inv_param->solution_type);
       qkxTM_vectorTmp->downloadFromCuda(*x);                          // now on qkxTM_vectorTmp we keep phi solution vector
       if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	 qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
       } 
       delete tmp_u;

       stopClock = MPI_Wtime();
       printfQuda("Inversion time is %f minutes\n",(stopClock-startClock)/60.);

       startClock = MPI_Wtime();
       threepStochPion_WilsonLinks(*qkxTM_vectorTmp, *xi_unsmeared3D ,*uprop,*uprop3D , deviceWilsonPath,fixSinkTime,threep_filename, Nmom, momElem,NmomSink,momSink);

       /*
#ifdef CHECK_PION
       FILE *ptr_phi,*ptr_xi;
       ptr_phi=fopen("phi.dat","w");
       ptr_xi=fopen("xi.dat","w");
       qkxTM_vectorTmp->download();
       xi_unsmeared3D->download();

       for(int iv = 0 ; iv < G_localVolume ; iv++)
	 for(int mu = 0 ; mu < G_nSpin ; mu++)
	   for(int c1 = 0 ; c1 < G_nColor ; c1++){
	     fprintf(ptr_phi,"%+e %+e\n", qkxTM_vectorTmp->H_elem()[iv*G_nSpin*G_nColor*2 + mu*G_nColor*2 + c1*2  + 0],
		     qkxTM_vectorTmp->H_elem()[iv*G_nSpin*G_nColor*2 + mu*G_nColor*2 + c1*2  + 1]);
	   }

       for(int iv = 0 ; iv < G_localVolume3D ; iv++)
	 for(int mu = 0 ; mu < G_nSpin ; mu++)
	   for(int c1 = 0 ; c1 < G_nColor ; c1++){
	     fprintf(ptr_xi,"%+e %+e\n", xi_unsmeared3D->H_elem()[iv*G_nSpin*G_nColor*2 + mu*G_nColor*2 + c1*2  + 0],
		     xi_unsmeared3D->H_elem()[iv*G_nSpin*G_nColor*2 + mu*G_nColor*2 + c1*2  + 1]);
	   }

       fclose(ptr_phi);
       fclose(ptr_xi);
#endif
       */
       stopClock = MPI_Wtime();
       printfQuda("Contraction time is %f minutes\n",(stopClock-startClock)/60.);

       printfQuda("Finish contractions (Nstoch,mu,c1) : (%d,%d,%d)\n",istoch,mu,c1);
       
       
     } // close loop over stochastic vectors
 




 //////////////////////////////////////////////// twop ///////////////////////
 
 char filename_out_pion[257];
 int NmomSinkTwop = NmomSink + 1;
 int (*momSinkTwop)[3];
 momSinkTwop = (int(*)[3]) calloc(NmomSinkTwop*3,sizeof(int));

 momSinkTwop[0][0]=0;
 momSinkTwop[0][1]=0;
 momSinkTwop[0][2]=0;

 for(int i = 0 ; i < NmomSink ; i++)
   for(int j = 0 ; j < 3 ; j++)
     momSinkTwop[i+1][j] = momSink[i][j];

 sprintf(filename_out_pion,"%s_%s.dat",twop_filename,"pion");

 for(int nu = 0 ; nu < 4 ; nu++)
   for(int c2 = 0 ; c2 < 3 ; c2++){
     qkxTM_vectorTmp->copyPropagator(*uprop,nu,c2);
     Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
     uprop->absorbVector(*qkxTM_vectorGauss,nu,c2);
   }


#ifdef CHECK_PION
 uprop->download();
 FILE *ptr_propSS;
 ptr_propSS=fopen("fwd_propSS.dat","w");
 for(int iv = 0 ; iv < G_localVolume ; iv++)
   for(int mu = 0 ; mu < G_nSpin ; mu++)                // always work with format colors inside spins
     for(int nu = 0 ; nu < G_nSpin ; nu++)
       for(int c1 = 0 ; c1 < G_nColor ; c1++)
	 for(int c2 = 0 ; c2 < G_nColor ; c2++){
	     fprintf(ptr_propSS,"%+e %+e\n",uprop->H_elem()[iv*G_nSpin*G_nSpin*G_nColor*G_nColor*2 + mu*G_nSpin*G_nColor*G_nColor*2 + nu*G_nColor*G_nColor*2 + c1*G_nColor*2 + c2*2 + 0],uprop->H_elem()[iv*G_nSpin*G_nSpin*G_nColor*G_nColor*2 + mu*G_nSpin*G_nColor*G_nColor*2 + nu*G_nColor*G_nColor*2 + c1*G_nColor*2 + c2*2 + 1] );
	   }
 fclose(ptr_propSS);
#endif

 performContractionsPion(*uprop, NmomSinkTwop, momSinkTwop, filename_out_pion);


 free(momSinkTwop);

 delete uprop3D;
 delete xi_unsmeared3D;

 gsl_rng_free(rng);
 free(input_vector);
 delete uprop;
 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;
  
  freeGaugeQuda();
  cudaFree(deviceWilsonPath);
  // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());
  
  profileInvert[QUDA_PROFILE_TOTAL].Stop();
  
  
}


void quda::check_wilson_links(void **gauge, void **gaugeAPE)
{         
  int V = 16*16*16*32;
  QKXTM_Gauge *qkxTM_gauge = new QKXTM_Gauge();
  qkxTM_gauge->packGauge(gaugeAPE);
  qkxTM_gauge->loadGauge();               // now we have the gauge on device 
  double *deviceWilsonPath = createWilsonPath(*qkxTM_gauge,2);
  delete qkxTM_gauge;
  double* host_pointer=(double*) malloc((V*9*16/2)*2*sizeof(double));
  cudaMemcpy(host_pointer,deviceWilsonPath,(V*9*16/2)*2*sizeof(double),cudaMemcpyDeviceToHost);
  double **kale = (double**) gaugeAPE;

  for(int dz = 0 ; dz < 8 ; dz++)
    for(int c2 = 0 ; c2 < 3 ; c2++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	printf("%d %d %d  %+e %+e\n",dz,c2,c1,host_pointer[((dz)*3*3*V*2 + (c2)*3*V*2 + (c1)*V*2 + 0)], host_pointer[((dz)*3*3*V*2 + (c2)*3*V*2 + (c1)*V*2 + 1)]);

  printf("\n\n");
  for(int z =0 ; z < 8 ; z++)
    for(int c2 = 0 ; c2 < 3 ; c2++)
      for(int c1 =0 ; c1 < 3 ; c1++){
	int iv = z*16*16 + (0)*16 + (0);
	printf("%d %d %d  %+e %+e\n",z,c2,c1, kale[2][ iv*3*3*2 + c2*3*2 + c1*2 + 0],kale[2][ iv*3*3*2 + c2*3*2 + c1*2 + 1]);
      }
}



void quda::HYP3D(void **gaugeHYP)  
{         
  
  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();    // temporary gauge to pefrorm APE smearing
  QKXTM_Gauge *qkxTM_gaugeHYP = new QKXTM_Gauge();  // because we want a backup in ram
  
  QKXTM_Propagator *prp1 = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing
  QKXTM_Propagator *prp2 = new QKXTM_Propagator(); 

  double plaq;
  qkxTM_gaugeTmp->packGauge(gaugeHYP);       
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device

  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  HYP3D_smearing(*qkxTM_gaugeHYP,*qkxTM_gaugeTmp, *prp1,*prp2); // use *uprop temporary for ape smearing

  plaq = qkxTM_gaugeHYP->calculatePlaq();
  printfQuda("Plaquette HYP smeared is %e\n",plaq);

  delete qkxTM_gaugeTmp;
  delete qkxTM_gaugeHYP;
  delete prp1;
  delete prp2;
}


void quda::invertWriteProps_SS(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
			    int *sourcePosition, char *prop_path)  
{         

  char tempFilename[257];

  //before we perform inversions we need to calculate the plaquette and APE smearing

  QKXTM_Gauge *qkxTM_gaugeTmp = new QKXTM_Gauge();    // temporary gauge to pefrorm APE smearing
  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(); 
  QKXTM_Propagator *prop = new QKXTM_Propagator();   // forward definition of uprop to use it in APE smearing


  double plaq;
  qkxTM_gaugeTmp->packGauge(gaugeAPE);       
  qkxTM_gaugeTmp->loadGauge();               // now we have the gauge on device




  plaq = qkxTM_gaugeTmp->calculatePlaq();
  printfQuda("Plaquette is %e\n",plaq);

  APE_smearing(*qkxTM_gaugeAPE,*qkxTM_gaugeTmp, *prop); // use *uprop temporary for ape smearing

  delete qkxTM_gaugeTmp;                                // dont need it anymore
  delete prop;
  // from now on we have on device pointer the smeared gauge in gpu form
  // on primary host pointer we have the smeared gauge in gpu form
  // on backup host pointer we have the unsmeared gauge in gpu form
  // if you want to have smeared gauge on gpu use loadGauge if you want unsmeared gauge on gpu use loadGaugeFromBackup
  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);

  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
  if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);

 // for up quark test


 // for loop  

 QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
 QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));



 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);

   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark   
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);      
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorGauss->download();
   sprintf(tempFilename,"%s_up.%04d",prop_path,ip);
   qkxTM_vectorGauss->write(tempFilename);
   delete tmp_up;
   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   // down
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);


   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);

   qkxTM_vectorGauss->uploadToCuda(*b);               // re-upload source for inversion for down quark       
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorGauss->download();
   sprintf(tempFilename,"%s_down.%04d",prop_path,ip);
   qkxTM_vectorGauss->write(tempFilename);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   
   
 }



 free(input_vector);
 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;
  
  freeGaugeQuda();
  // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());
  
  profileInvert[QUDA_PROFILE_TOTAL].Stop();
  

}

void quda::invertWritePropsNoApe_SS(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
			    int *sourcePosition, char *prop_path)  
{         

  char tempFilename[257];

  //before we perform inversions we need to calculate the plaquette and APE smearing

  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(); 
  QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
  QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


  double plaq;
  qkxTM_gaugeAPE->packGauge(gaugeAPE);       
  qkxTM_gaugeAPE->loadGauge();               // now we have the gauge on device

  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);

  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
  if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);

 // for up quark test


 // for loop  



 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));



 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);

   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark   
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);      
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorGauss->download();
   sprintf(tempFilename,"%s_up.%04d",prop_path,ip);
   qkxTM_vectorGauss->write(tempFilename);
   delete tmp_up;
   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   // down
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);


   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);

   qkxTM_vectorGauss->uploadToCuda(*b);               // re-upload source for inversion for down quark       
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorGauss->download();
   sprintf(tempFilename,"%s_down.%04d",prop_path,ip);
   qkxTM_vectorGauss->write(tempFilename);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   
   
 }



 free(input_vector);
 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;
  
  freeGaugeQuda();
  // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());
  
  profileInvert[QUDA_PROFILE_TOTAL].Stop();
  

}

void quda::invertWritePropsNoApe_SL(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,
			    int *sourcePosition, char *prop_path)  
{         

  char tempFilename[257];

  //before we perform inversions we need to calculate the plaquette and APE smearing

  QKXTM_Gauge *qkxTM_gaugeAPE = new QKXTM_Gauge(); 
  QKXTM_Vector *qkxTM_vectorTmp = new QKXTM_Vector();        // because we use gaussian smearing we can avoid the 2 vectors
  QKXTM_Vector *qkxTM_vectorGauss = new QKXTM_Vector();


  double plaq;
  qkxTM_gaugeAPE->packGauge(gaugeAPE);       
  qkxTM_gaugeAPE->loadGauge();               // now we have the gauge on device

  plaq = qkxTM_gaugeAPE->calculatePlaq();
  printfQuda("Plaquette smeared is %e\n",plaq);

  //////////////////////////////////////////////////////////
  // now we need to load the gauge field for inversion 
  loadGaugeQuda((void*)gauge, gauge_param);
  ////////////////////////////////////

  profileInvert[QUDA_PROFILE_TOTAL].Start();
  if (!initialized) errorQuda("QUDA not initialized");
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(inv_param);         // this checks if we had call the loadGauge
  checkInvertParam(inv_param);


  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION) || (inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (inv_param->solution_type == QUDA_MAT_SOLUTION) || (inv_param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (inv_param->solve_type == QUDA_DIRECT_SOLVE) || (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE);

  inv_param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) inv_param->spinorGiB *= 2;
  inv_param->spinorGiB *= (inv_param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (inv_param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    inv_param->spinorGiB *= (inv_param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  inv_param->secs = 0;
  inv_param->gflops = 0;
  inv_param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert[QUDA_PROFILE_H2D].Start();

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();
  
  // create cudaColorSpinorField vectors

  ColorSpinorParam qkxTMParam; 
  qkxTMParam.nColor = 3;
  qkxTMParam.nSpin = 4;
  qkxTMParam.nDim = 4;
  for(int d = 0 ; d < qkxTMParam.nDim ; d++)qkxTMParam.x[d] = X[d];
  qkxTMParam.pad = inv_param->sp_pad;
  qkxTMParam.precision = inv_param->cuda_prec;
  qkxTMParam.twistFlavor = QUDA_TWIST_NO;        // change it later
  if(!pc_solution){
    qkxTMParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }else{
    qkxTMParam.x[0] /= 2;
    qkxTMParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  }
  
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  
  qkxTMParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  qkxTMParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;

  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  
  qkxTMParam.gammaBasis = inv_param->gamma_basis;
  qkxTMParam.create = QUDA_ZERO_FIELD_CREATE;
  qkxTMParam.v = NULL;
  qkxTMParam.norm = NULL;
  qkxTMParam.verbose = inv_param->verbosity;

  b = new cudaColorSpinorField(qkxTMParam);
  x = new cudaColorSpinorField(qkxTMParam);


  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  

  

 if (pc_solution && !pc_solve) {
   errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
 }
 
 if (!mat_solution && !pc_solution && pc_solve) {
   errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
 }
 

 if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
  if( inv_param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

 DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
 Solver *solve = Solver::create(*inv_param, m, mSloppy, mPre, profileInvert);

 // for up quark test


 // for loop  



 void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));



 for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = sourcePosition[i] - comm_coords(i) * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);

   qkxTM_vectorGauss->uploadToCuda(*b);           // uses it and for down quark   
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_up);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);      
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   //   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorTmp->download();
   sprintf(tempFilename,"%s_up.%04d",prop_path,ip);
   qkxTM_vectorTmp->write(tempFilename);
   delete tmp_up;
   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   // down
   qkxTM_vectorTmp->flagsToFalse();                                                 // hack
   qkxTM_vectorTmp->packVector(input_vector);
   qkxTM_vectorTmp->loadVector();
   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);


   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);

   qkxTM_vectorGauss->uploadToCuda(*b);               // re-upload source for inversion for down quark       
   zeroCuda(*x);
   dirac.prepare(in, out, *x, *b, inv_param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, inv_param->solution_type);
   qkxTM_vectorTmp->downloadFromCuda(*x);
   if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION || inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     qkxTM_vectorTmp->scaleVector(2*inv_param->kappa);
   }

   //   Gaussian_smearing(*qkxTM_vectorGauss, *qkxTM_vectorTmp , *qkxTM_gaugeAPE);
   qkxTM_vectorTmp->download();
   sprintf(tempFilename,"%s_down.%04d",prop_path,ip);
   qkxTM_vectorTmp->write(tempFilename);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   
   
 }



 free(input_vector);
 delete qkxTM_vectorTmp;
 delete qkxTM_vectorGauss;
 delete qkxTM_gaugeAPE;
 delete solve;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;
  
  freeGaugeQuda();
  // popVerbosity();

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());
  
  profileInvert[QUDA_PROFILE_TOTAL].Stop();
  

}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 
#include <sys/time.h>

void initCommsQuda(int argc, char **argv, const int *X, int nDim) {
#ifdef MULTI_GPU
  comm_create(argc, argv);
  comm_set_gridsize(X, nDim);  
  comm_init();
#endif
}

void endCommsQuda() {
#ifdef MULTI_GPU
  comm_cleanup();
#endif
}

/*
  The following functions are for the Fortran interface.
*/
/*
void init_quda_(int *dev) { initQuda(*dev); }
void end_quda_() { endQuda(); }
void load_gauge_quda_(void *h_gauge, QudaGaugeParam *param) { loadGaugeQuda(h_gauge, param); }
void free_gauge_quda_() { freeGaugeQuda(); }
//void load_clover_quda_(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param) 
//{ loadCloverQuda(h_clover, h_clovinv, inv_param); }
//void free_clover_quda_(void) { freeCloverQuda(); }
void dslash_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param,
		  QudaParity *parity) { dslashQuda(h_out, h_in, inv_param, *parity); }
//void clover_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param,
//		  QudaParity *parity, int *inverse) { cloverQuda(h_out, h_in, inv_param, *parity, *inverse); }
void mat_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param)
{ MatQuda(h_out, h_in, inv_param); }
void mat_dag_mat_quda_(void *h_out, void *h_in, QudaInvertParam *inv_param)
{ MatDagMatQuda(h_out, h_in, inv_param); }
void invert_quda_(void *hp_x, void *hp_b, QudaInvertParam *param) 
{ invertQuda(hp_x, hp_b, param); }    
void new_quda_gauge_param_(QudaGaugeParam *param) {
  *param = newQudaGaugeParam();
}
void new_quda_invert_param_(QudaInvertParam *param) {
  *param = newQudaInvertParam();
}
void comm_set_gridsize_(int *grid) {
#ifdef MULTI_GPU
  comm_set_gridsize(grid, 4);
#endif
}
*/
