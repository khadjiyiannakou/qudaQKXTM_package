#ifndef _DSLASH_QUDA_H
#define _DSLASH_QUDA_H

#include <quda_internal.h>
#include <tune_quda.h>
#include <face_quda.h>
#include <gauge_field.h>

namespace quda {

  /**
     @param tune Sets whether to tune the dslash kernels or not
     @param verbose The verbosity level to use in the dslash tuning functions
  */
  void setDslashTuning(QudaTune tune, QudaVerbosity verbose);

  /**
     @param pack Sets whether to use a kernel to pack the T dimension
   */
  void setKernelPackT(bool pack);

  /**
     @return Whether the T dimension is kernel packed or not
   */
  bool getKernelPackT();

#ifdef DSLASH_PROFILING
  void printDslashProfile();
#endif

  void setFace(const FaceBuffer &face);

  bool getDslashLaunch();

  void createDslashEvents();
  void destroyDslashEvents();

  void initLatticeConstants(const LatticeField &lat);
  void initGaugeConstants(const cudaGaugeField &gauge);
  void initSpinorConstants(const cudaColorSpinorField &spinor);
  void initDslashConstants();


  // plain Wilson Dslash  
  void wilsonDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in,
			const int oddBit, const int daggerBit, const cudaColorSpinorField *x,
			const double &k, const int *commDim);


  // twisted mass Dslash  
  void twistedMassDslashCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const cudaColorSpinorField *in,
			     const int parity, const int dagger, const cudaColorSpinorField *x, 
			     const double &kappa, const double &mu, const double &a, const int *commDim);

  // solo twist term
  void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
		       const int dagger, const double &kappa, const double &mu,
		       const QudaTwistGamma5Type);

  // face packing routines
  void packFace(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger, 
		const int parity, const cudaStream_t &stream);

}

#endif // _DSLASH_QUDA_H
