#include <comm_quda.h>
#include <quda_internal.h>
#include <quda.h>
#include <iostream>
#include <complex>
#include <cuda.h>
#include <color_spinor_field.h>
#include <dirac_quda.h>

#ifndef _LANCZOS_H
#define _LANCZOS_H

namespace quda{
  enum SMLM {SM,LM};

  typedef struct {
    int dimensionMatrix;
    int lanczosTotalVectors;
    int lanczosWantedVectors;
    int lanczosUnwantedVectors;
    int maxLanczosIter;
    double tolerance;
    long unsigned int seed;
    enum SMLM smlm;
  }lanczosInfo;

  class Lanczos;
  void initLanczos(lanczosInfo in_lanczosInfo, qudaQKXTMinfo latInfo);
  void finalizeLanczos();
  void printfLanczos();

  void IRLM_ES(void **gauge,QudaInvertParam *inv_param, QudaGaugeParam *gauge_param, qudaQKXTMinfo in_latInfo, lanczosInfo in_lInfo );

  enum MATRIXTYPE_LANCZOS{matrixNxM_L,matrixMxM_L,vectorN_L};

  class Lanczos {
  private:
    size_t size;
    Complex *d_mNxM ;
    double *d_mMxM ;
    Complex *d_vN ;
  public:
    Complex *D_mNxM() { return d_mNxM; }
    double  *D_mMxM() { return d_mMxM; }
    Complex *D_vN() { return d_vN; }
    size_t Size() { return size;}
    Lanczos(enum MATRIXTYPE_LANCZOS);
    ~Lanczos();


    void uploadToCuda(cudaColorSpinorField &cudaVector,int offset);
    void downloadFromCuda(cudaColorSpinorField &cudaVector);
    void applyMatrix(Dirac &A,cudaColorSpinorField &in, cudaColorSpinorField &out,Lanczos &inn, int offsetIn);
    void lanczos(int kstart, int kstop,Lanczos &V, Lanczos &T,Lanczos &r, Dirac &A, cudaColorSpinorField &in, cudaColorSpinorField &out);
    void eigMagmaAndSort(Lanczos &W, Lanczos &Q);
    void QR_Magma(Lanczos &Q);
    void matrixNxMmatrixMxLReal(Lanczos &V,int NL, int M,int L,bool transpose);
    void makeTridiagonal(int m , int l);
    void qrShiftsRotations(Lanczos &V, Lanczos &T, Lanczos &r);

  };

}

#endif
