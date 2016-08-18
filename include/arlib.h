#include <comm_quda.h>
#include <quda_internal.h>
#include <quda.h>
#include <iostream>
#include <complex>
#include <cuda.h>
#include <color_spinor_field.h>
#include <dirac_quda.h>

#ifndef _ARLIB_H
#define _ARLIB_H

namespace quda{
  enum SorL {S,L};

  typedef struct {
    int dimensionMatrix;
    int arnoldiTotalVectors;
    int arnoldiWantedVectors;
    int arnoldiUnwantedVectors;
    int maxArnoldiIter;
    double tolerance;
    long unsigned int seed;
    enum SorL sorl;
  }arlibInfo;

  class Arnoldi;
  void init_arnoldi(arlibInfo in_arinfo, qudaQKXTMinfo latInfo);
  void finalize_arnoldi();
  void printf_arlib();
  void Iram(void **gauge,QudaInvertParam *inv_param, QudaGaugeParam *gauge_param, qudaQKXTMinfo in_latInfo, arlibInfo in_arInfo );
  void clearNoiseCuda(Complex *A, int L, double tolerance);

  enum MATRIXTYPE{matrixNxK,matrixKxK,vectorN,matrixNxM,matrixMxM};

  class Arnoldi {
  private:
    size_t size;
    Complex *d_mNxK ;
    Complex *d_mKxK ;
    Complex *d_mNxM ;
    Complex *d_mMxM ;
    Complex *d_vN ;
  public:
    Complex *D_mNxK() { return d_mNxK; }
    Complex *D_mKxK() { return d_mKxK; }
    Complex *D_mNxM() { return d_mNxM; }
    Complex *D_mMxM() { return d_mMxM; }
    Complex *xD_vN() { return d_vN; }
    size_t Size() { return size;}
    Arnoldi(enum MATRIXTYPE);
    ~Arnoldi();

    void uploadToCuda(cudaColorSpinorField &cudaVector,int offset);
    void downloadFromCuda(cudaColorSpinorField &cudaVector, int offset);
    //    void matrixNxMmatrixMxL(Arnoldi &V,int NL, int M,int L,bool transpose);
    void matrixNxMmatrixMxLReal(Arnoldi &V,int NL, int M,int L,bool transpose);

    void applyMatrix(Dirac &A,cudaColorSpinorField &in, cudaColorSpinorField &out,Arnoldi &inn, int offsetIn, int offsetOut);
    void initArnold(Arnoldi &V, Arnoldi &H, Dirac &A, cudaColorSpinorField &in, cudaColorSpinorField &out);
    void arnold(int kstart, int kstop,Arnoldi &V, Arnoldi &H, Dirac &A, cudaColorSpinorField &in, cudaColorSpinorField &out);


    void eigMagmaAndSort(Arnoldi &W, Arnoldi &Q);
    void QR_Magma(Arnoldi &Q);
    void qrShiftsRotations(Arnoldi &V, Arnoldi &H);
        
  };

}

#endif
