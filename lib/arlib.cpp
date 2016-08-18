#include <stdlib.h>
#include <string.h>
#include <qudaQKXTM.h>
#include <quda.h>
#include <dirac_quda.h>
#include <arlib.h>
#include <errno.h>
#include <mpi.h>
#ifndef CHECK_PARAM
#define CHECK_PARAM
#include <check_params.h>
#endif
#include <blas_quda.h>
#include <curand.h>
#include <magma.h>
#include <magma_operators.h>
#include <flops.h>
#include <magma_lapack.h>
#include <limits>
#define PRECISION_z
#define COMPLEX


using namespace quda;

extern float G_deviceMemory;
extern int G_nColor;
extern int G_nSpin;
extern int G_nDim;
extern int G_strideFull;
extern double G_alphaAPE;
extern double G_alphaGauss;
extern int G_localVolume;
extern int G_totalVolume;
extern int G_nsmearAPE;
extern int G_nsmearGauss;
extern bool G_dimBreak[QUDAQKXTM_DIM];
extern int G_localL[QUDAQKXTM_DIM];
extern int G_totalL[QUDAQKXTM_DIM];
extern int G_nProc[QUDAQKXTM_DIM];
extern int G_plusGhost[QUDAQKXTM_DIM];
extern int G_minusGhost[QUDAQKXTM_DIM];
extern int G_surface3D[QUDAQKXTM_DIM];
extern bool G_init_qudaQKXTM_flag;
extern int G_nsmearHYP;
extern double G_omega1HYP;
extern double G_omega2HYP;
// for mpi use global  variables 
extern MPI_Group G_fullGroup , G_spaceGroup , timeGroup;
extern MPI_Comm G_spaceComm , G_timeComm;
extern int G_localRank;
extern int G_localSize;
extern int G_timeRank;
extern int G_timeSize;
arlibInfo arInfo;

#define MAGMA_MALLOC_PIN(ptr,type,size) \
  if(MAGMA_SUCCESS != magma_malloc_pinned( (void**) &ptr, (size)*sizeof(type) )){ \
  magma_finalize();							\
  errorQuda("!!!! magma_malloc_pinned failed for: %s\n", #ptr );	\
  }

#define MAGMA_FREE_PIN( ptr) magma_free_pinned( ptr )

#define CUDA_MALLOC_ZERO( pointer, size )\
  cudaMalloc((void**)&pointer,size);	 \
  cudaMemset((void*)pointer,0,size);

static void init_eig(){
  int M = arInfo.arnoldiTotalVectors;

  Complex *h_H = (Complex*)calloc(M*M,sizeof(Complex));
  Complex *h_W = (Complex*)calloc(M,sizeof(Complex));
  Complex *h_Q = (Complex*)calloc(M*M,sizeof(Complex));

  magma_int_t info,lwork,nb;
  nb    = magma_get_zgehrd_nb(M);
  lwork = M*(1 + nb);
  lwork = std::max( lwork, M*(5 + 2*M) );

  double *rwork = (double*)malloc(2*M*sizeof(double));
  magmaDoubleComplex *hwork = (magmaDoubleComplex*) malloc(lwork*sizeof(magmaDoubleComplex));

#ifndef DEVICE_EIG
  lapackf77_zgeev( lapack_vec_const(MagmaNoVec),lapack_vec_const(MagmaVec), &M,(magmaDoubleComplex*) h_H, &M,(magmaDoubleComplex*) h_W,NULL,&M,(magmaDoubleComplex*) h_Q, &M,(magmaDoubleComplex*) hwork, &lwork, rwork, &info );
  if (info != 0)
    errorQuda("lapack_zgeev returned error \n");
#else
  magma_zgeev(MagmaNoVec,MagmaVec,(magma_int_t)M,h_H,(magma_int_t)M,h_W,NULL,(magma_int_t)M,h_Q,(magma_int_t)M,hwork,lwork,rwork,&info);
  if (info != 0)
    errorQuda("magma_zgeev returned error %d: %s.\n", (int) info, magma_strerror( info ));
#endif

  free(h_H);
  free(h_W);
  free(h_Q);
  free(rwork);
  free(hwork);

}

void quda::init_arnoldi(arlibInfo in_arinfo, qudaQKXTMinfo latInfo){
  double time;
  time = MPI_Wtime();
  arInfo = in_arinfo;
  init_qudaQKXTM(&latInfo);
  magma_init();
  cublasInit();
  init_eig();
  printf_qudaQKXTM();
  printf_arlib();
  time = MPI_Wtime() - time;
  printfQuda("Initialization took %f sec\n",time);
}

void quda::finalize_arnoldi(){
  magma_finalize();
  cublasShutdown();
}


void quda::printf_arlib(){

  printfQuda("The Matrix has dimensions %dX%d\n",arInfo.dimensionMatrix*G_nProc[0]*G_nProc[1]*G_nProc[2]*G_nProc[3],arInfo.dimensionMatrix*G_nProc[0]*G_nProc[1]*G_nProc[2]*G_nProc[3]);
  printfQuda("The total working vectors you ask is %d\n",arInfo.arnoldiTotalVectors);
  printfQuda("You seek for %d eigenvalues\n",arInfo.arnoldiWantedVectors);
  printfQuda("The expanded working space is %d vectors\n",arInfo.arnoldiUnwantedVectors);
  printfQuda("The maximum number of iteration is %d\n",arInfo.maxArnoldiIter);
  printfQuda("The tolerance for the arnoldi residual vector is %e\n",arInfo.tolerance);
  printfQuda("The first vector will be randomly initialized using seed  %ld\n",arInfo.seed);
  if(arInfo.sorl == S)  printfQuda("You are looking for the smallest eigenvalues\n");
  if(arInfo.sorl == L)  printfQuda("You are looking for the largest eigenvalues\n");
}

Arnoldi::Arnoldi(enum MATRIXTYPE matrixTYPE): d_mNxK(NULL),d_mKxK(NULL),d_mNxM(NULL),d_mMxM(NULL),d_vN(NULL){

  if(matrixTYPE == matrixNxM){
    size = G_localVolume * G_nSpin * G_nColor * (arInfo.arnoldiTotalVectors+1) * sizeof(Complex);
    //    cudaMalloc((void**)&d_mNxM,size);
    CUDA_MALLOC_ZERO(d_mNxM,size);
    checkCudaError();
    G_deviceMemory += size/(1024.*1024.);
    printfQuda("Device memory in used is %f MB A \n",G_deviceMemory);
  }
  else if(matrixTYPE == matrixNxK){
    size = G_localVolume *  G_nSpin * G_nColor * (arInfo.arnoldiWantedVectors+1) * sizeof(Complex);
    //    cudaMalloc((void**)&d_mNxK,size);
    CUDA_MALLOC_ZERO(d_mNxK,size);
    checkCudaError();
    G_deviceMemory += size/(1024.*1024.);
    printfQuda("Device memory in used is %f MB A \n",G_deviceMemory);
  }
  else if(matrixTYPE == matrixMxM){
    size = arInfo.arnoldiTotalVectors*arInfo.arnoldiTotalVectors * sizeof(Complex);
    //    cudaMalloc((void**)&d_mMxM,size);
    CUDA_MALLOC_ZERO(d_mMxM,size);
    checkCudaError();
    G_deviceMemory += size/(1024.*1024.);
    //    printfQuda("Device memory in used is %f MB A \n",G_deviceMemory);
  }
  else if(matrixTYPE == matrixKxK){
    size = arInfo.arnoldiWantedVectors *arInfo.arnoldiWantedVectors * sizeof(Complex);
    //    cudaMalloc((void**)&d_mKxK,size);
    CUDA_MALLOC_ZERO(d_mKxK,size);
    checkCudaError();
    G_deviceMemory += size/(1024.*1024.);
    //   printfQuda("Device memory in used is %f MB A \n",G_deviceMemory);
  }
  else if(matrixTYPE == vectorN){
    size = G_localVolume * G_nSpin * G_nColor * sizeof(Complex);
    //    cudaMalloc((void**)&d_vN,size);
    CUDA_MALLOC_ZERO(d_vN,size);
    checkCudaError();
    G_deviceMemory += size/(1024.*1024.);
    // printfQuda("Device memory in used is %f MB A \n",G_deviceMemory);
  }
  else{
    errorQuda("Wrong choice for arnoldi matrices\n");
  }
  checkCudaError();
}

Arnoldi::~Arnoldi(){
  if(d_mNxM != NULL){
    cudaFree(d_mNxM);
  printfQuda("Device memory in used is %f MB D \n",G_deviceMemory);    
  }
  if(d_mNxK != NULL)cudaFree(d_mNxK);
  if(d_mMxM != NULL)cudaFree(d_mMxM);
  if(d_mKxK != NULL)cudaFree(d_mKxK);
  if(d_vN != NULL)cudaFree(d_vN);
  checkCudaError();
  G_deviceMemory -= size/(1024.*1024.);

}

static Dirac* initQudaMatrix(QudaInvertParam *inv_param, QudaGaugeParam *gauge_param, void **gauge){
  loadGaugeQuda((void*)gauge, gauge_param);
  checkInvertParam(inv_param);

  // create Dirac operator
  DiracParam diracParam;
  setDiracParam(diracParam,inv_param,false);
  Dirac *dirac = Dirac::create(diracParam);
  setDslashTuning(inv_param->tune, getVerbosity());
  setBlasTuning(inv_param->tune, getVerbosity());
  checkCudaError();

  return dirac;
}

static cudaColorSpinorField* initQudaVector(QudaInvertParam *inv_param,QudaGaugeParam *gauge_param){

  //  printfQuda("%d %d %d %d\n",G_localL[0],G_localL[1],G_localL[2],G_localL[3]);

  cudaColorSpinorField *vector;
  ColorSpinorParam param;
  param.nColor = G_nColor;
  param.nSpin = G_nSpin;
  param.nDim = G_nDim;
  for(int d = 0 ; d< param.nDim;d++)param.x[d] = G_localL[d];
  param.pad = inv_param->sp_pad;
  param.precision = inv_param->cuda_prec;
  param.twistFlavor = inv_param->twist_flavor;
  param.siteSubset = QUDA_FULL_SITE_SUBSET;
  if(inv_param->dirac_order != QUDA_DIRAC_ORDER)
    errorQuda("qudaQKXTM package supports only dirac order");
  param.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  if(inv_param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS)
    errorQuda("qudaQKXTM package supports only ukqcd");
  param.gammaBasis = inv_param->gamma_basis;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.v = NULL;
  param.norm = NULL;
  param.verbose = inv_param->verbosity;
  vector = new cudaColorSpinorField(param);
  checkCudaError();
  return vector;
}

void Arnoldi::applyMatrix(Dirac &A,cudaColorSpinorField &in, cudaColorSpinorField &out,Arnoldi &inn, int offsetIn, int offsetOut){
  checkCudaError();
  inn.uploadToCuda(in,offsetIn);
  A.MdagM(out,in);
  this->downloadFromCuda(out,offsetOut);
}


static double normMagma(Complex* vec,int size){
  double norma;
  double norma_all;
  norma = magma_dznrm2(size,(magmaDoubleComplex*) vec,1);
  norma = norma*norma;
  MPI_Allreduce((void*) &norma,(void*) &norma_all,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  norma_all = sqrt(norma_all);
  return norma_all;
}

static Complex dotMagma(Complex* vec1, Complex* vec2, int size){
  magmaDoubleComplex dot;
  Complex dot_all;
  dot = magma_zdotc(size,(magmaDoubleComplex*) vec1, 1, (magmaDoubleComplex*) vec2,1);
  MPI_Allreduce((void*) &dot,(void*) &dot_all,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  return dot_all;
}



static void noiseCleaner(Complex* A, int L){
  /*
  double machine_epsilon = std::numeric_limits<double>::epsilon();
  double matrixOrder = 0.;

    for(int i = 0 ; i < L ; i++)
        matrixOrder += sqrt(norm(A[i]));

    for(int i = 0 ; i < L ; i++){
      if( sqrt(norm(A[i])) < (sqrt(machine_epsilon)/matrixOrder)/1000 )
          A[i] = (Complex) {0.,0.};
      }


    for(int i = 0 ; i < L ; i++){
      if( fabs(A[i].real()) < (sqrt(machine_epsilon)/matrixOrder)/1000 ) A[i].real() = 0;
      if( fabs(A[i].imag()) < (sqrt(machine_epsilon)/matrixOrder)/1000 ) A[i].imag() = 0;
      }
  */

  double matrixOrder = 0.;

    for(int i = 0 ; i < L ; i++)
        matrixOrder += sqrt(norm(A[i]));

    for(int i = 0 ; i < L ; i++){
      if( sqrt(norm(A[i])) < arInfo.tolerance/matrixOrder )
          A[i] = (Complex) {0.,0.};
      }


    for(int i = 0 ; i < L ; i++){
      if( fabs(A[i].real()) < arInfo.tolerance/matrixOrder ) A[i].real() = 0;
      if( fabs(A[i].imag()) < arInfo.tolerance/matrixOrder ) A[i].imag() = 0;
      }

}





//#define __DEBUG_ARLIB
#define __DEVICE_NOISE_CLEANER__

void Arnoldi::initArnold(Arnoldi &V, Arnoldi &H, Dirac &A, cudaColorSpinorField &in, cudaColorSpinorField &out){
  // initialize the first vector with random numbers
  int M = arInfo.arnoldiTotalVectors;
  int NL = arInfo.dimensionMatrix;

  /*
  curandGenerator_t gen;
  curandCreateGenerator(&gen,CURAND_RNG_QUASI_SOBOL64);
  curandSetPseudoRandomGeneratorSeed(gen,arInfo.seed);
  curandGenerateUniformDouble(gen,(double*) V.D_mNxM(),NL*sizeof(Complex));
  */
  Complex *h_v = (Complex*)malloc(NL*sizeof(Complex));
  for(int i = 0 ; i < NL ; i++){
    h_v[i].real() =  1.0;
    h_v[i].imag() = 0.0;
  }

  cudaMemcpy((void*)V.D_mNxM(),(void*)h_v,NL*sizeof(Complex),cudaMemcpyHostToDevice);
  // normalize the vector ( we need dotproduct)
  double beta = normMagma(V.D_mNxM(),NL);
  //  printfQuda("!!!!!!!!!!!!!!!!! %f\n",beta);
  // scale a vector to normalize it
  magma_zdscal((magma_int_t)NL,(1./beta),(magmaDoubleComplex*)V.D_mNxM(),1);
  // apply matrix on initial vector to calculate the second vector
  V.applyMatrix(A,in,out,V, 0, 1);
  // calculate the dot product of two vectors
  Complex alpha = dotMagma(V.D_mNxM(),V.D_mNxM() + NL, NL);  
  // set the alpha on Arnoldi matrix
  cudaMemcpy((void*)H.D_mMxM(),(void*)&alpha,sizeof(Complex),cudaMemcpyHostToDevice);
  // perform operation y = y + ax
  magma_zaxpy ((magma_int_t)NL,MAGMA_Z_MAKE(-alpha.real(),-alpha.imag()),(magmaDoubleComplex*) V.D_mNxM(), 1, (magmaDoubleComplex*) (V.D_mNxM() + NL),1 );
  // check orthogonality
  Complex alphaNew = dotMagma(V.D_mNxM(),V.D_mNxM() + NL, NL);
  alpha = alphaNew + alpha;
  cudaMemcpy((void*)H.D_mMxM(),(void*)&alpha,sizeof(Complex),cudaMemcpyHostToDevice);
  magma_zaxpy ((magma_int_t)NL,MAGMA_Z_MAKE(-alphaNew.real(),-alphaNew.imag()),(magmaDoubleComplex*) V.D_mNxM(), 1, (magmaDoubleComplex*) (V.D_mNxM() + NL),1 );

#ifndef __DEVICE_NOISE_CLEANER__
  Complex *h_H = (Complex*)malloc(M*M*sizeof(Complex));
  cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  noiseCleaner(h_H,M*M);
  cudaMemcpy((void*)H.D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
  free(h_H);
#else
  clearNoiseCuda(H.D_mMxM(),M*M,arInfo.tolerance);
#endif
  checkCudaError();
}

static double timeApplyMatrix;
static double accumApplyMatrix = 0;

static double timeApplyYPAX;
static double accumApplyYPAX = 0;

static double timeApplyNorm;
static double accumApplyNorm = 0;


static double timeCleaning;
static double accumCleaning = 0;
//#define __PINNED_MEMORY__

#define __EFFECTIVE_LANCZOS__
//#define __SELECTIVE_REFINEMENT__

void Arnoldi::arnold(int kstart, int kstop, Arnoldi &V, Arnoldi &H, Dirac &A, cudaColorSpinorField &in, cudaColorSpinorField &out){

  int M = arInfo.arnoldiTotalVectors;
  int NL = arInfo.dimensionMatrix;
  Arnoldi *S = new Arnoldi(matrixMxM);



#ifdef __PINNED_MEMORY__
  Complex *h_H;
  cudaHostAlloc((void**)&h_H,M*M*sizeof(Complex),cudaHostAllocDefault);
  Complex *beta;
  cudaHostAlloc((void**)&beta,sizeof(Complex),cudaHostAllocDefault);
#else
  Complex *h_H = (Complex*)calloc(M*M,sizeof(Complex));
  Complex *beta = (Complex*)calloc(1,sizeof(Complex));
#endif

  for(int j = kstart; j < kstop ; j++){
    int jm1 = j-1;
    int jp1 = j+1;



    timeApplyNorm = MPI_Wtime();
    beta[0].real() = normMagma(V.D_mNxM()+j*NL,NL);
    beta[0].imag() = 0.;
    accumApplyNorm += MPI_Wtime() - timeApplyNorm;

#ifndef __EFFECTIVE_LANCZOS__
    cudaMemcpy((void*) &(H.D_mMxM()[j*M+jm1]),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);
#else
    cudaMemcpy((void*) &(H.D_mMxM()[j*M+jm1]),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);
    cudaMemcpy((void*) &(H.D_mMxM()[jm1*M+j]),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);
#endif

    timeApplyNorm = MPI_Wtime();
    magma_zdscal((magma_int_t)NL,(1./beta[0].real()),(magmaDoubleComplex*) (V.D_mNxM() + j*NL),1);
    accumApplyNorm += MPI_Wtime() - timeApplyNorm;

    timeApplyMatrix = MPI_Wtime();
    V.applyMatrix(A,in,out,V, j, jp1);
    accumApplyMatrix += MPI_Wtime() - timeApplyMatrix;

    timeApplyNorm = MPI_Wtime();
#ifndef __EFFECTIVE_LANCZOS__
    for(int i = 0 ; i <= j ; i++){
      beta[0] = dotMagma(V.D_mNxM() + i*NL,V.D_mNxM() + jp1*NL, NL);
      cudaMemcpy((void*) (H.D_mMxM() + i*M+j),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);
    }
#else
    beta[0] = dotMagma(V.D_mNxM() + j*NL,V.D_mNxM() + jp1*NL, NL);
    cudaMemcpy((void*) (H.D_mMxM() + j*M+j),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);

    //    beta[0] = dotMagma(V.D_mNxM() + jm1*NL,V.D_mNxM() + jp1*NL, NL);
    //    cudaMemcpy((void*) (H.D_mMxM() + j*M+jm1),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);
    //  cudaMemcpy((void*) (H.D_mMxM() + jm1*M+j),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);

#endif

#ifndef __DEVICE_NOISE_CLEANER__
      cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
      noiseCleaner(h_H,M*M);
      cudaMemcpy((void*)H.D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
      clearNoiseCuda(H.D_mMxM(),M*M,arInfo.tolerance);
#endif



    accumApplyNorm += MPI_Wtime() - timeApplyNorm;
    // perform y =beta * y + alpha * A * x

    timeApplyYPAX = MPI_Wtime();
    magma_zgemv(MagmaNoTrans,(magma_int_t)NL,(magma_int_t)(j+1),MAGMA_Z_MAKE(-1.,0.),(magmaDoubleComplex*)V.D_mNxM(),(magma_int_t)NL,(magmaDoubleComplex*)(H.D_mMxM()+j),M, MAGMA_Z_MAKE(1.,0.),(magmaDoubleComplex*) (V.D_mNxM() + jp1*NL ),1);
    accumApplyYPAX += MPI_Wtime() - timeApplyYPAX;

    // check orthogonality using iterative refinement
#ifdef __SELECTIVE_REFINEMENT__
    timeApplyNorm = MPI_Wtime();
    for(int i = 0 ; i <= j ; i++){
      if( abs(i-j) >= 2 )continue;
      beta[0] = dotMagma(V.D_mNxM() + i*NL,V.D_mNxM() + jp1*NL, NL);
      cudaMemcpy((void*) &(S->D_mMxM()[i*M+j]),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);
            if( abs(i-j) == 1) cudaMemcpy((void*) &(S->D_mMxM()[j*M+i]),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);

    }
    accumApplyNorm += MPI_Wtime() - timeApplyNorm;

      timeApplyYPAX = MPI_Wtime();
      magma_zgemv(MagmaNoTrans,(magma_int_t)NL,(magma_int_t)(j+1),MAGMA_Z_MAKE(-1.,0.),(magmaDoubleComplex*)V.D_mNxM(),(magma_int_t)NL,(magmaDoubleComplex*)(S->D_mMxM()+j),(magma_int_t) M, MAGMA_Z_MAKE(1.,0.),(magmaDoubleComplex*) (V.D_mNxM() + jp1*NL ),1);
      accumApplyYPAX += MPI_Wtime() - timeApplyYPAX;
      for(int i = 0 ; i <= j ; i++)
	magma_zaxpy ((magma_int_t)(i+1),MAGMA_Z_MAKE(1.,0.),(magmaDoubleComplex*) S->D_mMxM() + j, M, (magmaDoubleComplex*) H.D_mMxM() + j,M );

      cudaMemset((void*)S->D_mMxM(),0,M*M*sizeof(Complex));
#else

      timeApplyNorm = MPI_Wtime();
      for(int i = 0 ; i <= j ; i++){
      beta[0] = dotMagma(V.D_mNxM() + i*NL,V.D_mNxM() + jp1*NL, NL);
      cudaMemcpy((void*) &(S->D_mMxM()[i*M+j]),(void*)beta,sizeof(Complex),cudaMemcpyHostToDevice);
      }
      accumApplyNorm += MPI_Wtime() - timeApplyNorm;

      
      timeApplyYPAX = MPI_Wtime();
      magma_zgemv(MagmaNoTrans,(magma_int_t)NL,(magma_int_t)(j+1),MAGMA_Z_MAKE(-1.,0.),(magmaDoubleComplex*)V.D_mNxM(),(magma_int_t)NL,(magmaDoubleComplex*)(S->D_mMxM()+j),(magma_int_t) M, MAGMA_Z_MAKE(1.,0.),(magmaDoubleComplex*) (V.D_mNxM() + jp1*NL ),1);
      accumApplyYPAX += MPI_Wtime() - timeApplyYPAX;
      for(int i = 0 ; i <= j ; i++)
	magma_zaxpy ((magma_int_t)(i+1),MAGMA_Z_MAKE(1.,0.),(magmaDoubleComplex*) S->D_mMxM() + j, M, (magmaDoubleComplex*) H.D_mMxM() + j,M );



#ifndef __DEVICE_NOISE_CLEANER__
      cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
      noiseCleaner(h_H,M*M);
      cudaMemcpy((void*)H.D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
      clearNoiseCuda(H.D_mMxM(),M*M,arInfo.tolerance);
#endif


#endif

    ///////////////////////////


  }

  /*
    FILE *ptr;
    ptr = fopen("kale.dat","a");

    cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < M ; i++){
      for(int j = 0 ; j < M ; j++)
	fprintf(ptr,"%+16.15e %+16.15e\n",h_H[i*M+j].real(),h_H[i*M+j].imag());
    }
    fclose(ptr);
  */

  timeCleaning = MPI_Wtime();
#ifndef __DEVICE_NOISE_CLEANER__
  cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  noiseCleaner(h_H,M*M);
  cudaMemcpy((void*)H.D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
  clearNoiseCuda(H.D_mMxM(),M*M,arInfo.tolerance);
#endif
  accumCleaning += MPI_Wtime() - timeCleaning;

#ifdef __PINNED_MEMORY__
  cudaFreeHost(h_H);
  cudaFreeHost(beta);
#else
  free(h_H);
  free(beta);
#endif
  delete S;
  checkCudaError();
}




static void sorting(Complex *h_W, Complex *h_Q){
  int M = arInfo.arnoldiTotalVectors;

  Complex *w = (Complex*)malloc(M*sizeof(Complex));
  double *w_mag = (double*)malloc(M*sizeof(double));
  double *w_mag_tmp = (double*)malloc(M*sizeof(double));

  int *pos = (int*)malloc(M*sizeof(int));
  Complex *q = (Complex*)malloc(M*M*sizeof(Complex));
  memcpy(q,h_Q,M*M*sizeof(Complex));

  for(int i = 0 ; i < M ; i++) w[i] = h_W[i];
  for(int i = 0 ; i < M ; i++) w_mag[i] = sqrt(norm(w[i]));
  memcpy(w_mag_tmp,w_mag,M*sizeof(double));

  if(arInfo.sorl == S){
    for(int i = 0 ; i < M-1 ; i++)
      for(int j = i + 1 ; j < M ; j++){
	if(w_mag[i] < w_mag[j]){
	  double a = w_mag[i];
	  w_mag[i] = w_mag[j];
	  w_mag[j] = a;
	}
      }
  }
  else if ( arInfo.sorl == L ){
    for(int i = 0 ; i < M-1 ; i++)
      for(int j = i + 1 ; j < M ; j++){
	if(w_mag[i] > w_mag[j]){
	  double a = w_mag[i];
	  w_mag[i] = w_mag[j];
	  w_mag[j] = a;
	}
      }
  }
  
  for(int i = 0 ; i < M ; i++)
    for(int j = 0 ; j < M ; j++)
      if(w_mag[i] == w_mag_tmp[j])
	pos[i] = j;

  for(int i = 0 ; i < M ; i++) h_W[i] = w[pos[i]];

  for(int i = 0 ; i < M ; i++)
    for(int j = 0 ; j < M ; j++)
      h_Q[i*M+j] = q[i*M+pos[j]];

  free(w);
  free(w_mag);
  free(w_mag_tmp);
  free(pos);
  free(q);
}

static void transposeMatrix(Complex *A,int M, int N){
  Complex *Atmp = (Complex*)malloc(M*N*sizeof(Complex));
  memcpy(Atmp,A,M*N*sizeof(Complex));
  for(int i = 0 ; i < M ; i++)
    for(int j = 0 ; j < N ; j++)
      A[j*N+i] = Atmp[i*M+j];
  free(Atmp);
}

//#define DEVICE_EIG
//#define DEVICE_QR

void Arnoldi::eigMagmaAndSort(Arnoldi &W, Arnoldi &Q){
  int M = arInfo.arnoldiTotalVectors;

  Complex *h_H = (Complex*)malloc(M*M*sizeof(Complex));
  Complex *h_W = (Complex*)malloc(M*sizeof(Complex));
  Complex *h_Q = (Complex*)malloc(M*M*sizeof(Complex));
  magma_int_t info,lwork,nb;
  nb    = magma_get_zgehrd_nb(M);
  lwork = M*(1 + nb);
  lwork = std::max( lwork, M*(5 + 2*M) );

  double *rwork = (double*)malloc(2*M*sizeof(double));
  magmaDoubleComplex *hwork = (magmaDoubleComplex*) malloc(lwork*sizeof(magmaDoubleComplex));

  cudaMemcpy((void*) h_H,(void*) this->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  noiseCleaner(h_H,M*M);
  transposeMatrix(h_H,M,M);

#ifndef DEVICE_EIG
  lapackf77_zgeev( lapack_vec_const(MagmaNoVec),lapack_vec_const(MagmaVec), &M,(magmaDoubleComplex*) h_H, &M,(magmaDoubleComplex*) h_W,NULL,&M,(magmaDoubleComplex*) h_Q, &M,(magmaDoubleComplex*) hwork, &lwork, rwork, &info );
  if (info != 0)
    errorQuda("lapack_zgeev returned error \n");
#else
  magma_zgeev(MagmaNoVec,MagmaVec,(magma_int_t)M,(magmaDoubleComplex*)h_H,(magma_int_t)M,(magmaDoubleComplex*)h_W,NULL,(magma_int_t)M,(magmaDoubleComplex*)h_Q,(magma_int_t)M,hwork,lwork,rwork,&info);
  if (info != 0)
    errorQuda("magma_zgeev returned error %d: %s.\n", (int) info, magma_strerror( info ));
#endif

  noiseCleaner(h_W,M);
  noiseCleaner(h_Q,M*M);
  // now before we return them to device we will sort them to magnitude descending order
  transposeMatrix(h_Q,M,M);
  sorting(h_W,h_Q);
  cudaMemset((void*) W.D_mMxM(), 0, M*M*sizeof(Complex));
  for(int i = 0 ; i < M ; i++)
    cudaMemcpy((void*) &(W.D_mMxM()[i*M+i]),(void*) &(h_W[i]),sizeof(Complex),cudaMemcpyHostToDevice);
  cudaMemcpy((void*) Q.D_mMxM(), (void*) h_Q, M*M*sizeof(Complex),cudaMemcpyHostToDevice);
  
  free(hwork);
  free(rwork);
  free(h_H);
  free(h_Q);
  free(h_W);

  checkCudaError();
}




void Arnoldi::QR_Magma(Arnoldi &Q){
  magma_int_t M = arInfo.arnoldiTotalVectors;

  magmaDoubleComplex *tau = (magmaDoubleComplex*)malloc(M*sizeof(magmaDoubleComplex));
  magma_int_t info;
  magma_int_t nb = magma_get_zgeqrf_nb(M);

#ifndef DEVICE_QR
  magma_int_t lwork;
  magma_int_t lwork2;
  magmaDoubleComplex *h_work,tmp[1];
  magmaDoubleComplex *h_work2;
  //  double work[1];
  lwork = -1;
  lapackf77_zgeqrf(&M, &M, NULL, &M, NULL, tmp, &lwork, &info);
  lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
  lwork = std::max( lwork, std::max( M*nb, 2*nb*nb ));
  magmaDoubleComplex* h_A = (magmaDoubleComplex*)malloc(M*M*sizeof(magmaDoubleComplex));
  cudaMemcpy((void*)h_A,(void*)this->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  h_work = (magmaDoubleComplex*)malloc(lwork*sizeof(magmaDoubleComplex));
  lapackf77_zgeqrf(&M, &M, h_A, &M, tau, h_work, &lwork,&info);
  if(info != 0)
    errorQuda("error in qr lapack\n");
  
  lwork2  = (M + 2*M+nb)*nb;
  h_work2 = (magmaDoubleComplex*)malloc(lwork2*sizeof(magmaDoubleComplex));
  lapackf77_zungqr( &M, &M, &M, h_A, &M, tau, h_work2, &lwork2, &info );
  if(info != 0)
    errorQuda("error in qr lapack\n");
  cudaMemcpy((void*) Q.D_mMxM(),(void*) h_A,M*M*sizeof(Complex),cudaMemcpyHostToDevice);

  free(h_A);
  free(h_work);    
  free(h_work2);
#else
  magma_int_t TN = (2*M + (M+31)/32*32 )*nb;
  Arnoldi *H = new Arnoldi(matrixMxM);
  cudaMemcpy((void*)H->D_mMxM(),(void*)this->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToDevice);
  magmaDoubleComplex* dT;

  CUDA_MALLOC_ZERO(dT,TN*sizeof(Complex));
  magma_zgeqrf_gpu(M,M,(magmaDoubleComplex*) H->D_mMxM(), M,tau,dT,&info);
  if (info != 0)
    errorQuda("magma_zgeqrf2_gpu returned error %d: %s.\n", (int) info, magma_strerror( info ));
  magma_zungqr_gpu(M,M,M,(magmaDoubleComplex*) H->D_mMxM(),M,tau,dT,nb,&info);
  if (info != 0)
    errorQuda("magma_zungqr_gpu returned error %d: %s.\n", (int) info, magma_strerror( info ));
  cudaMemcpy((void*) Q.D_mMxM(), (void*) H->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToDevice);
  cudaFree(dT);
  delete H;
  checkCudaError();
#endif

  free(tau);

}
#define DEVICE_MATRIX_MUL_INPLACE

/*
void Arnoldi::qrShiftsRotations(Arnoldi &V, Arnoldi &H){


  magma_int_t M = arInfo.arnoldiTotalVectors;
  magma_int_t P = arInfo.arnoldiUnwantedVectors;
  magma_int_t NL = arInfo.dimensionMatrix;
  magma_int_t K = arInfo.arnoldiWantedVectors;
  magmablas_ztranspose_inplace(M,(magmaDoubleComplex*)H.D_mMxM(),M);

  magmaDoubleComplex *e;
  magmaDoubleComplex *etmp;
  magmaDoubleComplex one = MAGMA_Z_MAKE(1.,0.);
  magmaDoubleComplex zero = MAGMA_Z_MAKE(0.,0.);

  CUDA_MALLOC_ZERO(e,M*sizeof(Complex));
  CUDA_MALLOC_ZERO(etmp,M*sizeof(Complex));

  cudaMemset((void*)e,0,M*sizeof(Complex));
  cudaMemcpy((void*)(e+M-1),(void*)&one,sizeof(Complex),cudaMemcpyHostToDevice);
  
  magmaDoubleComplex *unitMatrix;
  CUDA_MALLOC_ZERO(unitMatrix,M*M*sizeof(Complex));
  cudaMemset((void*)unitMatrix,0,M*M*sizeof(Complex));
  for(int i = 0 ; i < M ; i++) cudaMemcpy((void*)(unitMatrix+i*M+i),(void*) &one,sizeof(Complex),cudaMemcpyHostToDevice);

  magmaDoubleComplex *Tmp;
  CUDA_MALLOC_ZERO(Tmp,M*M*sizeof(Complex));
  cudaMemset((void*)Tmp,0,M*M*sizeof(Complex));

  Arnoldi *Hshifted = new Arnoldi(matrixMxM);
  Arnoldi *Q = new Arnoldi(matrixMxM);
  Complex *h_H = (Complex*)malloc(M*M*sizeof(Complex));

  for(int jj = 0 ; jj < P ; jj++){
    cudaMemcpy((void*)Hshifted->D_mMxM(),(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToDevice);
    magmaDoubleComplex alpha;
    cudaMemcpy((void*)&alpha,(void*) (this->D_mMxM() + jj*M+jj), sizeof(Complex), cudaMemcpyDeviceToHost );
    magmablas_zgeadd(M,M,-alpha,unitMatrix,M,(magmaDoubleComplex*) Hshifted->D_mMxM(),M);

    // clear noise in Hshifted
#ifndef __DEVICE_NOISE_CLEANER__  
    cudaMemcpy((void*)h_H,(void*)Hshifted->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    noiseCleaner(h_H,M*M);
    cudaMemcpy((void*)Hshifted->D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
    clearNoiseCuda(Hshifted->D_mMxM(),M*M);
#endif
    ////////////////////////

    Hshifted->QR_Magma(*Q);


    // Q' * H * Q
#ifndef __DEVICE_NOISE_CLEANER__
    cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    noiseCleaner(h_H,M*M);
    cudaMemcpy((void*)H.D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
    clearNoiseCuda(H.D_mMxM(),M*M);
#endif
    magmablas_zgemm(MagmaConjTrans,MagmaNoTrans,M,M,M,one,(magmaDoubleComplex*)Q->D_mMxM(),M,(magmaDoubleComplex*)H.D_mMxM(),M,zero,Tmp,M);
    magmablas_zgemm(MagmaNoTrans,MagmaNoTrans,M,M,M,one,Tmp,M,(magmaDoubleComplex*)Q->D_mMxM(),M,zero,(magmaDoubleComplex*)H.D_mMxM(),M);

#ifndef __DEVICE_NOISE_CLEANER__
    cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    noiseCleaner(h_H,M*M);
    cudaMemcpy((void*)H.D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
    clearNoiseCuda(H.D_mMxM(),M*M);
#endif

    //#define DEVICE_MATRIX_MUL_INPLACE
    // V * Q
#ifdef DEVICE_MATRIX_MUL_INPLACE
    Q->matrixNxMmatrixMxL(V,NL,M,M,true);
#else
    Complex *h_V = (Complex*)calloc(NL*M,sizeof(Complex));
    cudaMemcpy((void*)h_V,(void*)V.D_mNxM(),NL*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    Complex *h_Q = (Complex*)calloc(M*M,sizeof(Complex));
    cudaMemcpy((void*)h_Q,(void*)Q->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    Complex *tmp = (Complex*)calloc(M,sizeof(Complex));
    
    for(int i = 0 ; i < NL ; i++){
      for(int j = 0 ; j < M ; j++) tmp[j] = h_V[j*NL+i];
      for(int l = 0 ; l < M ; l++){
	Complex sum = (Complex) {0.,0.};
	for(int j = 0 ; j < M ; j++) sum = sum + tmp[j] * h_Q[l*M+j];
	h_V[l*NL+i] = sum;
      }
    }

    cudaMemcpy((void*)V.D_mNxM(),(void*)h_V,NL*M*sizeof(Complex),cudaMemcpyHostToDevice);

    free(tmp);
    free(h_V);
    free(h_Q);
#endif
    
    // e*Q
    cudaMemcpy((void*)etmp,(void*)e,M*sizeof(Complex),cudaMemcpyDeviceToDevice);
    magma_zgemv(MagmaTrans,M,M,one,(magmaDoubleComplex*)Q->D_mMxM(),M,etmp,1, zero,e,1);

  }


  magmaDoubleComplex alpha;
  magmaDoubleComplex heta;

  cudaMemcpy((void*)&alpha,(void*)&(e[K-1]),sizeof(Complex),cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)&heta,(void*)&(H.D_mMxM()[K*M+K-1]),sizeof(Complex),cudaMemcpyDeviceToHost);

  magma_zscal(NL,alpha,(magmaDoubleComplex*) (V.D_mNxM() + M*NL ),1);

  magma_zscal(NL,heta,(magmaDoubleComplex*) (V.D_mNxM() + K*NL ),1);
  magma_zaxpy (NL,one,(magmaDoubleComplex*) (V.D_mNxM() + M*NL ), 1, (magmaDoubleComplex*) (V.D_mNxM() + K*NL),1 );

  magmablas_ztranspose_inplace(M,(magmaDoubleComplex*)H.D_mMxM(),M);

#ifndef __DEVICE_NOISE_CLEANER__
  cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  noiseCleaner(h_H,M*M);
  cudaMemcpy((void*)H.D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
  clearNoiseCuda(H.D_mMxM(),M*M);
#endif

  delete Q;
  delete Hshifted;
  cudaFree(e);
  cudaFree(etmp);
  cudaFree(unitMatrix);
  cudaFree(Tmp);
  free(h_H);
  checkCudaError();
}
*/


void Arnoldi::qrShiftsRotations(Arnoldi &V, Arnoldi &H){


  magma_int_t M = arInfo.arnoldiTotalVectors;
  magma_int_t P = arInfo.arnoldiUnwantedVectors;
  magma_int_t NL = arInfo.dimensionMatrix;
  magma_int_t K = arInfo.arnoldiWantedVectors;
  magmablas_ztranspose_inplace(M,(magmaDoubleComplex*)H.D_mMxM(),M);

  magmaDoubleComplex *e;
  magmaDoubleComplex *etmp;
  magmaDoubleComplex one = MAGMA_Z_MAKE(1.,0.);
  magmaDoubleComplex zero = MAGMA_Z_MAKE(0.,0.);

  CUDA_MALLOC_ZERO(e,M*sizeof(Complex));
  CUDA_MALLOC_ZERO(etmp,M*sizeof(Complex));

  cudaMemset((void*)e,0,M*sizeof(Complex));
  cudaMemcpy((void*)(e+M-1),(void*)&one,sizeof(Complex),cudaMemcpyHostToDevice);
  
  magmaDoubleComplex *unitMatrix;
  CUDA_MALLOC_ZERO(unitMatrix,M*M*sizeof(Complex));
  cudaMemset((void*)unitMatrix,0,M*M*sizeof(Complex));
  for(int i = 0 ; i < M ; i++) cudaMemcpy((void*)(unitMatrix+i*M+i),(void*) &one,sizeof(Complex),cudaMemcpyHostToDevice);

  magmaDoubleComplex *Tmp;
  CUDA_MALLOC_ZERO(Tmp,M*M*sizeof(Complex));
  cudaMemset((void*)Tmp,0,M*M*sizeof(Complex));



  Arnoldi *Hshifted = new Arnoldi(matrixMxM);
  Arnoldi *Q = new Arnoldi(matrixMxM);
  Arnoldi *Qnew = new Arnoldi(matrixMxM);
  cudaMemcpy((void*)Qnew->D_mMxM(),(void*)unitMatrix,M*M*sizeof(Complex),cudaMemcpyDeviceToDevice);
  Complex *h_H = (Complex*)malloc(M*M*sizeof(Complex));

  for(int jj = 0 ; jj < P ; jj++){

    cudaMemcpy((void*)Hshifted->D_mMxM(),(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToDevice);
    magmaDoubleComplex alpha;
    cudaMemcpy((void*)&alpha,(void*) (this->D_mMxM() + jj*M+jj), sizeof(Complex), cudaMemcpyDeviceToHost );
    magmablas_zgeadd(M,M,-alpha,unitMatrix,M,(magmaDoubleComplex*) Hshifted->D_mMxM(),M);

    // clear noise in Hshifted
#ifndef __DEVICE_NOISE_CLEANER__
    cudaMemcpy((void*)h_H,(void*)Hshifted->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    noiseCleaner(h_H,M*M);
    cudaMemcpy((void*)Hshifted->D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
  clearNoiseCuda(Hshifted->D_mMxM(),M*M,arInfo.tolerance);
#endif
    ////////////////////////

    Hshifted->QR_Magma(*Q);

    magmablas_zgemm(MagmaConjTrans,MagmaNoTrans,M,M,M,one,(magmaDoubleComplex*)Q->D_mMxM(),M,(magmaDoubleComplex*)H.D_mMxM(),M,zero,Tmp,M);
    magmablas_zgemm(MagmaNoTrans,MagmaNoTrans,M,M,M,one,Tmp,M,(magmaDoubleComplex*)Q->D_mMxM(),M,zero,(magmaDoubleComplex*)H.D_mMxM(),M);

#ifndef __DEVICE_NOISE_CLEANER__
    cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    noiseCleaner(h_H,M*M);
    cudaMemcpy((void*)H.D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
  clearNoiseCuda(H.D_mMxM(),M*M,arInfo.tolerance);
#endif
    magmablas_zgemm(MagmaNoTrans,MagmaNoTrans,M,M,M,one,(magmaDoubleComplex*)Qnew->D_mMxM(),M,(magmaDoubleComplex*)Q->D_mMxM(),M,zero,Tmp,M);
    cudaMemcpy((void*)Qnew->D_mMxM(),(void*)Tmp,M*M*sizeof(Complex),cudaMemcpyDeviceToDevice);
    //#define DEVICE_MATRIX_MUL_INPLACE
    // V * Q
    
    // e*Q
    cudaMemcpy((void*)etmp,(void*)e,M*sizeof(Complex),cudaMemcpyDeviceToDevice);
    magma_zgemv(MagmaTrans,M,M,one,(magmaDoubleComplex*)Q->D_mMxM(),M,etmp,1, zero,e,1);

  }


  Qnew->matrixNxMmatrixMxLReal(V,NL,M,M,true);

  magmaDoubleComplex alpha;
  magmaDoubleComplex heta;

  cudaMemcpy((void*)&alpha,(void*)&(e[K-1]),sizeof(Complex),cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)&heta,(void*)&(H.D_mMxM()[K*M+K-1]),sizeof(Complex),cudaMemcpyDeviceToHost);

  magma_zscal(NL,alpha,(magmaDoubleComplex*) (V.D_mNxM() + M*NL ),1);

  magma_zscal(NL,heta,(magmaDoubleComplex*) (V.D_mNxM() + K*NL ),1);
  magma_zaxpy (NL,one,(magmaDoubleComplex*) (V.D_mNxM() + M*NL ), 1, (magmaDoubleComplex*) (V.D_mNxM() + K*NL),1 );

  magmablas_ztranspose_inplace(M,(magmaDoubleComplex*)H.D_mMxM(),M);

#ifndef __DEVICE_NOISE_CLEANER__
  cudaMemcpy((void*)h_H,(void*)H.D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  noiseCleaner(h_H,M*M);
  cudaMemcpy((void*)H.D_mMxM(),(void*)h_H,M*M*sizeof(Complex),cudaMemcpyHostToDevice);
#else
  clearNoiseCuda(H.D_mMxM(),M*M,arInfo.tolerance);
#endif

  delete Q;
  delete Hshifted;
  delete Qnew;
  cudaFree(e);
  cudaFree(etmp);
  cudaFree(unitMatrix);
  cudaFree(Tmp);
  free(h_H);
  checkCudaError();
}


void quda::Iram(void **gauge,QudaInvertParam *inv_param, QudaGaugeParam *gauge_param, qudaQKXTMinfo in_latInfo, arlibInfo in_arInfo ){



  init_arnoldi(in_arInfo,in_latInfo); 
  printfQuda("Start iteration phase\n");
  double time;
  time = MPI_Wtime();

  Dirac *dirac = initQudaMatrix(inv_param,gauge_param,gauge);

  cudaColorSpinorField *vec1 = initQudaVector(inv_param,gauge_param);
  cudaColorSpinorField *vec2 = initQudaVector(inv_param,gauge_param);

  Arnoldi *V = new Arnoldi(matrixNxM);
  Arnoldi *H = new Arnoldi(matrixMxM);
  Arnoldi *W = new Arnoldi(matrixMxM);
  Arnoldi *Q = new Arnoldi(matrixMxM);
  Arnoldi *caller = NULL;

  int K = arInfo.arnoldiWantedVectors;
  int M = arInfo.arnoldiTotalVectors;
  int NL = arInfo.dimensionMatrix;

  Complex *eigenVaules = (Complex*)malloc(M*M*sizeof(Complex));
  

  int iter = 0;
  double res = 1.;

  //  test(*V,*H,*dirac,*vec1,*vec2);  

  /*
  caller->initArnold(*V,*H,*dirac,*vec1,*vec2);
  caller->arnold(1,K,*V,*H,*dirac,*vec1,*vec2);
  caller->arnold(K,M,*V,*H,*dirac,*vec1,*vec2);
  H->eigMagmaAndSort(*W,*Q);
  W->qrShiftsRotations(*V,*H);

  Complex *h_H = (Complex*)calloc(M*M,sizeof(Complex));
  Complex *h_V = (Complex*)calloc(NL,(M+1)*sizeof(Complex));
  Complex *h_W = (Complex*)calloc(M*M,sizeof(Complex));
  Complex *h_Q = (Complex*)calloc(M*M,sizeof(Complex));

  cudaMemcpy((void*)h_H,(void*)H->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)h_V,(void*)V->D_mNxM(),NL*(M+1)*sizeof(Complex),cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)h_W,(void*)W->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)h_Q,(void*)Q->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);


  for(int i = 0 ; i < M ; i++){
    for(int j = 0 ; j < M ; j++)
      printf("%+e %+e\t",h_H[i*M+j].real(),h_H[i*M+j].imag());
    printf("\n");
  }
    printf("\n");


  for(int i = 0 ; i < M ; i++){
    for(int j = 0 ; j < M ; j++)
      printf("%+e %+e\t",h_W[i*M+j].real(),h_W[i*M+j].imag());
    printf("\n");
  }
    printf("\n");

  for(int i = 0 ; i < M ; i++){
    for(int j = 0 ; j < M ; j++)
      printf("%+e %+e\t",h_Q[i*M+j].real(),h_Q[i*M+j].imag());
    printf("\n");
  }


  FILE *ptr;
  ptr = fopen("kale.dat","w");

  for(int i = 0 ; i < M+1 ; i++){
    for(int j = 0 ; j < NL ; j++)
      fprintf(ptr,"%+e %+e\n",h_V[i*NL+j].real(),h_V[i*NL+j].imag());
  }
  */

  Complex *h_H = (Complex*)calloc(M*M,sizeof(Complex));
  
  caller->initArnold(*V,*H,*dirac,*vec1,*vec2);
  caller->arnold(1,K,*V,*H,*dirac,*vec1,*vec2);


  Complex *h_W = (Complex*)calloc(M*M,sizeof(Complex));
  //  Complex *h_H = (Complex*)calloc(M*M,sizeof(Complex));


  double timeArnoldi=0, accumArnoldi=0;
  double timeEig=0, accumEig =0;
  double timeQRshifts = 0, accumQRshifts = 0;

  while( (res > arInfo.tolerance) && (iter < arInfo.maxArnoldiIter) ){
    timeArnoldi = MPI_Wtime();
    caller->arnold(K,M,*V,*H,*dirac,*vec1,*vec2);
    accumArnoldi += MPI_Wtime() - timeArnoldi;

    /*
    if(iter == 1){
    cudaMemcpy((void*)h_H,(void*)H->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < M ; i++){
      for(int j = 0 ; j < M ; j++)
	printf("%+e  ",h_H[i*M+j].real());
      printf("\n");
    }
    return;
    }
    */
    /*
    if(iter == 1){
      for(int i = 0 ; i < M ; i++){
	res = normMagma(V->D_mNxM()+i*NL,NL);
	printfQuda("iter = %d, res = %16.15e\n",iter,res);
      }
    return;
    }
    */
    /*
    cudaMemcpy((void*)h_H,(void*)H->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < M ; i++){
      for(int j = 0 ; j < M ; j++)
	printf("%+e %+e\t",h_H[i*M+j].real(),h_H[i*M+j].imag());
      printf("\n");
    }
    */
    timeEig = MPI_Wtime();
    H->eigMagmaAndSort(*W,*Q);
    accumEig += MPI_Wtime() - timeEig;




    /*
    printf("iter = %d ",iter);
    for(int i = 0 ; i < M ; i++)printf("%+e %+e\t",h_W[i*M+i].real(),h_W[i*M+i].imag());
    printf("\n");
    */

    timeQRshifts = MPI_Wtime();
    W->qrShiftsRotations(*V,*H);
    accumQRshifts += MPI_Wtime() - timeQRshifts;



    /*
  cudaMemcpy((void*)h_H,(void*)H->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  for(int i = 0 ; i < M ; i++){
    for(int j = 0 ; j < M ; j++)
      printf("%+e %+e\t",h_H[i*M+j].real(),h_H[i*M+j].imag());
    printf("\n");
  }
  free(h_H);
  return;
    */

    /*
    cudaMemcpy((void*)h_H,(void*)H->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  for(int i = 0 ; i < M ; i++){
    for(int j = 0 ; j < M ; j++)
      printf("%+e %+e\t",h_H[i*M+j].real(),h_H[i*M+j].imag());
    printf("\n");
  }
    */
  //    printf("\n");


    //    H->eigMagmaAndSort(*W,*Q);

    res = normMagma(V->D_mNxM()+M*NL,NL);

    printfQuda("iter = %d, res = %16.15e\n",iter,res);
    iter++;
  }
  printfQuda("Cleaning took %f sec\n",accumCleaning);
  printfQuda("Norm and Dot took %f sec\n",accumApplyNorm);
  printfQuda("YPAX application took %f sec\n",accumApplyYPAX);
  printfQuda("Matrix application took %f sec\n",accumApplyMatrix);
  printfQuda("Arnoldi took %f sec\n",accumArnoldi);
  printfQuda("Eig took %f sec\n",accumEig);
  printfQuda("QRshifts took %f sec\n",accumQRshifts);


  /*
  Q->matrixNxMmatrixMxK(*V,NL,M,K);
  for(int i = 0 ; i < K ; i++){
    double norma = normMagma(V->D_mNxM() + i*NL,NL);
    magma_zdscal(NL,(1./norma),(magmaDoubleComplex*)(V->D_mNxM() + i*NL),1);
  }
  */

  H->eigMagmaAndSort(*W,*Q);
  cudaMemcpy((void*)eigenVaules,(void*)W->D_mMxM(),M*M*sizeof(Complex),cudaMemcpyDeviceToHost);
  for(int i = 0 ; i < K ; i++)
    printfQuda("eigvalue %d : %f \n",i,eigenVaules[(M-1-i)*M + (M-1-i)].real());



  time = MPI_Wtime() - time;
  printfQuda("Execution time took %f sec\n",time);

  delete vec1;
  delete vec2;
  delete V;
  delete H;
  delete W;
  delete Q;
  free(eigenVaules);
  delete dirac;

  finalize_arnoldi();

}


