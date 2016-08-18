#include <stdlib.h>
#include <string.h>
#include <qudaQKXTM.h>
#include <quda.h>
#include <dirac_quda.h>
#include <lanczos.h>
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
#include <malloc_quda.h>

//#define PRECISION_z
//#define COMPLEX


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
lanczosInfo lInfo;

#define MAGMA_MALLOC_PIN(ptr,type,size) \
  if(MAGMA_SUCCESS != magma_malloc_pinned( (void**) &ptr, (size)*sizeof(type) )){ \
  magma_finalize();							\
  errorQuda("!!!! magma_malloc_pinned failed for: %s\n", #ptr );	\
  }

#define MAGMA_FREE_PIN( ptr) magma_free_pinned( ptr )

#define CUDA_MALLOC_ZERO( pointer, size )\
  cudaMalloc((void**)&pointer,size);	 \
  cudaMemset((void*)pointer,0,size);

static void initEig(){
  int M = lInfo.lanczosTotalVectors;

  double *h_H = (double*)calloc(M*M,sizeof(double));
  double *h_W_r = (double*)calloc(M,sizeof(double));
  double *h_W_i = (double*)calloc(M,sizeof(double));
  double *h_Q = (double*)calloc(M*M,sizeof(double));

  magma_int_t info,lwork,nb;
  nb    = magma_get_dgehrd_nb(M);
  lwork = M*(1 + nb);
  lwork = std::max( lwork, M*(5 + 2*M) );

  double *hwork = (double*) malloc(lwork*sizeof(double));


  lapackf77_dgeev( lapack_vec_const(MagmaNoVec),lapack_vec_const(MagmaVec), &M, h_H, &M, h_W_r,h_W_i,NULL,&M, h_Q, &M, hwork, &lwork, &info );
  if (info != 0)
    errorQuda("lapack_dgeev returned error \n");

  free(h_H);
  free(h_W_r);
  free(h_W_i);
  free(h_Q);
  free(hwork);

}



void quda::initLanczos(lanczosInfo in_lanczosInfo, qudaQKXTMinfo latInfo){
  double time;
  time = MPI_Wtime();
  lInfo = in_lanczosInfo;
  init_qudaQKXTM(&latInfo);
  magma_init();
  cublasInit();
  initEig();
  printf_qudaQKXTM();
  printfLanczos();
  time = MPI_Wtime() - time;
  printfQuda("Initialization took %f sec\n",time);
}

void quda::finalizeLanczos(){
  magma_finalize();
  cublasShutdown();
}


void quda::printfLanczos(){

  printfQuda("The Matrix has dimensions %d X %d\n",lInfo.dimensionMatrix*G_nProc[0]*G_nProc[1]*G_nProc[2]*G_nProc[3],lInfo.dimensionMatrix*G_nProc[0]*G_nProc[1]*G_nProc[2]*G_nProc[3]);
  printfQuda("The total working vectors you ask is %d\n",lInfo.lanczosTotalVectors);
  printfQuda("You seek for %d eigenvalues\n",lInfo.lanczosWantedVectors);
  printfQuda("The expanded working space is %d vectors\n",lInfo.lanczosUnwantedVectors);
  printfQuda("The maximum number of iteration is %d\n",lInfo.maxLanczosIter);
  printfQuda("The tolerance for the lanczos residual vector is %e\n",lInfo.tolerance);
  printfQuda("The first vector will be randomly initialized using seed  %ld\n",lInfo.seed);
  if(lInfo.smlm == SM)  printfQuda("You are looking for the smallest eigenvalues\n");
  if(lInfo.smlm == LM)  printfQuda("You are looking for the largest eigenvalues\n");
}


Lanczos::Lanczos(enum MATRIXTYPE_LANCZOS matrixTYPE): d_mNxM(NULL),d_mMxM(NULL),d_vN(NULL){

  if(matrixTYPE == matrixNxM_L){
    size = G_localVolume * G_nSpin * G_nColor * (lInfo.lanczosTotalVectors) * sizeof(Complex);
    d_mNxM =(Complex*) device_malloc(size);
    cudaMemset((void*)d_mNxM,0,size);
    checkCudaError();
    G_deviceMemory += size/(1024.*1024.);
    printfQuda("Device memory in used is %f MB A \n",G_deviceMemory);
  }
  else if(matrixTYPE == matrixMxM_L){
    checkCudaError();
    size = lInfo.lanczosTotalVectors*lInfo.lanczosTotalVectors * sizeof(double);
    checkCudaError();
    d_mMxM =(double*) device_malloc(size);
    checkCudaError();
    cudaMemset((void*)d_mMxM,0,size);
    checkCudaError();

  }
  else if(matrixTYPE == vectorN_L){
    size = G_localVolume * G_nSpin * G_nColor * sizeof(Complex);
    d_vN =(Complex*) device_malloc(size);
    cudaMemset((void*)d_vN,0,size);
    checkCudaError();
    G_deviceMemory += size/(1024.*1024.);
    printfQuda("Device memory in used is %f MB A \n",G_deviceMemory);
  }
  else{
    errorQuda("Wrong choice for Lanczos matrices\n");
  }
  checkCudaError();
}


Lanczos::~Lanczos(){
  if(d_mNxM != NULL){
    device_free(d_mNxM);
    G_deviceMemory -= size/(1024.*1024.);
    printfQuda("Device memory in used is %f MB D \n",G_deviceMemory);    
  }
  if(d_mMxM != NULL){
    device_free(d_mMxM);
  }
  if(d_vN != NULL){
    device_free(d_vN);
    G_deviceMemory -= size/(1024.*1024.);
    printfQuda("Device memory in used is %f MB D \n",G_deviceMemory);    
  }
  checkCudaError();
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



void Lanczos::applyMatrix(Dirac &A,cudaColorSpinorField &in, cudaColorSpinorField &out,Lanczos &inn, int offsetIn){
  inn.uploadToCuda(in,offsetIn);
  A.MdagM(out,in);
  this->downloadFromCuda(out);
  checkCudaError();
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

/*
#define __LANCZOS_QUDA__
#ifndef __LANCZOS_QUDA__
void Lanczos::lanczos(int kstart, int kstop, Lanczos &V, Lanczos &T, Lanczos &r, Dirac &A, cudaColorSpinorField &in, cudaColorSpinorField &out){

  int M = lInfo.lanczosTotalVectors;
  int NL = lInfo.dimensionMatrix;
  double beta;
  Complex alpha;
  double *sigma = (double*)calloc(M,sizeof(double));
  double *S;
  CUDA_MALLOC_ZERO(S,M*sizeof(double));
  Complex *h_r = (Complex*)malloc(NL*sizeof(Complex));

  if(kstart == 0){
    for(int i = 0 ; i < NL ; i++){
      h_r[i].real() =  1.0;
      h_r[i].imag() = 0.0;
    }
    cudaMemcpy((void*)r.D_vN(),(void*)h_r,NL*sizeof(Complex),cudaMemcpyHostToDevice);
  }

  for(int j = kstart; j < kstop; j++){
    beta = normMagma(r.D_vN(),NL);
    magma_zdscal((magma_int_t)NL,(1./beta),(magmaDoubleComplex*)r.D_vN(),1);
    magma_zcopy((magma_int_t)NL,(magmaDoubleComplex*)r.D_vN(),1,(magmaDoubleComplex*)(V.D_mNxM()+j*NL),1);
    //    checkCudaError();

    if( j != 0 ){
      cudaMemcpy( (void*)(T.D_mMxM() + j*M+(j-1)),(void*)&beta,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy( (void*)(T.D_mMxM() + (j-1)*M+j),(void*)(T.D_mMxM() + j*M+j-1),sizeof(double),cudaMemcpyDeviceToDevice);
    }

    //    checkCudaError();
    if( j == 0){
      r.applyMatrix(A,in,out,V,0);
    }
    else{
      r.applyMatrix(A,in,out,V,j);
      magma_zaxpy((magma_int_t)NL,MAGMA_Z_MAKE(-beta,0),(magmaDoubleComplex*)(V.D_mNxM() + (j-1)*NL),1,(magmaDoubleComplex*)r.D_vN(),1);
    }

    alpha = dotMagma(V.D_mNxM() + j*NL,r.D_vN(),NL);
    magma_zaxpy((magma_int_t)NL,MAGMA_Z_MAKE(-alpha.real(),0),(magmaDoubleComplex*)(V.D_mNxM() + j*NL),1,(magmaDoubleComplex*)r.D_vN(),1);
    cudaMemcpy((void*)(T.D_mMxM() + j*M+j),(void*)&alpha,sizeof(double),cudaMemcpyHostToDevice);

    if( j != 0 ){
      for(int i = 0 ; i <= j ; i++) sigma[i] = real(dotMagma(V.D_mNxM() + i*NL,r.D_vN(),NL));
      cudaMemcpy((void*)S,(void*)sigma,(j+1)*sizeof(double),cudaMemcpyHostToDevice);
      magma_zgemv(MagmaNoTrans,(magma_int_t)NL,(magma_int_t)(j+1),MAGMA_Z_MAKE(-1.,0.),(magmaDoubleComplex*)V.D_mNxM(),(magma_int_t)NL,(magmaDoubleComplex*)S,1, MAGMA_Z_MAKE(1.,0.),(magmaDoubleComplex*)r.D_vN(),1);
      sigma[0] = real(dotMagma(V.D_mNxM() + j*NL,r.D_vN(),NL));
      alpha = beta + sigma[0];
      cudaMemcpy((void*)(T.D_mMxM() + j*M+j-1),(void*)&alpha,sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)(T.D_mMxM() + (j-1)*M+j),(void*)(T.D_mMxM() + j*M+j-1),sizeof(double),cudaMemcpyDeviceToDevice);
    }

  }

  free(h_r);
  free(sigma);
  cudaFree(S);
}
*/

void Lanczos::lanczos(int kstart, int kstop, Lanczos &V, Lanczos &T, Lanczos &r, Dirac &A, cudaColorSpinorField &in, cudaColorSpinorField &out){

  int M = lInfo.lanczosTotalVectors;
  int NL = lInfo.dimensionMatrix;

  double *alpha = (double*)calloc(kstop,sizeof(double));
  double *beta = (double*)calloc(kstop,sizeof(double));
  double ff;

  Complex *h_r = (Complex*)malloc(NL*sizeof(Complex));
  if(kstart == 0){
    for(int i = 0 ; i < NL ; i++){
      h_r[i].real() =  1.0;
      h_r[i].imag() = 0.0;
    }
    cudaMemcpy((void*)r.D_vN(),(void*)h_r,NL*sizeof(Complex),cudaMemcpyHostToDevice);
  }
  ff = normMagma(r.D_vN(),NL);
  if(kstart != 0)beta[kstart-1] = ff;

  //  if(kstart != 0)cudaMemcpy((void*)&(beta[kstart-1]),(void*)(T.D_mMxM()+(kstart-1)*M+kstart),sizeof(double),cudaMemcpyDeviceToHost);

  cudaMemset((void*)(V.D_mNxM()+kstart*NL),0,NL*sizeof(Complex));
  magma_zaxpy((magma_int_t)NL,MAGMA_Z_MAKE(1/ff,0),(magmaDoubleComplex*)r.D_vN(),1,(magmaDoubleComplex*)(V.D_mNxM()+kstart*NL),1);

  for(int k = kstart ; k < kstop ; k++)
    if(k == 0){
      r.applyMatrix(A,in,out,V,0);
      alpha[0] = real(dotMagma(V.D_mNxM(),r.D_vN(),NL));
      magma_zaxpy((magma_int_t)NL,MAGMA_Z_MAKE(-alpha[0],0),(magmaDoubleComplex*)V.D_mNxM(),1,(magmaDoubleComplex*)r.D_vN(),1);
      beta[0] = normMagma(r.D_vN(),NL);

      cudaMemset((void*)(V.D_mNxM()+1*NL),0,NL*sizeof(Complex));
      magma_zaxpy((magma_int_t)NL,MAGMA_Z_MAKE(1/beta[0],0),(magmaDoubleComplex*)r.D_vN(),1,(magmaDoubleComplex*)(V.D_mNxM()+1*NL),1);

    }
    else{
      r.applyMatrix(A,in,out,V,k);
      magma_zaxpy((magma_int_t)NL,MAGMA_Z_MAKE(-beta[k-1],0),(magmaDoubleComplex*)(V.D_mNxM()+(k-1)*NL),1,(magmaDoubleComplex*)r.D_vN(),1);
      alpha[k] = real(dotMagma(V.D_mNxM()+k*NL,r.D_vN(),NL));
      magma_zaxpy((magma_int_t)NL,MAGMA_Z_MAKE(-alpha[k],0),(magmaDoubleComplex*)(V.D_mNxM()+(k)*NL),1,(magmaDoubleComplex*)r.D_vN(),1);
      beta[k] = normMagma(r.D_vN(),NL);

      // perform reorthogonalization
      Complex xp(0.,0.);
      for(int i = 0 ; i < k+1 ; i++){
	xp = dotMagma(V.D_mNxM()+i*NL,r.D_vN(),NL);
	//	printfQuda("%d %+e %+e\n",k,real(xp),imag(xp));
	//	if(fabs(real(xp)) > 1e-13 || fabs(imag(xp)) > 1e-13)
	//  warningQuda("It seems that somehow we loose orthogonality\n");
	magma_zaxpy((magma_int_t)NL,MAGMA_Z_MAKE(-xp.real(),-xp.imag()),(magmaDoubleComplex*)(V.D_mNxM()+i*NL),1,(magmaDoubleComplex*)r.D_vN(),1);
      }
      ///////////////////////////////

      if(k+1 < kstop){
	cudaMemset((void*)(V.D_mNxM()+(k+1)*NL),0,NL*sizeof(Complex));
	magma_zaxpy((magma_int_t)NL,MAGMA_Z_MAKE(1/beta[k],0),(magmaDoubleComplex*)r.D_vN(),1,(magmaDoubleComplex*)(V.D_mNxM()+(k+1)*NL),1);
      }

    }
  
  // set alpha and beta on T
  for(int i = kstart ; i < kstop ; i++){
    if( kstart==0 ){
      cudaMemcpy((void*)(T.D_mMxM()+(i+1)*M+i),(void*)&(beta[i]),sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)(T.D_mMxM()+(i)*M+i+1),(void*)(T.D_mMxM()+(i+1)*M+i),sizeof(double),cudaMemcpyDeviceToDevice);
    }
    else{
      cudaMemcpy((void*)(T.D_mMxM()+(i-1)*M+i),(void*)&(beta[i-1]),sizeof(double),cudaMemcpyHostToDevice);
      cudaMemcpy((void*)(T.D_mMxM()+(i)*M+i-1),(void*)(T.D_mMxM()+(i-1)*M+i),sizeof(double),cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy((void*)(T.D_mMxM()+i*M+i),(void*)&(alpha[i]),sizeof(double),cudaMemcpyHostToDevice);
  }

  free(h_r);
  free(alpha);
  free(beta);
}



static void sorting(double *h_W, double *h_Q){
  int M = lInfo.lanczosTotalVectors;

  double *w = (double*)malloc(M*sizeof(double));
  double *w_abs = (double*)malloc(M*sizeof(double));
  double *w_abs_tmp = (double*)malloc(M*sizeof(double));

  int *pos = (int*)malloc(M*sizeof(int));
  double *q = (double*)malloc(M*M*sizeof(double));
  memcpy(q,h_Q,M*M*sizeof(double));

  for(int i = 0 ; i < M ; i++) w[i] = h_W[i];
  for(int i = 0 ; i < M ; i++) w_abs[i] = fabs(w[i]);
  memcpy(w_abs_tmp,w_abs,M*sizeof(double));

  if(lInfo.smlm == SM){
    for(int i = 0 ; i < M-1 ; i++)
      for(int j = i + 1 ; j < M ; j++){
	if(w_abs[i] < w_abs[j]){
	  double a = w_abs[i];
	  w_abs[i] = w_abs[j];
	  w_abs[j] = a;
	}
      }
  }
  else if ( lInfo.smlm == LM ){
    for(int i = 0 ; i < M-1 ; i++)
      for(int j = i + 1 ; j < M ; j++){
	if(w_abs[i] > w_abs[j]){
	  double a = w_abs[i];
	  w_abs[i] = w_abs[j];
	  w_abs[j] = a;
	}
      }
  }
  
  for(int i = 0 ; i < M ; i++)
    for(int j = 0 ; j < M ; j++)
      if(w_abs[i] == w_abs_tmp[j])
	pos[i] = j;

  for(int i = 0 ; i < M ; i++) h_W[i] = w[pos[i]];

  for(int i = 0 ; i < M ; i++)
    for(int j = 0 ; j < M ; j++)
      h_Q[i*M+j] = q[i*M+pos[j]];

  free(w);
  free(w_abs);
  free(w_abs_tmp);
  free(pos);
  free(q);
}


static void transposeMatrix(double *A,int M, int N){
  double *Atmp = (double*)malloc(M*N*sizeof(double));
  memcpy(Atmp,A,M*N*sizeof(double));
  for(int i = 0 ; i < M ; i++)
    for(int j = 0 ; j < N ; j++)
      A[j*N+i] = Atmp[i*M+j];
  free(Atmp);
}


void Lanczos::eigMagmaAndSort(Lanczos &W, Lanczos &Q){
  int M = lInfo.lanczosTotalVectors;

  double *h_T = (double*)malloc(M*M*sizeof(double));
  double *h_W_r = (double*)malloc(M*sizeof(double));
  double *h_W_i = (double*)malloc(M*sizeof(double));
  double *h_Q = (double*)malloc(M*M*sizeof(double));

  magma_int_t info,lwork,nb;
  nb    = magma_get_dgehrd_nb(M);
  lwork = M*(1 + nb);
  lwork = std::max( lwork, M*(5 + 2*M) );

  double *hwork = (double*) malloc(lwork*sizeof(double));

  cudaMemcpy((void*) h_T,(void*) this->D_mMxM(),M*M*sizeof(double),cudaMemcpyDeviceToHost);

  transposeMatrix(h_T,M,M);

  lapackf77_dgeev( lapack_vec_const(MagmaNoVec),lapack_vec_const(MagmaVec), &M, h_T, &M, h_W_r,h_W_i,NULL,&M, h_Q, &M, hwork, &lwork, &info );

  if (info != 0)
    errorQuda("lapack_zgeev returned error \n");


  transposeMatrix(h_Q,M,M);
  sorting(h_W_r,h_Q);

  cudaMemset((void*) W.D_mMxM(), 0, M*M*sizeof(double));

  for(int i = 0 ; i < M ; i++)
    cudaMemcpy((void*) &(W.D_mMxM()[i*M+i]),(void*) &(h_W_r[i]),sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy((void*) Q.D_mMxM(), (void*) h_Q, M*M*sizeof(double),cudaMemcpyHostToDevice);
  
  free(hwork);
  free(h_T);
  free(h_Q);
  free(h_W_r);
  free(h_W_i);

  checkCudaError();
}


void Lanczos::QR_Magma(Lanczos &Q){

  magma_int_t M = lInfo.lanczosTotalVectors;
  double *tau = (double*)malloc(M*sizeof(double));
  magma_int_t info;
  magma_int_t nb = magma_get_dgeqrf_nb(M);

  magma_int_t lwork;
  magma_int_t lwork2;

  double *h_work,tmp[1];
  double *h_work2;

  lwork = -1;
  lapackf77_dgeqrf(&M, &M, NULL, &M, NULL, tmp, &lwork, &info);
  lwork = (magma_int_t) tmp[0] ;
  lwork = std::max( lwork, std::max( M*nb, 2*nb*nb ));

  double* h_A = (double*)malloc(M*M*sizeof(double));
  cudaMemcpy((void*)h_A,(void*)this->D_mMxM(),M*M*sizeof(double),cudaMemcpyDeviceToHost);

  h_work = (double*)malloc(lwork*sizeof(double));
  lapackf77_dgeqrf(&M, &M, h_A, &M, tau, h_work, &lwork,&info);
  if(info != 0)
    errorQuda("error in qr lapack\n");
  
  lwork2  = (M + 2*M+nb)*nb;
  h_work2 = (double*)malloc(lwork2*sizeof(double));
  lapackf77_dorgqr( &M, &M, &M, h_A, &M, tau, h_work2, &lwork2, &info );

  if(info != 0)
    errorQuda("error in qr lapack\n");
  cudaMemcpy((void*) Q.D_mMxM(),(void*) h_A,M*M*sizeof(double),cudaMemcpyHostToDevice);

  free(h_A);
  free(h_work);    
  free(h_work2);
  free(tau);

}

/*
static void noiseCleaner(double* A, int L){
  double matrixOrder = 0.;

    for(int i = 0 ; i < L ; i++)
      matrixOrder += fabs(A[i]);

    for(int i = 0 ; i < L ; i++){
      if( fabs(A[i]) < lInfo.tolerance/matrixOrder )
          A[i] = 0;
      }


    for(int i = 0 ; i < L ; i++){
      if( fabs(A[i]) < lInfo.tolerance/matrixOrder ) A[i] = 0;
      }

}
*/


void Lanczos::qrShiftsRotations(Lanczos &V, Lanczos &T, Lanczos &r){

  magma_int_t M = lInfo.lanczosTotalVectors;
  magma_int_t P = lInfo.lanczosUnwantedVectors;
  magma_int_t NL = lInfo.dimensionMatrix;
  magma_int_t K = lInfo.lanczosWantedVectors;


  double one = 1.;
  double zero = 0.;
  
  double *unitMatrix;
  CUDA_MALLOC_ZERO(unitMatrix,M*M*sizeof(double));
  cudaMemset((void*)unitMatrix,0,M*M*sizeof(double));

  for(int i = 0 ; i < M ; i++) cudaMemcpy((void*)(unitMatrix+i*M+i),(void*) &one,sizeof(double),cudaMemcpyHostToDevice);

  double *Tmp;
  CUDA_MALLOC_ZERO(Tmp,M*M*sizeof(double));
  cudaMemset((void*)Tmp,0,M*M*sizeof(double));

  checkCudaError();
  Lanczos *Tshifted = new Lanczos(matrixMxM_L);
  checkCudaError();
  Lanczos *Q = new Lanczos(matrixMxM_L);
  checkCudaError();
  Lanczos *Qnew = new Lanczos(matrixMxM_L);
  checkCudaError();
  cudaMemcpy((void*)Qnew->D_mMxM(),(void*)unitMatrix,M*M*sizeof(double),cudaMemcpyDeviceToDevice);

  double *h_T = (double*)calloc(M*M,sizeof(double));

  for(int jj = 0 ; jj < P ; jj++){
    cudaMemcpy((void*)Tshifted->D_mMxM(),(void*)T.D_mMxM(),M*M*sizeof(double),cudaMemcpyDeviceToDevice);
    double alpha;
    cudaMemcpy((void*)&alpha,(void*) (this->D_mMxM() + jj*M+jj), sizeof(double), cudaMemcpyDeviceToHost );
    magmablas_dgeadd(M,M,-alpha,unitMatrix,M,Tshifted->D_mMxM(),M);
    Tshifted->QR_Magma(*Q);
    magmablas_dgemm(MagmaConjTrans,MagmaNoTrans,M,M,M,one,Q->D_mMxM(),M,T.D_mMxM(),M,zero,Tmp,M);
    magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,M,M,M,one,Tmp,M,Q->D_mMxM(),M,zero,T.D_mMxM(),M);

    // make T tridiagonal and clear noise
    T.makeTridiagonal(M,M);
    cudaMemset((void*)(T.D_mMxM()+(M-1-jj)*M+M-1-jj-1),0,sizeof(double));
    cudaMemset((void*)(T.D_mMxM()+(M-1-jj-1)*M+M-1-jj),0,sizeof(double));
    ////////

    magmablas_dgemm(MagmaNoTrans,MagmaNoTrans,M,M,M,one,Qnew->D_mMxM(),M,Q->D_mMxM(),M,zero,Tmp,M);
    cudaMemcpy((void*)Qnew->D_mMxM(),(void*)Tmp,M*M*sizeof(double),cudaMemcpyDeviceToDevice);
  }

  /*
  // test
  cudaMemcpy((void*)h_T,(void*)T.D_mMxM(),M*M*sizeof(double),cudaMemcpyDeviceToHost);
  FILE *ptr;
  ptr = fopen("T.dat","a");
  for(int i = 0 ; i < M ; i++){
    for(int j = 0 ; j < M ; j++)
      fprintf(ptr,"%+e\t",h_T[i*M+j]);
    fprintf(ptr,"\n");
  }
  fprintf(ptr,"\n");
  fclose(ptr);
    //////////
    */

  Qnew->matrixNxMmatrixMxLReal(V,NL,M,M,true);

  //Qnew->matrixNxMmatrixMxLRealTexture(V,NL,M,M,true);

  double alpha;
  cudaMemcpy((void*)&alpha,(void*)(Qnew->D_mMxM() + (K-1)*M+M-1),sizeof(double),cudaMemcpyDeviceToHost);
  magma_zdscal(NL,alpha,(magmaDoubleComplex*)r.D_vN(),1);
  cudaMemcpy((void*)&alpha,(void*)(T.D_mMxM() + K*M+K-1),sizeof(double),cudaMemcpyDeviceToHost);
  magma_zaxpy (NL,MAGMA_Z_MAKE(alpha,0),(magmaDoubleComplex*)(V.D_mNxM() + K*NL), 1,(magmaDoubleComplex*)r.D_vN(),1 );

  free(h_T);
  delete Q;
  delete Tshifted;
  delete Qnew;
  cudaFree(unitMatrix);
  cudaFree(Tmp);
  //  free(h_T);
  checkCudaError();
}


void quda::IRLM_ES(void **gauge,QudaInvertParam *inv_param, QudaGaugeParam *gauge_param,qudaQKXTMinfo in_latInfo,lanczosInfo in_lInfo){

  size_t freeMem;
  size_t totalMem;

  cudaMemGetInfo(&freeMem,&totalMem);
  printfQuda("Check that we start with small initial memory occupied (Free = %f Mb),(Percentage = %f %%)\n",freeMem/(1024.*1024.),100*(freeMem/(double)totalMem));

  initLanczos(in_lInfo,in_latInfo);
  printfQuda("Start iteration phase\n");

  Dirac *dirac = initQudaMatrix(inv_param,gauge_param,gauge);
  cudaColorSpinorField *vec1 = initQudaVector(inv_param,gauge_param);
  cudaColorSpinorField *vec2 = initQudaVector(inv_param,gauge_param);
  int K = lInfo.lanczosWantedVectors;
  int M = lInfo.lanczosTotalVectors;
  int NL = lInfo.dimensionMatrix;

  cudaMemGetInfo(&freeMem,&totalMem);
  printfQuda("Memory status after operator allocation (Free = %f Mb),(Percentage = %f %%)\n",freeMem/(1024.*1024.),100*(freeMem/(double)totalMem));


  double *h_T = (double*)malloc(M*M*sizeof(double));
  Lanczos *V = new Lanczos(matrixNxM_L);
  Lanczos *T = new Lanczos(matrixMxM_L);
  Lanczos *W = new Lanczos(matrixMxM_L);
  Lanczos *Q = new Lanczos(matrixMxM_L);
  Lanczos *r = new Lanczos(vectorN_L);
  Lanczos *caller = NULL;

  int iter = 0;
  double res = 1.;

  cudaMemGetInfo(&freeMem,&totalMem);
  printfQuda("Memory status after lanczos vectors allocation (Free = %f Mb),(Percentage = %f %%)\n",freeMem/(1024.*1024.),100*(freeMem/(double)totalMem));


  caller->lanczos(0,K,*V,*T,*r,*dirac,*vec1,*vec2);



  while( (res > lInfo.tolerance) && (iter < lInfo.maxLanczosIter) ){
    caller->lanczos(K,M,*V,*T,*r,*dirac,*vec1,*vec2);



    /*
    if(iter == 1){
      for(int i = 0 ; i < M ; i++){
	res = normMagma(V->D_mNxM() + i*NL,NL);
	printfQuda("iter = %d, res = %16.15e\n",iter,res);
      }
    return;
    }
    */

    T->eigMagmaAndSort(*W,*Q);



    W->qrShiftsRotations(*V,*T,*r);



    /*
    cudaMemcpy((void*)h_T,(void*)T->D_mMxM(),M*M*sizeof(double),cudaMemcpyDeviceToHost);    
    for(int i = 0 ; i < M ; i++){
      for(int j = 0 ; j < M ; j++)
	printf("%+e  ",h_T[i*M+j]);
      printf("\n");
    }

    return;
    */
    res = normMagma(r->D_vN(),NL);
    printfQuda("iter = %d, res = %16.15e\n",iter,res);
    iter++;
  }

  double *eigenvalues = (double*)calloc(M*M,sizeof(double));
  cudaMemcpy((void*)eigenvalues,(void*)W->D_mMxM(),M*M*sizeof(double),cudaMemcpyDeviceToHost);
  for(int i = 0 ; i < K ; i++)
    printfQuda("eigvalue %d : %e \n",i,eigenvalues[(M-1-i)*M + (M-1-i)]);


  T->eigMagmaAndSort(*W,*Q);
  free(eigenvalues);

  delete dirac;
  delete vec1;
  delete vec2;
  delete V;
  delete T;
  delete W;
  delete Q;
  delete r;
  finalizeLanczos();

  cudaMemGetInfo(&freeMem,&totalMem);
  printfQuda("Memory status before exit (Free = %f Mb),(Percentage = %f %%)\n",freeMem/(1024.*1024.),100*(freeMem/(double)totalMem));

}

