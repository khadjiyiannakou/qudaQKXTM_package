#include <qudaQKXTM.h>
#include <errno.h>
#include <arlib.h>
#include <limits>

#define THREADS_PER_BLOCK 32
using namespace quda;

__constant__ double2 c_matrixQ[50*50];
__constant__ double c_machineEpsilon;

extern __constant__ bool c_dimBreak[4];
extern __constant__ int c_nColor;
extern __constant__ int c_nDim;
extern __constant__ int c_localL[4];
extern __constant__ int c_plusGhost[4];
extern __constant__ int c_minusGhost[4];
extern __constant__ int c_stride;
extern __constant__ int c_surface[4];
extern __constant__ int c_nSpin;
extern __constant__ double c_alphaAPE;
extern __constant__ double c_alphaGauss;
extern __constant__ int c_threads;
extern __constant__ int c_eps[6][3];
extern __constant__ int c_sgn_eps[6];
extern __constant__ int c_procPosition[4];
extern __constant__ int c_sourcePosition[4];
extern __constant__ int c_totalL[4];

extern int G_localVolume;
extern int G_nSpin;
extern int G_nColor;


#if (__COMPUTE_CAPABILITY__ >= 130)
__inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
  int4 v = tex1Dfetch(t,i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}
#else
__inline__ __device__ double2 fetch_double2(texture<int4, 1> t, int i)
{
  // do nothing 
  return make_double2(0.0, 0.0);
}
#endif

__device__ inline double2 operator*(const double a , const double2 b){
  double2 res;
  res.x = a*b.x;
  res.y = a*b.y;
  return res;
} 

__device__ inline double2 operator*(const int a , const double2 b){
  double2 res;
  res.x = a*b.x;
  res.y = a*b.y;
  return res;
} 

__device__ inline double2 operator*(const double2 a, const double2 b){
  double2 res;
  res.x = a.x*b.x - a.y*b.y;
  res.y = a.x*b.y + a.y*b.x;
  return res;
}

__device__ inline double2 operator+(const double2 a, const double2 b){
  double2 res;
  res.x = a.x + b.x;
  res.y = a.y + b.y;
  return res;
}

__device__ inline double2 operator-(const double2 a, const double2 b){
  double2 res;
  res.x = a.x - b.x;
  res.y = a.y - b.y;
  return res;
}

__device__ inline double2 conj(const double2 a){
  double2 res;
  res.x = a.x;
  res.y = -a.y;
  return res;
}

__device__ inline double norm(const double2 a){
  double res;
  res = sqrt(a.x*a.x + a.y*a.y);
  return res;
}

__device__ inline double norm2(const double2 a){
  double res;
  res = a.x*a.x + a.y*a.y;
  return res;
}

__device__ inline double2 cpow(const double2 x , const double a){
  double2 res;
  res.x = pow(norm(x),a) * cos( atan2(x.y,x.x) * a);
  res.y = pow(norm(x),a) * sin( atan2(x.y,x.x) * a);
  return res;
}

__device__ inline double2 operator/(const double2 x, const double2 y){
  double2 res;
  res.x = (x.x * y.x + x.y * y.y) / (y.x * y.x + y.y * y.y);
  res.y = (x.y * y.x - x.x * y.y) / (y.x * y.x + y.y * y.y);
  return res;
}

__global__ void uploadToCuda_kernel_(double2 *in, double2 *outEven, double2 *outOdd){

#include <uploadToCuda_core.h>

}


__global__ void downloadFromCuda_kernel_(double2 *out, double2 *inEven, double2 *inOdd){

#include <downloadFromCuda_core.h>

}

__global__ void matrixNxMmatrixMxL_kernel(double2 *mNxM, int NL, int M, int L, bool transpose){

#include <matrixNxMmatrixMxL_core.h>

}

__global__ void noiseCleaner_kernel(double2 *A){

#include <noiseCleaner_core.h>

}

void Arnoldi::uploadToCuda(cudaColorSpinorField &cudaVector, int offset){
  
 
    double *pointEven = (double*) cudaVector.Even().V(); // take the pointer to even and odd memory location
    double *pointOdd = (double*) cudaVector.Odd().V();

    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (G_localVolume/2 + blockDim.x -1)/blockDim.x , 1 , 1);     // half G_localVolume threads now
    uploadToCuda_kernel_<<<gridDim,blockDim>>>( (double2*) (d_mNxM + offset*G_localVolume*G_nSpin*G_nColor) , (double2*) pointEven, (double2*) pointOdd);
    cudaDeviceSynchronize();
  checkCudaError();
  
}

void Arnoldi::downloadFromCuda(cudaColorSpinorField &cudaVector, int offset){

  double *pointEven = (double*) cudaVector.Even().V(); // take the pointer to even and odd memory location
  double *pointOdd = (double*) cudaVector.Odd().V();

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (G_localVolume/2 + blockDim.x -1)/blockDim.x , 1 , 1);     // half G_localVolume threads now

  downloadFromCuda_kernel_<<<gridDim,blockDim>>>( (double2*) (d_mNxM + offset*G_localVolume*G_nSpin*G_nColor) , (double2*) pointEven, (double2*) pointOdd);

  cudaDeviceSynchronize();
  checkCudaError();


}

void Arnoldi::matrixNxMmatrixMxL(Arnoldi &V,int NL, int M,int L,bool transpose){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (G_localVolume*G_nSpin*G_nColor + blockDim.x -1)/blockDim.x , 1 , 1);
  cudaMemcpyToSymbol(c_matrixQ , d_mMxM , M*M*sizeof(double2),0,cudaMemcpyDeviceToDevice );
  checkCudaError();

  matrixNxMmatrixMxL_kernel<<<gridDim,blockDim,blockDim.x*M*sizeof(Complex)>>>( (double2*) V.D_mNxM(), NL , M, L ,transpose);


  cudaDeviceSynchronize();
  checkCudaError();

}

void quda::clearNoiseCuda(Complex *A, int L){
  dim3 blockDim( L , 1, 1);
  dim3 gridDim( 1 , 1 , 1);
  double machineEpsilon = std::numeric_limits<double>::epsilon();
  cudaMemcpyToSymbol(c_machineEpsilon , &machineEpsilon , sizeof(double),0,cudaMemcpyHostToDevice );
  noiseCleaner_kernel<<<gridDim,blockDim,blockDim.x*sizeof(double)>>>((double2*)A);
  cudaDeviceSynchronize();
  checkCudaError();
}
