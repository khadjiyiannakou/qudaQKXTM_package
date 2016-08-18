#include <qudaQKXTM.h>
#include <errno.h>
#include <mpi.h>
#include <cuPrintf.cu>


#define THREADS_PER_BLOCK 32
#define PI 3.141592653589793

using namespace quda;

// $$ Section 1: Texture References $$
/* texture block */
texture<int4, 1> gaugeTexPlaq;      // use this texture reference only for plaquette
texture<int4, 1> gaugeTexAPE;    // use this for APE smearing , this texture will be binded and unbinded   
texture<int4, 1> vectorTexGauss; // this texture needed for gaussian smearing
texture<int4, 1> propagatorTexAPE; // APE smearing need a propagator structure
texture<int4, 1> gaugeTexNorm2;
texture<int4, 1> vectorTexNorm2;     // to find the norm
texture<int4, 1> propagatorTexNorm2;
texture<int4, 1> propagatorTexOne;       // for contractions
texture<int4, 1> propagatorTexTwo;
texture<int4, 1> correlationTex;
texture<int4, 1> propagator3DTex1;
texture<int4, 1> propagator3DTex2;
texture<int4, 1> seqPropagatorTex;
texture<int4, 1> fwdPropagatorTex;
texture<int4, 1> gaugeDerivativeTex;
texture<int4, 1> phiVectorStochTex;
texture<int4, 1> propStochTex;
texture<int4, 1> insLineFourierTex;
texture<int4, 1> uprop3DStochTex;
texture<int4, 1> dprop3DStochTex;
texture<int4, 1> sprop3DStochTex;
texture<int4, 1> insLineMomTex;
texture<int4, 1> xiVector3DStochTex;
texture<int4, 1> gaugePath;           // bind standard texture for wilson path
texture<int4, 1>gaugeTexHYP;
texture<int4, 1>propagatorTexHYP;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// $$ Section 2: Constant Refeneces $$
/* block for device constants */
__constant__ bool c_dimBreak[4];
__constant__ int c_nColor;
__constant__ int c_nDim;
__constant__ int c_localL[4];
__constant__ int c_plusGhost[4];
__constant__ int c_minusGhost[4];
__constant__ int c_stride;
__constant__ int c_surface[4];
__constant__ int c_nSpin;
__constant__ double c_alphaAPE;
__constant__ double c_alphaGauss;
__constant__ int c_threads;
__constant__ int c_eps[6][3];
__constant__ int c_sgn_eps[6];
__constant__ int c_procPosition[4];
__constant__ int c_sourcePosition[4];
__constant__ int c_totalL[4];

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// $$ Section 3: Device /*Inline*/ Functions $$

/* Block for device kernels */
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

// $$ Section 4: Device Kernels $$

#include <core_def.h>

__global__ void calculatePlaq_kernel(double *partial_plaq){

#include <plaquette_core.h>

}

__global__ void APE_kernel_1(double2 *prp , double2 *out){

#include <APE_core_1.h>

}

#ifdef _PROPAGATOR_APE_TEX
__global__ void APE_kernel_2(double2 *out){

#include <APE_core_2.h>  

}
#else
__global__ void APE_kernel_2(double2 *prop,double2 *out){

#include <APE_core_2.h>  

}
#endif

__global__ void Gauss_kernel(double2 *out){

  #include <Gauss_core.h>

}

__global__ void norm2Gauge_kernel(double *cache){

#include <norm2Gauge_core.h>

}

__global__ void norm2Vector_kernel(double *cache){

#include <norm2Vector_core.h>

}

__global__ void norm2Propagator_kernel(double *cache){

#include <norm2Propagator_core.h>

}

__global__ void uploadToCuda_kernel(double2 *in, double2 *outEven, double2 *outOdd){

#include <uploadToCuda_core.h>

}


__global__ void downloadFromCuda_kernel(double2 *out, double2 *inEven, double2 *inOdd){

#include <downloadFromCuda_core.h>

}

__global__ void scaleVector_kernel(double2 *inOut,double a){

#include <scaleVector_core.h>

}

__global__ void rotateToPhysicalBase_kernel(double2 *inOut, int sign){

#include <rotateToPhysicalBase_core.h>

}

__global__ void contract_Type1_kernel(double2 *out){

#include <contract_Type1_core.h>

}

__global__ void fourierCorr_kernel(double2 *block ,int it ,int nx , int ny , int nz){

#include <fourierCorr_core.h>

}

__global__ void fourierCorr_kernel2(double2 *block ,int it ,int nx , int ny , int nz){

#include <fourierCorr_core2.h>

}

__global__ void fourierCorr_kernel3(double2 *block ,int it , int nx , int ny , int nz){

#include <fourierCorr_core3.h>

}

__global__ void fourierCorr_kernel4(double2 *block , int nx , int ny , int nz){

#include <fourierCorr_core4.h>

}

__global__ void seqSourceFixSinkPart1_kernel( double2 *out, int timeslice ,int c_nu, int c_c2, whatProjector typeProj, whatParticle testParticle ){

#include <seqSourceFixSinkPart1_core.h>

}


__global__ void seqSourceFixSinkPart2_kernel( double2 *out, int timeslice ,int c_nu, int c_c2, whatProjector typeProj, whatParticle testParticle  ){

#include <seqSourceFixSinkPart2_core.h>

}

__global__ void conjugate_vector_kernel( double2 *inOut ){

#include <conjugate_vector_core.h>

}


__global__ void conjugate_propagator_kernel( double2 *inOut ){

#include <conjugate_propagator_core.h>

}

__global__ void apply_gamma5_vector_kernel( double2 *inOut ){

#include <apply_gamma5_vector_core.h>

}
__global__ void apply_gamma_transf_vector_kernel( double2 *inOut ){

#include <apply_gamma_transf_vector_core.h>

}

__global__ void apply_gamma5_propagator_kernel( double2 *inOut ){

#include <apply_gamma5_propagator_core.h>

}

__global__ void fixSinkContractions_local_kernel( double2 *out, int flag, whatParticle testParticle, int partFlag){ // partFlag must be 1 or 2

#include <fixSinkContractions_local_core.h>  

}

__global__ void fixSinkContractions_oneD_kernel( double2 *out ,int flag,int dir ,whatParticle testParticle,int partFlag){

#include <fixSinkContractions_oneD_core.h>

}

__global__ void fixSinkContractions_noether_kernel( double2 *out ,int dir ,whatParticle testParticle,int partFlag){

#include <fixSinkContractions_noether_core.h>

}

__global__ void fixSinkContractions_nonLocal_kernel(double2 *out, double2 *deviceWilsonLinks, int dl, whatParticle testParticle, int partFlag,int direction){

#include <fixSinkContractions_nonLocal_core.h>

}

__global__ void insLine_local_kernel( double2 *out , int iflag , int partFlag ){

#include <insLine_local_core.h>

}

__global__ void insLine_oneD_kernel(double2 *out, int iflag , int dir){

  #include <insLine_oneD_core.h>

}


__global__ void insLine_oneD_kernel_new(double2 *out, int dir){

  #include <insLine_oneD_core_new.h>

}

__global__ void insLine_noether_kernel(double2 *out, int dir){

#include <insLine_noether_core.h>

}


__global__ void contract3pf_Type1_1_kernel( double2 *out, int iflag, int index1 , int index2){

  #include <contract3pf_Type1_1_core.h>

}

__global__ void contract3pf_Type1_2_kernel( double2 *out, int iflag, int index1){

#include <contract3pf_Type1_2_core.h>

}

__global__ void partial_lvl1_Contract3pf_Type1_1_kernel(double2 *out, int index1, int index2){

#include <partial_lvl1_Contract3pf_Type1_1_core.h>

}

__global__ void partial_lvl3_Contract3pf_Type1_1_kernel(double2 *out,int gamma,int gamma1, int index1, int index2){

#include <partial_lvl3_Contract3pf_Type1_1_core.h>

}

__global__ void partial_lvl3_Contract3pf_Type1_2_kernel(double2 *out,int gamma,int gamma1, int index1){

#include <partial_lvl3_Contract3pf_Type1_2_core.h>

}

__global__ void createWilsonPath_kernel(double2 *deviceWilsonPath,int direction){

#include <createWilsonPath_core.h>
}

__global__ void createWilsonPath_kernel_all(double2 *deviceWilsonPath){

#include <createWilsonPath_allDirections_core.h>
}

__global__ void insLine_Nonlocal_kernel(double2 *out, double2 *deviceWilsonLinks, int dl, int dir){

#include <insLine_Nonlocal_core.h>

}

__global__ void HYP3D_kernel_1(double2 *prp1){

#include <HYP3D_core_1.h>

}

__global__ void HYP3D_kernel_2(double2 *prp2){

#include <HYP3D_core_2.h>

}

__global__ void HYP3D_kernel_3(double2 *prp2, double omega2){

#include <HYP3D_core_3.h>

}

__global__ void HYP3D_kernel_4(double2 *prp1,double2 *out){

#include <HYP3D_core_4.h>

}

__global__ void HYP3D_kernel_5(double2 *out, double omega1){

#include <HYP3D_core_5.h>

}


__global__ void apply_momentum_kernel(double2 *vector, int nx , int ny , int nz){

#include <apply_momentum_core.h>

}

///////////////////////////////////////////////////////////////////////////////

// $$ Section 5: Static Global Variables $$

///////////////////////////////////////////////////
/* Block for static global variables */

static float deviceMemory = 0.;
static int nColor;
static int nSpin;
static int nDim;
static int strideFull;
static double alphaAPE;
static double alphaGauss;
static int localVolume;
static int totalVolume;
static int nsmearAPE;
static int nsmearGauss;
static bool dimBreak[QUDAQKXTM_DIM];
static int localL[QUDAQKXTM_DIM];
static int totalL[QUDAQKXTM_DIM];
static int nProc[QUDAQKXTM_DIM];
static int plusGhost[QUDAQKXTM_DIM];
static int minusGhost[QUDAQKXTM_DIM];
static int surface3D[QUDAQKXTM_DIM];
static bool init_qudaQKXTM_flag = false;
static int nsmearHYP;
static double omega1HYP;
static double omega2HYP;
// for mpi use global static variables
static MPI_Group fullGroup , spaceGroup , timeGroup;
static MPI_Comm spaceComm , timeComm;
static int localRank;
static int localSize;
static int timeRank;
static int timeSize;

//////////////////////////////////////////////////

// $$ Section 6: Initialize qudaQKXTM $$ 

// initialization function for qudaQKXTM lib
void quda::init_qudaQKXTM(qudaQKXTMinfo *info){

  if(init_qudaQKXTM_flag == false){
    nColor = 3;
    nSpin = 4;
    nDim = QUDAQKXTM_DIM;
    alphaAPE = info->alphaAPE;
    alphaGauss = info->alphaGauss;
    nsmearAPE = info->nsmearAPE;
    nsmearGauss = info->nsmearGauss;
    nsmearHYP = info->nsmearHYP;
    omega1HYP = info->omega1HYP;
    omega2HYP = info->omega2HYP;
    // from now on depends on lattice and break format we choose

    for(int i = 0 ; i < nDim ; i++)
      nProc[i] = comm_dim(i);
    
        for(int i = 0 ; i < nDim ; i++){   // take local and total lattice
      localL[i] = info->lL[i];
      totalL[i] = nProc[i] * localL[i];
    }
  
    localVolume = 1;
    totalVolume = 1;
    for(int i = 0 ; i < nDim ; i++){
      localVolume *= localL[i];
      totalVolume *= totalL[i];
    }

    strideFull = localVolume;

    for (int i=0; i<nDim; i++) {
      surface3D[i] = 1;
      for (int j=0; j<nDim; j++) {
	if (i==j) continue;
	surface3D[i] *= localL[j];
      }
    }

  for(int i = 0 ; i < nDim ; i++)
    if( localL[i] == totalL[i] )
      surface3D[i] = 0;
    
    for(int i = 0 ; i < nDim ; i++){
      plusGhost[i] =0;
      minusGhost[i] = 0;
    }
    
#ifdef MULTI_GPU
    int lastIndex = localVolume;
    for(int i = 0 ; i < nDim ; i++)
      if( localL[i] < totalL[i] ){
	plusGhost[i] = lastIndex ;
	minusGhost[i] = lastIndex + surface3D[i];
	lastIndex += 2*surface3D[i];
      }
#endif
    

    for(int i = 0 ; i < nDim ; i++){
      if( localL[i] < totalL[i])
	dimBreak[i] = true;
      else
	dimBreak[i] = false;
    }

    const int eps[6][3]=
      {
	{0,1,2},
	{2,0,1},
	{1,2,0},
	{2,1,0},
	{0,2,1},
	{1,0,2}
      };
    
    const int sgn_eps[6]=
      {
	+1,+1,+1,-1,-1,-1
      };

    int procPosition[4];
    
    for(int i= 0 ; i < 4 ; i++)
      procPosition[i] = comm_coords(i);

    int sourcePosition[4];
    // put it zero but change it later
    for(int i = 0 ; i < 4 ; i++)
      sourcePosition[i] = info->sourcePosition[i];

    // initialization consist also from define device constants

    cudaMemcpyToSymbol(c_nColor, &nColor, sizeof(int) );
    cudaMemcpyToSymbol(c_nSpin, &nSpin, sizeof(int) );
    cudaMemcpyToSymbol(c_nDim, &nDim, sizeof(int) );
    cudaMemcpyToSymbol(c_stride, &strideFull, sizeof(int) );
    cudaMemcpyToSymbol(c_alphaAPE, &alphaAPE , sizeof(double) );
    cudaMemcpyToSymbol(c_alphaGauss, &alphaGauss , sizeof(double) );
    cudaMemcpyToSymbol(c_threads , &localVolume , sizeof(double) ); // may change

    cudaMemcpyToSymbol(c_dimBreak , dimBreak , QUDAQKXTM_DIM*sizeof(bool) );
    cudaMemcpyToSymbol(c_localL , localL , QUDAQKXTM_DIM*sizeof(int) );
    cudaMemcpyToSymbol(c_totalL , totalL , QUDAQKXTM_DIM*sizeof(int) );
    cudaMemcpyToSymbol(c_plusGhost , plusGhost , QUDAQKXTM_DIM*sizeof(int) );
    cudaMemcpyToSymbol(c_minusGhost , minusGhost , QUDAQKXTM_DIM*sizeof(int) );
    cudaMemcpyToSymbol(c_surface , surface3D , QUDAQKXTM_DIM*sizeof(int) );
    
    cudaMemcpyToSymbol(c_eps, &(eps[0][0]) , 6*3*sizeof(int) );
    cudaMemcpyToSymbol(c_sgn_eps, sgn_eps , 6*sizeof(int) );

    cudaMemcpyToSymbol(c_procPosition, procPosition, QUDAQKXTM_DIM*sizeof(int));
    cudaMemcpyToSymbol(c_sourcePosition, sourcePosition, QUDAQKXTM_DIM*sizeof(int));
    
    checkCudaError();

    // create groups of process to use mpi reduce only on spatial points
    MPI_Comm_group(MPI_COMM_WORLD, &fullGroup);

    int space3D_proc;
    space3D_proc = nProc[0] * nProc[1] * nProc[2];
    int *ranks = (int*) malloc(space3D_proc*sizeof(int));

    for(int i= 0 ; i < space3D_proc ; i++)
      ranks[i] = comm_coords(3) * space3D_proc + i;

    MPI_Group_incl(fullGroup,space3D_proc,ranks,&spaceGroup);
    MPI_Group_rank(spaceGroup,&localRank);
    MPI_Group_size(spaceGroup,&localSize);
    MPI_Comm_create(MPI_COMM_WORLD, spaceGroup , &spaceComm);

    // create group of process to use mpi gather
    int *ranksTime = (int*) malloc(nProc[3]*sizeof(int));

    for(int i=0 ; i < nProc[3] ; i++)
      ranksTime[i] = i*space3D_proc;
    
    MPI_Group_incl(fullGroup,nProc[3], ranksTime, &timeGroup);
    MPI_Group_rank(timeGroup, &timeRank);
    MPI_Group_size(timeGroup, &timeSize);
    MPI_Comm_create(MPI_COMM_WORLD, timeGroup, &timeComm);

    //////////////////////////////////////////////////////////////////////////////
    free(ranks);
    free(ranksTime);

    init_qudaQKXTM_flag = true;
    printfQuda("qudaQKXTM has been initialized\n");
  }
  else
    return;

}

int quda::comm_localRank(){
  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");
  return localRank;
}


void quda::printf_qudaQKXTM(){

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");
  printfQuda("Number of colors is %d\n",nColor);
  printfQuda("Number of spins is %d\n",nSpin);
  printfQuda("Number of dimensions is %d\n",nDim);
  printfQuda("Number of process in each direction is (x,y,z,t) %d x %d x %d x %d\n",nProc[0],nProc[1],nProc[2],nProc[3]);
  printfQuda("Total lattice is (x,y,z,t) %d x %d x %d x %d\n",totalL[0],totalL[1],totalL[2],totalL[3]);
  printfQuda("Local lattice is (x,y,z,t) %d x %d x %d x %d\n",localL[0],localL[1],localL[2],localL[3]);
  printfQuda("Total volume is %d\n",totalVolume);
  printfQuda("Local volume is %d\n",localVolume);
  printfQuda("Surface is (x,y,z,t) ( %d , %d , %d , %d)\n",surface3D[0],surface3D[1],surface3D[2],surface3D[3]);
  printfQuda("The plus Ghost points in directions (x,y,z,t) ( %d , %d , %d , %d )\n",plusGhost[0],plusGhost[1],plusGhost[2],plusGhost[3]);
  printfQuda("The Minus Ghost points in directixons (x,y,z,t) ( %d , %d , %d , %d )\n",minusGhost[0],minusGhost[1],minusGhost[2],minusGhost[3]);
  printfQuda("For APE smearing we use nsmear = %d , alpha = %lf\n",nsmearAPE,alphaAPE);
  printfQuda("For Gauss smearing we use nsmear = %d , alpha = %lf\n",nsmearGauss,alphaGauss);
  printfQuda("For HYP smearing we use nsmear = %d , omega1 = %lf , omega2 = %lf\n",nsmearHYP,omega1HYP,omega2HYP);
}




///////////////////  METHODS //////////////////////////////

//////////////////////// class QKXTM_Field /////////////////////////////

// $$ Section 7: Class QKXTM_Field $$ 

QKXTM_Field::QKXTM_Field():
  h_elem(NULL) , d_elem(NULL) , h_ext_ghost(NULL) , d_ext_ghost(NULL)
{

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");

  field_length = 1;

  ghost_length = 0; 
  
  for(int i = 0 ; i < nDim ; i++)
    ghost_length += 2*surface3D[i];

  total_length = localVolume + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(double);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(double);

  //  create_all();

}

QKXTM_Field::~QKXTM_Field(){
  //  destroy_all();
}


void QKXTM_Field::create_host(){
    h_elem = (double*) malloc(bytes_total_length);
    if(h_elem == NULL) errorQuda("Error with allocation host memory");
}

void QKXTM_Field::create_host_ghost(){
#ifdef MULTI_GPU
    if( comm_size() > 1){
      h_ext_ghost = (double*) malloc(bytes_ghost_length);
      if(h_ext_ghost == NULL)errorQuda("Error with allocation host memory");
    }
#endif
}

void QKXTM_Field::create_device(){
  cudaMalloc((void**)&d_elem,bytes_total_length);
  checkCudaError();
  deviceMemory += bytes_total_length/(1024*1024);               // device memory in MB
  printfQuda("Device memory in used is %f MB A \n",deviceMemory);
}

void QKXTM_Field::create_device_ghost(){
#ifdef MULTI_GPU
    if( comm_size() > 1){
      cudaMalloc((void**)&d_ext_ghost,bytes_ghost_length);
      checkCudaError();
      deviceMemory += bytes_ghost_length/(1024*1024);
      printfQuda("Device memory in used is %f MB A \n",deviceMemory);
    }
#endif
}

void QKXTM_Field::destroy_host(){
    free(h_elem);
    h_elem = NULL;
}

void QKXTM_Field::destroy_device(){
  if(d_elem != NULL){
    cudaFree(d_elem);
    checkCudaError();
    d_elem = NULL;
    deviceMemory -= bytes_total_length/(1024*1024);
    printfQuda("Device memory in used is %f MB D \n",deviceMemory);
  }
}

void QKXTM_Field::destroy_host_ghost(){
#ifdef MULTI_GPU
  if( (comm_size() > 1) ){
    free(h_ext_ghost);
    h_ext_ghost = NULL;
  } 
#endif
}

void QKXTM_Field::destroy_device_ghost(){
#ifdef MULTI_GPU
  if( comm_size() > 1 ){
    if(d_ext_ghost != NULL){
      cudaFree(d_ext_ghost);
      d_ext_ghost = NULL;
      checkCudaError();
      deviceMemory -= bytes_ghost_length/(1024*1024);
      printfQuda("Device memory in used is %f MB D \n",deviceMemory);
    }
  }
#endif
}

void QKXTM_Field::create_all(){
  create_host();
  create_host_ghost();
  create_device();
  //  create_device_ghost();            // with cudaMemcpy2D dont need it
  zero();
}

void QKXTM_Field::destroy_all(){
  destroy_host();
  destroy_host_ghost();
  destroy_device();
  //  destroy_device_ghost();
}


void QKXTM_Field::printInfo(){

  printfQuda("GPU memory needed is %f MB \n",bytes_total_length/(1024.0 * 1024.0));

}

void QKXTM_Field::zero(){
  memset(h_elem,0,bytes_total_length);
  cudaMemset(d_elem,0,bytes_total_length);
  checkCudaError();
}

// -----------------------------------------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////// Class QKXTM_Gauge ////////////////////////////////////////

// $$ Section 8: Class QKXTM_Gauge $$

QKXTM_Gauge::QKXTM_Gauge():
  gauge_binded_plaq(false) , packGauge_flag(false) , loadGauge_flag(false) , gauge_binded_ape(false), h_elem_backup(NULL)
{

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");

  ghost_length = 0; 
  field_length = nDim * nColor * nColor;
  
  for(int i = 0 ; i < nDim ; i++)
    ghost_length += 2*surface3D[i];

  total_length = localVolume + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(double);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(double);

  create_all();

}

QKXTM_Gauge::QKXTM_Gauge(NgaugeHost ngaugeHost):
  gauge_binded_plaq(false) , packGauge_flag(false) , loadGauge_flag(false) , gauge_binded_ape(false), h_elem_backup(NULL)
{

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");

  ghost_length = 0; 
  field_length = nDim * nColor * nColor;
  
  for(int i = 0 ; i < nDim ; i++)
    ghost_length += 2*surface3D[i];

  total_length = localVolume + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(double);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(double);

  if(ngaugeHost == QKXTM_N1){
    create_all();}
  else{
    create_all();
    h_elem_backup =(double*) malloc(bytes_total_length);
    if(h_elem_backup == NULL)errorQuda("Error allocate host memory for backup");
  }

}

QKXTM_Gauge::~QKXTM_Gauge(){
  destroy_all();
  if(h_elem_backup != NULL)free(h_elem_backup);
  if(gauge_binded_plaq == true) unbindGaugePlaq();
  if(gauge_binded_ape == true) unbindGaugeAPE();
  gauge_binded_plaq = false;
  gauge_binded_ape = false;
}

void QKXTM_Gauge::bindGaugePlaq(){
  if( gauge_binded_plaq == false ){
    cudaBindTexture(0,gaugeTexPlaq,d_elem,bytes_total_length);
    checkCudaError();
  }
  gauge_binded_plaq = true;
}

void QKXTM_Gauge::unbindGaugePlaq(){
  if(gauge_binded_plaq == true){
    cudaUnbindTexture(gaugeTexPlaq);
    checkCudaError();
  }
  gauge_binded_plaq = false;
}

void QKXTM_Gauge::bindGaugeAPE(){
  if( gauge_binded_ape == false ){
    cudaBindTexture(0,gaugeTexAPE,d_elem,bytes_total_length);
    checkCudaError();
  }
  gauge_binded_ape = true;
}

void QKXTM_Gauge::unbindGaugeAPE(){
  if(gauge_binded_ape == true){
    cudaUnbindTexture(gaugeTexAPE);
    checkCudaError();
  }
  gauge_binded_ape = false;
}


void QKXTM_Gauge::rebindGaugeAPE(){
  cudaUnbindTexture(gaugeTexAPE);
  cudaBindTexture(0,gaugeTexAPE,d_elem,bytes_total_length);
  checkCudaError();
}

void QKXTM_Gauge::packGauge(void **gauge){

  //  if(packGauge_flag == false){
    double **p_gauge = (double**) gauge;
    
    for(int dir = 0 ; dir < nDim ; dir++)
      for(int iv = 0 ; iv < localVolume ; iv++)
	for(int c1 = 0 ; c1 < nColor ; c1++)
	  for(int c2 = 0 ; c2 < nColor ; c2++)
	    for(int part = 0 ; part < 2 ; part++){
	      h_elem[dir*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + iv*2 + part] = p_gauge[dir][iv*nColor*nColor*2 + c1*nColor*2 + c2*2 + part];
	    }

    printfQuda("Gauge qkxTM packed on gpu form\n");
    //  packGauge_flag = true;
    // }

}


void QKXTM_Gauge::loadGauge(){

  //if((packGauge_flag == true) && (loadGauge_flag == false)){
    cudaMemcpy(d_elem,h_elem,(bytes_total_length - bytes_ghost_length), cudaMemcpyHostToDevice );
    checkCudaError();
    //  loadGauge_flag = true;
      printfQuda("Gauge qkxTM loaded on gpu\n");
      // }

}

void QKXTM_Gauge::justDownloadGauge(){

  //if((packGauge_flag == true) && (loadGauge_flag == false)){
    cudaMemcpy(h_elem,d_elem,(bytes_total_length - bytes_ghost_length), cudaMemcpyDeviceToHost );
    checkCudaError();
    //  loadGauge_flag = true;
      printfQuda("GaugeApe just downloaded\n");
      // }

}



void QKXTM_Gauge::packGaugeToBackup(void **gauge){


    double **p_gauge = (double**) gauge;
    
    for(int dir = 0 ; dir < nDim ; dir++)
      for(int iv = 0 ; iv < localVolume ; iv++)
	for(int c1 = 0 ; c1 < nColor ; c1++)
	  for(int c2 = 0 ; c2 < nColor ; c2++)
	    for(int part = 0 ; part < 2 ; part++){
	      h_elem_backup[dir*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + iv*2 + part] = p_gauge[dir][iv*nColor*nColor*2 + c1*nColor*2 + c2*2 + part];
	    }

    printfQuda("Gauge qkxTM packed on gpu form on backupHost\n");



}


void QKXTM_Gauge::loadGaugeFromBackup(){


    cudaMemcpy(d_elem,h_elem_backup,(bytes_total_length - bytes_ghost_length), cudaMemcpyHostToDevice );
    checkCudaError();

      printfQuda("Gauge qkxTM loaded on gpu from backupHost\n");


}


void QKXTM_Gauge::ghostToHost(){   // gpu collect ghost and send it to host

  // direction x ////////////////////////////////////
#ifdef MULTI_GPU
  if( localL[0] < totalL[0]){
    int position;
    int height = localL[1] * localL[2] * localL[3]; // number of blocks that we need
    size_t width = 2*sizeof(double);
    size_t spitch = localL[0]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;
  // set plus points to minus area
    //    position = (localL[0]-1)*localL[1]*localL[2]*localL[3];
    position = localL[0]-1;
    for(int i = 0 ; i < nDim ; i++)
      for(int c1 = 0 ; c1 < nColor ; c1++)
	for(int c2 = 0 ; c2 < nColor ; c2++){
	  d_elem_offset = d_elem + i*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + minusGhost[0]*nDim*nColor*nColor*2 + i*nColor*nColor*surface3D[0]*2 + c1*nColor*surface3D[0]*2 + c2*surface3D[0]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < nDim ; i++)
      for(int c1 = 0 ; c1 < nColor ; c1++)
	for(int c2 = 0 ; c2 < nColor ; c2++){
	  d_elem_offset = d_elem + i*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + plusGhost[0]*nDim*nColor*nColor*2 + i*nColor*nColor*surface3D[0]*2 + c1*nColor*surface3D[0]*2 + c2*surface3D[0]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}


  }
  // direction y ///////////////////////////////////

  if( localL[1] < totalL[1]){

    int position;
    int height = localL[2] * localL[3]; // number of blocks that we need
    size_t width = localL[0]*2*sizeof(double);
    size_t spitch = localL[1]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;

  // set plus points to minus area
    //    position = localL[0]*(localL[1]-1)*localL[2]*localL[3];
    position = localL[0]*(localL[1]-1);
    for(int i = 0 ; i < nDim ; i++)
      for(int c1 = 0 ; c1 < nColor ; c1++)
	for(int c2 = 0 ; c2 < nColor ; c2++){
	  d_elem_offset = d_elem + i*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + minusGhost[1]*nDim*nColor*nColor*2 + i*nColor*nColor*surface3D[1]*2 + c1*nColor*surface3D[1]*2 + c2*surface3D[1]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < nDim ; i++)
      for(int c1 = 0 ; c1 < nColor ; c1++)
	for(int c2 = 0 ; c2 < nColor ; c2++){
	  d_elem_offset = d_elem + i*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + plusGhost[1]*nDim*nColor*nColor*2 + i*nColor*nColor*surface3D[1]*2 + c1*nColor*surface3D[1]*2 + c2*surface3D[1]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}


  }
  
  // direction z //////////////////////////////////
  if( localL[2] < totalL[2]){

    int position;
    int height = localL[3]; // number of blocks that we need
    size_t width = localL[1]*localL[0]*2*sizeof(double);
    size_t spitch = localL[2]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;

  // set plus points to minus area
    // position = localL[0]*localL[1]*(localL[2]-1)*localL[3];
    position = localL[0]*localL[1]*(localL[2]-1);
    for(int i = 0 ; i < nDim ; i++)
      for(int c1 = 0 ; c1 < nColor ; c1++)
	for(int c2 = 0 ; c2 < nColor ; c2++){
	  d_elem_offset = d_elem + i*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + minusGhost[2]*nDim*nColor*nColor*2 + i*nColor*nColor*surface3D[2]*2 + c1*nColor*surface3D[2]*2 + c2*surface3D[2]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < nDim ; i++)
      for(int c1 = 0 ; c1 < nColor ; c1++)
	for(int c2 = 0 ; c2 < nColor ; c2++){
	  d_elem_offset = d_elem + i*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + plusGhost[2]*nDim*nColor*nColor*2 + i*nColor*nColor*surface3D[2]*2 + c1*nColor*surface3D[2]*2 + c2*surface3D[2]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}

  }

  //  printfQuda("before copy device to host\n");  
  // direction t /////////////////////////////////////
  if( localL[3] < totalL[3]){
    int position;
    int height = nDim*nColor*nColor;
    size_t width = localL[2]*localL[1]*localL[0]*2*sizeof(double);
    size_t spitch = localL[3]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;
  // set plus points to minus area
    position = localL[0]*localL[1]*localL[2]*(localL[3]-1);
    d_elem_offset = d_elem + position*2;
    h_elem_offset = h_elem + minusGhost[3]*nDim*nColor*nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
  // set minus points to plus area
    position = 0;
    d_elem_offset = d_elem + position*2;
    h_elem_offset = h_elem + plusGhost[3]*nDim*nColor*nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
  }

  checkCudaError();
#endif
}

void QKXTM_Gauge::cpuExchangeGhost(){ // cpus exchange links

#ifdef MULTI_GPU
  if(comm_size() > 1){

    MPI_Request request_recv[2*nDim];
    MPI_Request request_send[2*nDim];
    int back_nbr[4] = {X_BACK_NBR,Y_BACK_NBR,Z_BACK_NBR,T_BACK_NBR};             
    int fwd_nbr[4] = {X_FWD_NBR,Y_FWD_NBR,Z_FWD_NBR,T_FWD_NBR};

    double *pointer_receive = NULL;
    double *pointer_send = NULL;

    // direction x
    if(localL[0] < totalL[0]){
      //      double *pointer_receive = NULL;
      // double *pointer_send = NULL;
      size_t nbytes = surface3D[0]*nColor*nColor*nDim*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[0]-localVolume)*nColor*nColor*nDim*2;
      pointer_send = h_elem + minusGhost[0]*nColor*nColor*nDim*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[0], 0, &(request_recv[0]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[0], 0, &(request_send[0]));
      comm_wait(&(request_recv[0])); // blocking until receive finish
      comm_wait(&(request_send[0]));
      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[0]-localVolume)*nColor*nColor*nDim*2;
      pointer_send = h_elem + plusGhost[0]*nColor*nColor*nDim*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[0], 1, &(request_recv[1]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[0], 1, &(request_send[1]));
      comm_wait(&(request_recv[1])); // blocking until receive finish
      comm_wait(&(request_send[1]));

      pointer_receive = NULL;
      pointer_send = NULL;
    }
    // direction y
    if(localL[1] < totalL[1]){
      //      double *pointer_receive = NULL;
      // double *pointer_send = NULL;
      size_t nbytes = surface3D[1]*nColor*nColor*nDim*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[1]-localVolume)*nColor*nColor*nDim*2;
      pointer_send = h_elem + minusGhost[1]*nColor*nColor*nDim*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[1], 2, &(request_recv[2]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[1], 2, &(request_send[2]));
      comm_wait(&(request_recv[2])); // blocking until receive finish
      comm_wait(&(request_send[2]));
      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[1]-localVolume)*nColor*nColor*nDim*2;
      pointer_send = h_elem + plusGhost[1]*nColor*nColor*nDim*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[1], 3, &(request_recv[3]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[1], 3, &(request_send[3]));
      comm_wait(&(request_recv[3])); // blocking until receive finish
      comm_wait(&(request_send[3]));
      pointer_receive = NULL;
      pointer_send = NULL;

    }

    // direction z
    if(localL[2] < totalL[2]){
      //      double *pointer_receive = NULL;
      // double *pointer_send = NULL;
      size_t nbytes = surface3D[2]*nColor*nColor*nDim*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[2]-localVolume)*nColor*nColor*nDim*2;
      pointer_send = h_elem + minusGhost[2]*nColor*nColor*nDim*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[2], 4, &(request_recv[4]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[2], 4, &(request_send[4]));
      comm_wait(&(request_recv[4])); // blocking until receive finish
      comm_wait(&(request_send[4]));

      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[2]-localVolume)*nColor*nColor*nDim*2;
      pointer_send = h_elem + plusGhost[2]*nColor*nColor*nDim*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[2], 5, &(request_recv[5]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[2], 5, &(request_send[5]));
      comm_wait(&(request_recv[5])); // blocking until receive finish
      comm_wait(&(request_send[5]));
      pointer_receive = NULL;
      pointer_send = NULL;

    }


    // direction t
    if(localL[3] < totalL[3]){
      //      double *pointer_receive = NULL;
      // double *pointer_send = NULL;

      //      printfQuda("Here\n");

      size_t nbytes = surface3D[3]*nColor*nColor*nDim*2*sizeof(double);



      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[3]-localVolume)*nColor*nColor*nDim*2;
      pointer_send = h_elem + minusGhost[3]*nColor*nColor*nDim*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[3], 6, &(request_recv[6]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[3], 6, &(request_send[6]));
      comm_wait(&(request_recv[6])); // blocking until receive finish
      comm_wait(&(request_send[6]));

      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[3]-localVolume)*nColor*nColor*nDim*2;
      pointer_send = h_elem + plusGhost[3]*nColor*nColor*nDim*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[3], 7, &(request_recv[7]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[3], 7, &(request_send[7]));
      comm_wait(&(request_recv[7])); // blocking until receive finish
      comm_wait(&(request_send[7]));

      pointer_receive = NULL;
      pointer_send = NULL;

    }


  }
#endif

}

void QKXTM_Gauge::ghostToDevice(){ // simple cudamemcpy to send ghost to device
#ifdef MULTI_GPU
  if(comm_size() > 1){
    double *host = h_ext_ghost;
    double *device = d_elem + localVolume*nColor*nColor*nDim*2;
    cudaMemcpy(device,host,bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
#endif
}

double QKXTM_Gauge::norm2Host(){
  double res = 0.;

  for(int i = 0 ; i < nDim*nColor*nColor*localVolume ; i++){
    res += h_elem[i*2 + 0]*h_elem[i*2 + 0] + h_elem[i*2 + 1]*h_elem[i*2 + 1];
  }

#ifdef MULTI_GPU
  double globalRes;
  int rc = MPI_Allreduce(&res , &globalRes , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  return globalRes ;
#else
  return res;
#endif  

}

double QKXTM_Gauge::norm2Device(){

  double *h_partial = NULL;
  double *d_partial = NULL;

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);


  h_partial = (double*) malloc(gridDim.x * sizeof(double) ); // only real part
  if(h_partial == NULL) errorQuda("Error allocate memory for host partial plaq");

  cudaMalloc((void**)&d_partial, gridDim.x * sizeof(double));

  cudaBindTexture(0,gaugeTexNorm2,d_elem, Bytes() - BytesGhost() );
  norm2Gauge_kernel<<<gridDim,blockDim>>>(d_partial);
  cudaDeviceSynchronize();
  cudaUnbindTexture(gaugeTexNorm2);

  cudaMemcpy(h_partial, d_partial , gridDim.x * sizeof(double) , cudaMemcpyDeviceToHost);
  
  double norm2 = 0.;

  // simple host reduction

  for(int i = 0 ; i < gridDim.x ; i++)
    norm2 += h_partial[i];

  free(h_partial);
  cudaFree(d_partial);

  h_partial = NULL;
  d_partial = NULL;

  checkCudaError();

#ifdef MULTI_GPU
  double globalNorm2;
  int rc = MPI_Allreduce(&norm2 , &globalNorm2 , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for norm2");
  return globalNorm2 ;
#else
  return norm2;
#endif  
  
}

double QKXTM_Gauge::calculatePlaq(){


  if(gauge_binded_plaq == false) bindGaugePlaq();



  //  if(packGauge_flag == false) packGauge(gauge);    // you must to do it in the executable because I will calculate plaquette for APE gauge
  // if(loadGauge_flag == false) loadGauge();

  ghostToHost(); // collect surface from device and send it to host
  //  comm_barrier();
  cpuExchangeGhost(); // cpus exchange surfaces with previous and forward proc all dir

  ghostToDevice();   // now the host send surface to device 



  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  double *h_partial_plaq = NULL;
  double *d_partial_plaq = NULL;

  h_partial_plaq = (double*) malloc(gridDim.x * sizeof(double) ); // only real part
  if(h_partial_plaq == NULL) errorQuda("Error allocate memory for host partial plaq");

  cudaMalloc((void**)&d_partial_plaq, gridDim.x * sizeof(double));

  // cudaPrintfInit();

  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  calculatePlaq_kernel<<<gridDim,blockDim>>>(d_partial_plaq);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);

  //  if(comm_rank() == 0)  cudaPrintfDisplay(stdout,true);
  //  cudaPrintfEnd();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  //  printfQuda("Elapsed time for plaquette kernel is %f ms\n",elapsedTime);

  // now copy partial plaq to host
  cudaMemcpy(h_partial_plaq, d_partial_plaq , gridDim.x * sizeof(double) , cudaMemcpyDeviceToHost);
  


  double plaquette = 0.;

#ifdef MULTI_GPU
  double globalPlaquette = 0.;
#endif
  // simple host reduction on plaq
  for(int i = 0 ; i < gridDim.x ; i++)
    plaquette += h_partial_plaq[i];

  free(h_partial_plaq);
  cudaFree(d_partial_plaq);

  h_partial_plaq = NULL;
  d_partial_plaq = NULL;

  checkCudaError();

  unbindGaugePlaq();

#ifdef MULTI_GPU
  int rc = MPI_Allreduce(&plaquette , &globalPlaquette , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  return globalPlaquette/(totalVolume*nColor*6) ;
#else
  return plaquette/(totalVolume*nColor*6);
#endif  
  
}

void QKXTM_Gauge::checkSum(){
  justDownloadGauge(); //gpuformat
  double *M = H_elem();
  double sum_real,sum_imag;
  sum_real = 0.;
  sum_imag = 0.;
  
  int mu =0;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < 32 ; t++)
	for(int z = 0 ; z < 16 ; z++)
	  for(int y = 0 ; y < 16 ; y++)
	    for(int x = 0 ; x < 16 ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
		  int position = x + 16*y + 16*16*z + 16*16*16*t + 16*16*16*32*c2 + 16*16*16*32*3*c1 + 16*16*16*32*3*3*mu; 
		  sum_real += M[position*2 + 0];
		  sum_imag += M[position*2 + 1];
	}
  printf("%+e %+e\n",sum_real,sum_imag);

  mu =1;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < 32 ; t++)
	for(int z = 0 ; z < 16 ; z++)
	  for(int y = 0 ; y < 16 ; y++)
	    for(int x = 0 ; x < 16 ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
		  int position = x + 16*y + 16*16*z + 16*16*16*t + 16*16*16*32*c2 + 16*16*16*32*3*c1 + 16*16*16*32*3*3*mu; 
		  sum_real += M[position*2 + 0];
		  sum_imag += M[position*2 + 1];
	}
  printf("%+e %+e\n",sum_real,sum_imag);

  mu =2;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < 32 ; t++)
	for(int z = 0 ; z < 16 ; z++)
	  for(int y = 0 ; y < 16 ; y++)
	    for(int x = 0 ; x < 16 ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
		  int position = x + 16*y + 16*16*z + 16*16*16*t + 16*16*16*32*c2 + 16*16*16*32*3*c1 + 16*16*16*32*3*3*mu; 
		  sum_real += M[position*2 + 0];
		  sum_imag += M[position*2 + 1];
	}
  printf("%+e %+e\n",sum_real,sum_imag);


  mu =3;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < 32 ; t++)
	for(int z = 0 ; z < 16 ; z++)
	  for(int y = 0 ; y < 16 ; y++)
	    for(int x = 0 ; x < 16 ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
		  int position = x + 16*y + 16*16*z + 16*16*16*t + 16*16*16*32*c2 + 16*16*16*32*3*c1 + 16*16*16*32*3*3*mu; 
		  sum_real += M[position*2 + 0];
		  sum_imag += M[position*2 + 1];
	}
  printf("%+e %+e\n",sum_real,sum_imag);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void quda::APE_smearing(QKXTM_Gauge &gaugeAPE , QKXTM_Gauge &gaugeTmp){// this is a function not a routine which perform smearing , need two QKXTM_Gauge objects

  //  if(nsmearAPE == 0) errorQuda("You cant call APE_smearing with nsmearAPE = 0"); // for nsmearAPE == 0 just copy to APE



  QKXTM_Propagator *prop = new QKXTM_Propagator(); // the constructor allocate memory on gpu for propagator I will use it for staple
  QKXTM_Propagator &prp = *prop;                   // take reference class

#ifdef _PROPAGATOR_APE_TEX  
  prp.bindPropagatorAPE();   // need to bind propagator to texture because it will be input in kernel2
#endif 

  // create pointer to classes , only pointer no memory allocation because I didnt call the construnctor 
  QKXTM_Gauge *in = NULL; 
  QKXTM_Gauge *out = NULL;
  QKXTM_Gauge *tmp = NULL;

  in = &(gaugeTmp);
  out = &(gaugeAPE);

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  printfQuda("Perform APE smearing\n");

  for(int iter = 0 ; iter < nsmearAPE ; iter++){

    
    //rebind texture to "in" gauge field
    in->rebindGaugeAPE();                           // now texture show to "in" gauge
    //communicate "in" gauge field
    
    in->ghostToHost();
    in->cpuExchangeGhost();                        // perform communication of the gauge
    in->ghostToDevice();
    //    cudaPrintfInit();    
    //kernel_1 first phase of APE smearing
    APE_kernel_1<<<gridDim,blockDim>>>((double2*) prp.D_elem() ,(double2*) out->D_elem() );
    cudaDeviceSynchronize();                      // we need to block until the kernel finish
    //communicate propagator
    prp.ghostToHost();
    prp.cpuExchangeGhost();                        // perform communication of the gauge in propagator structure
    prp.ghostToDevice();
    
    //kernel_2 second phase of APE smearing and SU3 projection
#ifdef _PROPAGATOR_APE_TEX
    APE_kernel_2<<<gridDim,blockDim>>>((double2*) out->D_elem() );
#else
    APE_kernel_2<<<gridDim,blockDim>>>((double2*) prp.D_elem(),(double2*) out->D_elem() );
#endif
    cudaDeviceSynchronize();
    // if(comm_rank() == 0)cudaPrintfDisplay(stdout,true);
    //cudaPrintfEnd();
    
    tmp=in;
    in=out; 
    out=tmp; // swap glasses
    
    checkCudaError();
    
  }

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);

  //  printfQuda("Elapsed time for APE smearing kernel is %f ms\n",elapsedTime);

  if((nsmearAPE%2) == 0){
    out->unbindGaugeAPE();
    cudaMemcpy(gaugeAPE.D_elem(),gaugeTmp.D_elem(), gaugeAPE.Bytes() - gaugeAPE.BytesGhost(), cudaMemcpyDeviceToDevice);
  }

  else{
    out->unbindGaugeAPE();
    cudaMemcpy(gaugeAPE.D_elem() + 3*nColor*nColor*localVolume*2 , gaugeTmp.D_elem() + 3*nColor*nColor*localVolume*2 , nColor*nColor*localVolume*2*sizeof(double) , cudaMemcpyDeviceToDevice); 
  }
  
    checkCudaError();

  delete prop;

}


void quda::APE_smearing(QKXTM_Gauge &gaugeAPE , QKXTM_Gauge &gaugeTmp, QKXTM_Propagator &prp){// this is a function not a routine which perform smearing , need two QKXTM_Gauge objects

  //  if(nsmearAPE == 0) errorQuda("You cant call APE_smearing with nsmearAPE = 0"); // for nsmearAPE == 0 just copy to APE

  //  QKXTM_Propagator *prop = new QKXTM_Propagator(); // the constructor allocate memory on gpu for propagator I will use it for staple
  //  QKXTM_Propagator &prp = *prop;                   // take reference class
  
  if(nsmearAPE == 0) return;

#ifdef _PROPAGATOR_APE_TEX
  prp.bindPropagatorAPE();   // need to bind propagator to texture because it will be input in kernel2
#endif
  // create pointer to classes , only pointer no memory allocation because I didnt call the construnctor 
  QKXTM_Gauge *in = NULL; 
  QKXTM_Gauge *out = NULL;
  QKXTM_Gauge *tmp = NULL;

  in = &(gaugeTmp);
  out = &(gaugeAPE);

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  printfQuda("Perform APE smearing\n");

  for(int iter = 0 ; iter < nsmearAPE ; iter++){

    
    //rebind texture to "in" gauge field
    in->rebindGaugeAPE();                           // now texture show to "in" gauge
    //communicate "in" gauge field
    
    in->ghostToHost();
    in->cpuExchangeGhost();                        // perform communication of the gauge
    in->ghostToDevice();
    cudaPrintfInit();    
    //kernel_1 first phase of APE smearing
    APE_kernel_1<<<gridDim,blockDim>>>((double2*) prp.D_elem() ,(double2*) out->D_elem() );
    cudaDeviceSynchronize();                      // we need to block until the kernel finish
    //communicate propagator
    prp.ghostToHost();
    prp.cpuExchangeGhost();                        // perform communication of the gauge in propagator structure
    prp.ghostToDevice();
    
    //kernel_2 second phase of APE smearing and SU3 projection
#ifdef _PROPAGATOR_APE_TEX
    APE_kernel_2<<<gridDim,blockDim>>>((double2*) out->D_elem() );
#else
    APE_kernel_2<<<gridDim,blockDim>>>((double2*) prp.D_elem() ,(double2*) out->D_elem() );
#endif
    cudaDeviceSynchronize();
    if(comm_rank() == 0)cudaPrintfDisplay(stdout,true);
    cudaPrintfEnd();
    
    tmp=in;
    in=out; 
    out=tmp; // swap glasses
    
    checkCudaError();
    
  }

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);

  prp.unbindPropagatorAPE();
  //  printfQuda("Elapsed time for APE smearing kernel is %f ms\n",elapsedTime);

  if((nsmearAPE%2) == 0){
    out->unbindGaugeAPE();
    cudaMemcpy(gaugeAPE.D_elem(),gaugeTmp.D_elem(), gaugeAPE.Bytes() - gaugeAPE.BytesGhost(), cudaMemcpyDeviceToDevice);
  }

  else{
    out->unbindGaugeAPE();
    cudaMemcpy(gaugeAPE.D_elem() + 3*nColor*nColor*localVolume*2 , gaugeTmp.D_elem() + 3*nColor*nColor*localVolume*2 , nColor*nColor*localVolume*2*sizeof(double) , cudaMemcpyDeviceToDevice); 
  }
  
    checkCudaError();

    //  delete prop;
  //  printfQuda("after delete prop\n");
}



void quda::HYP3D_smearing(QKXTM_Gauge &gaugeHYP , QKXTM_Gauge &gaugeTmp, QKXTM_Propagator &prp1, QKXTM_Propagator &prp2){// this is a function not a routine which perform smearing , need two QKXTM_Gauge objects

  //  cudaBindTexture(0,gaugeTexAPE,d_elem,bytes_total_length);
  // cudaBindTexture(0,propagatorTexAPE,d_elem,bytes_total_length);
  
  if(nsmearHYP == 0)return;
  // create pointer to classes , only pointer no memory allocation because I didnt call the construnctor 
  QKXTM_Gauge *in = NULL; 
  QKXTM_Gauge *out = NULL;
  QKXTM_Gauge *tmp = NULL;

  in = &(gaugeTmp);
  out = &(gaugeHYP);

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  printfQuda("Perform HYP 3D smearing\n");

  for(int iter = 0 ; iter < nsmearHYP ; iter++){

    //step 1
    cudaBindTexture(0,gaugeTexHYP,in->D_elem(),in->Bytes());    
    in->ghostToHost();
    in->cpuExchangeGhost();
    in->ghostToDevice();
    //step2
    HYP3D_kernel_1<<<gridDim,blockDim>>>((double2*) prp1.D_elem());
    cudaDeviceSynchronize();
    


    //step3
    cudaBindTexture(0,propagatorTexHYP,prp1.D_elem(),prp1.Bytes());
    prp1.ghostToHost();
    prp1.cpuExchangeGhost();
    prp1.ghostToDevice();
    //step4
    HYP3D_kernel_2<<<gridDim,blockDim>>>((double2*) prp2.D_elem() );
    cudaDeviceSynchronize();



    //step5
    cudaUnbindTexture(propagatorTexHYP);
    cudaBindTexture(0,propagatorTexHYP,prp2.D_elem(),prp2.Bytes());
    HYP3D_kernel_3<<<gridDim,blockDim>>>((double2*) prp2.D_elem(),omega2HYP);
    cudaDeviceSynchronize();


    prp2.ghostToHost();
    prp2.cpuExchangeGhost();
    prp2.ghostToDevice();
    // check the sum

    //step6
    HYP3D_kernel_4<<<gridDim,blockDim>>>((double2*) prp1.D_elem(),(double2*) out->D_elem());
    cudaDeviceSynchronize();


    //    out->checkSum();
    //step7
    cudaUnbindTexture(propagatorTexHYP);
    cudaBindTexture(0,propagatorTexHYP,prp1.D_elem(),prp1.Bytes());
    prp1.ghostToHost();
    prp1.cpuExchangeGhost();
    prp1.ghostToDevice();
    //step8
    HYP3D_kernel_5<<<gridDim,blockDim>>>((double2*)out->D_elem(),omega1HYP);
    cudaDeviceSynchronize();
    //step9
    cudaUnbindTexture(propagatorTexHYP);
    cudaUnbindTexture(gaugeTexHYP);
    
    tmp=in;
    in=out; 
    out=tmp; // swap classes
    
    checkCudaError();
    
  }

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);


  if((nsmearHYP%2) == 0){
    cudaMemcpy(gaugeHYP.D_elem(),gaugeTmp.D_elem(), gaugeHYP.Bytes() - gaugeHYP.BytesGhost(), cudaMemcpyDeviceToDevice);
  }

  else{
    cudaMemcpy(gaugeHYP.D_elem() + 3*nColor*nColor*localVolume*2 , gaugeTmp.D_elem() + 3*nColor*nColor*localVolume*2 , nColor*nColor*localVolume*2*sizeof(double) , cudaMemcpyDeviceToDevice); 
  }
  
    checkCudaError();

}


double* quda::createWilsonPath(QKXTM_Gauge &gauge,int direction ){
  double* deviceWilsonPath = NULL;

  cudaBindTexture(0,gaugePath,gauge.D_elem(),gauge.Bytes());
  checkCudaError();
  cudaMalloc((void**)&deviceWilsonPath,(localVolume*9*totalL[direction]/2)*2*sizeof(double) ); //we choose z direction and \Delta{Z} until the half spatial direction
  checkCudaError();

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  createWilsonPath_kernel<<<gridDim,blockDim>>>((double2*) deviceWilsonPath,direction);
  cudaDeviceSynchronize();  

  cudaUnbindTexture(gaugePath);
  checkCudaError();
  return deviceWilsonPath;
}

double* quda::createWilsonPath(QKXTM_Gauge &gauge){
  double* deviceWilsonPath = NULL;

  cudaBindTexture(0,gaugePath,gauge.D_elem(),gauge.Bytes());
  checkCudaError();
  if( (totalL[0] != totalL[1]) || (totalL[0] != totalL[2])){
    printfQuda("Lattice length must be equal in spatial directions\n");
  } 
  cudaMalloc((void**)&deviceWilsonPath,(localVolume*9*(totalL[0]/2)*3)*2*sizeof(double) ); //we choose z direction and \Delta{Z} until the half spatial direction
  checkCudaError();

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  createWilsonPath_kernel_all<<<gridDim,blockDim>>>((double2*) deviceWilsonPath);
  cudaDeviceSynchronize();  

  cudaUnbindTexture(gaugePath);
  checkCudaError();
  return deviceWilsonPath;
}




// -----------------------------------------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------------------------------------

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////// class QKXTM_Vector ////////////////////////////////////////////////////////////////

// $$ Section 9: Class QKXTM_Vector $$

QKXTM_Vector::QKXTM_Vector():
  vector_binded_gauss(false) , packVector_flag(false) , loadVector_flag(false) 
{

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");

  ghost_length = 0; 
  field_length = nSpin*nColor;
  
  for(int i = 0 ; i < nDim ; i++)
    ghost_length += 2*surface3D[i];

  total_length = localVolume + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(double);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(double);

  create_all();

}

QKXTM_Vector::~QKXTM_Vector(){
  destroy_all();
  if(vector_binded_gauss == true) unbindVectorGauss();
  vector_binded_gauss = false;
}

void QKXTM_Vector::packVector(void *vector){

  if(packVector_flag == false){
    double *p_vector = (double*) vector;
    
      for(int iv = 0 ; iv < localVolume ; iv++)
	for(int mu = 0 ; mu < nSpin ; mu++)                // always work with format colors inside spins
	  for(int c1 = 0 ; c1 < nColor ; c1++)
	    for(int part = 0 ; part < 2 ; part++){
	      h_elem[mu*nColor*localVolume*2 + c1*localVolume*2 + iv*2 + part] = p_vector[iv*nSpin*nColor*2 + mu*nColor*2 + c1*2 + part];
	    }

    printfQuda("Vector qkxTM packed on gpu form\n");
    packVector_flag = true;
  }

}

void QKXTM_Vector::loadVector(){

  if(packVector_flag == true && loadVector_flag == false){   
      cudaMemcpy(d_elem,h_elem,(bytes_total_length - bytes_ghost_length), cudaMemcpyHostToDevice );
      checkCudaError();
      loadVector_flag = true;
      printfQuda("Vector qkxTM loaded on gpu\n");
    }

}


void QKXTM_Vector::ghostToHost(){   // gpu collect ghost and send it to host

  // direction x ////////////////////////////////////
#ifdef MULTI_GPU
  if( localL[0] < totalL[0]){
    int position;
    int height = localL[1] * localL[2] * localL[3]; // number of blocks that we need
    size_t width = 2*sizeof(double);
    size_t spitch = localL[0]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;
  // set plus points to minus area
    //    position = (localL[0]-1)*localL[1]*localL[2]*localL[3];
    position = (localL[0]-1);
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + minusGhost[0]*nSpin*nColor*2 + mu*nColor*surface3D[0]*2 + c1*surface3D[0]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;

      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + plusGhost[0]*nSpin*nColor*2 + mu*nColor*surface3D[0]*2 + c1*surface3D[0]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}


  }
  // direction y ///////////////////////////////////

  if( localL[1] < totalL[1]){

    int position;
    int height = localL[2] * localL[3]; // number of blocks that we need
    size_t width = localL[0]*2*sizeof(double);
    size_t spitch = localL[1]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;

  // set plus points to minus area
    // position = localL[0]*(localL[1]-1)*localL[2]*localL[3];
    position = localL[0]*(localL[1]-1);
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + minusGhost[1]*nSpin*nColor*2 + mu*nColor*surface3D[1]*2 + c1*surface3D[1]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}

  // set minus points to plus area
    position = 0;

      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + plusGhost[1]*nSpin*nColor*2 + mu*nColor*surface3D[1]*2 + c1*surface3D[1]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}

  }
  
  // direction z //////////////////////////////////
  if( localL[2] < totalL[2]){

    int position;
    int height = localL[3]; // number of blocks that we need
    size_t width = localL[1]*localL[0]*2*sizeof(double);
    size_t spitch = localL[2]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;

  // set plus points to minus area
    //    position = localL[0]*localL[1]*(localL[2]-1)*localL[3];
    position = localL[0]*localL[1]*(localL[2]-1);
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + minusGhost[2]*nSpin*nColor*2 + mu*nColor*surface3D[2]*2 + c1*surface3D[2]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}

  // set minus points to plus area
    position = 0;
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + plusGhost[2]*nSpin*nColor*2 + mu*nColor*surface3D[2]*2 + c1*surface3D[2]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}

  }


  // direction t /////////////////////////////////////
  if( localL[3] < totalL[3]){
    int position;
    int height = nSpin*nColor;
    size_t width = localL[2]*localL[1]*localL[0]*2*sizeof(double);
    size_t spitch = localL[3]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;
  // set plus points to minus area
    position = localL[0]*localL[1]*localL[2]*(localL[3]-1);
    d_elem_offset = d_elem + position*2;
    h_elem_offset = h_elem + minusGhost[3]*nSpin*nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
  // set minus points to plus area
    position = 0;
    d_elem_offset = d_elem + position*2;
    h_elem_offset = h_elem + plusGhost[3]*nSpin*nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
  }


#endif
}

void QKXTM_Vector::cpuExchangeGhost(){ // cpus exchange links

#ifdef MULTI_GPU
  if(comm_size() > 1){

    MPI_Request request_recv[2*nDim];
    MPI_Request request_send[2*nDim];
    int back_nbr[4] = {X_BACK_NBR,Y_BACK_NBR,Z_BACK_NBR,T_BACK_NBR};             
    int fwd_nbr[4] = {X_FWD_NBR,Y_FWD_NBR,Z_FWD_NBR,T_FWD_NBR};

    // direction x
    if(localL[0] < totalL[0]){
      double *pointer_receive = NULL;
      double *pointer_send = NULL;
      size_t nbytes = surface3D[0]*nSpin*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[0]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + minusGhost[0]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[0], 0, &(request_recv[0]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[0], 0, &(request_send[0]));
      comm_wait(&(request_recv[0])); // blocking until receive finish
      comm_wait(&(request_send[0]));

      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[0]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + plusGhost[0]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[0], 1, &(request_recv[1]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[0], 1, &(request_send[1]));
      comm_wait(&(request_recv[1])); // blocking until receive finish
      comm_wait(&(request_send[1]));
    }
    // direction y
    if(localL[1] < totalL[1]){
      double *pointer_receive = NULL;
      double *pointer_send = NULL;
      size_t nbytes = surface3D[1]*nSpin*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[1]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + minusGhost[1]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[1], 2, &(request_recv[2]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[1], 2, &(request_send[2]));
      comm_wait(&(request_recv[2])); // blocking until receive finish
      comm_wait(&(request_send[2]));
      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[1]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + plusGhost[1]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[1], 3, &(request_recv[3]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[1], 3, &(request_send[3]));
      comm_wait(&(request_recv[3])); // blocking until receive finish
      comm_wait(&(request_send[3]));
    }

    // direction z
    if(localL[2] < totalL[2]){
      double *pointer_receive = NULL;
      double *pointer_send = NULL;
      size_t nbytes = surface3D[2]*nSpin*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[2]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + minusGhost[2]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[2], 4, &(request_recv[4]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[2], 4, &(request_send[4]));
      comm_wait(&(request_recv[4])); // blocking until receive finish
      comm_wait(&(request_send[4]));
      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[2]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + plusGhost[2]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[2], 5, &(request_recv[5]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[2], 5, &(request_send[5]));
      comm_wait(&(request_recv[5])); // blocking until receive finish
      comm_wait(&(request_send[5]));
    }


    // direction t
    if(localL[3] < totalL[3]){
      double *pointer_receive = NULL;
      double *pointer_send = NULL;
      size_t nbytes = surface3D[3]*nSpin*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[3]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + minusGhost[3]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[3], 6, &(request_recv[6]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[3], 6, &(request_send[6]));
      comm_wait(&(request_recv[6])); // blocking until receive finish
      comm_wait(&(request_send[6]));
      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[3]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + plusGhost[3]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[3], 7, &(request_recv[7]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[3], 7, &(request_send[7]));
      comm_wait(&(request_recv[7])); // blocking until receive finish
      comm_wait(&(request_send[7]));
    }


  }
#endif

}

void QKXTM_Vector::ghostToDevice(){ // simple cudamemcpy to send ghost to device
#ifdef MULTI_GPU
  if(comm_size() > 1){
    double *host = h_ext_ghost;
    double *device = d_elem + localVolume*nSpin*nColor*2;
    cudaMemcpy(device,host,bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
#endif
}


void QKXTM_Vector::bindVectorGauss(){
  if( vector_binded_gauss == false ){
    cudaBindTexture(0,vectorTexGauss,d_elem,bytes_total_length);
    checkCudaError();
  }
  vector_binded_gauss = true;
}

void QKXTM_Vector::unbindVectorGauss(){
  if(vector_binded_gauss == true){
    cudaUnbindTexture(vectorTexGauss);
    checkCudaError();
  }
  vector_binded_gauss = false;
}


void QKXTM_Vector::rebindVectorGauss(){
  cudaUnbindTexture(vectorTexGauss);
  cudaBindTexture(0,vectorTexGauss,d_elem,bytes_total_length);
  checkCudaError();
}

void QKXTM_Vector::download(){

  cudaMemcpy(h_elem,d_elem,Bytes() - BytesGhost() , cudaMemcpyDeviceToHost);
  checkCudaError();

  double *vector_tmp = (double*) malloc( Bytes() - BytesGhost() );
  if(vector_tmp == NULL)errorQuda("Error in allocate memory of tmp vector");

      for(int iv = 0 ; iv < localVolume ; iv++)
	for(int mu = 0 ; mu < nSpin ; mu++)                // always work with format colors inside spins
	  for(int c1 = 0 ; c1 < nColor ; c1++)
	    for(int part = 0 ; part < 2 ; part++){
	      vector_tmp[iv*nSpin*nColor*2 + mu*nColor*2 + c1*2 + part] = h_elem[mu*nColor*localVolume*2 + c1*localVolume*2 + iv*2 + part];
	    }

      memcpy(h_elem,vector_tmp,Bytes() - BytesGhost());

  free(vector_tmp);
  vector_tmp = NULL;
}


double QKXTM_Vector::norm2Host(){
  double res = 0.;

  for(int i = 0 ; i < nSpin*nColor*localVolume ; i++){
    res += h_elem[i*2 + 0]*h_elem[i*2 + 0] + h_elem[i*2 + 1]*h_elem[i*2 + 1];
  }

#ifdef MULTI_GPU
  double globalRes;
  int rc = MPI_Allreduce(&res , &globalRes , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  return globalRes ;
#else
  return res;
#endif  

}


double QKXTM_Vector::norm2Device(){

  double *h_partial = NULL;
  double *d_partial = NULL;

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);


  h_partial = (double*) malloc(gridDim.x * sizeof(double) ); // only real part
  if(h_partial == NULL) errorQuda("Error allocate memory for host partial plaq");

  cudaMalloc((void**)&d_partial, gridDim.x * sizeof(double));

  cudaBindTexture(0,vectorTexNorm2,d_elem, Bytes() - BytesGhost() );
  norm2Vector_kernel<<<gridDim,blockDim>>>(d_partial);
  cudaDeviceSynchronize();
  cudaUnbindTexture(vectorTexNorm2);


  cudaMemcpy(h_partial, d_partial , gridDim.x * sizeof(double) , cudaMemcpyDeviceToHost);
  
  double norm2 = 0.;

  // simple host reduction

  for(int i = 0 ; i < gridDim.x ; i++)
    norm2 += h_partial[i];

  free(h_partial);
  cudaFree(d_partial);

  h_partial = NULL;
  d_partial = NULL;

  checkCudaError();

#ifdef MULTI_GPU
  double globalNorm2;
  int rc = MPI_Allreduce(&norm2 , &globalNorm2 , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for norm2");
  return globalNorm2 ;
#else
  return norm2;
#endif  
  
}

void QKXTM_Vector::uploadToCuda(cudaColorSpinorField &cudaVector){
  
 
    double *pointEven = (double*) cudaVector.Even().V(); // take the pointer to even and odd memory location
    double *pointOdd = (double*) cudaVector.Odd().V();

    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (localVolume/2 + blockDim.x -1)/blockDim.x , 1 , 1);     // half localVolume threads now
  // cudaPrintfInit();
    uploadToCuda_kernel<<<gridDim,blockDim>>>( (double2*) d_elem , (double2*) pointEven, (double2*) pointOdd);
    //cudaPrintfDisplay(stdout,true);
  //cudaPrintfEnd();
    cudaDeviceSynchronize();
 

  checkCudaError();
  
}

void QKXTM_Vector::downloadFromCuda(cudaColorSpinorField &cudaVector){

  double *pointEven = (double*) cudaVector.Even().V(); // take the pointer to even and odd memory location
  double *pointOdd = (double*) cudaVector.Odd().V();

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/2 + blockDim.x -1)/blockDim.x , 1 , 1);     // half localVolume threads now

  downloadFromCuda_kernel<<<gridDim,blockDim>>>( (double2*) d_elem , (double2*) pointEven, (double2*) pointOdd);

  cudaDeviceSynchronize();
  checkCudaError();


}

void QKXTM_Vector::flagsToFalse(){

  packVector_flag = false;
  loadVector_flag = false;
}

void QKXTM_Vector::scaleVector(double a){

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);     
  scaleVector_kernel<<<gridDim,blockDim>>>((double2*) d_elem, a);
  cudaDeviceSynchronize();
  checkCudaError();

}

void QKXTM_Vector::copyPropagator3D(QKXTM_Propagator3D &prop, int timeslice, int nu , int c2){

  double *pointer_src = NULL;
  double *pointer_dst = NULL;

  int localVolume3D = localL[0]*localL[1]*localL[2];

  for(int mu = 0 ; mu < nSpin ; mu++)
    for(int c1 = 0 ; c1 < nColor ; c1++){
	  pointer_dst = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + timeslice*localVolume3D*2 ;
	  pointer_src = prop.D_elem() + mu*nSpin*nColor*nColor*localVolume3D*2 + nu*nColor*nColor*localVolume3D*2 + c1*nColor*localVolume3D*2 + c2*localVolume3D*2;
	  cudaMemcpy(pointer_dst, pointer_src, localVolume3D*2 * sizeof(double), cudaMemcpyDeviceToDevice);
	}

  pointer_src = NULL;
  pointer_dst = NULL;

  checkCudaError();

}

void QKXTM_Vector::copyPropagator(QKXTM_Propagator &prop, int nu , int c2){

  double *pointer_src = NULL;
  double *pointer_dst = NULL;

 

  for(int mu = 0 ; mu < nSpin ; mu++)
    for(int c1 = 0 ; c1 < nColor ; c1++){
	  pointer_dst = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 ;
	  pointer_src = prop.D_elem() + mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2;
	  cudaMemcpy(pointer_dst, pointer_src, localVolume*2 *sizeof(double), cudaMemcpyDeviceToDevice);
	}

  pointer_src = NULL;
  pointer_dst = NULL;

  checkCudaError();

}

void QKXTM_Vector::conjugate(){

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  conjugate_vector_kernel<<<gridDim,blockDim>>>( (double2*) D_elem() );

  cudaDeviceSynchronize();
  checkCudaError();
}

void QKXTM_Vector::applyGamma5(){

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  apply_gamma5_vector_kernel<<<gridDim,blockDim>>>( (double2*) D_elem() );

  cudaDeviceSynchronize();
  checkCudaError();

}

void QKXTM_Vector::applyGammaTransformation(){

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  apply_gamma_transf_vector_kernel<<<gridDim,blockDim>>>( (double2*) D_elem() );

  cudaDeviceSynchronize();
  checkCudaError();

}

void QKXTM_Vector::applyMomentum(int nx, int ny, int nz){

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);  

  apply_momentum_kernel<<<gridDim,blockDim>>>( (double2*) D_elem() , nx , ny , nz);

  cudaDeviceSynchronize();
  checkCudaError();

}



// new addition
#include <lime.h>

static void qcd_swap_8(double *Rd, int N)
{
   register char *i,*j,*k;
   char swap;
   char *max;
   char *R = (char*) Rd;

   max = R+(N<<3);
   for(i=R;i<max;i+=8)
   {
      j=i; k=j+7;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
   }
}

static int qcd_isBigEndian()
{
   union{
     char C[4];
     int  R   ;
        }word;
   word.R=1;
   if(word.C[3]==1) return 1;
   if(word.C[0]==1) return 0;

   return -1;
}



void QKXTM_Vector::write(char *filename){

  FILE *fid;
  int error_in_header=0;
  LimeWriter *limewriter;
  LimeRecordHeader *limeheader = NULL;
  int ME_flag=0, MB_flag=0, limeStatus;
  u_int64_t message_length;
  MPI_Offset offset;
  MPI_Datatype subblock;  //MPI-type, 5d subarray  
  MPI_File mpifid;
  MPI_Status status;
  int sizes[5], lsizes[5], starts[5];
  long int i;
  int chunksize,mu,c1;
  char *buffer;
  int x,y,z,t;
  char tmp_string[2048];

  if(comm_rank() == 0){ // master will write the lime header
    fid = fopen(filename,"w");
    if(fid == NULL){
      fprintf(stderr,"Error open file to write propagator in %s \n",__func__);
      comm_exit(-1);
    }
    else{
      limewriter = limeCreateWriter(fid);
      if(limewriter == (LimeWriter*)NULL) {
	fprintf(stderr, "Error in %s. LIME error in file for writing! in %s\n", __func__);
	error_in_header=1;
	comm_exit(-1);
      }
      else
	{
	  sprintf(tmp_string, "DiracFermion_Sink");
	  message_length=(long int) strlen(tmp_string);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "propagator-type", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
            {
              fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_exit(-1);
            }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
              error_in_header=1;
	      comm_exit(-1);
            }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
              error_in_header=1;
	      comm_exit(-1);
            }

	  sprintf(tmp_string, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n\t<field>diracFermion</field>\n\t<precision>64</precision>\n\t<flavours>1</flavours>\n\t<lx>%d</lx>\n\t<ly>%d</ly>\n\t<lz>%d</lz>\n\t<lt>%d</lt>\n\t<spin>4</spin>\n\t<colour>3</colour>\n</etmcFormat>", totalL[0], totalL[1], totalL[2], totalL[3]);

	  message_length=(long int) strlen(tmp_string); 
	  MB_flag=1; ME_flag=1;

	  limeheader = limeCreateHeader(MB_flag, ME_flag, "quda-propagator-format", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
            {
              fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_exit(-1);
            }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
              error_in_header=1;
	      comm_exit(-1);
            }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
              error_in_header=1;
	      comm_exit(-1);
            }
	  
	  message_length = totalVolume*4*3*2*sizeof(double);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", message_length);
	  limeStatus = limeWriteRecordHeader( limeheader, limewriter);
	  if(limeStatus < 0 )
            {
              fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
              error_in_header=1;
            }
	  limeDestroyHeader( limeheader );
	}
      message_length=1;
      limeWriteRecordData(tmp_string, &message_length, limewriter);
      limeDestroyWriter(limewriter);
      offset = ftell(fid)-1;
      fclose(fid);
    }
  }

  MPI_Bcast(&offset,sizeof(MPI_Offset),MPI_BYTE,0,MPI_COMM_WORLD);
  
  sizes[0]=totalL[3];
  sizes[1]=totalL[2];
  sizes[2]=totalL[1];
  sizes[3]=totalL[0];
  sizes[4]=4*3*2;
  lsizes[0]=localL[3];
  lsizes[1]=localL[2];
  lsizes[2]=localL[1];
  lsizes[3]=localL[0];
  lsizes[4]=sizes[4];
  starts[0]=comm_coords(3)*localL[3];
  starts[1]=comm_coords(2)*localL[2];
  starts[2]=comm_coords(1)*localL[1];
  starts[3]=comm_coords(0)*localL[0];
  starts[4]=0;  

  MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,MPI_DOUBLE,&subblock);
  MPI_Type_commit(&subblock);
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY, MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_FLOAT, subblock, "native", MPI_INFO_NULL);

  chunksize=4*3*2*sizeof(double);
  buffer = (char*) malloc(chunksize*localVolume);

  if(buffer==NULL)  
    {
      fprintf(stderr,"Error in %s! Out of memory\n", __func__);
      comm_exit(-1);
    }

  i=0;
                        
  for(t=0; t<localL[3];t++)
    for(z=0; z<localL[2];z++)
      for(y=0; y<localL[1];y++)
	for(x=0; x<localL[0];x++)
	  for(mu=0; mu<4; mu++)
	    for(c1=0; c1<3; c1++) // works only for QUDA_DIRAC_ORDER (color inside spin)
	      {
		((double *)buffer)[i] = h_elem[t*localL[2]*localL[1]*localL[0]*4*3*2 + z*localL[1]*localL[0]*4*3*2 + y*localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 0];
		((double *)buffer)[i+1] = h_elem[t*localL[2]*localL[1]*localL[0]*4*3*2 + z*localL[1]*localL[0]*4*3*2 + y*localL[0]*4*3*2 + x*4*3*2 + mu*3*2 + c1*2 + 1];
		i+=2;
	      }
  if(!qcd_isBigEndian())
    qcd_swap_8((double*) buffer,2*4*3*localVolume);

  MPI_File_write_all(mpifid, buffer, 4*3*2*localVolume, MPI_DOUBLE, &status);
  free(buffer);
  MPI_File_close(&mpifid);
  MPI_Type_free(&subblock);


}
//



void quda::Gaussian_smearing(QKXTM_Vector &vectorGauss , QKXTM_Vector &vectorTmp , QKXTM_Gauge &gaugeAPE){

  //  if(nsmearGauss == 0) errorQuda("You cant run Gaussian_smearing with nsmearGauss = 0"); // for nsmearGauss == 0 just copy

  // first communicate APE gauge 
  gaugeAPE.ghostToHost();
  gaugeAPE.cpuExchangeGhost();
  gaugeAPE.ghostToDevice();
  
  QKXTM_Vector *in = NULL;
  QKXTM_Vector *out = NULL;
  QKXTM_Vector *tmp = NULL;

  in = &(vectorTmp);
  out = &(vectorGauss);

  gaugeAPE.bindGaugeAPE();

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  // cudaPrintfInit();
  cudaEvent_t start,stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  printfQuda("Permform Gaussian smearing\n");
  for(int iter = 0 ; iter < nsmearGauss ; iter++){


    in->ghostToHost();
    in->cpuExchangeGhost();
    in->ghostToDevice();

    in->rebindVectorGauss();

    Gauss_kernel<<<gridDim,blockDim>>>((double2*) out->D_elem());
    //    cudaPrintfDisplay(stdout,true);

    cudaDeviceSynchronize();
    checkCudaError();

    tmp = in;
    in = out;
    out = tmp;
  }
  // cudaPrintfEnd();

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime,start,stop);

  //  printfQuda("Elapsed time for APE smearing kernel is %f ms\n",elapsedTime);


  if( (nsmearGauss%2) == 0){
    cudaMemcpy(vectorGauss.D_elem() , vectorTmp.D_elem() , vectorGauss.Bytes() - vectorGauss.BytesGhost() , cudaMemcpyDeviceToDevice);
  }

  gaugeAPE.unbindGaugeAPE();

}


void quda::seqSourceFixSinkPart1(QKXTM_Vector &vec, QKXTM_Propagator3D &prop1, QKXTM_Propagator3D &prop2, int timeslice,int nu,int c2, whatProjector typeProj , whatParticle testParticle){

    //    cudaPrintfInit();    
    // if(comm_rank() == 0)cudaPrintfDisplay(stdout,true);
    //cudaPrintfEnd();

  //cudaPrintfInit();    
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);  // now is localVolume3D

  cudaBindTexture(0, propagator3DTex1, prop1.D_elem(), prop1.Bytes());
  cudaBindTexture(0, propagator3DTex2, prop2.D_elem(), prop2.Bytes());

  seqSourceFixSinkPart1_kernel<<<gridDim,blockDim>>>( (double2*) vec.D_elem(), timeslice , nu, c2, typeProj , testParticle );
  cudaDeviceSynchronize();
  
  // if(comm_rank() == 0)cudaPrintfDisplay(stdout,true);

  cudaUnbindTexture(propagator3DTex1);
  cudaUnbindTexture(propagator3DTex2);
  //  cudaPrintfEnd();
  checkCudaError();
}


void quda::seqSourceFixSinkPart2(QKXTM_Vector &vec, QKXTM_Propagator3D &prop1, int timeslice,int nu,int c2, whatProjector typeProj, whatParticle testParticle){

    //    cudaPrintfInit();    
    // if(comm_rank() == 0)cudaPrintfDisplay(stdout,true);
    //cudaPrintfEnd();

  //cudaPrintfInit();    
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);  // now is localVolume3D

  cudaBindTexture(0, propagator3DTex1, prop1.D_elem(), prop1.Bytes());

  seqSourceFixSinkPart2_kernel<<<gridDim,blockDim>>>( (double2*) vec.D_elem(), timeslice , nu, c2, typeProj, testParticle );
  cudaDeviceSynchronize();
  
  // if(comm_rank() == 0)cudaPrintfDisplay(stdout,true);

  cudaUnbindTexture(propagator3DTex1);
 
  //  cudaPrintfEnd();
  checkCudaError();
}



// -----------------------------------------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------------------------------------

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////// class QKXTM_Propagator ////////////////////////////////////////////////////////////////

// $$ Section 10: Class QKXTM_Propagator $$

QKXTM_Propagator::QKXTM_Propagator():
  propagator_binded_ape(false) , packPropagator_flag(false) , loadPropagator_flag(false) 
{

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");

  ghost_length = 0; 
  field_length = nSpin*nSpin*nColor*nColor;
  
  for(int i = 0 ; i < nDim ; i++)
    ghost_length += 2*surface3D[i];

  total_length = localVolume + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(double);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(double);

  create_all();

}

QKXTM_Propagator::~QKXTM_Propagator(){
  destroy_all();
  if(propagator_binded_ape == true) unbindPropagatorAPE();
  propagator_binded_ape = false;
}


void QKXTM_Propagator::packPropagator(void *propagator){

  if(packPropagator_flag == false){
    double *p_propagator = (double*) propagator;
    
      for(int iv = 0 ; iv < localVolume ; iv++)
	for(int mu = 0 ; mu < nSpin ; mu++)                // always work with format colors inside spins
	  for(int nu = 0 ; nu < nSpin ; nu++)
	    for(int c1 = 0 ; c1 < nColor ; c1++)
	      for(int c2 = 0 ; c2 < nColor ; c2++)
		for(int part = 0 ; part < 2 ; part++){
		  h_elem[mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + iv*2 + part] = p_propagator[iv*nSpin*nSpin*nColor*nColor*2 + mu*nSpin*nColor*nColor*2 + nu*nColor*nColor*2 + c1*nColor*2 + c2*2 + part];
		}

    printfQuda("Propagator qkxTM packed on gpu form\n");
    packPropagator_flag = true;
  }

}

void QKXTM_Propagator::loadPropagator(){

  if(packPropagator_flag == true && loadPropagator_flag == false){   
      cudaMemcpy(d_elem,h_elem,(bytes_total_length - bytes_ghost_length), cudaMemcpyHostToDevice );
      checkCudaError();
      loadPropagator_flag = true;
      printfQuda("Propagator qkxTM loaded on gpu\n");
    }

}



void QKXTM_Propagator::ghostToHost(){   // gpu collect ghost and send it to host

  // direction x ////////////////////////////////////
#ifdef MULTI_GPU
  if( localL[0] < totalL[0]){
    int position;
    int height = localL[1] * localL[2] * localL[3]; // number of blocks that we need
    size_t width = 2*sizeof(double);
    size_t spitch = localL[0]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;
  // set plus points to minus area
    //    position = (localL[0]-1)*localL[1]*localL[2]*localL[3];
    position = (localL[0]-1);
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int nu = 0 ; nu < nSpin ; nu++)
	  for(int c1 = 0 ; c1 < nColor ; c1++)
	    for(int c2 = 0 ; c2 < nColor ; c2++){
	      d_elem_offset = d_elem + mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	      h_elem_offset = h_elem + minusGhost[0]*nSpin*nSpin*nColor*nColor*2 + mu*nSpin*nColor*nColor*surface3D[0]*2 + nu*nColor*nColor*surface3D[0]*2 + c1*nColor*surface3D[0]*2 + c2*surface3D[0]*2;
	      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	    }
  // set minus points to plus area
    position = 0;

      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int nu = 0 ; nu < nSpin ; nu++)
	  for(int c1 = 0 ; c1 < nColor ; c1++)
	    for(int c2 = 0 ; c2 < nColor ; c2++){
	      d_elem_offset = d_elem + mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	      h_elem_offset = h_elem + plusGhost[0]*nSpin*nSpin*nColor*nColor*2 + mu*nSpin*nColor*nColor*surface3D[0]*2 + nu*nColor*nColor*surface3D[0]*2 + c1*nColor*surface3D[0]*2 + c2*surface3D[0]*2;
	      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	    }


  }
  // direction y ///////////////////////////////////

  if( localL[1] < totalL[1]){

    int position;
    int height = localL[2] * localL[3]; // number of blocks that we need
    size_t width = localL[0]*2*sizeof(double);
    size_t spitch = localL[1]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;

  // set plus points to minus area
    //    position = localL[0]*(localL[1]-1)*localL[2]*localL[3];
    position = localL[0]*(localL[1]-1);
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int nu = 0 ; nu < nSpin ; nu++)
	  for(int c1 = 0 ; c1 < nColor ; c1++)
	    for(int c2 = 0 ; c2 < nColor ; c2++){
	      d_elem_offset = d_elem + mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	      h_elem_offset = h_elem + minusGhost[1]*nSpin*nSpin*nColor*nColor*2 + mu*nSpin*nColor*nColor*surface3D[1]*2 + nu*nColor*nColor*surface3D[1]*2 + c1*nColor*surface3D[1]*2 + c2*surface3D[1]*2;
	      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	    }

  // set minus points to plus area
    position = 0;
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int nu = 0 ; nu < nSpin ; nu++)
	  for(int c1 = 0 ; c1 < nColor ; c1++)
	    for(int c2 = 0 ; c2 < nColor ; c2++){
	      d_elem_offset = d_elem + mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	      h_elem_offset = h_elem + plusGhost[1]*nSpin*nSpin*nColor*nColor*2 + mu*nSpin*nColor*nColor*surface3D[1]*2 + nu*nColor*nColor*surface3D[1]*2 + c1*nColor*surface3D[1]*2 + c2*surface3D[1]*2;
	      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	    }

  }
  
  // direction z //////////////////////////////////
  if( localL[2] < totalL[2]){

    int position;
    int height = localL[3]; // number of blocks that we need
    size_t width = localL[1]*localL[0]*2*sizeof(double);
    size_t spitch = localL[2]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;

  // set plus points to minus area
    //    position = localL[0]*localL[1]*(localL[2]-1)*localL[3];
    position = localL[0]*localL[1]*(localL[2]-1);
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int nu = 0 ; nu < nSpin ; nu++)
	  for(int c1 = 0 ; c1 < nColor ; c1++)
	    for(int c2 = 0 ; c2 < nColor ; c2++){
	      d_elem_offset = d_elem + mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	      h_elem_offset = h_elem + minusGhost[2]*nSpin*nSpin*nColor*nColor*2 + mu*nSpin*nColor*nColor*surface3D[2]*2 + nu*nColor*nColor*surface3D[2]*2 + c1*nColor*surface3D[2]*2 + c2*surface3D[2]*2;
	      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	    }

  // set minus points to plus area
    position = 0;

      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int nu = 0 ; nu < nSpin ; nu++)
	  for(int c1 = 0 ; c1 < nColor ; c1++)
	    for(int c2 = 0 ; c2 < nColor ; c2++){
	      d_elem_offset = d_elem + mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + position*2;  
	      h_elem_offset = h_elem + plusGhost[2]*nSpin*nSpin*nColor*nColor*2 + mu*nSpin*nColor*nColor*surface3D[2]*2 + nu*nColor*nColor*surface3D[2]*2 + c1*nColor*surface3D[2]*2 + c2*surface3D[2]*2;
	      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	    }
  }


  // direction t /////////////////////////////////////
  if( localL[3] < totalL[3]){
    int position;
    int height = nSpin*nSpin*nColor*nColor;
    size_t width = localL[2]*localL[1]*localL[0]*2*sizeof(double);
    size_t spitch = localL[3]*width;
    size_t dpitch = width;
    double *h_elem_offset = NULL;
    double *d_elem_offset = NULL;
  // set plus points to minus area
    position = localL[0]*localL[1]*localL[2]*(localL[3]-1);
    d_elem_offset = d_elem + position*2;
    h_elem_offset = h_elem + minusGhost[3]*nSpin*nSpin*nColor*nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
  // set minus points to plus area
    position = 0;
    d_elem_offset = d_elem + position*2;
    h_elem_offset = h_elem + plusGhost[3]*nSpin*nSpin*nColor*nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
  }


#endif
}

void QKXTM_Propagator::cpuExchangeGhost(){ // cpus exchange links

#ifdef MULTI_GPU
  if(comm_size() > 1){

    MPI_Request request_recv[2*nDim];
    MPI_Request request_send[2*nDim];
    int back_nbr[4] = {X_BACK_NBR,Y_BACK_NBR,Z_BACK_NBR,T_BACK_NBR};             
    int fwd_nbr[4] = {X_FWD_NBR,Y_FWD_NBR,Z_FWD_NBR,T_FWD_NBR};

    double *pointer_receive = NULL;
    double *pointer_send = NULL;
    
    // direction x
    if(localL[0] < totalL[0]){
      size_t nbytes = surface3D[0]*nSpin*nSpin*nColor*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[0]-localVolume)*nSpin*nSpin*nColor*nColor*2;
      pointer_send = h_elem + minusGhost[0]*nSpin*nSpin*nColor*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[0], 0, &(request_recv[0]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[0], 0, &(request_send[0]));
      comm_wait(&(request_recv[0])); // blocking until receive finish
      comm_wait(&(request_send[0]));
      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[0]-localVolume)*nSpin*nSpin*nColor*nColor*2;
      pointer_send = h_elem + plusGhost[0]*nSpin*nSpin*nColor*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[0], 1, &(request_recv[1]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[0], 1, &(request_send[1]));
      comm_wait(&(request_recv[1])); // blocking until receive finish
      comm_wait(&(request_send[1]));

      pointer_receive = NULL;
      pointer_send = NULL;
    }
    // direction y
    if(localL[1] < totalL[1]){
      //      double *pointer_receive = NULL;
      // double *pointer_send = NULL;
      size_t nbytes = surface3D[1]*nSpin*nSpin*nColor*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[1]-localVolume)*nSpin*nSpin*nColor*nColor*2;
      pointer_send = h_elem + minusGhost[1]*nSpin*nSpin*nColor*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[1], 2, &(request_recv[2]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[1], 2, &(request_send[2]));
      comm_wait(&(request_recv[2])); // blocking until receive finish
      comm_wait(&(request_send[2]));
      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[1]-localVolume)*nSpin*nSpin*nColor*nColor*2;
      pointer_send = h_elem + plusGhost[1]*nSpin*nSpin*nColor*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[1], 3, &(request_recv[3]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[1], 3, &(request_send[3]));
      comm_wait(&(request_recv[3])); // blocking until receive finish
      comm_wait(&(request_send[3]));

      pointer_receive = NULL;
      pointer_send = NULL;

    }

    // direction z
    if(localL[2] < totalL[2]){
      //      double *pointer_receive = NULL;
      // double *pointer_send = NULL;
      size_t nbytes = surface3D[2]*nSpin*nSpin*nColor*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[2]-localVolume)*nSpin*nSpin*nColor*nColor*2;
      pointer_send = h_elem + minusGhost[2]*nSpin*nSpin*nColor*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[2], 4, &(request_recv[4]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[2], 4, &(request_send[4]));
      comm_wait(&(request_recv[4])); // blocking until receive finish
      comm_wait(&(request_send[4]));
      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[2]-localVolume)*nSpin*nSpin*nColor*nColor*2;
      pointer_send = h_elem + plusGhost[2]*nSpin*nSpin*nColor*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[2], 5, &(request_recv[5]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[2], 5, &(request_send[5]));
      comm_wait(&(request_recv[5])); // blocking until receive finish
      comm_wait(&(request_send[5]));
      pointer_receive = NULL;
      pointer_send = NULL;

    }


    // direction t
    if(localL[3] < totalL[3]){
      //      double *pointer_receive = NULL;
      // double *pointer_send = NULL;
      size_t nbytes = surface3D[3]*nSpin*nSpin*nColor*nColor*2*sizeof(double);
      
      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[3]-localVolume)*nSpin*nSpin*nColor*nColor*2;
      pointer_send = h_elem + minusGhost[3]*nSpin*nSpin*nColor*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[3], 6, &(request_recv[6]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[3], 6, &(request_send[6]));
      comm_wait(&(request_recv[6])); // blocking until receive finish
      comm_wait(&(request_send[6])); 
      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[3]-localVolume)*nSpin*nSpin*nColor*nColor*2;
      pointer_send = h_elem + plusGhost[3]*nSpin*nSpin*nColor*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[3], 7, &(request_recv[7]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[3], 7, &(request_send[7]));
      comm_wait(&(request_recv[7])); // blocking until receive finish
      comm_wait(&(request_send[7]));

      pointer_receive = NULL;
      pointer_send = NULL;
    }


  }
#endif

}

void QKXTM_Propagator::ghostToDevice(){ // simple cudamemcpy to send ghost to device
#ifdef MULTI_GPU
  if(comm_size() > 1){
    double *host = h_ext_ghost;
    double *device = d_elem + localVolume*nSpin*nSpin*nColor*nColor*2;
    cudaMemcpy(device,host,bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
#endif
}


void QKXTM_Propagator::bindPropagatorAPE(){
  if( propagator_binded_ape == false ){
    cudaBindTexture(0,propagatorTexAPE,d_elem,bytes_total_length);
    checkCudaError();
  }
  propagator_binded_ape = true;
}

void QKXTM_Propagator::unbindPropagatorAPE(){
  if(propagator_binded_ape == true){
    cudaUnbindTexture(propagatorTexAPE);
    checkCudaError();
  }
  propagator_binded_ape = false;
}


void QKXTM_Propagator::rebindPropagatorAPE(){
  cudaUnbindTexture(propagatorTexAPE);
  cudaBindTexture(0,propagatorTexAPE,d_elem,bytes_total_length);
  checkCudaError();
}


double QKXTM_Propagator::norm2Host(){
  double res = 0.;

  for(int i = 0 ; i < nSpin*nSpin*nColor*nColor*localVolume ; i++){
    res += h_elem[i*2 + 0]*h_elem[i*2 + 0] + h_elem[i*2 + 1]*h_elem[i*2 + 1];
  }

#ifdef MULTI_GPU
  double globalRes;
  int rc = MPI_Allreduce(&res , &globalRes , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  return globalRes ;
#else
  return res;
#endif  

}


double QKXTM_Propagator::norm2Device(){

  double *h_partial = NULL;
  double *d_partial = NULL;

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);


  h_partial = (double*) malloc(gridDim.x * sizeof(double) ); // only real part
  if(h_partial == NULL) errorQuda("Error allocate memory for host partial plaq");

  cudaMalloc((void**)&d_partial, gridDim.x * sizeof(double));

  cudaBindTexture(0,propagatorTexNorm2,d_elem, Bytes() - BytesGhost() );
  norm2Propagator_kernel<<<gridDim,blockDim>>>(d_partial);
  cudaDeviceSynchronize();
  cudaUnbindTexture(propagatorTexNorm2);


  cudaMemcpy(h_partial, d_partial , gridDim.x * sizeof(double) , cudaMemcpyDeviceToHost);
  
  double norm2 = 0.;

  // simple host reduction

  for(int i = 0 ; i < gridDim.x ; i++)
    norm2 += h_partial[i];

  free(h_partial);
  cudaFree(d_partial);

  h_partial = NULL;
  d_partial = NULL;

  checkCudaError();

#ifdef MULTI_GPU
  double globalNorm2;
  int rc = MPI_Allreduce(&norm2 , &globalNorm2 , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for norm2");
  return globalNorm2 ;
#else
  return norm2;
#endif  
  
}

void QKXTM_Propagator::absorbVector(QKXTM_Vector &vec, int nu, int c2){

  double *pointProp;
  double *pointVec;

  for(int mu = 0 ; mu < nSpin ; mu++)
    for(int c1 = 0 ; c1 < nColor ; c1++){
      pointProp = d_elem + mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2;
      pointVec = vec.D_elem() + mu*nColor*localVolume*2 + c1*localVolume*2;
      cudaMemcpy(pointProp,pointVec,localVolume*2*sizeof(double),cudaMemcpyDeviceToDevice); 
    }

  checkCudaError();

}

void QKXTM_Propagator::download(){

  cudaMemcpy(h_elem,d_elem,Bytes() - BytesGhost() , cudaMemcpyDeviceToHost);
  checkCudaError();

  double *propagator_tmp = (double*) malloc( Bytes() - BytesGhost() );
  if(propagator_tmp == NULL)errorQuda("Error in allocate memory of tmp propagator");

      for(int iv = 0 ; iv < localVolume ; iv++)
	for(int mu = 0 ; mu < nSpin ; mu++)                // always work with format colors inside spins
	  for(int nu = 0 ; nu < nSpin ; nu++)
	    for(int c1 = 0 ; c1 < nColor ; c1++)
	      for(int c2 = 0 ; c2 < nColor ; c2++)
		for(int part = 0 ; part < 2 ; part++){
		  propagator_tmp[iv*nSpin*nSpin*nColor*nColor*2 + mu*nSpin*nColor*nColor*2 + nu*nColor*nColor*2 + c1*nColor*2 + c2*2 + part] = h_elem[mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + iv*2 + part];
		}

      memcpy(h_elem,propagator_tmp,Bytes() - BytesGhost());

  free(propagator_tmp);
  propagator_tmp = NULL;
}

void QKXTM_Propagator::rotateToPhysicalBasePlus(){

  printfQuda("Perform rotation to physical base using + sign\n");

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);                                                         
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  rotateToPhysicalBase_kernel<<<gridDim,blockDim>>>( (double2*) d_elem , +1);  //kernel 
  cudaDeviceSynchronize();
  checkCudaError();

}

void QKXTM_Propagator::rotateToPhysicalBaseMinus(){

  printfQuda("Perform rotation to physical base using - sign\n");

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);                                                         
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  rotateToPhysicalBase_kernel<<<gridDim,blockDim>>>( (double2*) d_elem , -1);  //kernel 
  cudaDeviceSynchronize();
  checkCudaError();

}

void QKXTM_Propagator::conjugate(){

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  conjugate_propagator_kernel<<<gridDim,blockDim>>>( (double2*) D_elem() );

  cudaDeviceSynchronize();
  checkCudaError();
}

void QKXTM_Propagator::applyGamma5(){

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  apply_gamma5_propagator_kernel<<<gridDim,blockDim>>>( (double2*) D_elem() );

  cudaDeviceSynchronize();
  checkCudaError();

}

void QKXTM_Propagator::checkSum(){
  download();
  double *M = H_elem();
  double sum_real,sum_imag;
  sum_real = 0.;
  sum_imag = 0.;
  
  int mu =2;
  int nu =0;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < localL[3] ; t++)
	for(int z = 0 ; z < localL[2] ; z++)
	  for(int y = 0 ; y < localL[1] ; y++)
	    for(int x = 0 ; x < localL[0] ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
	      int prp_position = c2 + 3*c1 + 3*3*nu + 3*3*4*mu + 3*3*4*4*x + 3*3*4*4*localL[0]*y + 3*3*4*4*localL[0]*localL[1]*z + 3*3*4*4*localL[0]*localL[1]*localL[2]*t;
	      sum_real += M[prp_position*2 + 0];
	      sum_imag += M[prp_position*2 + 1];
	}
      printf("%d %+e %+e\n",comm_rank(),sum_real,sum_imag);


  mu =2; nu =1;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < localL[3] ; t++)
	for(int z = 0 ; z < localL[2] ; z++)
	  for(int y = 0 ; y < localL[1] ; y++)
	    for(int x = 0 ; x < localL[0] ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
	      int prp_position = c2 + 3*c1 + 3*3*nu + 3*3*4*mu + 3*3*4*4*x + 3*3*4*4*localL[0]*y + 3*3*4*4*localL[0]*localL[1]*z + 3*3*4*4*localL[0]*localL[1]*localL[2]*t;
	      sum_real += M[prp_position*2 + 0];
	      sum_imag += M[prp_position*2 + 1];
	}
      printf("%d %+e %+e\n",comm_rank(),sum_real,sum_imag);

  mu =1; nu =2;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < localL[3] ; t++)
	for(int z = 0 ; z < localL[2] ; z++)
	  for(int y = 0 ; y < localL[1] ; y++)
	    for(int x = 0 ; x < localL[0] ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
	      int prp_position = c2 + 3*c1 + 3*3*nu + 3*3*4*mu + 3*3*4*4*x + 3*3*4*4*localL[0]*y + 3*3*4*4*localL[0]*localL[1]*z + 3*3*4*4*localL[0]*localL[1]*localL[2]*t;
	      sum_real += M[prp_position*2 + 0];
	      sum_imag += M[prp_position*2 + 1];
	}
      printf("%d %+e %+e\n",comm_rank(),sum_real,sum_imag);


  mu =1; nu =0;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < localL[3] ; t++)
	for(int z = 0 ; z < localL[2] ; z++)
	  for(int y = 0 ; y < localL[1] ; y++)
	    for(int x = 0 ; x < localL[0] ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
	      int prp_position = c2 + 3*c1 + 3*3*nu + 3*3*4*mu + 3*3*4*4*x + 3*3*4*4*localL[0]*y + 3*3*4*4*localL[0]*localL[1]*z + 3*3*4*4*localL[0]*localL[1]*localL[2]*t;
	      sum_real += M[prp_position*2 + 0];
	      sum_imag += M[prp_position*2 + 1];
	}
      printf("%d %+e %+e\n",comm_rank(),sum_real,sum_imag);

  mu =0; nu =2;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < localL[3] ; t++)
	for(int z = 0 ; z < localL[2] ; z++)
	  for(int y = 0 ; y < localL[1] ; y++)
	    for(int x = 0 ; x < localL[0] ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
	      int prp_position = c2 + 3*c1 + 3*3*nu + 3*3*4*mu + 3*3*4*4*x + 3*3*4*4*localL[0]*y + 3*3*4*4*localL[0]*localL[1]*z + 3*3*4*4*localL[0]*localL[1]*localL[2]*t;
	      sum_real += M[prp_position*2 + 0];
	      sum_imag += M[prp_position*2 + 1];
	}
      printf("%d %+e %+e\n",comm_rank(),sum_real,sum_imag);

  mu =0; nu =1;
  sum_real = 0.;sum_imag = 0.;
      for(int t = 0 ; t < localL[3] ; t++)
	for(int z = 0 ; z < localL[2] ; z++)
	  for(int y = 0 ; y < localL[1] ; y++)
	    for(int x = 0 ; x < localL[0] ; x++)
	      for(int c1 =0 ; c1 < 3 ; c1++)
		for(int c2 =0 ; c2 < 3 ; c2++){
	      int prp_position = c2 + 3*c1 + 3*3*nu + 3*3*4*mu + 3*3*4*4*x + 3*3*4*4*localL[0]*y + 3*3*4*4*localL[0]*localL[1]*z + 3*3*4*4*localL[0]*localL[1]*localL[2]*t;
	      sum_real += M[prp_position*2 + 0];
	      sum_imag += M[prp_position*2 + 1];
	}
      printf("%d %+e %+e\n",comm_rank(),sum_real,sum_imag);

}

//////////////////////////////////////////// class QKXTM_Correlator ////////////

// Section 11: Class QKXTM_Correlator $$

//////////////////////////////////////////////////////////////////////////
QKXTM_Correlator::QKXTM_Correlator()
{

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");

  ghost_length = 0; 
  field_length = nSpin*nSpin;
  
  for(int i = 0 ; i < nDim ; i++)
    ghost_length += 2*surface3D[i];

  total_length = localVolume + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(double);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(double);

  create_all();

}

QKXTM_Correlator::~QKXTM_Correlator(){
  destroy_all();
}

void QKXTM_Correlator::download(){

  cudaMemcpy(h_elem,d_elem,Bytes() - BytesGhost() , cudaMemcpyDeviceToHost);
  checkCudaError();

  double *corr_tmp = (double*) malloc( Bytes() - BytesGhost() );
  if(corr_tmp == NULL)errorQuda("Error in allocate memory of tmp correlator");

  for(int iv = 0 ; iv < localVolume ; iv++)
    for(int mu = 0 ; mu < nSpin ; mu++)                // always work with format colors inside spins
      for(int nu = 0 ; nu < nSpin ; nu++)
	for(int part = 0 ; part < 2 ; part++){
	  corr_tmp[iv*nSpin*nSpin*2 + mu*nSpin*2 + nu*2 + part] = h_elem[mu*nSpin*localVolume*2 + nu*localVolume*2 + iv*2 + part];
	}

      memcpy(h_elem,corr_tmp,Bytes() - BytesGhost());

  free(corr_tmp);
  corr_tmp = NULL;
}

// spatial volume reduction ( first try with only zero momentum)

void QKXTM_Correlator::fourierCorr(double *corrMom, int Nmom , int momElem[][3]){
  // corrMom must be allocated with localL[3]*Nmom*4*4*2
  // slowest is time then momentum then gamma then gamma1 then r,i

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);  // now is localVolume3D


  cudaBindTexture(0, correlationTex, d_elem, Bytes() );
  double *h_partial_block = NULL;
  double *d_partial_block = NULL;
  h_partial_block = (double*) malloc(4*4*gridDim.x*2 * sizeof(double) );                  // for complex *2
  if(h_partial_block == NULL) errorQuda("error allocate memory for host partial block");
  cudaMalloc((void**)&d_partial_block, 4*4*gridDim.x*2 * sizeof(double) );

  double reduction[4*4*2];
  double globalReduction[4*4*2];

  for(int it = 0 ; it < localL[3] ; it++){
    
    for(int imom = 0 ; imom < Nmom ; imom++){


      fourierCorr_kernel<<<gridDim,blockDim>>>((double2*) d_partial_block ,it ,momElem[imom][0] , momElem[imom][1] , momElem[imom][2] ); // source position and proc position is in constant memory
      cudaDeviceSynchronize();
      cudaMemcpy(h_partial_block , d_partial_block , 4*4*gridDim.x*2 * sizeof(double) , cudaMemcpyDeviceToHost);
      
      memset(reduction , 0 , 4*4*2 * sizeof(double) );
      
      for(int gamma = 0 ; gamma < 4 ; gamma++)
	for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++)
	  for(int i =0 ; i < gridDim.x ; i++){
	    reduction[gamma*4*2 + gamma1*2 + 0] += h_partial_block[gamma*4*gridDim.x*2 + gamma1*gridDim.x*2 + i*2 + 0];
	    reduction[gamma*4*2 + gamma1*2 + 1] += h_partial_block[gamma*4*gridDim.x*2 + gamma1*gridDim.x*2 + i*2 + 1];
	  }

      MPI_Reduce(&(reduction[0]) , &(globalReduction[0]) , 4*4*2 , MPI_DOUBLE , MPI_SUM , 0 , spaceComm);           // only local root has the right value
      
      if(localRank == 0){
	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){
	    corrMom[it*Nmom*4*4*2 + imom*4*4*2 + gamma*4*2 + gamma1*2 + 0] = globalReduction[gamma*4*2 + gamma1*2 + 0];
	    corrMom[it*Nmom*4*4*2 + imom*4*4*2 + gamma*4*2 + gamma1*2 + 1] = globalReduction[gamma*4*2 + gamma1*2 + 1];
	  }
      }

    }  // for all momenta


  } // for all local timeslice


  cudaUnbindTexture(correlationTex);

  free(h_partial_block);
  cudaFree(d_partial_block);
  checkCudaError();

  h_partial_block = NULL;
  d_partial_block = NULL;

}





void QKXTM_Correlator::packCorrelator(void *corr){


    double *p_corr = (double*) corr;
    
      for(int iv = 0 ; iv < localVolume ; iv++)
	for(int mu = 0 ; mu < nSpin ; mu++)
	  for(int nu = 0 ; nu < nSpin ; nu++)
	    for(int part = 0 ; part < 2 ; part++){
	      h_elem[mu*nSpin*localVolume*2 + nu*localVolume*2 + iv*2 + part] = p_corr[iv*nSpin*nSpin*2 + mu*nSpin*2 + nu*2 + part];
	    }

    printfQuda("Correlator qkxTM packed on gpu form\n");

}


void QKXTM_Correlator::loadCorrelator(){

  cudaMemcpy(d_elem,h_elem,(bytes_total_length - bytes_ghost_length), cudaMemcpyHostToDevice );
  checkCudaError();
  printfQuda("Correlator qkxTM loaded on gpu\n");

}

////////////////////////////////////////////////////////  Contractions ///////////////////////////////////////

void quda::corrProton(QKXTM_Propagator &uprop, QKXTM_Propagator &dprop ,QKXTM_Correlator &corr){

  printfQuda("Perform contractions for Proton\n");

  cudaBindTexture(0,propagatorTexOne,uprop.D_elem(),uprop.Bytes());     // one will be up prop  
  cudaBindTexture(0,propagatorTexTwo,dprop.D_elem(),dprop.Bytes());     // two will be down prop  

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);                                                         
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  contract_Type1_kernel<<<gridDim,blockDim>>>((double2*) corr.D_elem() );
  cudaDeviceSynchronize();

  cudaUnbindTexture(propagatorTexOne);
  cudaUnbindTexture(propagatorTexTwo);
    
  checkCudaError();
}


void quda::corrNeutron(QKXTM_Propagator &uprop, QKXTM_Propagator &dprop ,QKXTM_Correlator &corr){

  printfQuda("Perform contractions for Neutron\n");

  cudaBindTexture(0,propagatorTexOne,dprop.D_elem(),dprop.Bytes());     
  cudaBindTexture(0,propagatorTexTwo,uprop.D_elem(),uprop.Bytes());     

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);                                                         
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

  contract_Type1_kernel<<<gridDim,blockDim>>>((double2*) corr.D_elem() );
  
  cudaDeviceSynchronize();

  cudaUnbindTexture(propagatorTexOne);
  cudaUnbindTexture(propagatorTexTwo);
    
  checkCudaError();
}


void quda::performContractions(QKXTM_Propagator &uprop, QKXTM_Propagator &dprop , int Nmom, int momElem[][3] , char *filenameProton , char *filenameNeutron){

  if(init_qudaQKXTM == false)errorQuda("You must initialize qudaQKXTM first");

  FILE *fileProton, *fileNeutron;
    
  if(comm_rank() == 0){
    fileProton = fopen(filenameProton,"w");
    fileNeutron = fopen(filenameNeutron,"w");
    if(fileProton == NULL || fileNeutron == NULL){
      fprintf(stderr,"Error open file paths for writting\n");
      comm_exit(-1);
    }
  }

  QKXTM_Correlator *corr = new QKXTM_Correlator();
  double *corr_fourier = (double*) calloc(localL[3]*Nmom*4*4*2,sizeof(double));
  double *corr_fourier_full = (double*) calloc(totalL[3]*Nmom*4*4*2,sizeof(double));

  corrProton(uprop,dprop,*corr);
  corr->fourierCorr(corr_fourier,Nmom,momElem);
  
  int error = 0;
  
  if(timeRank >= 0 && timeRank < nProc[3] ){
    error = MPI_Gather(corr_fourier,localL[3]*Nmom*4*4*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*4*4*2,MPI_DOUBLE,0,timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");    
  }

  if(comm_rank() == 0){
    for(int it = 0 ; it < totalL[3] ; it++)
      for(int imom = 0 ; imom < Nmom ; imom++)
	for(int gamma = 0 ; gamma < nSpin ; gamma++)
	  for(int gamma1 = 0 ; gamma1 < nSpin ; gamma1++){
	    fprintf(fileProton,"%d   %+d %+d %+d   %d %d \t %+e %+e\n",it,momElem[imom][0],momElem[imom][1],momElem[imom][2],gamma,gamma1,corr_fourier_full[it*Nmom*4*4*2 + imom*4*4*2 + gamma*4*2 + gamma1*2 + 0] , corr_fourier_full[it*Nmom*4*4*2 + imom*4*4*2 + gamma*4*2 + gamma1*2 + 1]);
	}
  }
  
  comm_barrier();

  corrNeutron(uprop,dprop,*corr);
  corr->fourierCorr(corr_fourier,Nmom,momElem);
  
  if(timeRank >= 0 && timeRank < nProc[3] ){
    error = MPI_Gather(corr_fourier,localL[3]*Nmom*4*4*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*4*4*2,MPI_DOUBLE,0,timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");    
  }

  if(comm_rank() == 0){
    for(int it = 0 ; it < totalL[3] ; it++)
      for(int imom = 0 ; imom < Nmom ; imom++)
	for(int gamma = 0 ; gamma < nSpin ; gamma++)
	  for(int gamma1 = 0 ; gamma1 < nSpin ; gamma1++){
	    fprintf(fileNeutron,"%d   %+d %+d %+d   %d %d \t %+e %+e\n",it,momElem[imom][0],momElem[imom][1],momElem[imom][2],gamma,gamma1,corr_fourier_full[it*Nmom*4*4*2 + imom*4*4*2 + gamma*4*2 + gamma1*2 + 0] , corr_fourier_full[it*Nmom*4*4*2 + imom*4*4*2 + gamma*4*2 + gamma1*2 + 1]);
	}
  }
  

  
   delete corr;
  free(corr_fourier);
  free(corr_fourier_full);

}



void quda::fixSinkFourier(double *corr,double *corrMom, int Nmom , int momElem[][3]){
  // corrMom must be allocated with localL[3]*Nmom*2
  // slowest is time then momentum then r,i

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);  // now is localVolume3D


  cudaBindTexture(0, correlationTex, corr, localVolume*2*sizeof(double) );
  double *h_partial_block = NULL;
  double *d_partial_block = NULL;
  h_partial_block = (double*) malloc(gridDim.x*2 * sizeof(double) );                  // for complex *2
  if(h_partial_block == NULL) errorQuda("error allocate memory for host partial block");
  cudaMalloc((void**)&d_partial_block, gridDim.x*2 * sizeof(double) );

  double reduction[2];
  double globalReduction[2];

  for(int it = 0 ; it < localL[3] ; it++){
    
    for(int imom = 0 ; imom < Nmom ; imom++){


      fourierCorr_kernel2<<<gridDim,blockDim>>>((double2*) d_partial_block ,it ,momElem[imom][0] , momElem[imom][1] , momElem[imom][2] ); // source position and proc position is in constant memory
      cudaDeviceSynchronize();
      cudaMemcpy(h_partial_block , d_partial_block , gridDim.x*2 * sizeof(double) , cudaMemcpyDeviceToHost);
      
      memset(reduction , 0 , 2 * sizeof(double) );
      
	  for(int i =0 ; i < gridDim.x ; i++){
	    reduction[0] += h_partial_block[i*2 + 0];
	    reduction[1] += h_partial_block[i*2 + 1];
	  }

      MPI_Reduce(&(reduction[0]) , &(globalReduction[0]) , 2 , MPI_DOUBLE , MPI_SUM , 0 , spaceComm);           // only local root has the right value
      
      if(localRank == 0){
	corrMom[it*Nmom*2 + imom*2 + 0] = globalReduction[0];
	corrMom[it*Nmom*2 + imom*2 + 1] = globalReduction[1];	  
      }

    }  // for all momenta


  } // for all local timeslice


  cudaUnbindTexture(correlationTex);

  free(h_partial_block);
  cudaFree(d_partial_block);
  checkCudaError();

  h_partial_block = NULL;
  d_partial_block = NULL;

}


void quda::fixSinkContractions(QKXTM_Propagator &seqProp, QKXTM_Propagator &prop , QKXTM_Gauge &gauge,whatProjector typeProj , char *filename , int Nmom , int momElem[][3] , whatParticle testParticle, int partFlag){
  

  if(typeProj == QKXTM_TYPE1)
    sprintf(filename,"%s_%s",filename,"type1");
  else
    sprintf(filename,"%s_%s",filename,"type2");

  if(testParticle == QKXTM_PROTON)
    sprintf(filename,"%s_%s",filename,"proton");
  else
    sprintf(filename,"%s_%s",filename,"neutron");

  char filename_local[257] , filename_noether[257] , filename_oneD[257];
  sprintf(filename_local,"%s_%s.dat",filename,"local");
  sprintf(filename_noether,"%s_%s.dat",filename,"noether");
  sprintf(filename_oneD,"%s_%s.dat",filename,"oneD");
  FILE *fileLocal , *fileNoether , *fileOneD;
  
  if(comm_rank() == 0){
    fileLocal = fopen(filename_local,"w");
    fileNoether = fopen(filename_noether,"w");
    fileOneD = fopen(filename_oneD, "w");

    if(fileLocal == NULL || fileOneD == NULL){
      fprintf(stderr,"Error open file for writting\n");
      comm_exit(-1);
    }

  }

  seqProp.applyGamma5();
  seqProp.conjugate();

  // execution domain
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

   // holds correlator in position space
  double *d_corr;
  cudaError error;
  error = cudaMalloc((void**)&d_corr, localVolume*2*sizeof(double));
  if(error != cudaSuccess)errorQuda("Error allocate device memory for correlator");

  double *corr_fourier = (double*) calloc(localL[3]*Nmom*2,sizeof(double)); 
  double *corr_fourier_full = (double*) calloc(totalL[3]*Nmom*2,sizeof(double));


  // to speed up contraction we use texture binding for seq-prop and prop
  cudaBindTexture(0, seqPropagatorTex, seqProp.D_elem(), seqProp.Bytes());
  cudaBindTexture(0, fwdPropagatorTex, prop.D_elem(), prop.Bytes());



  // +++++++++++++++  local operators   +++++++++++++++++++// (10 local operator )
  // for local operators we use       1 , g1 , g2 , g3 , g4 , g5 , g5g1 , g5g2 , g5g3 , g5g4
  // so we map operators to integers  0 , 1  , 2 ,  3  , 4  , 5  , 6    , 7    , 8    , 9
  
  for(int iflag = 0 ; iflag < 10 ; iflag++){

    fixSinkContractions_local_kernel<<<gridDim,blockDim>>>((double2*) d_corr , iflag, testParticle, partFlag);
    cudaDeviceSynchronize(); // to make sure that we have the data in corr

    fixSinkFourier(d_corr,corr_fourier,Nmom,momElem);

    int error = 0;
    
    if(timeRank >= 0 && timeRank < nProc[3] ){
      error = MPI_Gather(corr_fourier,localL[3]*Nmom*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*2,MPI_DOUBLE,0,timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
        
    if(comm_rank() == 0){
      for(int it = 0 ; it < totalL[3] ; it++)
	for(int imom = 0 ; imom < Nmom ; imom++){
	  fprintf(fileLocal,"%d %d   %+d %+d %+d \t %+e %+e\n",iflag,it,momElem[imom][0],momElem[imom][1],momElem[imom][2],corr_fourier_full[it*Nmom*2 + imom*2 + 0] , corr_fourier_full[it*Nmom*2 + imom*2 + 1]);
	    }
    }
    
  }

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

  // communication
  cudaBindTexture(0, gaugeDerivativeTex, gauge.D_elem(), gauge.Bytes());

  gauge.ghostToHost();
  gauge.cpuExchangeGhost(); // communicate gauge
  gauge.ghostToDevice();
  comm_barrier();          // just in case
  prop.ghostToHost();
  prop.cpuExchangeGhost(); // communicate forward propagator
  prop.ghostToDevice();
  comm_barrier();          // just in case
  seqProp.ghostToHost();
  seqProp.cpuExchangeGhost(); // communicate sequential propagator
  seqProp.ghostToDevice();
  comm_barrier();          // just in case

  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

  // +++++++++++++++++++ conserved current    +++++++++++++++++++++++++++++++++++++++++//
  // mapping gamma indices
  // g1 , g2 , g3 , g4
  // 0  , 1  , 2  , 3

  for(int idir = 0 ; idir < 4 ; idir++){

    fixSinkContractions_noether_kernel<<<gridDim,blockDim>>>((double2*) d_corr , idir, testParticle, partFlag);
    cudaDeviceSynchronize(); // to make sure that we have the data in corr

    fixSinkFourier(d_corr,corr_fourier,Nmom,momElem);

    int error = 0;
    
    if(timeRank >= 0 && timeRank < nProc[3] ){
      error = MPI_Gather(corr_fourier,localL[3]*Nmom*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*2,MPI_DOUBLE,0,timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
        
    if(comm_rank() == 0){
      for(int it = 0 ; it < totalL[3] ; it++)
	for(int imom = 0 ; imom < Nmom ; imom++){
	  fprintf(fileNoether,"%d %d   %+d %+d %+d \t %+e %+e\n",idir,it,momElem[imom][0],momElem[imom][1],momElem[imom][2],corr_fourier_full[it*Nmom*2 + imom*2 + 0] , corr_fourier_full[it*Nmom*2 + imom*2 + 1]);
	    }
    }
    
  }


  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

  // +++++++++++++++++++ derivative operators ++++++++++++++++++++++++++++++++//


  // for derivative operators we have for gamma matrices g1,g2,g3,g4 ,g5g1,g5g2,g5g3,g5g4 => 4+4 combinations 
  // for derivative index we have 4 index D^0 , D^1 , D^2 , D^3
  // for total we have 8*4=32 combinations
  
  // mapping gamma indices, (derivative will have a seperate index)
  // g1 , g2 , g3 , g4 , g5g1 , g5g2 , g5g3 , g5g4 
  // 0  , 1  , 2  , 3  , 4      , 5      , 6      , 7      

  for(int iflag = 0 ; iflag < 8 ; iflag++){ // iflag perform loop over gammas

    for(int dir = 0 ; dir < 4 ; dir++){

      fixSinkContractions_oneD_kernel<<<gridDim,blockDim>>>((double2*) d_corr , iflag, dir , testParticle, partFlag);
      cudaDeviceSynchronize(); // to make sure that we have the data in corr

      fixSinkFourier(d_corr,corr_fourier,Nmom,momElem);

      int error = 0;
    
      if(timeRank >= 0 && timeRank < nProc[3] ){
	error = MPI_Gather(corr_fourier,localL[3]*Nmom*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*2,MPI_DOUBLE,0,timeComm);
	if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
      }
        
      if(comm_rank() == 0){
	for(int it = 0 ; it < totalL[3] ; it++)
	  for(int imom = 0 ; imom < Nmom ; imom++){
	    fprintf(fileOneD,"%d %d %d   %+d %+d %+d \t %+e %+e\n",iflag,dir,it,momElem[imom][0],momElem[imom][1],momElem[imom][2],corr_fourier_full[it*Nmom*2 + imom*2 + 0] , corr_fourier_full[it*Nmom*2 + imom*2 + 1]);
	  }
      }


    }

  }



  // ------------------------------------------------------------------------------------------
  cudaUnbindTexture(seqPropagatorTex);
  cudaUnbindTexture(fwdPropagatorTex);
  cudaUnbindTexture(gaugeDerivativeTex);
  cudaFree(d_corr);
  checkCudaError();
  free(corr_fourier_full);
  free(corr_fourier);

  if(comm_rank() == 0){
    fclose(fileLocal);
    fclose(fileNoether);
    fclose(fileOneD);
  }

}


void quda::fixSinkContractions_nonLocal(QKXTM_Propagator &seqProp, QKXTM_Propagator &prop , QKXTM_Gauge &gauge,whatProjector typeProj , char *filename , int Nmom , int momElem[][3] , whatParticle testParticle, int partFlag, double *deviceWilsonPath,int direction){
  

  if(typeProj == QKXTM_TYPE1)
    sprintf(filename,"%s_%s",filename,"type1");
  else
    sprintf(filename,"%s_%s",filename,"type2");

  if(testParticle == QKXTM_PROTON)
    sprintf(filename,"%s_%s",filename,"proton");
  else
    sprintf(filename,"%s_%s",filename,"neutron");

  char filename_nonLocal[257];
  sprintf(filename_nonLocal,"%s_%s.dat",filename,"nonLocal");
  FILE *fileNonLocal;
  
  if(comm_rank() == 0){
    fileNonLocal = fopen(filename_nonLocal,"w");

    if(fileNonLocal == NULL){
      fprintf(stderr,"Error open file for writting\n");
      comm_exit(-1);
    }

  }

  seqProp.applyGamma5();
  seqProp.conjugate();

  // execution domain
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

   // holds correlator in position space
  double *d_corr;
  cudaError error;
  error = cudaMalloc((void**)&d_corr, localVolume*2*sizeof(double));
  if(error != cudaSuccess)errorQuda("Error allocate device memory for correlator");

  double *corr_fourier = (double*) calloc(localL[3]*Nmom*2,sizeof(double)); 
  double *corr_fourier_full = (double*) calloc(totalL[3]*Nmom*2,sizeof(double));


  // to speed up contraction we use texture binding for seq-prop and prop
  cudaBindTexture(0, seqPropagatorTex, seqProp.D_elem(), seqProp.Bytes());
  cudaBindTexture(0, fwdPropagatorTex, prop.D_elem(), prop.Bytes());

  cudaBindTexture(0, gaugeDerivativeTex, gauge.D_elem(), gauge.Bytes());
  
  for(int dl = 0 ; dl < totalL[direction]/2 ; dl++){

    fixSinkContractions_nonLocal_kernel<<<gridDim,blockDim>>>((double2*) d_corr ,(double2*) deviceWilsonPath, dl, testParticle, partFlag,direction);
    cudaDeviceSynchronize(); // to make sure that we have the data in corr

    fixSinkFourier(d_corr,corr_fourier,Nmom,momElem);

    int error = 0;
    
    if(timeRank >= 0 && timeRank < nProc[3] ){
      error = MPI_Gather(corr_fourier,localL[3]*Nmom*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*2,MPI_DOUBLE,0,timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
        
    if(comm_rank() == 0){
      for(int it = 0 ; it < totalL[3] ; it++)
	for(int imom = 0 ; imom < Nmom ; imom++){
	  fprintf(fileNonLocal,"%d %d   %+d %+d %+d \t %+e %+e\n",dl,it,momElem[imom][0],momElem[imom][1],momElem[imom][2],corr_fourier_full[it*Nmom*2 + imom*2 + 0] , corr_fourier_full[it*Nmom*2 + imom*2 + 1]);
	    }
    }
    
  }




  // ------------------------------------------------------------------------------------------
  cudaUnbindTexture(seqPropagatorTex);
  cudaUnbindTexture(fwdPropagatorTex);
  cudaUnbindTexture(gaugeDerivativeTex);
  cudaFree(d_corr);
  checkCudaError();
  free(corr_fourier_full);
  free(corr_fourier);

  if(comm_rank() == 0){
    fclose(fileNonLocal);
  }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////


// $$ Section 12: Class QKXTM_Propagator3D $$


/////////////////////////////////////////////////////// class Propagator 3D /////////
QKXTM_Propagator3D::QKXTM_Propagator3D()
{

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");

 
  field_length = nSpin*nSpin*nColor*nColor;
  

  bytes_total_length = localL[0]*localL[1]*localL[2]*field_length*2*sizeof(double);

  create_host();
  create_device();
  zero();
}

QKXTM_Propagator3D::~QKXTM_Propagator3D(){
    destroy_host();
   destroy_device();
}

void QKXTM_Propagator3D::absorbTimeSlice(QKXTM_Propagator &prop, int timeslice){
  double *pointer_src = NULL;
  double *pointer_dst = NULL;

  int localVolume3D = localL[0]*localL[1]*localL[2];

  for(int mu = 0 ; mu < nSpin ; mu++)
    for(int nu = 0 ; nu < nSpin ; nu++)
      for(int c1 = 0 ; c1 < nColor ; c1++)
	for(int c2 = 0 ; c2 < nColor ; c2++){
	  pointer_dst = d_elem + mu*nSpin*nColor*nColor*localVolume3D*2 + nu*nColor*nColor*localVolume3D*2 + c1*nColor*localVolume3D*2 + c2*localVolume3D*2;
	  pointer_src = prop.D_elem() + mu*nSpin*nColor*nColor*localVolume*2 + nu*nColor*nColor*localVolume*2 + c1*nColor*localVolume*2 + c2*localVolume*2 + timeslice*localVolume3D*2;
	  cudaMemcpy(pointer_dst, pointer_src, localVolume3D*2*sizeof(double), cudaMemcpyDeviceToDevice);
	}

  pointer_src = NULL;
  pointer_dst = NULL;

  checkCudaError();
}

void QKXTM_Propagator3D::absorbVectorTimeSlice(QKXTM_Vector &vec, int timeslice, int nu , int c2){

  double *pointer_src = NULL;
  double *pointer_dst = NULL;

  int localVolume3D = localL[0]*localL[1]*localL[2];

  for(int mu = 0 ; mu < nSpin ; mu++)
    for(int c1 = 0 ; c1 < nColor ; c1++){
      pointer_dst = d_elem + mu*nSpin*nColor*nColor*localVolume3D*2 + nu*nColor*nColor*localVolume3D*2 + c1*nColor*localVolume3D*2 + c2*localVolume3D*2;	  
      pointer_src = vec.D_elem() + mu*nColor*localVolume*2 + c1*localVolume*2 + timeslice*localVolume3D*2;
      cudaMemcpy(pointer_dst, pointer_src, localVolume3D*2 * sizeof(double), cudaMemcpyDeviceToDevice);
    }

  pointer_src = NULL;
  pointer_dst = NULL;

  checkCudaError();

}


void QKXTM_Propagator3D::download(){

  cudaMemcpy(h_elem,d_elem,Bytes() , cudaMemcpyDeviceToHost);
  checkCudaError();
  int localVolume3D = localL[0]*localL[1]*localL[2];
  double *propagator3D_tmp = (double*) malloc( Bytes()  );
  if(propagator3D_tmp == NULL)errorQuda("Error in allocate memory of tmp propagator");

      for(int iv = 0 ; iv < localVolume3D ; iv++)
	for(int mu = 0 ; mu < nSpin ; mu++)                // always work with format colors inside spins
	  for(int nu = 0 ; nu < nSpin ; nu++)
	    for(int c1 = 0 ; c1 < nColor ; c1++)
	      for(int c2 = 0 ; c2 < nColor ; c2++)
		for(int part = 0 ; part < 2 ; part++){
		  propagator3D_tmp[iv*nSpin*nSpin*nColor*nColor*2 + mu*nSpin*nColor*nColor*2 + nu*nColor*nColor*2 + c1*nColor*2 + c2*2 + part] = h_elem[mu*nSpin*nColor*nColor*localVolume3D*2 + nu*nColor*nColor*localVolume3D*2 + c1*nColor*localVolume3D*2 + c2*localVolume3D*2 + iv*2 + part];
		}

      memcpy(h_elem,propagator3D_tmp,Bytes() );

  free(propagator3D_tmp);
  propagator3D_tmp = NULL;
}



void QKXTM_Propagator3D::justCopyToHost(){
  cudaMemcpy(H_elem() , D_elem() , Bytes() , cudaMemcpyDeviceToHost);
  checkCudaError();
}

void QKXTM_Propagator3D::justCopyToDevice(){
  cudaMemcpy(D_elem() , H_elem() , Bytes() , cudaMemcpyHostToDevice);
  checkCudaError();
}

void QKXTM_Propagator3D::broadcast(int tsink){
  
  justCopyToHost(); // transfer data to host so we can communicate
  comm_barrier();
  int bcastRank = tsink/localL[3];
  int localVolume3D = localL[0]*localL[1]*localL[2];
  int error = MPI_Bcast(H_elem() , 4*4*3*3*localVolume3D*2 , MPI_DOUBLE , bcastRank , timeComm ); // broadcast the data from node that has the tsink to other nodes
  if(error != MPI_SUCCESS)errorQuda("Error in mpi broadcasting");
  justCopyToDevice();

}


// $$ Section 13: Class QKXTM_Vector3D $$

//////////////////////////////////////////////////// class Vector3D
QKXTM_Vector3D::QKXTM_Vector3D()
{

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");

 
  field_length = nSpin*nColor;
  

  bytes_total_length = localL[0]*localL[1]*localL[2]*field_length*2*sizeof(double);

  create_host();
  create_device();
  zero();
}

QKXTM_Vector3D::~QKXTM_Vector3D(){
    destroy_host();
   destroy_device();
}


void QKXTM_Vector3D::absorbTimeSlice(QKXTM_Vector &vec, int timeslice){
  double *pointer_src = NULL;
  double *pointer_dst = NULL;

  int localVolume3D = localL[0]*localL[1]*localL[2];

  for(int mu = 0 ; mu < nSpin ; mu++)
      for(int c1 = 0 ; c1 < nColor ; c1++){

	  pointer_dst = d_elem + mu*nColor*localVolume3D*2 +  c1*localVolume3D*2;
	  pointer_src = vec.D_elem() + mu*nColor*localVolume*2 +  c1*localVolume*2 + timeslice*localVolume3D*2;
	  cudaMemcpy(pointer_dst, pointer_src, localVolume3D*2*sizeof(double), cudaMemcpyDeviceToDevice);
	}

  pointer_src = NULL;
  pointer_dst = NULL;

  checkCudaError();
}

void QKXTM_Vector3D::justCopyToHost(){
  cudaMemcpy(H_elem() , D_elem() , Bytes() , cudaMemcpyDeviceToHost);
  checkCudaError();
}

void QKXTM_Vector3D::justCopyToDevice(){
  cudaMemcpy(D_elem() , H_elem() , Bytes() , cudaMemcpyHostToDevice);
  checkCudaError();
}

void QKXTM_Vector3D::broadcast(int tsink){
  
  justCopyToHost(); // transfer data to host so we can communicate
  comm_barrier();
  int bcastRank = tsink/localL[3];
  int localVolume3D = localL[0]*localL[1]*localL[2];
  int error = MPI_Bcast(H_elem() , 4*3*localVolume3D*2 , MPI_DOUBLE , bcastRank , timeComm ); // broadcast the data from node that has the tsink to other nodes
  if(error != MPI_SUCCESS)errorQuda("Error in mpi broadcasting");
  justCopyToDevice();

}





//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************
//****************************************************************************************************************************************************************************

// $$ Section 14: Stochastic Connected Diagrams $$

// Contents
// 1) insLineFourier 
// 2) write_3pf_local
// 3) write_3pf_oneD
// 4) QKXTM_Vector3D::fourier
// 5) partialContract3pf_upart_proton
// 6) partialContract3pf_upart_neutron
// 7) partialContract3pf_dpart_neutron
// 8) partialContract3pf_dpart_proton
// 9) finalize_contract3pf_mixLevel
// 10) finalize_contract3pf_oneLevel
// 11) threepStochUpart
// 12) threepStochDpart




///////////////////////////////////////////// functions for stochastic three point functions //////////

#define WRITE_BINARY

QKXTM_VectorX8::QKXTM_VectorX8()
{

  if(init_qudaQKXTM_flag == false) errorQuda("You must initialize init_qudaQKXTM first");

  ghost_length = 0; 
  field_length = 8*nSpin*nColor;
  
  total_length = localVolume + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(double);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(double);

  create_all();

}

QKXTM_VectorX8::~QKXTM_VectorX8(){
  destroy_all();
}


void quda::insLineFourier(double *insLineMom , double *insLine, int Nmom , int momElem[][3]){
  // insLineMom time,spin,color

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);  // now is localVolume3D


  cudaBindTexture(0, insLineFourierTex, insLine, 4*3*localVolume*2*sizeof(double) );

  double *h_partial_block = NULL;
  double *d_partial_block = NULL;
  h_partial_block = (double*) malloc(4*3*gridDim.x*2 * sizeof(double) );                  // for complex *2
  if(h_partial_block == NULL) errorQuda("error allocate memory for host partial block");
  cudaMalloc((void**)&d_partial_block, 4*3*gridDim.x*2 * sizeof(double) );

  double reduction[4*3*2];
  double globalReduction[4*3*2];

  for(int it = 0 ; it < localL[3] ; it++){
    for(int imom = 0 ; imom < Nmom ; imom++){
    
      fourierCorr_kernel3<<<gridDim,blockDim>>>((double2*) d_partial_block ,it , momElem[imom][0] , momElem[imom][1] , momElem[imom][2] ); // future include mom here
      cudaDeviceSynchronize();
      cudaMemcpy(h_partial_block , d_partial_block , 4*3*gridDim.x*2 * sizeof(double) , cudaMemcpyDeviceToHost);
      
      memset(reduction , 0 , 4*3*2 * sizeof(double) );
      
      for(int gamma = 0 ; gamma < 4 ; gamma++)
	for(int c1 = 0 ; c1 < 3 ; c1++)
	  for(int i =0 ; i < gridDim.x ; i++){
	    reduction[gamma*3*2 + c1*2 + 0] += h_partial_block[gamma*3*gridDim.x*2 + c1*gridDim.x*2 + i*2 + 0];
	    reduction[gamma*3*2 + c1*2 + 1] += h_partial_block[gamma*3*gridDim.x*2 + c1*gridDim.x*2 + i*2 + 1];
	  }

      MPI_Reduce(&(reduction[0]) , &(globalReduction[0]) , 4*3*2 , MPI_DOUBLE , MPI_SUM , 0 , spaceComm);           // only local root has the right value
     
      
      if(localRank == 0){
	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int c1 = 0 ; c1 < 3 ; c1++){
	    insLineMom[it*Nmom*4*3*2 + imom*4*3*2 + gamma*3*2 + c1*2 + 0] = globalReduction[gamma*3*2 + c1*2 + 0];
	    insLineMom[it*Nmom*4*3*2 + imom*4*3*2 + gamma*3*2 + c1*2 + 1] = globalReduction[gamma*3*2 + c1*2 + 1];
	  }
      }


    }
  } // for all local timeslice


  cudaUnbindTexture(insLineFourierTex);

  free(h_partial_block);
  cudaFree(d_partial_block);
  checkCudaError();

  h_partial_block = NULL;
  d_partial_block = NULL;

}



// we must calculate insetion line for all the operators
#define MAX_PARTICLES 18

/*
static void write_3pf_local(FILE *file_ptr, double *results, int iflag , int Nmom , int momElem[][3] ){

  double *corr_fourier_full = (double*) calloc(totalL[3]*Nmom*Nmom*4*4*2,sizeof(double));
  
  int error = 0;
  
  if(timeRank >= 0 && timeRank < nProc[3] ){
    error = MPI_Gather(results,localL[3]*Nmom*Nmom*4*4*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*Nmom*4*4*2,MPI_DOUBLE,0,timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");    
  }

  if(comm_rank() == 0){
    for(int it = 0 ; it < totalL[3] ; it++)
      for(int imom1 = 0 ; imom1 < Nmom ; imom1++)
	for(int imom2 = 0 ; imom2 < Nmom ; imom2++)
	  for(int gamma = 0 ; gamma < nSpin ; gamma++)
	    for(int gamma1 = 0 ; gamma1 < nSpin ; gamma1++){
	      fprintf(file_ptr,"%d %d   %+d %+d %+d  %+d %+d %+d   %d %d \t %+e %+e\n",iflag,it,momElem[imom1][0],momElem[imom1][1],momElem[imom1][2],
		      momElem[imom2][0],momElem[imom2][1],momElem[imom2][2],
		      gamma,gamma1,corr_fourier_full[it*Nmom*Nmom*4*4*2 + imom1*Nmom*4*4*2 + imom2*4*4*2 + gamma*4*2 + gamma1*2 + 0] , 
		      corr_fourier_full[it*Nmom*Nmom*4*4*2 + imom1*Nmom*4*4*2 +imom2*4*4*2 + gamma*4*2 + gamma1*2 + 1]);
	    }
  }
  comm_barrier();
  free(corr_fourier_full);
}


static void write_3pf_oneD(FILE *file_ptr, double *results, int iflag ,int dir , int Nmom , int momElem[][3] ){

  double *corr_fourier_full = (double*) calloc(totalL[3]*Nmom*Nmom*4*4*2,sizeof(double));
  
  int error = 0;
  
  if(timeRank >= 0 && timeRank < nProc[3] ){
    error = MPI_Gather(results,localL[3]*Nmom*Nmom*4*4*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*Nmom*4*4*2,MPI_DOUBLE,0,timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");    
  }

  if(comm_rank() == 0){
    for(int it = 0 ; it < totalL[3] ; it++)
      for(int imom1 = 0 ; imom1 < Nmom ; imom1++)
	for(int imom2 = 0 ; imom2 < Nmom ; imom2++)
	  for(int gamma = 0 ; gamma < nSpin ; gamma++)
	    for(int gamma1 = 0 ; gamma1 < nSpin ; gamma1++){
	      fprintf(file_ptr,"%d %d %d   %+d %+d %+d  %+d %+d %+d   %d %d \t %+e %+e\n",iflag,dir,it,momElem[imom1][0],momElem[imom1][1],momElem[imom1][2],
		      momElem[imom2][0],momElem[imom2][1],momElem[imom2][2],
		      gamma,gamma1,corr_fourier_full[it*Nmom*Nmom*4*4*2 + imom1*Nmom*4*4*2 + imom2*4*4*2 + gamma*4*2 + gamma1*2 + 0] , 
		      corr_fourier_full[it*Nmom*Nmom*4*4*2 + imom1*Nmom*4*4*2 +imom2*4*4*2 + gamma*4*2 + gamma1*2 + 1]);
	    }
  }
  comm_barrier();
  free(corr_fourier_full);
}
*/

static void write_3pf_Nonlocal_zeroMomIns(FILE *file_ptr, double *results,int dir, int iflag , int NmomSink , int momElemSink[][3] ){

  double *corr_fourier_full = (double*) calloc(totalL[3]*NmomSink*4*4*2,sizeof(double));
  
  int error = 0;
  
  if(timeRank >= 0 && timeRank < nProc[3] ){
    error = MPI_Gather(results,localL[3]*NmomSink*4*4*2,MPI_DOUBLE,corr_fourier_full,localL[3]*NmomSink*4*4*2,MPI_DOUBLE,0,timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");    
  }

  if(comm_rank() == 0){


    for(int it = 0 ; it < totalL[3] ; it++)
      for(int imom1 = 0 ; imom1 < NmomSink ; imom1++)
	  for(int gamma = 0 ; gamma < nSpin ; gamma++)
	    for(int gamma1 = 0 ; gamma1 < nSpin ; gamma1++){
	      fprintf(file_ptr,"%d %d %d   %+d %+d %+d  %d %d \t %+e %+e\n",dir,iflag,it,momElemSink[imom1][0],momElemSink[imom1][1],momElemSink[imom1][2],
		      gamma,gamma1,corr_fourier_full[it*NmomSink*4*4*2 + imom1*4*4*2 + gamma*4*2 + gamma1*2 + 0] , 
		      corr_fourier_full[it*NmomSink*4*4*2 + imom1*4*4*2 + gamma*4*2 + gamma1*2 + 1]);
	    }


  }
  comm_barrier();
  free(corr_fourier_full);
}


static void write_3pf_local_zeroMomSink(FILE *file_ptr, double *results, int iflag , int Nmom , int momElem[][3] ){

  double *corr_fourier_full = (double*) calloc(totalL[3]*Nmom*4*4*2,sizeof(double));
  
  int error = 0;
  
  if(timeRank >= 0 && timeRank < nProc[3] ){
    error = MPI_Gather(results,localL[3]*Nmom*4*4*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*4*4*2,MPI_DOUBLE,0,timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");    
  }

  if(comm_rank() == 0){

#ifdef WRITE_BINARY

  fwrite((void*) corr_fourier_full, sizeof(double) , totalL[3]*Nmom*4*4*2,file_ptr);

#else

    for(int it = 0 ; it < totalL[3] ; it++)
      for(int imom1 = 0 ; imom1 < Nmom ; imom1++)
	  for(int gamma = 0 ; gamma < nSpin ; gamma++)
	    for(int gamma1 = 0 ; gamma1 < nSpin ; gamma1++){
	      fprintf(file_ptr,"%d %d   %+d %+d %+d  %d %d \t %+e %+e\n",iflag,it,momElem[imom1][0],momElem[imom1][1],momElem[imom1][2],
		      gamma,gamma1,corr_fourier_full[it*Nmom*4*4*2 + imom1*4*4*2 + gamma*4*2 + gamma1*2 + 0] , 
		      corr_fourier_full[it*Nmom*4*4*2 + imom1*4*4*2 + gamma*4*2 + gamma1*2 + 1]);
	    }
#endif

  }
  comm_barrier();
  free(corr_fourier_full);
}


static void write_3pf_oneD_zeroMomSink(FILE *file_ptr, double *results, int iflag ,int dir , int Nmom , int momElem[][3] ){

  double *corr_fourier_full = (double*) calloc(totalL[3]*Nmom*4*4*2,sizeof(double));
  
  int error = 0;
  
  if(timeRank >= 0 && timeRank < nProc[3] ){
    error = MPI_Gather(results,localL[3]*Nmom*4*4*2,MPI_DOUBLE,corr_fourier_full,localL[3]*Nmom*4*4*2,MPI_DOUBLE,0,timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");    
  }

  if(comm_rank() == 0){

#ifdef WRITE_BINARY

  fwrite((void*) corr_fourier_full, sizeof(double) , totalL[3]*Nmom*4*4*2,file_ptr);

#else
    for(int it = 0 ; it < totalL[3] ; it++)
      for(int imom1 = 0 ; imom1 < Nmom ; imom1++)
	  for(int gamma = 0 ; gamma < nSpin ; gamma++)
	    for(int gamma1 = 0 ; gamma1 < nSpin ; gamma1++){
	      fprintf(file_ptr,"%d %d %d   %+d %+d %+d  %d %d \t %+e %+e\n",iflag,dir,it,momElem[imom1][0],momElem[imom1][1],momElem[imom1][2],
		      gamma,gamma1,corr_fourier_full[it*Nmom*4*4*2 + imom1*4*4*2  + gamma*4*2 + gamma1*2 + 0] , 
		      corr_fourier_full[it*Nmom*4*4*2 + imom1*4*4*2 + gamma*4*2 + gamma1*2 + 1]);
	    }
#endif

  }
  comm_barrier();
  free(corr_fourier_full);
}


void QKXTM_Vector3D::fourier(double *vecMom, int Nmom , int momElem[][3]){
  // vecMom must be allocated with Nmom*4*3*2
  // slowest is momentum then gamma then c1 then r,i

  //  cudaError error;

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);  // now is localVolume3D


   cudaBindTexture(0, correlationTex, d_elem, Bytes() );
   //if(error != cudaSuccess)fprintf(stderr,"Error bind texture\n");

  double *h_partial_block = NULL;
  double *d_partial_block = NULL;
  h_partial_block = (double*) malloc(4*3*gridDim.x*2 * sizeof(double) );                  // for complex *2
  if(h_partial_block == NULL) errorQuda("error allocate memory for host partial block");
  cudaMalloc((void**)&d_partial_block, 4*3*gridDim.x*2 * sizeof(double) );
  //if(error != cudaSuccess)fprintf(stderr,"Error malloc\n");

  double reduction[4*3*2];
  double globalReduction[4*3*2];


    
    for(int imom = 0 ; imom < Nmom ; imom++){

      fourierCorr_kernel4<<<gridDim,blockDim>>>((double2*) d_partial_block ,momElem[imom][0] , momElem[imom][1] , momElem[imom][2] ); // source position and proc position is in constant memory
      cudaDeviceSynchronize();
      cudaMemcpy(h_partial_block , d_partial_block , 4*3*gridDim.x*2 * sizeof(double) , cudaMemcpyDeviceToHost);
      //if(error != cudaSuccess)fprintf(stderr,"Error memcpy\n");

      memset(reduction , 0 , 4*3*2 * sizeof(double) );
      
      for(int gamma = 0 ; gamma < 4 ; gamma++)
	for(int c1 = 0 ; c1 < 3 ; c1++)
	  for(int i =0 ; i < gridDim.x ; i++){
	    reduction[gamma*3*2 + c1*2 + 0] += h_partial_block[gamma*3*gridDim.x*2 + c1*gridDim.x*2 + i*2 + 0];
	    reduction[gamma*3*2 + c1*2 + 1] += h_partial_block[gamma*3*gridDim.x*2 + c1*gridDim.x*2 + i*2 + 1];
	  }

      MPI_Reduce(&(reduction[0]) , &(globalReduction[0]) , 4*3*2 , MPI_DOUBLE , MPI_SUM , 0 , spaceComm);           // only local root has the right value
      
      if(localRank == 0){
	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int c1 = 0 ; c1 < 3 ; c1++){
	    vecMom[ imom*4*3*2 + gamma*3*2 + c1*2 + 0] = globalReduction[gamma*3*2 + c1*2 + 0];
	    vecMom[ imom*4*3*2 + gamma*3*2 + c1*2 + 1] = globalReduction[gamma*3*2 + c1*2 + 1];
	  }
      }

    }  // for all momenta


  cudaUnbindTexture(correlationTex);

  free(h_partial_block);
  cudaFree(d_partial_block);
  //if(error != cudaSuccess)fprintf(stderr,"Error cuda free\n");
  checkCudaError();

  h_partial_block = NULL;
  d_partial_block = NULL;

}


static void partialContract3pf_upart_proton(double *proton_level1,double *proton_level3, QKXTM_Vector3D &vec3D , int Nmom,int momElem[][3]){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);

  partial_lvl1_Contract3pf_Type1_1_kernel<<<gridDim,blockDim>>>((double2*) vec3D.D_elem() , 1 , 2);
  cudaDeviceSynchronize();
  checkCudaError();
  vec3D.fourier(proton_level1,Nmom,momElem);

  for(int gamma = 0 ; gamma < 4 ; gamma++)
    for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){
      partial_lvl3_Contract3pf_Type1_1_kernel<<<gridDim,blockDim>>>((double2*) vec3D.D_elem(),gamma,gamma1, 1 , 2);
      cudaDeviceSynchronize();
      checkCudaError();
      double *ptr = proton_level3 + gamma*4*Nmom*4*3*2 + gamma1*Nmom*4*3*2;
      vec3D.fourier(ptr,Nmom,momElem);
    }
  checkCudaError();
}

static void partialContract3pf_upart_neutron(double *neutron_level3, QKXTM_Vector3D &vec3D , int Nmom,int momElem[][3]){

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);

  for(int gamma = 0 ; gamma < 4 ; gamma++)
    for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){
      partial_lvl3_Contract3pf_Type1_2_kernel<<<gridDim,blockDim>>>((double2*) vec3D.D_elem(),gamma,gamma1, 2);
      cudaDeviceSynchronize();
      checkCudaError();
      double *ptr = neutron_level3 + gamma*4*Nmom*4*3*2 + gamma1*Nmom*4*3*2;
      vec3D.fourier(ptr,Nmom,momElem);
    }

  checkCudaError();
}


static void partialContract3pf_dpart_neutron(double *neutron_level1,double *neutron_level3, QKXTM_Vector3D &vec3D , int Nmom,int momElem[][3]){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);

  partial_lvl1_Contract3pf_Type1_1_kernel<<<gridDim,blockDim>>>((double2*) vec3D.D_elem() , 2 , 1);
  cudaDeviceSynchronize();
  checkCudaError();
  vec3D.fourier(neutron_level1,Nmom,momElem);

  for(int gamma = 0 ; gamma < 4 ; gamma++)
    for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){
      partial_lvl3_Contract3pf_Type1_1_kernel<<<gridDim,blockDim>>>((double2*) vec3D.D_elem(),gamma,gamma1, 2 , 1);
      cudaDeviceSynchronize();
      checkCudaError();
      double *ptr = neutron_level3 + gamma*4*Nmom*4*3*2 + gamma1*Nmom*4*3*2;
      vec3D.fourier(ptr,Nmom,momElem);
    }
  checkCudaError();
}

static void partialContract3pf_dpart_proton(double *proton_level3, QKXTM_Vector3D &vec3D , int Nmom,int momElem[][3]){

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume/localL[3] + blockDim.x -1)/blockDim.x , 1 , 1);

  for(int gamma = 0 ; gamma < 4 ; gamma++)
    for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){
      partial_lvl3_Contract3pf_Type1_2_kernel<<<gridDim,blockDim>>>((double2*) vec3D.D_elem(),gamma,gamma1, 1);
      cudaDeviceSynchronize();
      checkCudaError();
      double *ptr = proton_level3 + gamma*4*Nmom*4*3*2 + gamma1*Nmom*4*3*2;
      vec3D.fourier(ptr,Nmom,momElem);
    }
  checkCudaError();

}



static void finalize_contract3pf_mixLevel(Complex *res,Complex *Iins, Complex *lvl3, Complex *lvl1, int Nmom, int momElem[][3]){

  memset(res,0,localL[3]*Nmom*4*4*2*sizeof(double));

  for(int it = 0 ; it < localL[3] ; it++)
    for(int imom1 = 0 ; imom1 < Nmom ; imom1++){

	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){

	    for(int color = 0 ; color < 3 ; color++){
	      res[it*Nmom*4*4 + imom1*4*4 + gamma*4 + gamma1] = res[it*Nmom*4*4 + imom1*4*4 + gamma*4 + gamma1] +
		Iins[it*Nmom*4*3 + imom1*4*3 + gamma1*3 + color] * lvl1[gamma*3 + color];
	      for(int spin = 0 ; spin < 4 ; spin++){
		res[it*Nmom*4*4 + imom1*4*4 + gamma*4 + gamma1] = res[it*Nmom*4*4 + imom1*4*4 + gamma*4 + gamma1] +
		  Iins[it*Nmom*4*3 + imom1*4*3 + spin*3 + color] * lvl3[gamma*4*4*3 + gamma1*4*3 + spin*3 + color];
	      }
	    }


	  }

      }

}

static void finalize_contract3pf_oneLevel(Complex *res,Complex *Iins, Complex *lvl3, int Nmom, int momElem[][3]){

  memset(res,0,localL[3]*Nmom*4*4*2*sizeof(double));

  for(int it = 0 ; it < localL[3] ; it++)
    for(int imom1 = 0 ; imom1 < Nmom ; imom1++){

	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){

	    for(int color = 0 ; color < 3 ; color++)
	      for(int spin = 0 ; spin < 4 ; spin++){
		res[it*Nmom*4*4 + imom1*4*4 + gamma*4 + gamma1] = res[it*Nmom*4*4 + imom1*4*4 + gamma*4 + gamma1] +
		  Iins[it*Nmom*4*3 + imom1*4*3 + spin*3 + color] * lvl3[gamma*4*4*3 + gamma1*4*3 + spin*3 + color];
	      
	      }

	  }

      }

}



static void finalize_contract3pf_mixLevel_SinkMom(Complex *res,Complex *Iins, Complex *lvl3, Complex *lvl1, int NmomSink, int momElemSink[][3]){

  memset(res,0,localL[3]*NmomSink*4*4*2*sizeof(double));

  for(int it = 0 ; it < localL[3] ; it++)
    for(int imom1 = 0 ; imom1 < NmomSink ; imom1++){

	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){

	    for(int color = 0 ; color < 3 ; color++){
	      res[it*NmomSink*4*4 + imom1*4*4 + gamma*4 + gamma1] = res[it*NmomSink*4*4 + imom1*4*4 + gamma*4 + gamma1] +
		Iins[it*4*3 + gamma1*3 + color] * lvl1[ imom1*4*3 + gamma*3 + color];
	      for(int spin = 0 ; spin < 4 ; spin++){
		res[it*NmomSink*4*4 + imom1*4*4 + gamma*4 + gamma1] = res[it*NmomSink*4*4 + imom1*4*4 + gamma*4 + gamma1] +
		  Iins[it*4*3 + spin*3 + color] * lvl3[gamma*4*NmomSink*4*3 + gamma1*NmomSink*4*3 + imom1*4*3 + spin*3 + color];
	      }
	    }


	  }

      }

}

static void finalize_contract3pf_oneLevel_SinkMom(Complex *res,Complex *Iins, Complex *lvl3, int NmomSink, int momElemSink[][3]){

  memset(res,0,localL[3]*NmomSink*4*4*2*sizeof(double));

  for(int it = 0 ; it < localL[3] ; it++)
    for(int imom1 = 0 ; imom1 < NmomSink ; imom1++){

	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){

	    for(int color = 0 ; color < 3 ; color++)
	      for(int spin = 0 ; spin < 4 ; spin++){
		res[it*NmomSink*4*4 + imom1*4*4 + gamma*4 + gamma1] = res[it*NmomSink*4*4 + imom1*4*4 + gamma*4 + gamma1] +
		  Iins[it*4*3 + spin*3 + color] * lvl3[gamma*4*NmomSink*4*3 + gamma1*NmomSink*4*3 + imom1*4*3 + spin*3 + color];
	      
	      }

	  }

      }

}

//#define NEW_VERSION

void quda::threepStochUpart( QKXTM_Vector &phi , QKXTM_Vector3D &xi ,QKXTM_Propagator &uprop , QKXTM_Propagator3D &uprop3D , QKXTM_Propagator3D &dprop3D, QKXTM_Gauge &gauge, int fixTime , char *filename ,int Nmom , int momElem[][3]){

  // ! fix time is the absolute sink time
  int NmomSink = 1;
  int momElemSink[1][3];
  momElemSink[0][0] = 0;
  momElemSink[0][1] = 0;
  momElemSink[0][2] = 0;
  
  char particles_filename[MAX_PARTICLES][257];
  char particles_filename_noether[MAX_PARTICLES][257];
  char particles_filename_oneD[MAX_PARTICLES][257];

  FILE *file_local[MAX_PARTICLES];
  FILE *file_noether[MAX_PARTICLES];
  FILE *file_oneD[MAX_PARTICLES];

  sprintf(particles_filename[0],"%s_%s",filename,"proton_local.dat");
  sprintf(particles_filename[1],"%s_%s",filename,"neutron_local.dat");

  sprintf(particles_filename_noether[0],"%s_%s",filename,"proton_noether.dat");
  sprintf(particles_filename_noether[1],"%s_%s",filename,"neutron_noether.dat");

  sprintf(particles_filename_oneD[0],"%s_%s",filename,"proton_oneD.dat");
  sprintf(particles_filename_oneD[1],"%s_%s",filename,"neutron_oneD.dat");

  if(comm_rank() == 0){

#ifdef WRITE_BINARY

    file_local[0] = fopen(particles_filename[0],"ab");
    file_local[1] = fopen(particles_filename[1],"ab");

    file_noether[0] = fopen(particles_filename_noether[0],"ab");
    file_noether[1] = fopen(particles_filename_noether[1],"ab");

    file_oneD[0] = fopen(particles_filename_oneD[0],"ab");
    file_oneD[1] = fopen(particles_filename_oneD[1],"ab");

#else
    file_local[0] = fopen(particles_filename[0],"a");
    file_local[1] = fopen(particles_filename[1],"a");

    file_noether[0] = fopen(particles_filename_noether[0],"a");
    file_noether[1] = fopen(particles_filename_noether[1],"a");

    file_oneD[0] = fopen(particles_filename_oneD[0],"a");
    file_oneD[1] = fopen(particles_filename_oneD[1],"a");

#endif

    if(file_local[0] == NULL || file_local[1] == NULL || file_oneD[0] == NULL || file_oneD[1] == NULL || file_noether[0] == NULL || file_noether[1] == NULL){
      fprintf(stderr,"Error open files for writting : %s\n",strerror(errno));
      MPI_Abort(MPI_COMM_WORLD,-1);
    }

  }
  
  // here we will calculate part of the contraction ----------------

  printfQuda("Start partial contraction\n");

  QKXTM_Vector3D *levelVec = new QKXTM_Vector3D();
  
  //  QKXTM_Vector3D *levelVec = new QKXTM_Vector3D[8];
  // QKXTM_Vector3D *levelVec = malloc(8*sizeof(QKXTM_Vector3D));
  //levelVec[0]->QKXTM_Vector3D();

  cudaBindTexture(0, uprop3DStochTex,uprop3D.D_elem(), uprop3D.Bytes());
  cudaBindTexture(0, dprop3DStochTex,dprop3D.D_elem(), dprop3D.Bytes());
  cudaBindTexture(0, xiVector3DStochTex,  xi.D_elem(), xi.Bytes());


  double *proton_level1 = (double*) malloc(NmomSink*4*3*2*sizeof(double));
  double *proton_level3 = (double*) malloc(4*4*NmomSink*4*3*2*sizeof(double));
  double *neutron_level3 = (double*) malloc(4*4*NmomSink*4*3*2*sizeof(double));

  if(proton_level1 == NULL || proton_level3 == NULL || neutron_level3 == NULL){
    fprintf(stderr,"Error allocate host memory for partial contraction\n");
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

  partialContract3pf_upart_proton(proton_level1,proton_level3,*levelVec,NmomSink,momElemSink);
  partialContract3pf_upart_neutron(neutron_level3,*levelVec,NmomSink,momElemSink);

  cudaUnbindTexture(xiVector3DStochTex);
  cudaUnbindTexture(uprop3DStochTex);
  cudaUnbindTexture(dprop3DStochTex);

  delete levelVec;
  printfQuda("Finish partial contraction\n");

  // ---------------------------------------------------------------

  // execution domain
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

   
  double *insLineMom = (double*) malloc(10*localL[3]*Nmom*4*3*2*sizeof(double));
  double *insLineNoetherMom = (double*) malloc(4*localL[3]*Nmom*4*3*2*sizeof(double));
  double *insLineOneDMom = (double*) malloc(8*4*localL[3]*Nmom*4*3*2*sizeof(double));
  if(insLineMom == NULL || insLineOneDMom == NULL || insLineNoetherMom == NULL){
    fprintf(stderr,"Error allocate host memory for insLineMom\n");
    MPI_Abort(MPI_COMM_WORLD,-1);
  }


  // +++++++++++++++  local operators   +++++++++++++++++++// (10 local operator )
  // for local operators we use       1 , g1 , g2 , g3 , g4 , g5 , g5g1 , g5g2 , g5g3 , g5g4
  // so we map operators to integers  0 , 1  , 2 ,  3  , 4  , 5  , 6    , 7    , 8    , 9
  cudaBindTexture(0, propStochTex, uprop.D_elem(), uprop.Bytes());
  cudaBindTexture(0, phiVectorStochTex, phi.D_elem(), phi.Bytes());
  
  QKXTM_Vector *insLine = new QKXTM_Vector();
  printfQuda("Start Insertion line\n");

  for(int iflag = 0 ; iflag < 10 ; iflag++){
    insLine_local_kernel<<<gridDim,blockDim>>>((double2*) insLine->D_elem() , iflag , 1);          // (1,2,3) (upart,dpart,spart)
    cudaDeviceSynchronize(); // to make sure that we have the data in corr
    checkCudaError();
    insLineFourier(insLineMom + iflag*localL[3]*Nmom*4*3*2 ,insLine->D_elem() , Nmom , momElem );
    cudaDeviceSynchronize();
    checkCudaError();
  }
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++// 

  // communication
  cudaBindTexture(0, gaugeDerivativeTex, gauge.D_elem(), gauge.Bytes());

  gauge.ghostToHost();
  gauge.cpuExchangeGhost(); // communicate gauge
  gauge.ghostToDevice();
  comm_barrier();          // just in case
  uprop.ghostToHost();
  uprop.cpuExchangeGhost(); // communicate propagator
  uprop.ghostToDevice();
  comm_barrier();          // just in case
  phi.ghostToHost();
  phi.cpuExchangeGhost(); // communicate stochastic vector
  phi.ghostToDevice();
  comm_barrier();          // just in case

  // +++++++++++++++++++ conserved current    ++++++++++++++++++++++++++++++++//
  // mapping gamma
  // g1 , g2 , g3 , g4
  // 0  , 1  , 2  , 3
  for(int idir = 0 ; idir < 4 ; idir++){
    insLine_noether_kernel<<<gridDim,blockDim>>>((double2*) insLine->D_elem() , idir);
    cudaDeviceSynchronize(); // to make sure that we have the data in corr
    checkCudaError();
    insLineFourier(insLineNoetherMom + idir*localL[3]*Nmom*4*3*2 ,insLine->D_elem() , Nmom , momElem );
    cudaDeviceSynchronize();
    checkCudaError();
  }

  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

  // +++++++++++++++++++ derivative operators ++++++++++++++++++++++++++++++++//

  // for derivative operators we have for gamma matrices g1,g2,g3,g4 ,g5g1,g5g2,g5g3,g5g4 => 4+4 combinations 
  // for derivative index we have 4 index D^0 , D^1 , D^2 , D^3
  // for total we have 8*4=32 combinations
  
  // mapping gamma indices, (derivative will have a seperate index)
  // g1 , g2 , g3 , g4 , g5g1 , g5g2 , g5g3 , g5g4 
  // 0  , 1  , 2  , 3  , 4      , 5      , 6      , 7      



  //#ifdef NEW_VERSION

  QKXTM_VectorX8 *insLineX8 = new QKXTM_VectorX8();
  
  for(int dir = 0 ; dir < 4 ; dir++){
    insLine_oneD_kernel_new<<<gridDim,blockDim>>>((double2*) insLineX8->D_elem(), dir);
    cudaDeviceSynchronize();
    checkCudaError();

    for(int iflag = 0 ; iflag < 8 ; iflag++){
      insLineFourier(insLineOneDMom + iflag*4*localL[3]*Nmom*4*3*2 + dir*localL[3]*Nmom*4*3*2, insLineX8->D_elem() + iflag*nSpin*nColor*localVolume*2, Nmom, momElem);
      cudaDeviceSynchronize();
      checkCudaError();
    }

  }

  delete insLineX8;

  
  //#else

  /*
  for(int iflag = 0 ; iflag < 8 ; iflag++) // iflag perform loop over gammas
    for(int dir = 0 ; dir < 4 ; dir++){

      // need to find a way to improve it
      insLine_oneD_kernel<<<gridDim,blockDim>>>((double2*) insLine->D_elem() , iflag ,dir); // we dont need part here because operators are vector , axial
      cudaDeviceSynchronize(); // to make sure that we have the data in corr
      checkCudaError();
      insLineFourier(insLineOneDMom + iflag*4*localL[3]*Nmom*4*3*2 + dir*localL[3]*Nmom*4*3*2 ,insLine->D_elem() , Nmom , momElem );
      cudaDeviceSynchronize();
      checkCudaError();
    }
  */

  delete insLine;  
  cudaUnbindTexture(gaugeDerivativeTex);
  printfQuda("Finish insertion line\n");
  cudaUnbindTexture(phiVectorStochTex);
  cudaUnbindTexture(propStochTex);


  //+++++++++++++++++++++++++++++++++++ finish insertion line

  double *res = (double*) malloc(localL[3]*Nmom*4*4*2*sizeof(double));

  // write local
  for(int iflag = 0 ; iflag < 10 ; iflag++){
   
    finalize_contract3pf_mixLevel((Complex*) res,(Complex*) (insLineMom + iflag*localL[3]*Nmom*4*3*2 ),(Complex*) proton_level3,(Complex*) proton_level1,  Nmom,  momElem);   
    write_3pf_local_zeroMomSink(file_local[0],res,iflag,Nmom,momElem);

    finalize_contract3pf_oneLevel((Complex*) res,(Complex*) (insLineMom + iflag*localL[3]*Nmom*4*3*2 ),(Complex*) neutron_level3,  Nmom,  momElem);   
    write_3pf_local_zeroMomSink(file_local[1],res,iflag,Nmom,momElem);
    
  }

  // write noether
  for(int idir = 0 ; idir < 4 ; idir++){
   
    finalize_contract3pf_mixLevel((Complex*) res,(Complex*) (insLineNoetherMom + idir*localL[3]*Nmom*4*3*2 ),(Complex*) proton_level3,(Complex*) proton_level1,  Nmom,  momElem);   
    write_3pf_local_zeroMomSink(file_noether[0],res,idir,Nmom,momElem);

    finalize_contract3pf_oneLevel((Complex*) res,(Complex*) (insLineNoetherMom + idir*localL[3]*Nmom*4*3*2 ),(Complex*) neutron_level3,  Nmom,  momElem);   
    write_3pf_local_zeroMomSink(file_noether[1],res,idir,Nmom,momElem);
    
  }


  // write derivatives
  for(int iflag = 0 ; iflag < 8 ; iflag++)
    for(int dir = 0 ; dir < 4 ; dir++){
   
      finalize_contract3pf_mixLevel((Complex*) res,(Complex*) (insLineOneDMom + iflag*4*localL[3]*Nmom*4*3*2 + dir*localL[3]*Nmom*4*3*2 ),(Complex*) proton_level3,(Complex*) proton_level1,  Nmom,  momElem);   
      write_3pf_oneD_zeroMomSink(file_oneD[0],res,iflag,dir,Nmom,momElem);

      finalize_contract3pf_oneLevel((Complex*) res,(Complex*) (insLineOneDMom + iflag*4*localL[3]*Nmom*4*3*2 + dir*localL[3]*Nmom*4*3*2 ),(Complex*) neutron_level3,  Nmom,  momElem);   
      write_3pf_oneD_zeroMomSink(file_oneD[1],res,iflag,dir,Nmom,momElem);
    
  }




  free(res);
  
  if(comm_rank()==0){
    fclose(file_local[0]);
    fclose(file_local[1]);
    fclose(file_noether[0]);
    fclose(file_noether[1]);
    fclose(file_oneD[0]);
    fclose(file_oneD[1]);
  }
  
  free(insLineMom);
  free(insLineNoetherMom);
  free(insLineOneDMom);
  free(proton_level1);
  free(proton_level3);
  free(neutron_level3);
  checkCudaError();
}




void quda::threepStochDpart( QKXTM_Vector &phi , QKXTM_Vector3D &xi ,QKXTM_Propagator &dprop , QKXTM_Propagator3D &uprop3D , QKXTM_Propagator3D &dprop3D, QKXTM_Gauge &gauge, int fixTime , char *filename ,int Nmom , int momElem[][3]){

  int NmomSink = 1;
  int momElemSink[1][3];
  momElemSink[0][0] = 0;
  momElemSink[0][1] = 0;
  momElemSink[0][2] = 0;

  
  // ! fix time is the absolute sink time
  char particles_filename[MAX_PARTICLES][257];
  char particles_filename_noether[MAX_PARTICLES][257];
  char particles_filename_oneD[MAX_PARTICLES][257];

  FILE *file_local[MAX_PARTICLES];
  FILE *file_noether[MAX_PARTICLES];
  FILE *file_oneD[MAX_PARTICLES];

  // FILE *file_oneD[MAX_PARTICLES];

  sprintf(particles_filename[0],"%s_%s",filename,"proton_local.dat");
  sprintf(particles_filename[1],"%s_%s",filename,"neutron_local.dat");

  sprintf(particles_filename_noether[0],"%s_%s",filename,"proton_noether.dat");
  sprintf(particles_filename_noether[1],"%s_%s",filename,"neutron_noether.dat");
  
  sprintf(particles_filename_oneD[0],"%s_%s",filename,"proton_oneD.dat");
  sprintf(particles_filename_oneD[1],"%s_%s",filename,"neutron_oneD.dat");

  if(comm_rank() == 0){

#ifdef WRITE_BINARY

    file_local[0] = fopen(particles_filename[0],"ab");
    file_local[1] = fopen(particles_filename[1],"ab");

    file_noether[0] = fopen(particles_filename_noether[0],"ab");
    file_noether[1] = fopen(particles_filename_noether[1],"ab");

    file_oneD[0] = fopen(particles_filename_oneD[0],"ab");
    file_oneD[1] = fopen(particles_filename_oneD[1],"ab");


#else

    file_local[0] = fopen(particles_filename[0],"a");
    file_local[1] = fopen(particles_filename[1],"a");

    file_noether[0] = fopen(particles_filename_noether[0],"a");
    file_noether[1] = fopen(particles_filename_noether[1],"a");

    file_oneD[0] = fopen(particles_filename_oneD[0],"a");
    file_oneD[1] = fopen(particles_filename_oneD[1],"a");

#endif

    if(file_local[0] == NULL || file_local[1] == NULL || file_oneD[0] == NULL || file_oneD[1] == NULL || file_noether[0] == NULL || file_noether[1] == NULL){
      fprintf(stderr,"Error open files for writting : %s\n",strerror(errno));
      MPI_Abort(MPI_COMM_WORLD,-1);
    }

  }
  


  // here we will calculate part of the contraction ----------------

  printfQuda("Start partial contraction\n");
  QKXTM_Vector3D *levelVec = new QKXTM_Vector3D();
  cudaBindTexture(0, uprop3DStochTex,uprop3D.D_elem(), uprop3D.Bytes());
  cudaBindTexture(0, dprop3DStochTex,dprop3D.D_elem(), dprop3D.Bytes());
  cudaBindTexture(0, xiVector3DStochTex,  xi.D_elem(), xi.Bytes());

  double *neutron_level1 = (double*) malloc(NmomSink*4*3*2*sizeof(double));
  double *neutron_level3 = (double*) malloc(4*4*NmomSink*4*3*2*sizeof(double));
  double *proton_level3 = (double*) malloc(4*4*NmomSink*4*3*2*sizeof(double));
  if(neutron_level1 == NULL || neutron_level3 == NULL || proton_level3 == NULL){
    fprintf(stderr,"Error allocate host memory for partial contraction\n");
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

  partialContract3pf_dpart_neutron(neutron_level1,neutron_level3,*levelVec,NmomSink,momElemSink);
  partialContract3pf_dpart_proton(proton_level3,*levelVec,NmomSink,momElemSink);

  cudaUnbindTexture(xiVector3DStochTex);
  cudaUnbindTexture(uprop3DStochTex);
  cudaUnbindTexture(dprop3DStochTex);

  delete levelVec;
  printfQuda("Finish partial contraction\n");

  // ---------------------------------------------------------------

  // execution domain
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

   
  double *insLineMom = (double*) malloc(10*localL[3]*Nmom*4*3*2*sizeof(double));
  double *insLineNoetherMom = (double*) malloc(4*localL[3]*Nmom*4*3*2*sizeof(double));
  double *insLineOneDMom = (double*) malloc(8*4*localL[3]*Nmom*4*3*2*sizeof(double));
  if(insLineMom == NULL || insLineOneDMom == NULL || insLineNoetherMom == NULL){
    fprintf(stderr,"Error allocate host memory for insLineMom\n");
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

  // +++++++++++++++  local operators   +++++++++++++++++++// (10 local operator )
  // for local operators we use       1 , g1 , g2 , g3 , g4 , g5 , g5g1 , g5g2 , g5g3 , g5g4
  // so we map operators to integers  0 , 1  , 2 ,  3  , 4  , 5  , 6    , 7    , 8    , 9
  cudaBindTexture(0, propStochTex, dprop.D_elem(), dprop.Bytes());
  cudaBindTexture(0, phiVectorStochTex, phi.D_elem(), phi.Bytes());
  QKXTM_Vector *insLine = new QKXTM_Vector();
  printfQuda("Start Insertion line\n");

  for(int iflag = 0 ; iflag < 10 ; iflag++){
    insLine_local_kernel<<<gridDim,blockDim>>>((double2*) insLine->D_elem() , iflag , 2);          // (1,2,3) (upart,dpart,spart)
    cudaDeviceSynchronize(); // to make sure that we have the data in corr
    checkCudaError();
    insLineFourier(insLineMom + iflag*localL[3]*Nmom*4*3*2 ,insLine->D_elem() , Nmom , momElem );
    cudaDeviceSynchronize();
    checkCudaError();
  }
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //
  // communication
  cudaBindTexture(0, gaugeDerivativeTex, gauge.D_elem(), gauge.Bytes());

  gauge.ghostToHost();
  gauge.cpuExchangeGhost(); // communicate gauge
  gauge.ghostToDevice();
  comm_barrier();          // just in case
  dprop.ghostToHost();
  dprop.cpuExchangeGhost(); // communicate propagator
  dprop.ghostToDevice();
  comm_barrier();          // just in case
  phi.ghostToHost();
  phi.cpuExchangeGhost(); // communicate stochastic vector
  phi.ghostToDevice();
  comm_barrier();          // just in case

  // +++++++++++++++++++ conserved current +++++++++++++++++++++++++++++++++++//
  // mapping gamma
  // g1 , g2 , g3 , g4
  // 0  , 1  , 2  , 3
  for(int idir = 0 ; idir < 4 ; idir++){
    insLine_noether_kernel<<<gridDim,blockDim>>>((double2*) insLine->D_elem() , idir);
    cudaDeviceSynchronize(); // to make sure that we have the data in corr
    checkCudaError();
    insLineFourier(insLineNoetherMom + idir*localL[3]*Nmom*4*3*2 ,insLine->D_elem() , Nmom , momElem );
    cudaDeviceSynchronize();
    checkCudaError();
  }

  // +++++++++++++++++++ derivative operators ++++++++++++++++++++++++++++++++//
  // for derivative operators we have for gamma matrices g1,g2,g3,g4 ,g5g1,g5g2,g5g3,g5g4 => 4+4 combinations 
  // for derivative index we have 4 index D^0 , D^1 , D^2 , D^3
  // for total we have 8*4=32 combinations
  
  // mapping gamma indices, (derivative will have a seperate index)
  // g1 , g2 , g3 , g4 , g5g1 , g5g2 , g5g3 , g5g4 
  // 0  , 1  , 2  , 3  , 4      , 5      , 6      , 7      


  //#ifdef NEW_VERSION

  QKXTM_VectorX8 *insLineX8 = new QKXTM_VectorX8();
  
  for(int dir = 0 ; dir < 4 ; dir++){
    insLine_oneD_kernel_new<<<gridDim,blockDim>>>((double2*) insLineX8->D_elem(), dir);
    cudaDeviceSynchronize();
    checkCudaError();

    for(int iflag = 0 ; iflag < 8 ; iflag++){
      insLineFourier(insLineOneDMom + iflag*4*localL[3]*Nmom*4*3*2 + dir*localL[3]*Nmom*4*3*2, insLineX8->D_elem() + iflag*nSpin*nColor*localVolume*2, Nmom, momElem);
      cudaDeviceSynchronize();
      checkCudaError();
    }

  }

  delete insLineX8;
  

  /*  
  //#else
  for(int iflag = 0 ; iflag < 8 ; iflag++) // iflag perform loop over gammas
    for(int dir = 0 ; dir < 4 ; dir++){

      // need to find a way to improve it
      insLine_oneD_kernel<<<gridDim,blockDim>>>((double2*) insLine->D_elem() , iflag ,dir); // we dont need part here because operators are vector , axial
      cudaDeviceSynchronize(); // to make sure that we have the data in corr
      checkCudaError();
      insLineFourier(insLineOneDMom + iflag*4*localL[3]*Nmom*4*3*2 + dir*localL[3]*Nmom*4*3*2 ,insLine->D_elem() , Nmom , momElem );
      cudaDeviceSynchronize();
      checkCudaError();
    }
  */

  //#endif
  
  cudaUnbindTexture(gaugeDerivativeTex);

  printfQuda("Finish insertion line\n");
  delete insLine;
  cudaUnbindTexture(phiVectorStochTex);
  cudaUnbindTexture(propStochTex);


  //+++++++++++++++++++++++++++++++++++ finish insertion line

  double *res = (double*) malloc(localL[3]*Nmom*4*4*2*sizeof(double));

  // write local
  for(int iflag = 0 ; iflag < 10 ; iflag++){
   
    finalize_contract3pf_mixLevel((Complex*) res,(Complex*) (insLineMom + iflag*localL[3]*Nmom*4*3*2 ),(Complex*) neutron_level3,(Complex*) neutron_level1,  Nmom,  momElem);   
    write_3pf_local_zeroMomSink(file_local[1],res,iflag,Nmom,momElem);

    finalize_contract3pf_oneLevel((Complex*) res,(Complex*) (insLineMom + iflag*localL[3]*Nmom*4*3*2 ),(Complex*) proton_level3,  Nmom,  momElem);   
    write_3pf_local_zeroMomSink(file_local[0],res,iflag,Nmom,momElem);
    
  }

  // write conserved
  for(int idir = 0 ; idir < 4 ; idir++){
   
    finalize_contract3pf_mixLevel((Complex*) res,(Complex*) (insLineNoetherMom + idir*localL[3]*Nmom*4*3*2 ),(Complex*) neutron_level3,(Complex*) neutron_level1,  Nmom,  momElem);   
    write_3pf_local_zeroMomSink(file_noether[1],res,idir,Nmom,momElem);

    finalize_contract3pf_oneLevel((Complex*) res,(Complex*) (insLineNoetherMom + idir*localL[3]*Nmom*4*3*2 ),(Complex*) proton_level3,  Nmom,  momElem);   
    write_3pf_local_zeroMomSink(file_noether[0],res,idir,Nmom,momElem);
    
  }


  // write derivatives
  for(int iflag = 0 ; iflag < 8 ; iflag++)
    for(int dir = 0 ; dir < 4 ; dir++){
   
      finalize_contract3pf_mixLevel((Complex*) res,(Complex*) (insLineOneDMom + iflag*4*localL[3]*Nmom*4*3*2 + dir*localL[3]*Nmom*4*3*2 ),(Complex*) neutron_level3,(Complex*) neutron_level1,  Nmom,  momElem);   
      write_3pf_oneD_zeroMomSink(file_oneD[1],res,iflag,dir,Nmom,momElem);

      finalize_contract3pf_oneLevel((Complex*) res,(Complex*) (insLineOneDMom + iflag*4*localL[3]*Nmom*4*3*2 + dir*localL[3]*Nmom*4*3*2 ),(Complex*) proton_level3,  Nmom,  momElem);   
      write_3pf_oneD_zeroMomSink(file_oneD[0],res,iflag,dir,Nmom,momElem);
    
  }

  free(res);
  
  if(comm_rank()==0){
    fclose(file_local[0]);
    fclose(file_local[1]);
    fclose(file_noether[0]);
    fclose(file_noether[1]);    
    fclose(file_oneD[0]);
    fclose(file_oneD[1]);
  }
  
  free(insLineMom);
  free(insLineNoetherMom);
  free(insLineOneDMom);
  free(neutron_level1);
  free(neutron_level3);
  free(proton_level3);
  checkCudaError();
}




//  
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void quda::threepStochUpart_WilsonLinks( QKXTM_Vector &phi , QKXTM_Vector3D &xi ,QKXTM_Propagator &uprop , QKXTM_Propagator3D &uprop3D , QKXTM_Propagator3D &dprop3D,double* deviceWilsonLinks, int fixTime , char *filename ,int Nmom , int momElem[][3], int NmomSink, int momSink[][3]){

  // ! fix time is the absolute sink time

  
  char particles_filename[MAX_PARTICLES][257];

  FILE *file_Nonlocal[MAX_PARTICLES];

  sprintf(particles_filename[0],"%s_%s",filename,"proton_Nonlocal.dat");
  sprintf(particles_filename[1],"%s_%s",filename,"neutron_Nonlocal.dat");


  if(comm_rank() == 0){

    file_Nonlocal[0] = fopen(particles_filename[0],"a");
    file_Nonlocal[1] = fopen(particles_filename[1],"a");

    if(file_Nonlocal[0] == NULL || file_Nonlocal[1] == NULL ){
      fprintf(stderr,"Error open files for writting : %s\n",strerror(errno));
      MPI_Abort(MPI_COMM_WORLD,-1);
    }

  }
  
  // here we will calculate part of the contraction ----------------

  printfQuda("Start partial contraction\n");

  QKXTM_Vector3D *levelVec = new QKXTM_Vector3D();
  
  //  QKXTM_Vector3D *levelVec = new QKXTM_Vector3D[8];
  // QKXTM_Vector3D *levelVec = malloc(8*sizeof(QKXTM_Vector3D));
  //levelVec[0]->QKXTM_Vector3D();

  cudaBindTexture(0, uprop3DStochTex,uprop3D.D_elem(), uprop3D.Bytes());
  cudaBindTexture(0, dprop3DStochTex,dprop3D.D_elem(), dprop3D.Bytes());
  cudaBindTexture(0, xiVector3DStochTex,  xi.D_elem(), xi.Bytes());


  double *proton_level1 = (double*) malloc(NmomSink*4*3*2*sizeof(double));
  double *proton_level3 = (double*) malloc(4*4*NmomSink*4*3*2*sizeof(double));
  double *neutron_level3 = (double*) malloc(4*4*NmomSink*4*3*2*sizeof(double));

  if(proton_level1 == NULL || proton_level3 == NULL || neutron_level3 == NULL){
    fprintf(stderr,"Error allocate host memory for partial contraction\n");
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

  partialContract3pf_upart_proton(proton_level1,proton_level3,*levelVec,NmomSink,momSink);
  partialContract3pf_upart_neutron(neutron_level3,*levelVec,NmomSink,momSink);

  cudaUnbindTexture(xiVector3DStochTex);
  cudaUnbindTexture(uprop3DStochTex);
  cudaUnbindTexture(dprop3DStochTex);

  delete levelVec;
  printfQuda("Finish partial contraction\n");

  // ---------------------------------------------------------------

  // execution domain
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

   
  double *insLineMom = (double*) malloc(3*(totalL[0]/2)*localL[3]*Nmom*4*3*2*sizeof(double));

  if(insLineMom == NULL){
    fprintf(stderr,"Error allocate host memory for insLineMom\n");
    MPI_Abort(MPI_COMM_WORLD,-1);
  }


  // +++++++++++++++  Non local operators   +++++++++++++++++++// 
  cudaBindTexture(0, propStochTex, uprop.D_elem(), uprop.Bytes());
  cudaBindTexture(0, phiVectorStochTex, phi.D_elem(), phi.Bytes());
  
  QKXTM_Vector *insLine = new QKXTM_Vector();
  printfQuda("Start Insertion line\n");

  for(int dir = 0 ; dir < 3 ; dir++)
    for(int dl = 0 ; dl < totalL[dir]/2 ; dl++){
      insLine_Nonlocal_kernel<<<gridDim,blockDim>>>((double2*) insLine->D_elem() ,(double2*) deviceWilsonLinks ,dl,dir);          // (1,2,3) (upart,dpart,spart)
      cudaDeviceSynchronize(); // to make sure that we have the data in corr
      checkCudaError();
      insLineFourier(insLineMom + dir*(totalL[dir]/2)*localL[3]*Nmom*4*3*2 + dl*localL[3]*Nmom*4*3*2 ,insLine->D_elem() , Nmom , momElem );
      cudaDeviceSynchronize();
      checkCudaError();
    }
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++// 


  printfQuda("Finish insertion line\n");
  delete insLine;
  cudaUnbindTexture(phiVectorStochTex);
  cudaUnbindTexture(propStochTex);


  //+++++++++++++++++++++++++++++++++++ finish insertion line

  double *res = (double*) malloc(localL[3]*NmomSink*4*4*2*sizeof(double));

  // write local
  for(int dir = 0 ; dir < 3 ; dir++)
    for(int iflag = 0 ; iflag < totalL[dir]/2 ; iflag++){
      
      finalize_contract3pf_mixLevel_SinkMom((Complex*) res,(Complex*) (insLineMom +dir*(totalL[dir]/2)*localL[3]*Nmom*4*3*2+ iflag*localL[3]*Nmom*4*3*2 ),(Complex*) proton_level3,(Complex*) proton_level1,  NmomSink,  momSink);   
      write_3pf_Nonlocal_zeroMomIns(file_Nonlocal[0],res,dir,iflag,NmomSink,momSink);

      finalize_contract3pf_oneLevel_SinkMom((Complex*) res,(Complex*) (insLineMom +dir*(totalL[dir]/2)*localL[3]*Nmom*4*3*2+ iflag*localL[3]*Nmom*4*3*2 ),(Complex*) neutron_level3,  NmomSink,  momSink);   
      write_3pf_Nonlocal_zeroMomIns(file_Nonlocal[1],res,dir,iflag,NmomSink,momSink);
    
    }



  free(res);
  
  if(comm_rank()==0){
    fclose(file_Nonlocal[0]);
    fclose(file_Nonlocal[1]);
     }
  
  free(insLineMom);
  free(proton_level1);
  free(proton_level3);
  free(neutron_level3);
  checkCudaError();
}



void quda::threepStochDpart_WilsonLinks( QKXTM_Vector &phi , QKXTM_Vector3D &xi ,QKXTM_Propagator &dprop , QKXTM_Propagator3D &uprop3D , QKXTM_Propagator3D &dprop3D, double* deviceWilsonLinks, int fixTime , char *filename ,int Nmom , int momElem[][3], int NmomSink, int momSink[][3]){

  
  // ! fix time is the absolute sink time
  char particles_filename[MAX_PARTICLES][257];

  FILE *file_Nonlocal[MAX_PARTICLES];

  // FILE *file_oneD[MAX_PARTICLES];

  sprintf(particles_filename[0],"%s_%s",filename,"proton_Nonlocal.dat");
  sprintf(particles_filename[1],"%s_%s",filename,"neutron_Nonlocal.dat");

  if(comm_rank() == 0){


    file_Nonlocal[0] = fopen(particles_filename[0],"a");
    file_Nonlocal[1] = fopen(particles_filename[1],"a");

    if(file_Nonlocal[0] == NULL || file_Nonlocal[1] == NULL){
      fprintf(stderr,"Error open files for writting : %s\n",strerror(errno));
      MPI_Abort(MPI_COMM_WORLD,-1);
    }

  }
  


  // here we will calculate part of the contraction ----------------

  printfQuda("Start partial contraction\n");
  QKXTM_Vector3D *levelVec = new QKXTM_Vector3D();
  cudaBindTexture(0, uprop3DStochTex,uprop3D.D_elem(), uprop3D.Bytes());
  cudaBindTexture(0, dprop3DStochTex,dprop3D.D_elem(), dprop3D.Bytes());
  cudaBindTexture(0, xiVector3DStochTex,  xi.D_elem(), xi.Bytes());

  double *neutron_level1 = (double*) malloc(NmomSink*4*3*2*sizeof(double));
  double *neutron_level3 = (double*) malloc(4*4*NmomSink*4*3*2*sizeof(double));
  double *proton_level3 = (double*) malloc(4*4*NmomSink*4*3*2*sizeof(double));
  if(neutron_level1 == NULL || neutron_level3 == NULL || proton_level3 == NULL){
    fprintf(stderr,"Error allocate host memory for partial contraction\n");
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

  partialContract3pf_dpart_neutron(neutron_level1,neutron_level3,*levelVec,NmomSink,momSink);
  partialContract3pf_dpart_proton(proton_level3,*levelVec,NmomSink,momSink);

  cudaUnbindTexture(xiVector3DStochTex);
  cudaUnbindTexture(uprop3DStochTex);
  cudaUnbindTexture(dprop3DStochTex);

  delete levelVec;
  printfQuda("Finish partial contraction\n");

  // ---------------------------------------------------------------

  // execution domain
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (localVolume + blockDim.x -1)/blockDim.x , 1 , 1);

   
  double *insLineMom = (double*) malloc(3*(totalL[0]/2)*localL[3]*Nmom*4*3*2*sizeof(double));


  if(insLineMom == NULL){
    fprintf(stderr,"Error allocate host memory for insLineMom\n");
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

  // +++++++++++++++  local operators   +++++++++++++++++++// (10 local operator )
  // for local operators we use       1 , g1 , g2 , g3 , g4 , g5 , g5g1 , g5g2 , g5g3 , g5g4
  // so we map operators to integers  0 , 1  , 2 ,  3  , 4  , 5  , 6    , 7    , 8    , 9
  cudaBindTexture(0, propStochTex, dprop.D_elem(), dprop.Bytes());
  cudaBindTexture(0, phiVectorStochTex, phi.D_elem(), phi.Bytes());
  QKXTM_Vector *insLine = new QKXTM_Vector();
  printfQuda("Start Insertion line\n");

  for(int dir = 0 ; dir < 3 ; dir++)
    for(int dl = 0 ; dl < totalL[dir]/2 ; dl++){
      insLine_Nonlocal_kernel<<<gridDim,blockDim>>>((double2*) insLine->D_elem() , (double2*) deviceWilsonLinks ,dl,dir);          // (1,2,3) (upart,dpart,spart)
      cudaDeviceSynchronize(); // to make sure that we have the data in corr
      checkCudaError();
      insLineFourier(insLineMom + dir*(totalL[dir]/2)*localL[3]*Nmom*4*3*2 + dl*localL[3]*Nmom*4*3*2 ,insLine->D_elem() , Nmom , momElem );
      cudaDeviceSynchronize();
      checkCudaError();
    }
  // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ //

  printfQuda("Finish insertion line\n");
  delete insLine;
  cudaUnbindTexture(phiVectorStochTex);
  cudaUnbindTexture(propStochTex);


  //+++++++++++++++++++++++++++++++++++ finish insertion line

  double *res = (double*) malloc(localL[3]*NmomSink*4*4*2*sizeof(double));

  // write nonlocal
  for(int dir = 0 ; dir < 3 ; dir++)
    for(int iflag = 0 ; iflag < totalL[0]/2 ; iflag++){
   
      finalize_contract3pf_mixLevel_SinkMom((Complex*) res,(Complex*) (insLineMom + dir*(totalL[dir]/2)*localL[3]*Nmom*4*3*2+ iflag*localL[3]*Nmom*4*3*2 ),(Complex*) neutron_level3,(Complex*) neutron_level1,  NmomSink,  momSink);   
      write_3pf_Nonlocal_zeroMomIns(file_Nonlocal[1],res,dir,iflag,NmomSink,momSink);

      finalize_contract3pf_oneLevel_SinkMom((Complex*) res,(Complex*) (insLineMom + dir*(totalL[dir]/2)*localL[3]*Nmom*4*3*2+ iflag*localL[3]*Nmom*4*3*2 ),(Complex*) proton_level3,  NmomSink,  momSink);   
      write_3pf_Nonlocal_zeroMomIns(file_Nonlocal[0],res,dir,iflag,NmomSink,momSink);
    
    }

  free(res);
  
  if(comm_rank()==0){
    fclose(file_Nonlocal[0]);
    fclose(file_Nonlocal[1]);
  }
  
  free(insLineMom);
  free(neutron_level1);
  free(neutron_level3);
  free(proton_level3);
  checkCudaError();

  
}


#undef MAX_PARTICLES


