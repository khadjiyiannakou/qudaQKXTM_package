#include <qkxTM.h>
#include <mpi.h>
#include <cuPrintf.cu>

#define THREADS_PER_BLOCK 32

using namespace quda;

//#include <qkxTM_gauge_constants.h>
//#include <qkxTM_gauge_textures.h>

texture<int4, 1> qkxTM_vectorTex;

__constant__ bool c_dimBreak[4];
__constant__ int c_nColor;
__constant__ int c_nDim;
__constant__ int c_localL[4];
__constant__ int c_plusGhost[4];
__constant__ int c_minusGhost[4];
__constant__ int c_stride;
__constant__ int c_surface[4];


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

/*
__global__ void calculatePlaq_kernel(double *partial_plaq ,int threads){

#include <plaquette_core.h>

}
*/


QKXTM_Vector::QKXTM_Vector(SmearingInfo *smearInfo):
  nColor(3) , nSpin(4) , nDim(4) , nsmearAPE(smearInfo->nsmearAPE),
  nsmearGauss(smearInfo->nsmearGauss), alphaAPE(smearInfo->alphaAPE),
  alphaGauss( smearInfo->alphaGauss) , init(false) , h_elem(NULL) , d_elem(NULL),
  init_host_alloc(false) , init_device_alloc(false) , init_ext_host_ghost(false) , 
  init_ext_device_ghost(false) , h_ext_ghost(NULL), d_ext_ghost(NULL),
  vector_binded(false), packVector_flag(false), loadVector_flag(false),
  device_constants_flag(false)
{
  
  for(int i = 0 ; i < nDim ; i++)
    nProc[i] = comm_dim(i);        // save number of procs in each direction

  for(int i = 0 ; i < nDim ; i++){   // take local and total lattice
    localL[i] = smearInfo->lL[i];
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
    surface[i] = 1;
    for (int j=0; j<nDim; j++) {
      if (i==j) continue;
      surface[i] *= localL[j];
    }
  }

  for(int i = 0 ; i < nDim ; i++)
    if( localL[i] == totalL[i] )
      surface[i] = 0;
  
  ghost_length = 0;

  for(int i = 0 ; i < nDim ; i++)
    ghost_length += 2*surface[i];
  
  total_length = localVolume + ghost_length;
  
  bytes_total_length = total_length*nSpin*nColor*2*sizeof(double); // for vector is different from gauge
  bytes_ghost_length = ghost_length*nSpin*nColor*2*sizeof(double);

  for(int i = 0 ; i < nDim ; i++){
    plusGhost[i] =0;
    minusGhost[i] = 0;
  }
  
#ifdef MULTI_GPU
  int lastIndex = localVolume;
  for(int i = 0 ; i < nDim ; i++)
    if( localL[i] < totalL[i] ){
      plusGhost[i] = lastIndex ;
      minusGhost[i] = lastIndex + surface[i];
      lastIndex += 2*surface[i];
    }
#endif

  init = true;
}

QKXTM_Vector::~QKXTM_Vector(){
  destroy_all();
  unbindVector();
  packVector_flag = false;
  loadVector_flag = false;
}

void QKXTM_Vector::create_host(){

  if(init_host_alloc == false){
    h_elem = (double*) malloc(bytes_total_length);
    if(h_elem == NULL) errorQuda("Error with allocation host memory");
  }
  init_host_alloc = true;
}

void QKXTM_Vector::create_host_ghost(){
#ifdef MULTI_GPU
  if(init_ext_host_ghost == false){
    if( comm_size() > 1){
      h_ext_ghost = (double*) malloc(bytes_ghost_length);
      if(h_ext_ghost == NULL)errorQuda("Error with allocation host memory");
    }
  }
#endif
  init_ext_host_ghost = true;
}


void QKXTM_Vector::create_device(){
  cudaError error;
  if(init_device_alloc == false){
    error = cudaMalloc((void**)&d_elem,bytes_total_length);
    if( error != cudaSuccess) errorQuda("Error with allocation device memory");
  }
  init_device_alloc = true;
}

void QKXTM_Vector::create_device_ghost(){
#ifdef MULTI_GPU
  cudaError error;
  if(init_ext_device_ghost == false){
    if( comm_size() > 1){
      error = cudaMalloc((void**)&d_ext_ghost,bytes_ghost_length);
      if( error != cudaSuccess) errorQuda("Error with allocation device memory");
    }
  }
#endif
  init_ext_device_ghost = true;
}

void QKXTM_Vector::destroy_host(){
  if(init_host_alloc == true){
    free(h_elem);
    h_elem = NULL;
    init_host_alloc = false;
  }
}

void QKXTM_Vector::destroy_device(){
  if(init_device_alloc == true){
    cudaFree(d_elem);
    d_elem = NULL;
    init_device_alloc = false;
  }
}

void QKXTM_Vector::destroy_host_ghost(){
#ifdef MULTI_GPU
  if( (comm_size() > 1) && (init_ext_host_ghost == true) ){
    free(h_ext_ghost);
  } 
#endif
  init_ext_host_ghost = false;
}

void QKXTM_Vector::destroy_device_ghost(){
#ifdef MULTI_GPU
  if( (comm_size() > 1) && (init_ext_device_ghost == true) ){
    cudaFree(d_ext_ghost);
  }
#endif
  init_ext_device_ghost = false;
}

void QKXTM_Vector::create_all(){
  create_host();
  create_host_ghost();
  create_device();
  create_device_ghost();
}

void QKXTM_Vector::destroy_all(){
  destroy_host();
  destroy_host_ghost();
  destroy_device();
  destroy_device_ghost();
}

void QKXTM_Vector::packVector(void *vector){

  if(init_host_alloc == true && packVector_flag == false){
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
  else{
    errorQuda("Error not create host pointer");
  }

}

void QKXTM_Vector::loadVector(){

  if(packVector_flag == true && loadVector_flag == false){
    cudaError error;
    if( (init_host_alloc == true) && (init_device_alloc == true) ){
      error = cudaMemcpy(d_elem,h_elem,(bytes_total_length - bytes_ghost_length), cudaMemcpyHostToDevice );
      if(error != cudaSuccess) errorQuda("Error problem with load Vector on device");
      loadVector_flag = true;
      printfQuda("Vector qkxTM loaded on gpu\n");
    }
    else{
      errorQuda("Error try to load vector without allocate properly first");
    }
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
    position = (localL[0]-1)*localL[1]*localL[2]*localL[3];
    
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + minusGhost[0]*nSpin*nColor*2 + mu*nColor*surface[0]*2 + c1*surface[0]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}
  // set minus points to plus area
    position = 0;

      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + plusGhost[0]*nSpin*nColor*2 + mu*nColor*surface[0]*2 + c1*surface[0]*2;
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
    position = localL[0]*(localL[1]-1)*localL[2]*localL[3];
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + minusGhost[1]*nSpin*nColor*2 + mu*nColor*surface[1]*2 + c1*surface[1]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}

  // set minus points to plus area
    position = 0;

      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + plusGhost[1]*nSpin*nColor*2 + mu*nColor*surface[1]*2 + c1*surface[1]*2;
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
    position = localL[0]*localL[1]*(localL[2]-1)*localL[3];
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + minusGhost[2]*nSpin*nColor*2 + mu*nColor*surface[2]*2 + c1*surface[2]*2;
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,cudaMemcpyDeviceToHost);
	}

  // set minus points to plus area
    position = 0;
      for(int mu = 0 ; mu < nSpin ; mu++)
	for(int c1 = 0 ; c1 < nColor ; c1++){
	  d_elem_offset = d_elem + mu*nColor*localVolume*2 + c1*localVolume*2 + position*2;  
	  h_elem_offset = h_elem + plusGhost[2]*nSpin*nColor*2 + mu*nColor*surface[2]*2 + c1*surface[2]*2;
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
      size_t nbytes = surface[0]*nSpin*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[0]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + minusGhost[0]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[0], 0, &(request_recv[0]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[0], 0, &(request_send[0]));
      comm_wait(&(request_recv[0])); // blocking until receive finish

      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[0]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + plusGhost[0]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[0], 1, &(request_recv[1]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[0], 1, &(request_send[1]));
      comm_wait(&(request_recv[1])); // blocking until receive finish
      
    }
    // direction y
    if(localL[1] < totalL[1]){
      double *pointer_receive = NULL;
      double *pointer_send = NULL;
      size_t nbytes = surface[1]*nSpin*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[1]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + minusGhost[1]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[1], 2, &(request_recv[2]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[1], 2, &(request_send[2]));
      comm_wait(&(request_recv[2])); // blocking until receive finish

      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[1]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + plusGhost[1]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[1], 3, &(request_recv[3]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[1], 3, &(request_send[3]));
      comm_wait(&(request_recv[3])); // blocking until receive finish
    }

    // direction z
    if(localL[2] < totalL[2]){
      double *pointer_receive = NULL;
      double *pointer_send = NULL;
      size_t nbytes = surface[2]*nSpin*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[2]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + minusGhost[2]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[2], 4, &(request_recv[4]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[2], 4, &(request_send[4]));
      comm_wait(&(request_recv[4])); // blocking until receive finish

      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[2]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + plusGhost[2]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[2], 5, &(request_recv[5]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[2], 5, &(request_send[5]));
      comm_wait(&(request_recv[5])); // blocking until receive finish
    }


    // direction t
    if(localL[3] < totalL[3]){
      double *pointer_receive = NULL;
      double *pointer_send = NULL;
      size_t nbytes = surface[3]*nSpin*nColor*2*sizeof(double);

      // send to plus
      pointer_receive = h_ext_ghost + (minusGhost[3]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + minusGhost[3]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, back_nbr[3], 6, &(request_recv[6]));
      comm_send_with_tag(pointer_send, nbytes, fwd_nbr[3], 6, &(request_send[6]));
      comm_wait(&(request_recv[6])); // blocking until receive finish

      // send to minus
      pointer_receive = h_ext_ghost + (plusGhost[3]-localVolume)*nSpin*nColor*2;
      pointer_send = h_elem + plusGhost[3]*nSpin*nColor*2;
      comm_recv_with_tag(pointer_receive, nbytes, fwd_nbr[3], 7, &(request_recv[7]));
      comm_send_with_tag(pointer_send, nbytes, back_nbr[3], 7, &(request_send[7]));
      comm_wait(&(request_recv[7])); // blocking until receive finish
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
  }
#endif
}

void QKXTM_Vector::bindVector(){
  cudaError error;
  if( vector_binded == false ){
    error = cudaBindTexture(0,qkxTM_vectorTex,d_elem,bytes_total_length);
    if(error != cudaSuccess) errorQuda("Problem bind Texture");
  }
  vector_binded = true;
}

void QKXTM_Vector::unbindVector(){
  cudaError error;
  if(vector_binded == true){
    error = cudaUnbindTexture(qkxTM_vectorTex);
    if(error != cudaSuccess) errorQuda("Problem unbind Texture");
  }
  vector_binded = false;
}

void QKXTM_Vector::device_constants(plaqParamKernel paramKernel){
  cudaError error;

  if(device_constants_flag == false){

    error = cudaMemcpyToSymbol(c_dimBreak,paramKernel.dimBreak, 4*sizeof(bool) );
    if(error != cudaSuccess) errorQuda("Problem copy to symbol in device memory");

    error = cudaMemcpyToSymbol(c_nColor,&(paramKernel.nColor), sizeof(int) );
    if(error != cudaSuccess) errorQuda("Problem copy to symbol in device memory");

    error = cudaMemcpyToSymbol(c_nDim,&(paramKernel.nDim), sizeof(int) );
    if(error != cudaSuccess) errorQuda("Problem copy to symbol in device memory");

    error = cudaMemcpyToSymbol(c_plusGhost,paramKernel.plusGhost, 4*sizeof(int) );
    if(error != cudaSuccess) errorQuda("Problem copy to symbol in device memory");

    error = cudaMemcpyToSymbol(c_minusGhost,paramKernel.minusGhost, 4*sizeof(int) );
    if(error != cudaSuccess) errorQuda("Problem copy to symbol in device memory");

    error = cudaMemcpyToSymbol(c_localL,paramKernel.localL, 4*sizeof(int) );
    if(error != cudaSuccess) errorQuda("Problem copy to symbol in device memory");

    error = cudaMemcpyToSymbol(c_stride,&strideFull, sizeof(int) );
    if(error != cudaSuccess) errorQuda("Problem copy to symbol in device memory");

    error = cudaMemcpyToSymbol(c_surface,surface, 4*sizeof(int) );
    if(error != cudaSuccess) errorQuda("Problem copy to symbol in device memory");

    device_constants_flag = true;
  }

}


double QKXTM_Vector::calculatePlaq(void **vector){

  plaqParamKernel paramKernel;

  for(int i = 0 ; i < nDim ; i++){
    if( localL[i] < totalL[i])
      paramKernel.dimBreak[i] = true;
    else
      paramKernel.dimBreak[i] = false;
  }

  paramKernel.nColor = nColor;
  paramKernel.nDim = nDim;
  
  for(int i = 0 ; i < nDim ; i++){
    paramKernel.localL[i] = localL[i]; 
    paramKernel.plusGhost[i] = plusGhost[i];
    paramKernel.minusGhost[i] = minusGhost[i];
  }
  
  paramKernel.threads = localVolume;


  create_all();
  if(device_constants_flag == false) device_constants(paramKernel);
  if(vector_binded == false) bindVector();
  if(packVector_flag == false) packVector(vector);
  if(loadVector_flag == false) loadVector();

  ghostToHost(); // collect surface from device and send it to host
  cpuExchangeGhost(); // cpus exchange surfaces with previous and forward proc all dir
  ghostToDevice();   // now the host send surface to device 

  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (paramKernel.threads + blockDim.x -1)/blockDim.x , 1 , 1);

  double *h_partial_plaq = NULL;
  double *d_partial_plaq = NULL;

  h_partial_plaq = (double*) malloc(gridDim.x * sizeof(double) ); // only real part
  if(h_partial_plaq == NULL) errorQuda("Error allocate memory for host partial plaq");

  cudaError error;
  error = cudaMalloc((void**)&d_partial_plaq, gridDim.x * sizeof(double));
  if(error != cudaSuccess) errorQuda("Error allocate device memory for partial plaq");

  // cudaPrintfInit();

  cudaEvent_t start,stop;
  float elapsedTime;
  error = cudaEventCreate(&start);
  if(error != cudaSuccess) errorQuda("Problem create Event for plaquette kernel");
  error = cudaEventCreate(&stop);
  if(error != cudaSuccess) errorQuda("Problem create Event for plaquette kernel");
  
  error = cudaEventRecord(start,0);
  if(error != cudaSuccess) errorQuda("Problem record Event for plaquette kernel");

  calculatePlaq_kernel<<<gridDim,blockDim>>>(d_partial_plaq,paramKernel.threads);

  if ( cudaSuccess != cudaGetLastError() ) errorQuda("Problem executing plaquette kernel");

  error = cudaEventRecord(stop,0);
  if(error != cudaSuccess) errorQuda("Problem record Event for plaquette kernel");

  error = cudaEventSynchronize(stop);
  if(error != cudaSuccess) errorQuda("Problem synch Event for plaquette kernel");

  error = cudaEventElapsedTime(&elapsedTime,start,stop);
  if(error != cudaSuccess) errorQuda("Problem take timing for plaquette kernel");
  //  if(comm_rank() == 0)  cudaPrintfDisplay(stdout,true);
  //  cudaPrintfEnd();

  error = cudaEventDestroy(start);
  if(error != cudaSuccess) errorQuda("Problem destroy event for plaquette kernel");

  error = cudaEventDestroy(stop);
  if(error != cudaSuccess) errorQuda("Problem destroy event for plaquette kernel");

  printfQuda("Elapsed time for plaquette kernel is %f ms\n",elapsedTime);

  // now copy partial plaq to host
  error = cudaMemcpy(h_partial_plaq, d_partial_plaq , gridDim.x * sizeof(double) , cudaMemcpyDeviceToHost);
  if( error != cudaSuccess ) errorQuda("Error copy partial plaq from device to host");

  double plaquette = 0.;

#ifdef MULTI_GPU
  double globalPlaquette = 0.;
#endif
  // simple host reduction on plaq
  for(int i = 0 ; i < gridDim.x ; i++)
    plaquette += h_partial_plaq[i];



  free(h_partial_plaq);
  cudaFree(d_partial_plaq);

#ifdef MULTI_GPU
  int rc = MPI_Allreduce(&plaquette , &globalPlaquette , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  return globalPlaquette/(totalVolume*nColor*6) ;
#else
  return plaquette/(totalVolume*nColor*6);
#endif  
  


}


void QKXTM_Vector::printInfo(){
  printfQuda("Number of colors is %d\n",nColor);
  printfQuda("Number of spins is %d\n",nSpin);
  printfQuda("Number of dimensions is %d\n",nDim);
  printfQuda("Number of process in each direction is (x,y,z,t) %d x %d x %d x %d\n",nProc[0],nProc[1],nProc[2],nProc[3]);
  printfQuda("Total lattice is (x,y,z,t) %d x %d x %d x %d\n",totalL[0],totalL[1],totalL[2],totalL[3]);
  printfQuda("Local lattice is (x,y,z,t) %d x %d x %d x %d\n",localL[0],localL[1],localL[2],localL[3]);
  printfQuda("Total volume is %d\n",totalVolume);
  printfQuda("Local volume is %d\n",localVolume);
  printfQuda("Surface is (x,y,z,t) ( %d , %d , %d , %d)\n",surface[0],surface[1],surface[2],surface[3]);
  printfQuda("Stride for GPU use is %d\n",strideFull);
  printfQuda("Ghost length with out ndim*ncolor*ncolor*2 is %d\n",ghost_length);
  printfQuda("Total length with out ndim*ncolor*ncolor*2 is %d\n",total_length);
  printfQuda("GPU memory needed is %f MB \n",bytes_total_length/(1024.0 * 1024.0));
  printfQuda("The plus Ghost points in directions (x,y,z,t) ( %d , %d , %d , %d )\n",plusGhost[0],plusGhost[1],plusGhost[2],plusGhost[3]);
  printfQuda("The Minus Ghost points in directixons (x,y,z,t) ( %d , %d , %d , %d )\n",minusGhost[0],minusGhost[1],minusGhost[2],minusGhost[3]);
  printfQuda("For APE smearing we use nsmear = %d , alpha = %lf\n",nsmearAPE,alphaAPE);
  printfQuda("For Gauss smearing we use nsmear = %d , alpha = %lf\n",nsmearGauss,alphaGauss);

}
