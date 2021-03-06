
__shared__ double2 shared_cache[THREADS_PER_BLOCK];

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= c_localL[0]*c_localL[1]*c_localL[2]) return;

int cacheIndex = threadIdx.x;

int x_id, y_id , z_id;
int r1,r2;

r1 = sid / c_localL[0];
x_id = sid - r1 * c_localL[0];
r2 = r1 / c_localL[1];
y_id = r1 - r2*c_localL[1];
z_id = r2;

int x,y,z;

x = x_id + c_procPosition[0] * c_localL[0] - c_sourcePosition[0];
y = y_id + c_procPosition[1] * c_localL[1] - c_sourcePosition[1];
z = z_id + c_procPosition[2] * c_localL[2] - c_sourcePosition[2];

double phase;

phase = ( ((double) nx*x)/c_totalL[0] + ((double) ny*y)/c_totalL[1] + ((double) nz*z)/c_totalL[2] ) * 2. * PI;

double2 expon;

expon.x = cos(phase);
expon.y = -sin(phase);

  
double2 corr[4*4];

for(int gamma = 0 ; gamma < 4 ; gamma++)
  for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++)
    corr[gamma*4+gamma1] = fetch_double2(correlationTex ,  gamma*4*c_stride + gamma1*c_stride + it*c_localL[0]*c_localL[1]*c_localL[2] + sid  );

for(int gamma = 0 ; gamma < 4 ; gamma++)
  for(int gamma1 = 0 ; gamma1 < 4 ; gamma1++){

    shared_cache[cacheIndex] = corr[gamma*4+gamma1] * expon;
    __syncthreads();

    int i = blockDim.x/2;
    
    while (i != 0){
      
      if(cacheIndex < i){
	shared_cache[cacheIndex].x += shared_cache[cacheIndex + i].x;
	shared_cache[cacheIndex].y += shared_cache[cacheIndex + i].y;
      }
      __syncthreads();
      i /= 2;
      
    }

    if(cacheIndex == 0)
      block[gamma*4*gridDim.x + gamma1*gridDim.x + blockIdx.x] = shared_cache[0];
  
  }
