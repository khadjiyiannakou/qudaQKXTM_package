int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

double2 phaseX, phaseY, phaseZ;

phaseX.x = cos( (2.*PI/c_totalL[0]) * px * zeta );
phaseX.y = sin( (2.*PI/c_totalL[0]) * px * zeta );

phaseY.x = cos( (2.*PI/c_totalL[1]) * py * zeta );
phaseY.y = sin( (2.*PI/c_totalL[1]) * py * zeta );

phaseZ.x = cos( (2.*PI/c_totalL[2]) * pz * zeta );
phaseZ.y = sin( (2.*PI/c_totalL[2]) * pz * zeta );

double2 G_0, G_1, G_2, G_3, G_4, G_5, G_6, G_7, G_8;


if(px != 0){
  
 G_0 = inOut[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid];
 G_1 = inOut[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid];
 G_2 = inOut[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid];
 G_3 = inOut[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid];
 G_4 = inOut[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid];
 G_5 = inOut[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid];
 G_6 = inOut[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid];
 G_7 = inOut[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid];
 G_8 = inOut[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid];

 inOut[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] = G_0 * phaseX;
 inOut[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] = G_1 * phaseX;
 inOut[0*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] = G_2 * phaseX;
 inOut[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] = G_3 * phaseX;
 inOut[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] = G_4 * phaseX;
 inOut[0*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] = G_5 * phaseX;
 inOut[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] = G_6 * phaseX;
 inOut[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] = G_7 * phaseX;
 inOut[0*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] = G_8 * phaseX;

 }

if(py != 0){

 G_0 = inOut[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid];
 G_1 = inOut[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid];
 G_2 = inOut[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid];
 G_3 = inOut[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid];
 G_4 = inOut[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid];
 G_5 = inOut[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid];
 G_6 = inOut[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid];
 G_7 = inOut[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid];
 G_8 = inOut[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid];

 inOut[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] = G_0 * phaseY;
 inOut[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] = G_1 * phaseY;
 inOut[1*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] = G_2 * phaseY;
 inOut[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] = G_3 * phaseY;
 inOut[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] = G_4 * phaseY;
 inOut[1*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] = G_5 * phaseY;
 inOut[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] = G_6 * phaseY;
 inOut[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] = G_7 * phaseY;
 inOut[1*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] = G_8 * phaseY;

 }

if(pz != 0){

 G_0 = inOut[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid];
 G_1 = inOut[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid];
 G_2 = inOut[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid];
 G_3 = inOut[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid];
 G_4 = inOut[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid];
 G_5 = inOut[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid];
 G_6 = inOut[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid];
 G_7 = inOut[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid];
 G_8 = inOut[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid];

 inOut[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 0*c_stride + sid] = G_0 * phaseZ;
 inOut[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 1*c_stride + sid] = G_1 * phaseZ;
 inOut[2*c_nColor*c_nColor*c_stride + 0*c_nColor*c_stride + 2*c_stride + sid] = G_2 * phaseZ;
 inOut[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 0*c_stride + sid] = G_3 * phaseZ;
 inOut[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 1*c_stride + sid] = G_4 * phaseZ;
 inOut[2*c_nColor*c_nColor*c_stride + 1*c_nColor*c_stride + 2*c_stride + sid] = G_5 * phaseZ;
 inOut[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 0*c_stride + sid] = G_6 * phaseZ;
 inOut[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 1*c_stride + sid] = G_7 * phaseZ;
 inOut[2*c_nColor*c_nColor*c_stride + 2*c_nColor*c_stride + 2*c_stride + sid] = G_8 * phaseZ;

 }
