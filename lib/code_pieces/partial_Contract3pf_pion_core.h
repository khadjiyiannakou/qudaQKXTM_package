
int sid = blockIdx.x*blockDim.x + threadIdx.x;
int space_stride = c_localL[0]*c_localL[1]*c_localL[2];

if (sid >= space_stride) return;



// perform contractions

double2 reg_out[4][3];

for(int gamma = 0 ; gamma < 4 ; gamma++)
  for(int c1 = 0 ; c1 < 3 ; c1++){
    reg_out[gamma][c1].x = 0.;     reg_out[gamma][c1].y = 0.; 
  }

double2 prop1[4][4][3][3];





   for(int mu = 0 ; mu < 4 ; mu++)
     for(int nu = 0 ; nu < 4 ; nu++)
       for(int c1 = 0 ; c1 < 3 ; c1++)
	 for(int c2 = 0 ; c2 < 3 ; c2++)
	   prop1[mu][nu][c1][c2] = fetch_double2(uprop3DStochTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * space_stride);
   



double2 xi[4][3];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
    xi[mu][c1] = fetch_double2(xiVector3DStochTex, sid + (mu*3+c1)*space_stride );

// now we must apply gamma5 on xi
for(int c1 = 0 ; c1 < 3 ; c1++){
  double2 backup_xi[4];
  backup_xi[0] = xi[0][c1];
  backup_xi[1] = xi[1][c1];
  backup_xi[2] = xi[2][c1];
  backup_xi[3] = xi[3][c1];

  xi[0][c1] = backup_xi[2];
  xi[1][c1] = backup_xi[3];
  xi[2][c1] = backup_xi[0];
  xi[3][c1] = backup_xi[1];
 }
  
if( index == 0 ){

  if(sid == 0){
   for(int mu = 0 ; mu < 4 ; mu++)
     for(int nu = 0 ; nu < 4 ; nu++)
       for(int c1 = 0 ; c1 < 3 ; c1++)
	 for(int c2 = 0 ; c2 < 3 ; c2++)
	   cuPrintf("%d %d %d %d %+e %+e\n",mu,nu,c1,c2,prop1[mu][nu][c1][c2].x,prop1[mu][nu][c1][c2].y);
   cuPrintf("\n");
   for(int mu = 0 ; mu < 4 ; mu++)
     for(int c1 = 0 ; c1 < 3 ; c1++)
       cuPrintf("%d %d %+e %+e\n",mu,c1,xi[mu][c1].x,xi[mu][c1].y);
  }

  for(int b = 0 ; b < 3 ; b++)
    for(int a = 0 ; a < 3 ; a++)
      for(int beta = 0 ; beta < 4 ; beta++)
	for(int alpha = 0 ; alpha < 4 ; alpha++)
	  reg_out[alpha][a] = reg_out[alpha][a] + conj(prop1[beta][alpha][b][a]) * xi[beta][b];

 }
 else if (index == 1)  {

  for(int b = 0 ; b < 3 ; b++)
    for(int a = 0 ; a < 3 ; a++)
      for(int beta = 0 ; beta < 4 ; beta++)
	for(int alpha = 0 ; alpha < 4 ; alpha++)
	  reg_out[alpha][a] = reg_out[alpha][a] + prop1[beta][alpha][b][a] * conj(xi[beta][b]); // because gamma_5^* = gamma_5


 }



// copy results to global memory
for(int alpha = 0 ; alpha < 4 ; alpha++)
  for(int a = 0 ; a < 3 ; a++)
    out[alpha*3*space_stride + a*space_stride + sid] = reg_out[alpha][a];


