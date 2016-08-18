
int sid = blockIdx.x*blockDim.x + threadIdx.x;
int space_stride = c_localL[0]*c_localL[1]*c_localL[2];

if (sid >= space_stride) return;


double2 Cg5[4][4];
double2 Cg5_bar[4][4];

for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++){
    Cg5[mu][nu].x = 0.; Cg5[mu][nu].y=0.;
    Cg5_bar[mu][nu].x = 0.; Cg5_bar[mu][nu].y=0.;
  }

Cg5[0][1].x = 1.;
Cg5[1][0].x = -1.;
Cg5[2][3].x = 1.;
Cg5[3][2].x = -1.;

Cg5_bar[0][1].x = -1.;
Cg5_bar[1][0].x = 1.;
Cg5_bar[2][3].x = -1.;
Cg5_bar[3][2].x = 1.;



double2 C_temp;
double2 Cg5Cg5bar_val[16*16];
unsigned short int Cg5Cg5bar_ind[16*16][4];
int counter = 0;

for(unsigned short int alpha = 0 ; alpha < 4 ; alpha++)
  for(unsigned short int beta = 0 ; beta < 4 ; beta++)
    for(unsigned short int beta1 = 0 ; beta1 < 4 ; beta1++)
      for(unsigned short int alpha1 = 0 ; alpha1 < 4 ; alpha1++){
	C_temp = Cg5[alpha][beta] * Cg5_bar[beta1][alpha1];
	if( norm(C_temp) > 1e-3 ){
	  Cg5Cg5bar_val[counter] = C_temp;
	  Cg5Cg5bar_ind[counter][0] = alpha;
	  Cg5Cg5bar_ind[counter][1] = beta;
	  Cg5Cg5bar_ind[counter][2] = beta1;
	  Cg5Cg5bar_ind[counter][3] = alpha1;
	  counter++;
	}

      }


// perform contractions
unsigned short int a,b,c,a1,b1,c1;
double2 reg_out[4][3];

for(int gamma = 0 ; gamma < 4 ; gamma++)
  for(int c1 = 0 ; c1 < 3 ; c1++){
    reg_out[gamma][c1].x = 0.;     reg_out[gamma][c1].y = 0.; 
  }

double2 prop1[4][4][3][3];
double2 prop2[4][4][3][3];


switch(index1){
 case 1:


   for(int mu = 0 ; mu < 4 ; mu++)
     for(int nu = 0 ; nu < 4 ; nu++)
       for(int c1 = 0 ; c1 < 3 ; c1++)
	 for(int c2 = 0 ; c2 < 3 ; c2++)
	   prop1[mu][nu][c1][c2] = fetch_double2(uprop3DStochTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * space_stride);
   

   break;
 case 2:

   for(int mu = 0 ; mu < 4 ; mu++)
     for(int nu = 0 ; nu < 4 ; nu++)
       for(int c1 = 0 ; c1 < 3 ; c1++)
	 for(int c2 = 0 ; c2 < 3 ; c2++)
	   prop1[mu][nu][c1][c2] = fetch_double2(dprop3DStochTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * space_stride);

   

   break;
 case 3:

   break;
 }


switch(index2){
 case 1:

   for(int mu = 0 ; mu < 4 ; mu++)
     for(int nu = 0 ; nu < 4 ; nu++)
       for(int c1 = 0 ; c1 < 3 ; c1++)
	 for(int c2 = 0 ; c2 < 3 ; c2++)
	   prop2[mu][nu][c1][c2] = fetch_double2(uprop3DStochTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * space_stride);
   
   break;
 case 2:

   for(int mu = 0 ; mu < 4 ; mu++)
     for(int nu = 0 ; nu < 4 ; nu++)
       for(int c1 = 0 ; c1 < 3 ; c1++)
	 for(int c2 = 0 ; c2 < 3 ; c2++)
	   prop2[mu][nu][c1][c2] = fetch_double2(dprop3DStochTex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * space_stride);
   

   break;
 case 3:

   break;
 }

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
  


for(int cc1 = 0 ; cc1 < 6 ; cc1++){
  a = c_eps[cc1][0];
  b = c_eps[cc1][1];
  c = c_eps[cc1][2];

  for(int cc2 = 0 ; cc2 < 6 ; cc2++){
    a1 = c_eps[cc2][0];
    b1 = c_eps[cc2][1];
    c1 = c_eps[cc2][2];



    for(int idx = 0 ; idx < counter ; idx++){
      
      unsigned short int alpha = Cg5Cg5bar_ind[idx][0];
      unsigned short int beta = Cg5Cg5bar_ind[idx][1];
      unsigned short int beta1 = Cg5Cg5bar_ind[idx][2];
      unsigned short int alpha1 = Cg5Cg5bar_ind[idx][3];
      
      
      for(int gamma = 0 ; gamma < 4 ; gamma++){
	
	
	
	double2 value =  c_sgn_eps[cc1] * c_sgn_eps[cc2] * Cg5Cg5bar_val[idx];
		
	reg_out[gamma][c1] = reg_out[gamma][c1] + value *
	  prop2[beta][beta1][b][b1] * xi[gamma][c] * prop1[alpha][alpha1][a][a1];
	
	//	      reg_out[gamma][gamma1] = reg_out[gamma][gamma1] + value *
	//		prop2[beta][beta1][b][b1] * xi[alpha][a] * prop1[gamma][gamma1][c][c1] * insLine[alpha1][a1];
	
	reg_out[gamma][c1] = reg_out[gamma][c1] - value *
	  prop2[beta][beta1][b][b1] * xi[alpha][a] * prop1[gamma][alpha1][c][a1];
		
	// reg_out[gamma][gamma1] = reg_out[gamma][gamma1] - value *
	//		prop2[beta][beta1][b][b1] * xi[gamma][c] * prop1[alpha][gamma1][a][c1] * insLine[alpha1][a1];
	
	
      }
    }
  }
 }

// copy results to global memory
for(int gamma = 0 ; gamma < 4 ; gamma++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
    out[gamma*3*space_stride + c1*space_stride + sid] = reg_out[gamma][c1];


