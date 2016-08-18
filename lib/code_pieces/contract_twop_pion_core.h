
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

double2 reg_out;
reg_out.x=0.;
reg_out.y=0.;

double2 prop[4][4][3][3];


for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	prop[mu][nu][c1][c2] = fetch_double2(propagatorTexOne,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
      }

for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
    for(int c1 = 0 ;c1 < 3 ; c1++)
      for(int c2 = 0 ;c2 < 3 ; c2++)
	reg_out = reg_out + prop[mu][nu][c1][c2] * conj(prop[mu][nu][c1][c2]);


out[sid] = reg_out;

