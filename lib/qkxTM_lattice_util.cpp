#include <qkxTM.h>
#include <lattice_util.h>
#include <blas_qkxTM.h>

#include <hopping_term.h>

using namespace qkxTM;

LatticeField::LatticeField(LatticeInfo *param)
  : nspins(NSPINS) , ncolors(NCOLORS), ndf(NDF),evenOdd_order(false)
{

  volume = param->L[0] * param->L[1] * param->L[2] * param->L[3];
  volumeCB = volume/2;
  surface[0] = param->L[1] * param->L[2] * param->L[3];
  surface[1] = param->L[0] * param->L[2] * param->L[3];
  surface[2] = param->L[0] * param->L[1] * param->L[3];
  surface[3] = param->L[0] * param->L[1] * param->L[2];
  for(int i = 0 ; i < NDIM ; i++) surfaceCB[i] = surface[i]/2;
  totalBytes = volume*nspins*ncolors*sizeof(Complex);
  for(int i = 0 ; i < NDIM ; i++) latticeLength[i] = param->L[i];

}

void LatticeField::getNormalCoords(const int latticePoint, int *x) const
{
  // 0 -> x
  // 1 -> y
  // 2 -> z
  // 3 -> t

  int z1, z2;
  z1 = latticePoint / latticeLength[0];
  x[0] = latticePoint - z1 * latticeLength[0];
  z2 = z1 / latticeLength[1];
  x[1] = z1 - z2 * latticeLength[1];
  x[3] = z2 / latticeLength[2];
  x[2] = z2 - x[3] * latticeLength[2];
}

void LatticeField::getFwdCoords(const int *x, int *fwd) const
{
  int NX = latticeLength[0];
  int NY = latticeLength[1];
  int NZ = latticeLength[2];
  //  int NT = latticeLength[3];
  int NXNY = NX*NY;
  int NXNYNZ = NX*NY*NZ;

  for(int dir = 0; dir < NDIM; dir++)
    {
      fwd[dir] = x[dir] + 1;
      if(fwd[dir] == latticeLength[dir])//Boundary check   
	fwd[dir] -= latticeLength[dir];
    }
  fwd[0] = fwd[0] + NX * x[1] + NXNY * x[2] + NXNYNZ * x[3];
  fwd[1] = x[0] + NX * fwd[1] + NXNY * x[2] + NXNYNZ * x[3];
  fwd[2] = x[0] + NX * x[1] + NXNY * fwd[2] + NXNYNZ * x[3];
  fwd[3] = x[0] + NX * x[1] + NXNY * x[2] + NXNYNZ * fwd[3];

}

void LatticeField::getBwdCoords(const int *x, int *bwd) const
{
  int NX = latticeLength[0];
  int NY = latticeLength[1];
  int NZ = latticeLength[2];
  //  int NT = latticeLength[3];
  int NXNY = NX*NY;
  int NXNYNZ = NX*NY*NZ;

  for(int dir = 0; dir < NDIM; dir++)
    {
      bwd[dir] = x[dir] - 1;
      if (x[dir] == 0)            //Boundary check   
	bwd[dir] += latticeLength[dir];
    }
  bwd[0] = bwd[0] + NX * x[1] + NXNY * x[2] + NXNYNZ * x[3];
  bwd[1] = x[0] + NX * bwd[1] + NXNY * x[2] + NXNYNZ * x[3];
  bwd[2] = x[0] + NX * x[1] + NXNY * bwd[2] + NXNYNZ * x[3];
  bwd[3] = x[0] + NX * x[1] + NXNY * x[2] + NXNYNZ * bwd[3];

}

void LatticeField::getFwdBwdCoords(const int *x, int fwdBwd[NDIM][NDIM])const
{

  int fwd[NDIM];
  int y[NDIM*NDIM];  // nothing to do with x,y,z coordinates
  int z[NDIM*NDIM];
  this->getFwdCoords(x,fwd);
  for(int i = 0 ; i < NDIM ; i++)
    this->getNormalCoords(fwd[i],&(y[i*4]));
  for(int i = 0 ; i < NDIM ; i++)
    this->getBwdCoords(&(y[i*4]),&(z[i*4]));

  for(int i = 0 ; i < NDIM ; i++)
    for(int j = 0 ; j < NDIM ; j++)
      fwdBwd[i][j] = z[i*4+j];

}

void LatticeField::getBwdFwdCoords(const int *x, int bwdFwd[NDIM][NDIM])const
{

  int bwd[NDIM];
  int y[NDIM*NDIM];  // nothing to do with x,y,z coordinates
  int z[NDIM*NDIM];

  this->getBwdCoords(x,bwd);
  for(int i = 0 ; i < NDIM ; i++)
    this->getNormalCoords(bwd[i],&(y[i*4]));
  for(int i = 0 ; i < NDIM ; i++)
    this->getFwdCoords(&(y[i*4]),&(z[i*4]));

  for(int i = 0 ; i < NDIM ; i++)
    for(int j = 0 ; j < NDIM ; j++)
      bwdFwd[i][j] = z[i*4+j];

}





GaugeField::GaugeField(LatticeInfo *param):
  LatticeField(param),p_gauge(NULL)
{
  totalBytes = volume*ncolors*ncolors*NDIM*sizeof(Complex);
}

void GaugeField::create()
{
  Complex** pointer; // number of direction are 4
  
  pointer = (Complex**)malloc(NDIM*sizeof(Complex*));

  if(pointer == NULL){
    fprintf(stderr,"Error with memory allocation of gauge field\n");
    exit(EXIT_FAILURE);
  }

  for( int i = 0 ; i < NDIM ; i++){
    pointer[i] = (Complex*)malloc(volume*ncolors*ncolors*sizeof(Complex));
    if( pointer[i] == NULL ){
      fprintf(stderr,"Error with memory allocation of gauge field\n");
      exit(EXIT_FAILURE);
    }
  }

  p_gauge = pointer;
}

GaugeField::~GaugeField(){
  for( int i = 0 ; i < NDIM ; i++)
    free(p_gauge[i]);
}

void GaugeField::copy(GaugeField &src){

  
  Complex **dst_p;
  Complex **src_p;

  dst_p = this->p_gauge;
  src_p = src.p_gauge;

    for(int i = 0 ; i < NDIM ; i++)
      memcpy(&(dst_p[i][0]),&(src_p[i][0]),volume*ncolors*ncolors*sizeof(Complex));

}

void GaugeField::zero(){

  for(int i = 0 ; i < NDIM ; i++)
    memset( &(this->p_gauge[i][0]) , 0 ,volume*ncolors*ncolors*sizeof(Complex));

}

void GaugeField::applyBoundary(LatticeInfo *latInfo){
  int sign = latInfo->boundaryT;
  int nx = latInfo->L[0];
  int ny = latInfo->L[1];
  int nz = latInfo->L[2];
  int nt = latInfo->L[3];

  Complex **gauge = this->p_gauge;

  if(sign == -1){
    
    for(int iv = nx*ny*nz*(nt-1) ; iv < nx*ny*nz*nt ; iv++)
      for(int c1 = 0 ; c1 < ncolors ; c1++)
	for(int c2 = 0 ; c2 < ncolors ; c2++){
	  gauge[3][MG(iv,c1,c2)].real() *= -1;
	  gauge[3][MG(iv,c1,c2)].imag() *= -1;
      }
  }

}

void GaugeField::unitField(){

  for(int mu = 0 ; mu < NDIM ; mu++)
    for(int iv = 0 ; iv < volume ; iv++)
      for(int c1 = 0 ; c1 < ncolors ; c1++)
	for(int c2 = 0 ; c2 < ncolors ; c2++){
	  p_gauge[mu][MG(iv,c1,c2)].real() = (c1 == c2) ? 1. : 0;
	  p_gauge[mu][MG(iv,c1,c2)].imag() = 0.;
	}

}

double GaugeField::calculatePlaquette()
{
  double plaquette;
  Complex temp;
  temp.real() = 0.;
  temp.imag() = 0.;

  Complex test;

  

  int x[NDIM];
  int fwd[NDIM];

  SU3 *A = new SU3();
  SU3 *B = new SU3();
  SU3 *C_dag = new SU3();
  SU3 *D_dag = new SU3();
  SU3 *E = new SU3();

  FILE *ptr_trace;
  ptr_trace=fopen("trace.dat","w");
  
  for(int iv = 0 ; iv < volume ; iv++){     // summation over the volume
    test.real()=0.;test.imag()=0.;
    for(int mu = 0 ; mu < 3 ; mu++)        // this two for loops implement the constrain \Sum_{nu>mu}
      for(int nu = mu+1 ; nu < 4 ; nu++){
	this->getNormalCoords(iv,x);
	this->getFwdCoords(x,fwd);

	/*	if(iv == 0 && mu == 0 && nu ==1){
	  printf("forward[0] is %d\n",fwd[0]);
	  for(int c1 = 0 ; c1 < 3 ; c1++)
	    for(int c2 = 0 ; c2 < 3; c2++)
	      printf("%e %e\n",p_gauge[nu][MG(fwd[mu],c1,c2)].real(),p_gauge[nu][MG(fwd[mu],c1,c2)].imag());
	      }*/

	memcpy(A->M,&(p_gauge[mu][MG(iv,0,0)]),9*sizeof(Complex));
	memcpy(B->M,&(p_gauge[nu][MG(fwd[mu],0,0)]),9*sizeof(Complex));
	memcpy(C_dag->M,&(p_gauge[mu][MG(fwd[nu],0,0)]),9*sizeof(Complex));
	memcpy(D_dag->M,&(p_gauge[nu][MG(iv,0,0)]),9*sizeof(Complex));
	C_dag->dagger();
	D_dag->dagger();
	*E = (*A)*(*B)*(*C_dag)*(*D_dag);
	temp += E->traceColor();
	test += E->traceColor();
      }
    fprintf(ptr_trace,"%d %d %d %d %d %e\n",iv,x[0],x[1],x[2],x[3],test.real());;
  }
  plaquette = (temp.real() ) / (ncolors*6*volume);    // 6 comes from the compinations of mu and nu that we have on the for loop  

  delete A;
  delete B;
  delete C_dag;
  delete D_dag;
  delete E;

  return plaquette;
}

void GaugeField::APEsmearing(Complex **U_tmp, Complex **U_ape, LatticeInfo *latInfo){

  
  SU3 *A = new SU3();
  SU3 *A_dag = new SU3();
  SU3 *B = new SU3();
  SU3 *C = new SU3();
  SU3 *C_dag = new SU3();
  SU3 *D = new SU3();
  SU3 *E = new SU3();
  SU3 *F = new SU3();

  int x[4] , fwdBwd[4][4] , fwd[4] , bwd[4];
  int nsmear = latInfo->NsmearAPE;
  double alpha = latInfo->alphaAPE;

  Complex **in = NULL;
  Complex **out = NULL;
  Complex **tmp = NULL;

  if(nsmear == 0) return;
  in = U_tmp;
  out = U_ape;

  for(int iter = 0 ; iter < nsmear ; iter++){
    fprintf(stdout,"smearing iteration %d\n",iter);
    for(int iv = 0 ; iv < volume ; iv++){
      for(int mu = 0 ; mu < 3 ; mu++) {       // not mu == 4
	D->zero();
	for(int nu = 0 ; nu < 3 ; nu++){
	  if(mu != nu){
	    this->getNormalCoords(iv,x);
	    this->getFwdCoords(x,fwd);
	    this->getBwdCoords(x,bwd);
	    this->getFwdBwdCoords(x,fwdBwd);

	    // forward staple
	    memcpy(A->M,&(in[nu][MG(iv,0,0)]),9*sizeof(Complex));
	    memcpy(B->M,&(in[mu][MG(fwd[nu],0,0)]),9*sizeof(Complex));
	    memcpy(C_dag->M,&(in[nu][MG(fwd[mu],0,0)]),9*sizeof(Complex));
	    C_dag->dagger();
	    *D = *D + (*A)*(*B)*(*C_dag);                          // U_\nu(x) * U_\mu(x+\nu) * ( U_\nu(x+\mu) )^dagger


	    // backward stable
	    memcpy(A_dag->M,&(in[nu][MG(bwd[nu],0,0)]),9*sizeof(Complex));
	    memcpy(B->M,&(in[mu][MG(bwd[nu],0,0)]),9*sizeof(Complex));
	    memcpy(C->M,&(in[nu][MG(fwdBwd[mu][nu],0,0)]),9*sizeof(Complex));
	    A_dag->dagger();
	    *D = *D + (*A_dag)*(*B)*(*C);                               // ( U_\nu(x-\nu))^dagger * U_\mu(x-\nu) * U_\nu(x+\mu-\nu)
	   
	    /*
	    *D = (*(A->dagger()))*(*B)*(*C);                               // ( U_\nu(x-\nu))^dagger * U_\mu(x-\nu) * U_\nu(x+\mu-\nu)
	    for(int c1 = 0 ; c1 < 3 ; c1++)
	      for(int c2 = 0 ; c2 < 3 ; c2++)
		printf("%+e %+e\n",D->M[MU(c1,c2)].real(),D->M[MU(c1,c2)].imag());
	    exit(-1);
	    */
	    
	  } // if
	} // for nu
	memcpy(E->M,&(in[mu][MG(iv,0,0)]),9*sizeof(Complex));
	*F = *E + (*D)*alpha;
	memcpy(&(out[mu][MG(iv,0,0)]) ,F->M , 9*sizeof(Complex) );
      } // for mu
    } // close volume

    projectSU3(out,latInfo);  
    tmp=in;
    in=out;
    out=tmp;

    
  }  // close iteration

    if( (nsmear%2) == 0){
      for(int i = 0 ; i < 3 ; i++)
	memcpy(U_ape[i],in[i],volume*ncolors*ncolors*sizeof(Complex));
    }
  
    memcpy(U_ape[3],U_tmp[3],volume*ncolors*ncolors*sizeof(Complex));



  delete A;
  delete A_dag;
  delete B;
  delete C;
  delete C_dag;
  delete D;
  delete E;
  delete F;

}

ColorSpinorField::ColorSpinorField(LatticeInfo *param):
  LatticeField(param),p_colorSpinor(NULL)
{
  ;
}


void ColorSpinorField::create()
{
  Complex** pointer; // number of direction are 4
  
  pointer = (Complex**)malloc(nspins*ncolors*sizeof(Complex*));

  if(pointer == NULL){
    fprintf(stderr,"Error with memory allocation of spinor field\n");
    exit(EXIT_FAILURE);
  }

  for( int i = 0 ; i < nspins*ncolors ; i++){
    pointer[i] = (Complex*)malloc(volume*sizeof(Complex));
    if( pointer[i] == NULL ){
      fprintf(stderr,"Error with memory allocation of gauge field\n");
      exit(EXIT_FAILURE);
    }
  }

  p_colorSpinor = pointer;
}

ColorSpinorField::~ColorSpinorField(){
  for( int i = 0 ; i < nspins*ncolors ; i++)
    free(p_colorSpinor[i]);
}

void ColorSpinorField::copy(ColorSpinorField &src){

  
  Complex **dst_p;
  Complex **src_p;

  dst_p = this->p_colorSpinor;
  src_p = src.p_colorSpinor;

  for(int mu = 0 ; mu < nspins ; mu++)
    for(int ic = 0 ; ic < ncolors ; ic++)
      memcpy(&(dst_p[MS(mu,ic)][0]),&(src_p[MS(mu,ic)][0]) , volume*sizeof(Complex) );

}

void ColorSpinorField::zero(){

  for(int mu = 0 ; mu < nspins ; mu++)
    for(int ic = 0 ; ic < ncolors ; ic++)
      memset(&(this->p_colorSpinor[MS(mu,ic)][0]), 0 , volume*sizeof(Complex) );

}

void ColorSpinorField::gaussianSmearing(Complex **Psi_tmp, Complex **Psi_sm, Complex **gauge , LatticeInfo *latInfo)
{
  int nsmear = latInfo->NsmearGauss;
  int alpha = latInfo->alphaGauss;
  double normalize = 1./(1 + 6*alpha);
  int x[4] , fwd[4], bwd[4];


  SU3 *U = new SU3();
  SU3 *U_back = new SU3();
  spinColor *psi_xpmu = new spinColor();
  spinColor *psi_xmmu = new spinColor();
  spinColor *psi_sum = new spinColor();
  spinColor *psi = new spinColor();

  Complex **in = NULL;
  Complex **out = NULL;
  Complex **tmp = NULL;

  if(nsmear == 0) return;
  in = Psi_tmp;
  out = Psi_sm;

  for(int iter = 0 ; iter < nsmear ; iter++){
    fprintf(stdout,"gauss smearing iteration %d\n",iter);
    for(int iv = 0 ; iv < volume ; iv++){
      psi_sum->zero();      
      for(int mu = 0 ; mu < 3 ; mu++){
	
	this->getNormalCoords(iv,x);
	this->getFwdCoords(x,fwd);
	this->getBwdCoords(x,bwd);
	
	for(int nu = 0 ; nu < nspins ; nu++)
	  for(int ic = 0 ; ic < ncolors ; ic++){
	    psi->M[MS(nu,ic)] = in[MS(nu,ic)][iv];
	    psi_xpmu->M[MS(nu,ic)] = in[MS(nu,ic)][fwd[mu]];
	    psi_xmmu->M[MS(nu,ic)] = in[MS(nu,ic)][bwd[mu]];
	}

	memcpy(U->M, &(gauge[mu][MG(iv,0,0)]), 9*sizeof(Complex));
	memcpy(U_back->M, &(gauge[mu][MG(bwd[mu],0,0)]), 9*sizeof(Complex));
	U_back->dagger();

	(*psi_sum) = (*psi_sum) + (*psi_xpmu) * (*U) + (*psi_xmmu) * (*U_back);

      } // close mu

      (*psi_sum) = (*psi_sum) * (alpha*normalize);
      (*psi) = (*psi) * normalize + (*psi_sum);

	for(int nu = 0 ; nu < nspins ; nu++)
	  for(int ic = 0 ; ic < ncolors ; ic++){
	    out[MS(nu,ic)][iv] = psi->M[MS(nu,ic)];
	  }

    } // close volume
    tmp = in;
    in = out;
    out = tmp;
  }               // close iteration

  if( (nsmear%2) == 0){
  for( int i = 0 ; i < nspins*ncolors ; i++)
    memcpy(Psi_sm[i],in[i], volume*sizeof(Complex) );
  }
  
  delete U;
  delete U_back;
  delete psi;
  delete psi_xpmu;
  delete psi_xmmu;
  delete psi_sum;

}



void ColorSpinorField::applyDslash(Complex **Psi, Complex **gauge){

  int x[NDIM], fwd[NDIM], bwd[NDIM];
  
  spinColor *phi = new spinColor();
  spinColor *R = new spinColor();
  spinColor *xi = new spinColor();

  for(int iv = 0 ; iv < volume ; iv++){
    
    getNormalCoords(iv,x);
    getFwdCoords(x,fwd);                                                                                                                   
    getBwdCoords(x,bwd);

    ZERO(R);
    //plus X direction
    PROJ_MINUS_X(phi,Psi);
    APPLY_LINK(xi,gauge,phi,0);
    COLLECT_MINUS_X(R,xi);
    //minus X direction
    PROJ_PLUS_X(phi,Psi);
    APPLY_LINK_DAG(xi,gauge,phi,0);
    COLLECT_PLUS_X(R,xi);
    //plus Y direction
    PROJ_MINUS_Y(phi,Psi);
    APPLY_LINK(xi,gauge,phi,1);
    COLLECT_MINUS_Y(R,xi);
    //minus Y direction
    PROJ_PLUS_Y(phi,Psi);
    APPLY_LINK_DAG(xi,gauge,phi,1);
    COLLECT_PLUS_Y(R,xi);
    //plus Z direction
    PROJ_MINUS_Z(phi,Psi);
    APPLY_LINK(xi,gauge,phi,2);
    COLLECT_MINUS_Z(R,xi);
    //minus Z direction
    PROJ_PLUS_Z(phi,Psi);
    APPLY_LINK_DAG(xi,gauge,phi,2);
    COLLECT_PLUS_Z(R,xi);
    //plus T direction
    PROJ_MINUS_T(phi,Psi);
    APPLY_LINK(xi,gauge,phi,3);
    COLLECT_MINUS_T(R,xi);
    //minus T direction
    PROJ_PLUS_T(phi,Psi);
    APPLY_LINK_DAG(xi,gauge,phi,3);
    COLLECT_PLUS_T(R,xi);
    ///////////////////////////////////////////////
    // save data to memory
    for(int mu = 0 ; mu < nspins ; mu++)
      for(int ic = 0 ; ic < ncolors ; ic++)
	this->p_colorSpinor[MS(mu,ic)][iv] = R->M[MS(mu,ic)];

  } // close volume loop

  delete phi;
  delete R;
  delete xi;
}

void ColorSpinorField::applyTwistAddDslash(Complex **Dslash, Complex **Psi, LatticeInfo *latInfo){

  spinColor *phi = new spinColor();
  Complex twistCoeff;
  
  twistCoeff.real() = 0.;
  twistCoeff.imag() = 2*(latInfo->kappa)*(latInfo->mu)*(latInfo->twistSign);

  for(int iv = 0 ; iv < volume; iv++){
    
    for(int ic = 0 ; ic < ncolors ; ic++){
      phi->M[MS(0,ic)] = Psi[MS(2,ic)][iv];
      phi->M[MS(1,ic)] = Psi[MS(3,ic)][iv];
      phi->M[MS(2,ic)] = Psi[MS(0,ic)][iv];
      phi->M[MS(3,ic)] = Psi[MS(1,ic)][iv];
    } // gamma_5 rotation

    for(int mu = 0 ; mu < nspins ; mu++)
      for(int ic = 0 ; ic < ncolors ; ic++){
	this->p_colorSpinor[MS(mu,ic)][iv] = (-latInfo->kappa)*Dslash[MS(mu,ic)][iv] + Psi[MS(mu,ic)][iv] + twistCoeff*phi->M[MS(mu,ic)];
      }

  } // close volume loop
  delete phi;
}


void ColorSpinorField::applyDslashDag(Complex **Psi, Complex **gauge){

  int x[NDIM], fwd[NDIM], bwd[NDIM];
  
  spinColor *phi = new spinColor();
  spinColor *R = new spinColor();
  spinColor *xi = new spinColor();

  for(int iv = 0 ; iv < volume ; iv++){
    
    getNormalCoords(iv,x);
    getFwdCoords(x,fwd);                                                                                                                   
    getBwdCoords(x,bwd);

    ZERO(R);

    //plus X direction
    DAG_PROJ_PLUS_X(phi,Psi);
    APPLY_LINK(xi,gauge,phi,0);
    COLLECT_PLUS_X(R,xi);
    //minus X direction
    DAG_PROJ_MINUS_X(phi,Psi);
    APPLY_LINK_DAG(xi,gauge,phi,0);
    COLLECT_MINUS_X(R,xi);
    //plus Y direction
    DAG_PROJ_PLUS_Y(phi,Psi);
    APPLY_LINK(xi,gauge,phi,1);
    COLLECT_PLUS_Y(R,xi);
    //minus Y direction
    DAG_PROJ_MINUS_Y(phi,Psi);
    APPLY_LINK_DAG(xi,gauge,phi,1);
    COLLECT_MINUS_Y(R,xi);
    //plus Z direction
    DAG_PROJ_PLUS_Z(phi,Psi);
    APPLY_LINK(xi,gauge,phi,2);
    COLLECT_PLUS_Z(R,xi);
    //minus Z direction
    DAG_PROJ_MINUS_Z(phi,Psi);
    APPLY_LINK_DAG(xi,gauge,phi,2);
    COLLECT_MINUS_Z(R,xi);
    //plus T direction
    DAG_PROJ_PLUS_T(phi,Psi);
    APPLY_LINK(xi,gauge,phi,3);
    COLLECT_PLUS_T(R,xi);
    //minus T direction
    DAG_PROJ_MINUS_T(phi,Psi);
    APPLY_LINK_DAG(xi,gauge,phi,3);
    COLLECT_MINUS_T(R,xi);

    ///////////////////////////////////////////////
    // save data to memory
    for(int mu = 0 ; mu < nspins ; mu++)
      for(int ic = 0 ; ic < ncolors ; ic++)
	this->p_colorSpinor[MS(mu,ic)][iv] = R->M[MS(mu,ic)];

  } // close volume loop

  delete phi;
  delete R;
  delete xi;
}

void ColorSpinorField::applyTwistDagAddDslashDag(Complex **DslashDag, Complex **Psi, LatticeInfo *latInfo){

  spinColor *phi = new spinColor();
  Complex twistCoeff;
  
  twistCoeff.real() = 0.;
  twistCoeff.imag() = -2*(latInfo->kappa)*(latInfo->mu)*(latInfo->twistSign);

  for(int iv = 0 ; iv < volume; iv++){
    
    for(int ic = 0 ; ic < ncolors ; ic++){
      phi->M[MS(0,ic)] = Psi[MS(2,ic)][iv];
      phi->M[MS(1,ic)] = Psi[MS(3,ic)][iv];
      phi->M[MS(2,ic)] = Psi[MS(0,ic)][iv];
      phi->M[MS(3,ic)] = Psi[MS(1,ic)][iv];
    } // gamma_5 rotation

    for(int mu = 0 ; mu < nspins ; mu++)
      for(int ic = 0 ; ic < ncolors ; ic++){
	this->p_colorSpinor[MS(mu,ic)][iv] = (-latInfo->kappa)*DslashDag[MS(mu,ic)][iv] + Psi[MS(mu,ic)][iv] + twistCoeff*phi->M[MS(mu,ic)];
      }

  } // close volume loop
  delete phi;
}


void ColorSpinorField::MdagM(Complex **vec,Complex **gauge, LatticeInfo *param){
  
  ColorSpinorField *tmp = new ColorSpinorField(param);
  tmp->create();
  this->applyDslash(vec,gauge);
  tmp->applyTwistAddDslash(this->P_colorSpinor(),vec,param);
  this->applyDslashDag(tmp->P_colorSpinor(),gauge);
  this->applyTwistDagAddDslashDag(this->P_colorSpinor(),tmp->P_colorSpinor(),param);

  delete tmp;
}
