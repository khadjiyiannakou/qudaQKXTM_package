#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <misc.h>

#include <face_quda.h>

#ifdef QMP_COMMS
#include <qmp.h>
#endif

#include <gauge_qio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#include <qudaQKXTM.h> 



extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;

extern char latfile[];

extern void usage(char** );

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d \n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     commDimPartitioned(0),
	     commDimPartitioned(1),
	     commDimPartitioned(2),
	     commDimPartitioned(3)); 
  
  return ;
  
}



int main(int argc, char **argv)
{

  using namespace quda;
  
  int i;
  for (i =1;i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION){
    prec_sloppy = prec;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    link_recon_sloppy = link_recon;
  }

  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);

  display_test_info();

  // *** QUDA parameters begin here.



  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH){
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = prec;
  QudaPrecision cuda_prec_sloppy = prec_sloppy;
  QudaPrecision cuda_prec_precondition = QUDA_HALF_PRECISION;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
 
  int myLs = 1; 


  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;
  inv_param.Ls =myLs;

  gauge_param.anisotropy = 1;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_sloppy;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.dslash_type = dslash_type;

  //  double mass = -0.4125;
  inv_param.kappa = 0.161231;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.mu = 0.008500;
    inv_param.twist_flavor = QUDA_TWIST_NO; // change later
  }

  // offsets used only by multi-shift solver

  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.solve_type = QUDA_NORMOP_PC_SOLVE;
    inv_param.inv_type = QUDA_CG_INVERTER;
  } else {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
  }

  inv_param.gcrNkrylov = 10;
  inv_param.tol = 1e-8;

  inv_param.maxiter = 5000;
  inv_param.reliable_delta = 1e-1; // ignored by multi-shift solver



  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  gauge_param.ga_pad = 0; // 24*24*24/2;
  inv_param.sp_pad = 0; // 24*24*24/2;
  inv_param.cl_pad = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif


  inv_param.verbosity = QUDA_VERBOSE;

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded

  setDims(gauge_param.X);
  Ls = 1;
  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);


  void *gauge[4];
  void *gauge_APE[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
    gauge_APE[dir] = malloc(V*gaugeSiteSize*gSize);
    if(gauge[dir] == NULL || gauge_APE[dir] == NULL){
      errorQuda("error allocate memory host gauge field\n");
    }

  }

  
  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    readLimeGauge(gauge, latfile, &gauge_param, &inv_param, gridsize_from_cmdline);        // first read gauge field without apply BC
    for(int dir = 0 ; dir < 4 ; dir++) memcpy( gauge_APE[dir] , gauge[dir] , V*gaugeSiteSize*gSize ); // copy gauge field to APE for smearing
    mapEvenOddToNormalGauge(gauge_APE,gauge_param,xdim,ydim,zdim,tdim);                             // because read conf func put in eo form we map it to normal for smearing
    applyBoundaryCondition(gauge, V/2 ,&gauge_param);                                                 // apply BC on the field that we will pass to inverter
  } else { // else generate a random SU(3) field
    errorQuda("error read gauge file");
  }


  // initialize the QUDA library
  initQuda(device);

  SmearingInfo smearInfo;

  smearInfo.nsmearAPE = 20;
  smearInfo.nsmearGauss = 50;
  smearInfo.alphaAPE = 0.5;
  smearInfo.alphaGauss = 4.0;
  smearInfo.lL[0] = xdim;
  smearInfo.lL[1] = ydim;
  smearInfo.lL[2] = zdim;
  smearInfo.lL[3] = tdim;

  init_qudaQKXTM(&smearInfo);
  printf_qudaQKXTM();

  QKXTM_Propagator *uprop = new QKXTM_Propagator();
  QKXTM_Propagator *dprop = new QKXTM_Propagator();

  char filename_out_up[257] = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/propUp";
  char filename_out_down[257] = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/propDown";

  sprintf(filename_out_up,"%s.%d.dat",filename_out_up,comm_rank());
  sprintf(filename_out_down,"%s.%d.dat",filename_out_down,comm_rank());

  FILE *ptr_out_up, *ptr_out_down;
  ptr_out_up = fopen(filename_out_up,"w");
  ptr_out_down = fopen(filename_out_down,"w");

  if(ptr_out_up == NULL || ptr_out_down == NULL)errorQuda("error cant open file");

  inverter(*uprop,*dprop,gauge,gauge_APE,&inv_param,&gauge_param);
  
  uprop->download();
  dprop->download();

  double *pointer_elem_up = uprop->H_elem();
  double *pointer_elem_down = dprop->H_elem();

  for(int iv = 0 ; iv < V ; iv++)
    for(int mu = 0 ; mu < 4 ; mu++)
      for(int nu = 0 ; nu < 4 ; nu++)
	for(int c1 = 0 ; c1 < 3 ; c1++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    fprintf(ptr_out_up,"%d %d %d %d %d %+e %+e\n",iv,mu,nu,c1,c2,pointer_elem_up[iv*4*4*3*3*2 + mu*4*3*3*2 + nu*3*3*2 + c1*3*2 + c2*2 + 0] , pointer_elem_up[iv*4*4*3*3*2 + mu*4*3*3*2 + nu*3*3*2 + c1*3*2 + c2*2 + 1]);
	    fprintf(ptr_out_down,"%d %d %d %d %d %+e %+e\n",iv,mu,nu,c1,c2,pointer_elem_down[iv*4*4*3*3*2 + mu*4*3*3*2 + nu*3*3*2 + c1*3*2 + c2*2 + 0] , pointer_elem_down[iv*4*4*3*3*2 + mu*4*3*3*2 + nu*3*3*2 + c1*3*2 + c2*2 + 1]);
      }

  ///////// clear ////////////
  delete uprop;
  delete dprop;
  for(int dir = 0 ; dir < 4 ; dir++){
    free(gauge[dir]);
    free(gauge_APE[dir]);
  }

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  endCommsQuda();

  return 0;
}
