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
#define MOM_MAX 1000
// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#include <qudaQKXTM.h> 
#include <mpi.h>


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

extern int src[];
extern int t_sink;
extern int Q_sq;
extern int nsmearAPE;
extern int nsmearGauss;
extern double alphaAPE;
extern double alphaGauss;
extern char twop_filename[];
extern char threep_filename[];
extern double muValue;
extern double kappa;

extern unsigned long int seed;
extern int Nstoch;

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

  
  // start timing
  double tstart,tstop;
  tstart = MPI_Wtime();

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


  inv_param.kappa = kappa;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.mu = muValue;
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
  inv_param.tol = 1e-7;

  inv_param.maxiter = 50000;
  inv_param.reliable_delta = 0.1; // ignored by multi-shift solver



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


  inv_param.verbosity = QUDA_SILENT;

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded

  setDims(gauge_param.X);
  Ls = 1;
  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);


  void *gauge[4];
  void *gauge_APE[4];

  qudaQKXTMinfo info;

  info.nsmearAPE = nsmearAPE;
  info.nsmearGauss = nsmearGauss;
  info.alphaAPE = alphaAPE;
  info.alphaGauss = alphaGauss;
  info.lL[0] = xdim;
  info.lL[1] = ydim;
  info.lL[2] = zdim;
  info.lL[3] = tdim;
  info.sourcePosition[0] = src[0];
  info.sourcePosition[1] = src[1];
  info.sourcePosition[2] = src[2];
  info.sourcePosition[3] = src[3];

  int Nmom = 1;
  int momElem[1][3];
  //  createMom(&Nmom, momElem);
  // if(Nmom > MOM_MAX)errorQuda("Error maximum number in momenta is %d\n",MOM_MAX);
  momElem[0][0] = 0;
  momElem[0][1] = 0;
  momElem[0][2] = 0;

  /*
  int mom[2][3];
  mom[0][0]=0;
  mom[0][1]=0;
  mom[0][2]=0;

  mom[1][0]=1;
  mom[1][1]=0;
  mom[1][2]=0;
  */

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
    applyBoundaryCondition(gauge, V/2 ,&gauge_param);                                                  // apply BC on the field that we will pass to inverter
  } else { // else generate a random SU(3) field
    errorQuda("error read gauge file");
  }


  // initialize the QUDA library
  initQuda(device);


  init_qudaQKXTM(&info);
  printf_qudaQKXTM();

  //  QKXTM_Propagator *uprop = new QKXTM_Propagator(); // allocate memory later
  // QKXTM_Propagator *dprop = new QKXTM_Propagator();

  whatParticle testParticle;
  testParticle = QKXTM_NEUTRON;

  printfQuda("Got source position (x,y,z,t) (%d,%d,%d,%d)\n",src[0],src[1],src[2],src[3]);
  printfQuda("Got tsink position: %d\n",t_sink);
  printfQuda("Got Q_sq=%d which corresponds to %d momenta combinations\n",Q_sq,Nmom);
  printfQuda("Got seed for ranlux: %ld\n",seed);
  printfQuda("Got number of noise vectors: %d\n",Nstoch);

  //  printfQuda("Got nsmearAPE=%d and alphaAPE=%f\n",nsmearAPE,alphaAPE);
  // printfQuda("Got nsmearGauss=%d and alphaGauss=%f\n",nsmearGauss,alphaGauss);

  int NmomSink = 6;
  int momSink[6][3];

  momSink[0][0] = 1;
  momSink[0][1] = 0;
  momSink[0][2] = 0;

  momSink[1][0] = -1;
  momSink[1][1] = 0;
  momSink[1][2] = 0;

  momSink[2][0] = 0;
  momSink[2][1] = 1;
  momSink[2][2] = 0;

  momSink[3][0] = 0;
  momSink[3][1] = -1;
  momSink[3][2] = 0;

  momSink[4][0] = 0;
  momSink[4][1] = 0;
  momSink[4][2] = 1;

  momSink[5][0] = 0;
  momSink[5][1] = 0;
  momSink[5][2] = -1;


  ThpTwp_stoch_WilsonLinks(gauge,gauge_APE,&inv_param,&gauge_param,info.sourcePosition,t_sink,testParticle,twop_filename,threep_filename, Nmom , momElem, seed, Nstoch, NmomSink,momSink);

  //  char filename_out_proton[257] = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_proton.dat";
  // char filename_out_neutron[257] = "/home/khadjiyiannakou/qudaQKXTM_package/bin/test/twop_neutron.dat";

  //  inverter(*uprop,*dprop,gauge,gauge_APE,&inv_param,&gauge_param,info.sourcePosition);
  // performContractions(*uprop, *dprop, 2, mom, filename_out_proton, filename_out_neutron);

  ///////// clear ////////////
  //  delete uprop;
  //delete dprop;


  for(int dir = 0 ; dir < 4 ; dir++){
    free(gauge[dir]);
    free(gauge_APE[dir]);
  }

  // finalize the QUDA library
  endQuda();

  tstop = MPI_Wtime();
  printfQuda("Program time is %f minutes\n",(tstop-tstart)/60.);
  // finalize the communications layer
  endCommsQuda();

  return 0;
}
