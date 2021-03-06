#include <comm_quda.h>
#include <quda_internal.h>
#include <quda.h>
#include <iostream>
#include <complex>
#include <cuda.h>
#include <color_spinor_field.h>

#ifndef _QUDAQKXTM_H
#define _QUDAQKXTM_H


#define QUDAQKXTM_DIM 4

#define LEXIC(it,iz,iy,ix,L) ( (it)*L[0]*L[1]*L[2] + (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_TZY(it,iz,iy,L) ( (it)*L[1]*L[2] + (iz)*L[1] + (iy) )
#define LEXIC_TZX(it,iz,ix,L) ( (it)*L[0]*L[2] + (iz)*L[0] + (ix) )
#define LEXIC_TYX(it,iy,ix,L) ( (it)*L[0]*L[1] + (iy)*L[0] + (ix) )
#define LEXIC_ZYX(iz,iy,ix,L) ( (iz)*L[0]*L[1] + (iy)*L[0] + (ix) )


//#ifndef __COMPLEX__KRIKITOS

//#define __COMPLEX__KRIKITOS
//#endif


namespace quda {

  //  typedef std::complex<double> Complex;

  typedef struct {
    int nsmearAPE;
    int nsmearGauss;
    double alphaAPE;
    double alphaGauss;
    int lL[QUDAQKXTM_DIM];
    int sourcePosition[QUDAQKXTM_DIM];
    int nsmearHYP;
    double omega1HYP;
    double omega2HYP;
  } qudaQKXTMinfo;

  enum whatParticle { QKXTM_PROTON, QKXTM_NEUTRON , QKXTM_BOTH };
  enum whatProjector { QKXTM_TYPE1, QKXTM_TYPE2, QKXTM_PROJ_G5G1, QKXTM_PROJ_G5G2, QKXTM_PROJ_G5G3 };
  enum NgaugeHost {QKXTM_N1 , QKXTM_N2};
  // forward declaration
  class QKXTM_Field;
  class QKXTM_Gauge;
  class QKXTM_Vector;
  class QKXTM_Propagator;
  class QKXTM_Correlator;
  class QKXTM_Propagator3D;
  class QKXTM_Vector3D;
  //////////////////////////////////////// functions /////////////////////////////////////////////

  void init_qudaQKXTM(qudaQKXTMinfo *info);
  void printf_qudaQKXTM();
  void APE_smearing(QKXTM_Gauge &gaugeAPE , QKXTM_Gauge &gaugeTmp);
  void APE_smearing(QKXTM_Gauge &gaugeAPE , QKXTM_Gauge &gaugeTmp, QKXTM_Propagator &prp);
  void Gaussian_smearing(QKXTM_Vector &vectorGauss , QKXTM_Vector &vectorTmp , QKXTM_Gauge &gaugeAPE);
  void inverter(QKXTM_Propagator &uprop, QKXTM_Propagator &dprop,void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param , int *sourcePosition);
  void invert_Vector_tmLQCD(QKXTM_Vector &vec ,void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param , int *sourcePosition);
  void ThpTwp(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param, int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename , char *threep_filename, int Nmom , int momElem[][3]);
  int comm_localRank();
  void corrProton(QKXTM_Propagator &uprop, QKXTM_Propagator &dprop ,QKXTM_Correlator &corr );
  void corrNeutron(QKXTM_Propagator &uprop, QKXTM_Propagator &dprop ,QKXTM_Correlator &corr );

  void performContractions(QKXTM_Propagator &uprop, QKXTM_Propagator &dprop , int Nmom, int momElem[][3] , char *filenameProton , char *filenameNeutron);

  void seqSourceFixSinkPart1(QKXTM_Vector &vec, QKXTM_Propagator3D &prop1, QKXTM_Propagator3D &prop2, int timeslice,int nu,int c2, whatProjector typeProj, whatParticle testParticle);

  //new
  //  void seqSourceFixSinkPion(QKXTM_Vector &vec, QKXTM_Propagator3D &prop,int timeslice,int nu,int c2);

  void seqSourceFixSinkPart2(QKXTM_Vector &vec, QKXTM_Propagator3D &prop1,  int timeslice,int nu,int c2, whatProjector typeProj, whatParticle testParticle);

  void fixSinkContractions(QKXTM_Propagator &seqProp, QKXTM_Propagator &prop ,QKXTM_Gauge &gauge ,whatProjector typeProj , char *filename , int Nmom , int momElem[][3], whatParticle testParticle, int partFlag );

  void fixSinkFourier(double *corr,double *corrMom, int Nmom , int momElem[][3]);
  void insLineFourier(double *insLineMom , double *insLine, int Nmom, int momElem[][3]);

  void ThpTwp_Pion(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename, char *threep_filename , int Nmom , int momElem[][3]);

  void ThpTwp_stoch(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param, int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename , char *threep_filename, int Nmom , int momElem[][3], unsigned long int seed, int Nstoch);



  void threepStochUpart( QKXTM_Vector &phi , QKXTM_Vector3D &xi ,QKXTM_Propagator &uprop , QKXTM_Propagator3D &uprop3D , QKXTM_Propagator3D &dprop3D, QKXTM_Gauge &gauge, int fixTime , char *filename,  int Nmom , int momElem[][3]);

  void threepStochDpart( QKXTM_Vector &phi , QKXTM_Vector3D &xi ,QKXTM_Propagator &dprop , QKXTM_Propagator3D &uprop3D , QKXTM_Propagator3D &dprop3D, QKXTM_Gauge &gauge, int fixTime ,char *filename,  int Nmom , int momElem[][3]);

  double* createWilsonPath(QKXTM_Gauge &gauge,int direction);
  double* createWilsonPathBwd(QKXTM_Gauge &gauge,int direction);

  double* createWilsonPath(QKXTM_Gauge &gauge);

  void ThpTwp_stoch_WilsonLinks(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param, int *sourcePosition, int fixSinkTime, whatParticle testParticle, char *twop_filename , char *threep_filename, int Nmom , int momElem[][3], unsigned long int seed, int Nstoch, int NmomSink, int momSink[][3]);

  void threepStochUpart_WilsonLinks( QKXTM_Vector &phi , QKXTM_Vector3D &xi ,QKXTM_Propagator &uprop , QKXTM_Propagator3D &uprop3D , QKXTM_Propagator3D &dprop3D, double* deviceWilsonPath, int fixTime , char *filename,  int Nmom , int momElem[][3], int NmomSink,int momSink[][3]);

  void threepStochDpart_WilsonLinks( QKXTM_Vector &phi , QKXTM_Vector3D &xi ,QKXTM_Propagator &dprop , QKXTM_Propagator3D &uprop3D , QKXTM_Propagator3D &dprop3D, double* deviceWilsonPath, int fixTime ,char *filename,  int Nmom , int momElem[][3], int NmomSink, int momSink[][3]);

  void check_wilson_links(void **gauge, void **gaugeAPE);


  void ThpTwp_nonLocal(void**, void**, QudaInvertParam*, QudaGaugeParam*, int*, int, quda::whatParticle, char*,char*, int, int (*)[3] ,int ,int ,int [][3] );
  void ThpTwp_Pion_nonLocal(void**, void**, QudaInvertParam*, QudaGaugeParam*, int*, int, quda::whatParticle, char*,char*, int, int (*)[3] ,int ,int ,int [][3] );
  
  void fixSinkContractions_nonLocal(QKXTM_Propagator &seqProp, QKXTM_Propagator &prop ,QKXTM_Gauge &gauge ,whatProjector typeProj , char *filename , int Nmom , int momElem[][3], whatParticle testParticle, int partFlag, double *deviceWilsonPath, int direction );

  void fixSinkContractions_nonLocal(QKXTM_Propagator &seqProp, QKXTM_Propagator &prop ,QKXTM_Gauge &gauge ,whatProjector typeProj , char *filename , int Nmom , int momElem[][3], whatParticle testParticle, int partFlag, double *deviceWilsonPath, double *deviceWilsonPathBwd, int direction );

  //new
  //  void fixSinkContractionsPion_nonLocal(QKXTM_Propagator &seqProp, QKXTM_Propagator &prop ,QKXTM_Gauge &gauge ,whatProjector typeProj , char *filename , int Nmom , int momElem[][3], double *deviceWilsonPath, int direction );

  void HYP3D_smearing(QKXTM_Gauge &gaugeHYP , QKXTM_Gauge &gaugeTmp, QKXTM_Propagator &prp1, QKXTM_Propagator &prp2);

  void HYP3D(void **gaugeHYP);

  void invertWriteProps_SS(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,int *sourcePosition, char *prop_path);

  void invertWritePropsNoApe_SS(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,int *sourcePosition, char *prop_path);
  void invertWritePropsNoApe_SL(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,int *sourcePosition, char *prop_path);

  void corrPion(QKXTM_Propagator &prop, double *corr);
  void performContractionsPion(QKXTM_Propagator &prop, int Nmom, int momElem[][3] , char *filenamePion);
  void threepStochPion_WilsonLinks( QKXTM_Vector &dphi , QKXTM_Vector3D &xi ,QKXTM_Propagator &uprop , QKXTM_Propagator3D &uprop3D , double* deviceWilsonLinks, int fixTime , char *filename ,int Nmom , int momElem[][3], int NmomSink, int momSink[][3]);
  void ThpTwp_stoch_Pion_WilsonLinks(void **gauge, void **gaugeAPE ,QudaInvertParam *inv_param , QudaGaugeParam *gauge_param,int *sourcePosition, int fixSinkTime, char *twop_filename,char *threep_filename , int Nmom , int momElem[][3], unsigned long int seed , int Nstoch,int NmomSink,int momSink[][3]);
  void UxMomentumPhase(QKXTM_Gauge &gaugeAPE, int px, int py, int pz, double zeta);
  ///////////////////////////////////////////////////// class QKXTM_Field ////////////////////////////////////////////
  
  class QKXTM_Field {           // base class use only for inheritance not polymorphism

  private:
    bool field_binded;

  protected:

    int field_length;
    int total_length;         // total length of the gauge including ghost zone
    int ghost_length;        // length of the ghost in all direction
    size_t bytes_total_length;
    size_t bytes_ghost_length;

    double *h_elem;
    double *d_elem;
    double *h_ext_ghost;
    double *d_ext_ghost;

    void create_host();
    void create_host_ghost();
    void destroy_host();
    void destroy_host_ghost();
    void create_device();
    void create_device_ghost();
    void destroy_device();
    void destroy_device_ghost();
    void create_all();
    void destroy_all();


  public:
    QKXTM_Field();
    ~QKXTM_Field();
    void zero();
    double* H_elem() const { return h_elem; }
    double* D_elem() const { return d_elem; }
    double* H_ext_ghost() const  { return h_ext_ghost; }
    double* D_ext_ghost() const  { return d_ext_ghost; }

    size_t Bytes() const { return bytes_total_length; }
    size_t BytesGhost() const { return bytes_ghost_length;}

    void printInfo();
    void fourierCorr(double *corr, double *corrMom, int Nmom , int momElem[][3]);
  };

  ///////////////////////////////////////////////////////// end QKXTM_Field //////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////// class QKXTM_Gauge /////////////////////////////////////////////////////////////////

  class QKXTM_Gauge : public QKXTM_Field {
  
  private:
    bool gauge_binded_plaq;
    bool packGauge_flag;
    bool loadGauge_flag;
    bool gauge_binded_ape;
    double *h_elem_backup;

  public:
    QKXTM_Gauge(); // class constructor
    QKXTM_Gauge(NgaugeHost ngaugeHost);
    ~QKXTM_Gauge();                       // class destructor

    void packGauge(void **gauge);
    void justDownloadGauge();
    void loadGauge();
    void packGaugeToBackup(void **gauge);
    void loadGaugeFromBackup();

    void ghostToHost();
    void cpuExchangeGhost();
    void ghostToDevice();
    double calculatePlaq();

    void bindGaugePlaq();
    void unbindGaugePlaq();
    void bindGaugeAPE();
    void unbindGaugeAPE();
    void rebindGaugeAPE();

    double norm2Host();
    double norm2Device();
    double* H_elem_backup() const { return h_elem_backup; }
    void checkSum();
  };


  //////////////////////////////////////////////////////////////////////////// end QKXTM_Gauge //////////////////////////


  ////////////////////////////////////////////////////////////////////// class vector ///////////////////////////////////////////

  class QKXTM_Vector : public QKXTM_Field {
  private:
    bool vector_binded_gauss;
    bool packVector_flag;
    bool loadVector_flag;


  public:
    QKXTM_Vector(); // class constructor
    ~QKXTM_Vector();                       // class destructor

    void packVector(void *vector);
    void loadVector();
    void ghostToHost();
    void cpuExchangeGhost();
    void ghostToDevice();

    void bindVectorGauss();
    void unbindVectorGauss();
    void rebindVectorGauss();

    double norm2Host();
    double norm2Device();

    void download(); //////////////////// take the vector from device to host
    void uploadToCuda(cudaColorSpinorField &cudaVector);
    void downloadFromCuda(cudaColorSpinorField &cudaVector);
    
    void flagsToFalse();
    void scaleVector(double a);
   
    void copyPropagator3D(QKXTM_Propagator3D &prop, int timeslice, int nu , int c2);
    void copyPropagator(QKXTM_Propagator &prop, int nu , int c2);

    void conjugate();
    void applyGamma5();
    void applyGammaTransformation();
    void applyMomentum(int nx, int ny, int nz);
    void write(char* filename);
    void getVectorProp3D(QKXTM_Propagator3D &prop1, int timeslice,int nu,int c2);
  };




///////////////////////////////////////////////////////////////////////// end QKXTM_Propagator ///////////////////////////

/////////////////////////////////////////////////////////////////////// class propagator ////////////////////////////////////

  class QKXTM_Propagator : public QKXTM_Field {
  private:
    bool propagator_binded_ape;
    bool packPropagator_flag;
    bool loadPropagator_flag;


  public:
    QKXTM_Propagator(); // class constructor
    ~QKXTM_Propagator();                       // class destructor

    void packPropagator(void *propagator);
    void loadPropagator();
    void ghostToHost();
    void cpuExchangeGhost();
    void ghostToDevice();

    void bindPropagatorAPE();
    void unbindPropagatorAPE();
    void rebindPropagatorAPE();

    double norm2Host();
    double norm2Device();

    void download();
    void absorbVector(QKXTM_Vector &vec, int nu, int c2);

    void rotateToPhysicalBasePlus();
    void rotateToPhysicalBaseMinus();

    void conjugate();
    void applyGamma5();
    void checkSum();
  };



  /////////////////////////////////////////////////////////////////

  class QKXTM_Correlator : public QKXTM_Field {

  public:
    QKXTM_Correlator(); // class constructor
    ~QKXTM_Correlator();                       // class destructor
    void download();
    void fourierCorr(double *corrMom, int Nmom , int momElem[][3]);
    void packCorrelator(void *corr);
    void loadCorrelator();
  };

  ///////////////////////////////////////////////////////
  class QKXTM_Propagator3D : public QKXTM_Field {

  public:
    QKXTM_Propagator3D();
    ~QKXTM_Propagator3D();
    
    void absorbTimeSlice(QKXTM_Propagator &prop, int timeslice);
    void absorbVectorTimeSlice(QKXTM_Vector &vec, int timeslice, int nu, int c2);
    void download();
    void justCopyToHost();
    void justCopyToDevice();
    void broadcast(int tsink);
    
  };

  ///////////////////////////////////////////////////////
  class QKXTM_Vector3D : public QKXTM_Field {
    
  public:
    QKXTM_Vector3D();
    ~QKXTM_Vector3D();
    
    void absorbTimeSlice(QKXTM_Vector &vec, int timeslice);
    void justCopyToHost();
    void justCopyToDevice();
    void broadcast(int tsink);
    void download();
    void fourier(double *vecMom, int Nmom, int momElem[][3]);

  };

  /////////////////////////////////////////////////
  class QKXTM_VectorX8 : public QKXTM_Field {

  public:
    QKXTM_VectorX8();
    ~QKXTM_VectorX8();

  };


}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
