include ../make.inc

OBJ_PATH = object_files

QUDA = ../lib/libquda.a


INC += -I../include -I. -Iheaders -I/fhgfs/prometheus/home/khadjiyiannakou/qudaQKXTM_package/lib/external_libs/lime-1.3.2/include -I/fhgfs/prometheus/home/khadjiyiannakou/qudaQKXTM_package/lib/external_libs/gsl-1.15/ -I/fhgfs/prometheus/home/khadjiyiannakou/magma-1.5.0/include -I/fhgfs/prometheus/home/build/eb140205/software/imkl/11.1.1.106/include -I/fhgfs/prometheus/home/khadjiyiannakou/qkxTM/include

LDFLAGS += -llime -L/fhgfs/prometheus/home/khadjiyiannakou/qudaQKXTM_package/lib/external_libs/lime-1.3.2/lib 
LDFLAGS += -lgsl -L/fhgfs/prometheus/home/khadjiyiannakou/qudaQKXTM_package/lib/external_libs/gsl-1.15/lib
LDFLAGS += -lcurand
LDFLAGS += -lmagma -L/fhgfs/prometheus/home/khadjiyiannakou/magma-1.5.0/lib -lmagma -lcublas
LDFLAGS += -L/fhgfs/prometheus/home/build/eb140205/software/imkl/11.1.1.106/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lstdc++ -liomp5 -lgfortran
LDFLAGS += -L/fhgfs/prometheus/home/khadjiyiannakou/qkxTM/lib -lqkxTM
#HDRS = blas_reference.h wilson_dslash_reference.h staggered_dslash_reference.h    \
	domain_wall_dslash_reference.h test_util.h dslash_util.h

#TESTS = blas_test invert_test gauge_test invert_test_L24T48 invert_test_L32T64 invert_test_L48T96 test_qkxTM_gauge
#TESTS = test_qkxTM_gauge test_prop test_corr test_fourier
#TESTS = test_final test_qkxTM_gauge twop twop_48x96 threepTwop
#TESTS = twop threepTwop threepTwop_stoch threepTwop_stoch_nonLocal threepTwop_nonLocal createWriteProps
#TESTS = test_WilsonPath
#TESTS = threepTwop_stoch_nonLocal_Pion threepTwop_stoch_nonLocal threepTwop_Pion_nonLocal
#TESTS = threepTwop_Pion threepTwop_stoch_nonLocal_Pion threepTwop_Pion_nonLocal createWritePropsSLReadApeGauge
#TESTS = threepTwop_Pion twop threepTwop
TESTS = threepTwop threepTwop_nonLocal
all: $(TESTS)

lanczos: $(OBJ_PATH)/lanczos.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)


arnoldi: $(OBJ_PATH)/arnoldi.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

threepTwop_stoch_nonLocal_Pion: $(OBJ_PATH)/threepTwop_stoch_nonLocal_Pion.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

threepTwop_Pion: $(OBJ_PATH)/threepTwop_Pion.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

threepTwop_Pion_nonLocal: $(OBJ_PATH)/threepTwop_Pion_nonLocal.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

threepTwop_nonLocal: $(OBJ_PATH)/threepTwop_nonLocal.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

createWriteProps: $(OBJ_PATH)/createWriteProps.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

createWritePropsReadApeGauge: $(OBJ_PATH)/createWritePropsReadApeGauge.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

createWritePropsSLReadApeGauge: $(OBJ_PATH)/createWritePropsSLReadApeGauge.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

test_WilsonPath: $(OBJ_PATH)/test_WilsonPath.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

threepTwop_stoch_nonLocal:  $(OBJ_PATH)/threepTwop_stoch_nonLocal.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

threepTwop_stoch: $(OBJ_PATH)/threepTwop_stoch.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

threepTwop: $(OBJ_PATH)/threepTwop.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

twop: $(OBJ_PATH)/twop.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

twop_48x96: $(OBJ_PATH)/twop_48x96.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

invert_test: $(OBJ_PATH)/invert_test.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

test_fourier: $(OBJ_PATH)/test_fourier.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

test_final: $(OBJ_PATH)/test_final.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 


test_corr: $(OBJ_PATH)/test_corr.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

test_qkxTM_gauge: $(OBJ_PATH)/test_qkxTM_gauge.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

test_prop: $(OBJ_PATH)/test_prop.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

invert_test_L24T48: $(OBJ_PATH)/invert_test_L24T48.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

invert_test_L32T64: $(OBJ_PATH)/invert_test_L32T64.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

invert_test_L48T96: $(OBJ_PATH)/invert_test_L48T96.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

gauge_test: $(OBJ_PATH)/gauge_test.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/wilson_dslash_reference.o $(OBJ_PATH)/blas_reference.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

blas_test: $(OBJ_PATH)/blas_test.o $(OBJ_PATH)/test_util.o $(OBJ_PATH)/misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

clean:
	-rm -f $(OBJ_PATH)/*.o $(TESTS)

$(OBJ_PATH)/%.o: %.c 
	$(CC) $(CFLAGS) $< -c -o $@

$(OBJ_PATH)/%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $< -c -o $@

$(OBJ_PATH)/%.o: %.cu 
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: all clean
