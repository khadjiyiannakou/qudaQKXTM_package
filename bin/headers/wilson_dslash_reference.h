#ifndef _WILSON_DSLASH_REFERENCE_H
#define _WILSON_DSLASH_REFERENCE_H

#include <enum_quda.h>
#include <quda.h>

#ifdef __cplusplus
extern "C" {
#endif

  void wil_dslash(void *res, void **gauge, void *spinorField, int oddBit,
		  int daggerBit, QudaPrecision precision, QudaGaugeParam &param);
  
  void wil_mat(void *out, void **gauge, void *in, double kappa, int daggerBit,
	       QudaPrecision precision, QudaGaugeParam &param);

  void wil_matpc(void *out, void **gauge, void *in, double kappa,
		 QudaMatPCType matpc_type,  int daggerBit, QudaPrecision precision, QudaGaugeParam &param);

  void tm_dslash(void *res, void **gauge, void *spinorField, double kappa,
		 double mu, QudaTwistFlavorType flavor, int oddBit,
		 int daggerBit, QudaPrecision sprecision, QudaGaugeParam &param);
  
  void tm_mat(void *out, void **gauge, void *in, double kappa, double mu,
	      QudaTwistFlavorType flavor, int daggerBit, QudaPrecision precision, QudaGaugeParam &param);

  void tm_matpc(void *out, void **gauge, void *in, double kappa, double mu,
		QudaTwistFlavorType flavor, QudaMatPCType matpc_type,  
		int daggerBit, QudaPrecision precision, QudaGaugeParam &param);

#ifdef __cplusplus
}
#endif

#endif // _WILSON_DSLASH_REFERENCE_H
