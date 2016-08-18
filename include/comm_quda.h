#ifndef _COMM_QUDA_H
#define _COMM_QUDA_H

#define BACK_NBR 1
#define FWD_NBR 2

#ifdef __cplusplus
extern "C" {
#endif

#define X_BACK_NBR 1
#define Y_BACK_NBR 2
#define Z_BACK_NBR 3
#define T_BACK_NBR 4
#define X_FWD_NBR  5
#define Y_FWD_NBR  6
#define Z_FWD_NBR  7
#define T_FWD_NBR  8

/* The following routines are implemented over MPI only. */

int             comm_dim_partitioned(int dir);
/*testing/debugging use only */
void            comm_dim_partitioned_set(int dir);
int             comm_dim(int);
int             comm_coords(int);
unsigned long	comm_send(void*, int, int, void*);
unsigned long	comm_send_to_rank(void*, int, int, void*);
unsigned long   comm_send_with_tag(void*, int, int, int, void*);
unsigned long	comm_recv(void*, int, int, void*);
unsigned long	comm_recv_from_rank(void*, int, int, void*);
unsigned long   comm_recv_with_tag(void*, int, int, int, void*);
int             comm_query(void*);
void		comm_wait(void*);
void		comm_allreduce(double* data);
void            comm_allreduce_int(int* data);
void		comm_allreduce_array(double* data, size_t size);
void		comm_allreduce_max(double* data);
int		comm_gpuid();
int		comm_get_neighbor_rank(int dx, int dy, int dz, int dt);

/* implemented over both MPI and QMP */

void            comm_create(int argc, char **argv);
void		comm_init(void);
void		comm_cleanup(void);
void		comm_exit(int);
char *          comm_hostname(void);
void            comm_set_gridsize(const int *X, int nDim);
int		comm_rank(void);
int		comm_size(void);
void		comm_barrier(void);
void            comm_broadcast(void *data, size_t nbytes);
  
#ifdef __cplusplus
}
#endif

#endif /* _COMM_QUDA_H */
