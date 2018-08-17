// GPU computing kernels
__global__ void	setup_kernel(unsigned long long seed,cru2 *CRU2);
__global__ void Init( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, 
						double *cnsr, double *cnsrnext, double *cs, double *csnext, double cjsr_b);
__global__ void Compute( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, 
						double *cnsr, double *cnsrnext, double *cs, double *csnext, double v, int step, double Ku, 
						double Ku2, double Kb, double xnai, double sv_ica, double sv_ncx);

__device__ int lcc_markov(curandState *state,int i, double cpx,double v);
__device__ double ncx(double cs, double v,double tperiod, double xnai, double *Ancx);	//Na_Ca exchanger 
__device__ double uptake(double ci, double cnsr);		//uptake
__device__ double lcccurrent(double v, double cp);	//Ica
__device__ int ryrgating (curandState *state, double Ku, double Ku2, double Kb, double cp, double cjsr, int * ncu, int * nou, 
							int * ncb, int * nob, int i, int j, int k, int step);
__device__ double myoca(double CaMyo, double MgMyo, double calciu, double dt);
__device__ double myomg(double CaMyo, double MgMyo, double calciu, double dt);
__device__ double tropf(double CaTf, double calciu, double dt);
__device__ double trops(double CaTs, double calciu, double dt);
__device__ double bucal(double CaCal, double calciu, double dt);
__device__ double budye(double CaDye, double calciu, double dt);
__device__ double busr(double CaSR, double calciu, double dt);
__device__ double busar(double CaSar, double calciu, double dt);
__device__ double busarh(double CaSarh, double calciu, double dt);
__device__ double busarj(double CaSar, double calciu, double dt);
__device__ double busarhj(double CaSarh, double calciu, double dt);