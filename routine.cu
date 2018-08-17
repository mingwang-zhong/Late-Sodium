// functions called by CPU, running on GPU

#include "buffers.cu"
#include "subroutine.cu"

__global__ void Init( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, double *cnsr, double *cnsrnext, 
						double *cs, double *csnext, double cjsr_b)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int ac = pos(i,j,k);
	
	#ifdef randomlcc
		curandState localState;
		localState=CRU2[ac].state;
	#endif
	
	#ifdef tapsi
		cjsr_b = 10;
	#endif

	CRU2[ac].randomi = 1.0;
	CRU2[ac].cp = ci_basal;
	cs[ac] = ci_basal;
	csnext[ac] = ci_basal;
	CRU2[ac].cjsr = cjsr_b;
	CRU2[ac].nspark = 0;
	CRU2[ac].Ancx = 0.025;

	double ratio;
	for ( int ii = 0; ii < 8; ++ii )
	{
		ci[ac*8+ii] = ci_basal;
		cnsr[ac*8+ii] = cjsr_b;
		cinext[ac*8+ii] = ci_basal;
		cnsrnext[ac*8+ii] = cjsr_b;

		CBU[ac*8+ii].catf = ktfon*ci_basal*Btf/(ktfon*ci_basal+ktfoff);
		CBU[ac*8+ii].cats = ktson*ci_basal*Bts/(ktson*ci_basal+ktsoff);
		CBU[ac*8+ii].cacal = kcalon*ci_basal*Bcal/(kcalon*ci_basal+kcaloff);
		CBU[ac*8+ii].cadye = kdyeon*ci_basal*Bdye/(kdyeon*ci_basal+kdyeoff);
		CBU[ac*8+ii].casr = ksron*ci_basal*Bsr/(ksron*ci_basal+ksroff);
		
		ratio = Mgi*Kmyoca/(ci_basal*Kmyomg);
		CBU[ac*8+ii].camyo = ci_basal*Bmyo/(Kmyoca+ci_basal*(ratio+1.0));
		CBU[ac*8+ii].mgmyo = CBU[ac*8+ii].camyo*ratio;
	}
	SBU[ac].casar = ksaron*ci_basal*Bsar/(ksaron*ci_basal+ksaroff);
	SBU[ac].casarh = ksarhon*ci_basal*Bsarh/(ksarhon*ci_basal+ksarhoff);
	SBU[ac].casarj = ksaron*ci_basal*Bsar/(ksaron*ci_basal+ksaroff);
	SBU[ac].casarhj = ksarhon*ci_basal*Bsarh/(ksarhon*ci_basal+ksarhoff);
	
	#ifdef randomlcc
		int randpoint = 0+(curand(&localState))%9;
	#endif

	#ifndef randomlcc
		int randpoint = svncp;
	#endif
	
	for(int ll=0; ll<8; ll++)
	{
		if ( ll < randpoint )
			CRU2[ac].lcc[ll] = 3;
		else
			CRU2[ac].lcc[ll] = 16;
	}

	
	double roo2 = ratedimer/(1.0+pow(kdimer/(cjsr_b),hilldimer));
	double kub = (-1.0+sqrt(1.0+8.0*BCSQN*roo2))/(4.0*roo2*BCSQN)/taub;
	double kbu = 1.0/tauu;

	double fracbound = 1/(1+kbu/kub);

	CRU2[ac].ncb = int(fracbound*nryr);
	CRU2[ac].ncu = nryr-int(fracbound*nryr);
	CRU2[ac].nob = 0;
	CRU2[ac].nou = 0;

}




#define FINESTEP 5
#define DTF (DT/FINESTEP)	
__global__ void Compute( cru *CRU, cru2 *CRU2, cytosol *CYT, cyt_bu *CBU, sl_bu *SBU, double *ci, double *cinext, double *cnsr, double *cnsrnext, 
						double *cs, double *csnext, double v, int step, double Ku, double Ku2, double Kb, double xnai, 
						double sv_ica, double sv_ncx)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int ac = pos(i,j,k);

	curandState localState;
	localState=CRU2[ac].state;

	double dotcjsr;
	double dotci[8];
	double dotcnsr[8];
	double xicoupn[8];
	double xicoupi[8];
	
	double sviup = sv_iup;


	#ifdef ISO
		sviup = sv_iup*1.75;
	#endif

	#ifdef tapsi
		sviup=0;
	#endif

	


	if((i*j*k)!=0 && i<nx-1 && j<ny-1 && k<nz-1)
	{	
		///////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////// ica /////////////////////////////////////
		if ( step%OUT_STEP==1 )
		{
			CRU2[ac].nl=0;
		}
		double ICa = 0;
		#ifndef perm
			int jj, ll, nlcp = 0;
			for (ll=0; CRU2[ac].lcc[ll]<16 && ll < 8; ll++)
			{
				jj = lcc_markov( &localState, CRU2[ac].lcc[ll], (CRU2[ac].cp+cs[ac]*icagamma)/(1+icagamma), v );
				CRU2[ac].lcc[ll] = jj;
				if (jj==0 )
				{
					++CRU2[ac].nl;
					++nlcp;
				}
			}
			ICa = sv_ica*(double)(nlcp)*lcccurrent(v,(CRU2[ac].cp+cs[ac]*icagamma)/(1+icagamma))/(1+icagamma);
		#endif

		double pmca = 0;
		double bcgcur = 0;

		CRU[ac].xica = ICa + pmca + bcgcur*(Vs/Vp);//I_pmca added;
		
		///////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////// RyR /////////////////////////////////////
		int problem = ryrgating( &localState,Ku, Ku2, Kb, CRU2[ac].cp, CRU2[ac].cjsr, &CRU2[ac].ncu, &CRU2[ac].nou, 
								 &CRU2[ac].ncb, &CRU2[ac].nob, i, j, k, step );
		
		if ( step*DT > nbeat*PCL + 100 + 16000 )
			CRU2[ac].po = 0.1;
		else
			CRU2[ac].po = (double)(CRU2[ac].nou+CRU2[ac].nob)/(double)(nryr);
		
		CRU2[ac].xire = 0.0147*svjmax*CRU2[ac].po*(CRU2[ac].cjsr-CRU2[ac].cp)/Vp/CRU2[ac].randomi;
		if	(CRU2[ac].xire<0) 	CRU2[ac].xire = 0;
		
		///////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////// inaca /////////////////////////////////////
		#ifndef perm
			CRU[ac].xinaca = sv_ncx*ncx( cs[ac]/1000.0, v, PCL, xnai, &CRU2[ac].Ancx );
		#else
			CRU[ac].xinaca = 0;
		#endif
		
		///////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////
		
		double diffjn0 = (CRU2[ac].cjsr-cnsr[ac*8])/tautr/2.;
		double diffjn1 = (CRU2[ac].cjsr-cnsr[ac*8+1])/tautr/2.;

		for ( int ii = 0; ii < 8; ++ii )
		{
			int act = ac*8+ii;

			CYT[act].xiup = sviup*uptake(ci[act],cnsr[act]);
			CYT[act].xileak = svileak*0.00001035*2*(cnsr[act]-ci[act])/(1.0+pow2(500.0/cnsr[act]));

			int north = (ii%2)?(pos(i,j,k+1)*8+ii-1):(act+1);
			int south = (ii%2)?(act-1):(pos(i,j,k-1)*8+ii+1);
			int east = ((ii/2)%2)?(pos(i,j+1,k)*8+ii-2):(act+2);
			int west = ((ii/2)%2)?(act-2):(pos(i,j-1,k)*8+ii+2);
			int top = ((ii/4)%2)?(pos(i+1,j,k)*8+ii-4):(act+4);
			int bottom = ((ii/4)%2)?(act-4):(pos(i-1,j,k)*8+ii+4);

			xicoupn[ii] =	(cnsr[north]-cnsr[act])/(taunt*xi) +
							(cnsr[south]-cnsr[act])/(taunt*xi) +
							(cnsr[east]-cnsr[act])/(taunt*xi) +
							(cnsr[west]-cnsr[act])/(taunt*xi) +
							(cnsr[top]-cnsr[act])/(taunl*xi) +
							(cnsr[bottom]-cnsr[act])/(taunl*xi) ;

			xicoupi[ii] =	(ci[north]-ci[act])/(tauit*xi) +
							(ci[south]-ci[act])/(tauit*xi) +
							(ci[east]-ci[act])/(tauit*xi) +
							(ci[west]-ci[act])/(tauit*xi) +
							(ci[top]-ci[act])/(tauil*xi) +
							(ci[bottom]-ci[act])/(tauil*xi) ;

			dotcnsr[ii]=(CYT[act].xiup-CYT[act].xileak)*Vi/Vnsr+xicoupn[ii];
				
			double buffers =	tropf(CBU[act].catf, ci[act], DT)+
								trops(CBU[act].cats, ci[act], DT)+
								bucal(CBU[act].cacal, ci[act], DT)+
								busr(CBU[act].casr, ci[act], DT)+
								myoca(CBU[act].camyo, CBU[act].mgmyo, ci[act], DT) ;

			dotci[ii]=	- CYT[act].xiup
						+ CYT[act].xileak
						- buffers
						+ xicoupi[ii];
		}
		dotcnsr[0] += diffjn0*Vjsr/Vnsr;
		dotcnsr[1] += diffjn1*Vjsr/Vnsr;


		///////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////////////
		csnext[ac] = cs[ac];
		for( int iii = 0; iii < FINESTEP; ++iii )
		{

			double diffpi0 = (CRU2[ac].cp-ci[ac*8])/taupi/2.0;
			double diffpi1 = (CRU2[ac].cp-ci[ac*8+1])/taupi/2.0;
			double diffsi0 = (csnext[ac]-ci[ac*8])/tausi/2.0;
			double diffsi1 = (csnext[ac]-ci[ac*8+1])/tausi/2.0;

			dotci[0] += diffsi0*(Vs/Vi)/FINESTEP;
			dotci[1] += diffsi1*(Vs/Vi)/FINESTEP;
			dotci[0] += diffpi0*(Vp/Vi)/FINESTEP;
			dotci[1] += diffpi1*(Vp/Vi)/FINESTEP;


			double diffps = (CRU2[ac].cp-csnext[ac])/taups;
			////////////////////// submembrane: dotcs /////////////////////////
			double csbuff = busar(SBU[ac].casar, csnext[ac], DTF) ;
			double csbuffh = busarh(SBU[ac].casarh, csnext[ac], DTF)	;

			double csdiff = (cs[pos(i,j,k+1)]+cs[pos(i,j,k-1)]-2*cs[ac])/(taust); //4.
	
			double dotcs =    CRU[ac].xinaca - bcgcur
							+ Vp/Vs*diffps - diffsi0 - diffsi1 + csdiff
							- csbuff - csbuffh ;

			csnext[ac] += dotcs*DTF;
			SBU[ac].casar += csbuff*DTF;
			SBU[ac].casarh += csbuffh*DTF;
			
			if( SBU[ac].casar < 0 )		SBU[ac].casar =0;
			if( SBU[ac].casarh < 0 )	SBU[ac].casarh =0;

			////////////////////// proximal space: dotcp ////////////////////// 
			double cpbuff = busarj(SBU[ac].casarj, CRU2[ac].cp, DTF);
			double cpbuffh = busarhj(SBU[ac].casarhj, CRU2[ac].cp, DTF);

			double dotcp =  CRU2[ac].xire - ICa
							- diffps -diffpi0 -diffpi1 
							- cpbuff - cpbuffh ;

			CRU2[ac].cp += dotcp*DTF;
			SBU[ac].casarj += cpbuff*DTF;
			SBU[ac].casarhj += cpbuffh*DTF;
			
			if( SBU[ac].casarj < 0 )		SBU[ac].casarj =0;
			if( SBU[ac].casarhj < 0 )		SBU[ac].casarhj =0;
		}


			if ( csnext[ac]<0 ) 			csnext[ac]=1e-6;
			if ( CRU2[ac].cp < 0 )  		CRU2[ac].cp=0;
	
		
		for ( int ii = 0; ii < 8; ++ii )
		{
			int act = ac*8+ii;
			
			cinext[act] = ci[act]+dotci[ii]*DT;
			cnsrnext[act]=cnsr[act]+dotcnsr[ii]*DT;

			CBU[act].catf += tropf(CBU[act].catf,ci[act], DT)*DT;
			CBU[act].cats += trops(CBU[act].cats,ci[act], DT)*DT;
			CBU[act].cacal += bucal(CBU[act].cacal,ci[act], DT)*DT;
			CBU[act].casr += busr(CBU[act].casr,ci[act], DT)*DT;
			CBU[act].camyo += myoca(CBU[act].camyo,CBU[act].mgmyo,ci[act], DT)*DT;
			CBU[act].mgmyo += myomg(CBU[act].camyo,CBU[act].mgmyo,ci[act], DT)*DT;
			CBU[act].cadye += budye(CBU[act].cadye,ci[act], DT)*DT;
			
			if( cinext[act]<0) 				cinext[act]=0;
			if( cnsrnext[act]<0) 			cnsrnext[act]=0;
			if( CBU[act].catf < 0 )			CBU[act].catf =0;
			if( CBU[act].cats < 0 )			CBU[act].cats =0;
			if( CBU[act].cacal < 0 )		CBU[act].cacal =0;
			if( CBU[act].casr < 0 )			CBU[act].casr =0;
			if( CBU[act].camyo < 0 )		CBU[act].camyo =0;
			if( CBU[act].mgmyo < 0 )		CBU[act].mgmyo =0;
			if( CBU[act].cadye < 0 )		CBU[act].cadye =0;
		
		}	
		

		dotcjsr = 1.0/(1.0 + (BCSQN*kbers*nM)/pow2((kbers+CRU2[ac].cjsr)) )*( -diffjn0-diffjn1 -CRU2[ac].xire*Vp/Vjsr*CRU2[ac].randomi);

		CRU2[ac].cjsr += dotcjsr*DT;
		CRU2[ac].Tcj = CRU2[ac].cjsr + BCSQN*nM*CRU2[ac].cjsr/(kbers+CRU2[ac].cjsr);
	}
	
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	//Boundary conditions

	#ifndef perm
		if(k==1)
		{
			for( int ii = 1; ii < 8; ii+=2 ){
				cinext[pos(i,j,0)*8+ii]=cinext[pos(i,j,1)*8+ii-1];
				cnsrnext[pos(i,j,0)*8+ii]=cnsrnext[pos(i,j,1)*8+ii-1];
			}
			csnext[pos(i,j,0)]=csnext[pos(i,j,1)];
		}
		if(k==nz-2)
		{
			for( int ii = 0; ii < 8; ii+=2 ){
				cinext[pos(i,j,nz-1)*8+ii]=cinext[pos(i,j,nz-2)*8+ii+1];
				cnsrnext[pos(i,j,nz-1)*8+ii]=cnsrnext[pos(i,j,nz-2)*8+ii+1];
			}
			csnext[pos(i,j,nz-1)]=csnext[pos(i,j,nz-2)];
		}
		if(j==1)
		{
			for( int ii = 2; ii < 8; ii+=4 ){
				cinext[pos(i,0,k)*8+ii]=cinext[pos(i,1,k)*8+ii-2];
				cinext[pos(i,0,k)*8+ii+1]=cinext[pos(i,1,k)*8+ii-1];
				cnsrnext[pos(i,0,k)*8+ii]=cnsrnext[pos(i,1,k)*8+ii-2];
				cnsrnext[pos(i,0,k)*8+ii+1]=cnsrnext[pos(i,1,k)*8+ii-1];
			}
		}
		if(j==ny-2)
		{
			for( int ii = 0; ii < 8; ii+=4 ){
				cinext[pos(i,ny-1,k)*8+ii]=cinext[pos(i,ny-2,k)*8+ii+2];
				cinext[pos(i,ny-1,k)*8+ii+1]=cinext[pos(i,ny-2,k)*8+ii+3];
				cnsrnext[pos(i,ny-1,k)*8+ii]=cnsrnext[pos(i,ny-2,k)*8+ii+2];
				cnsrnext[pos(i,ny-1,k)*8+ii+1]=cnsrnext[pos(i,ny-2,k)*8+ii+3];
			}
		}
		if(i==1)
		{
			for( int ii = 4; ii < 8; ++ii ){
				cinext[pos(0,j,k)*8+ii]=cinext[pos(1,j,k)*8+ii-4];
				cnsrnext[pos(0,j,k)*8+ii]=cnsrnext[pos(1,j,k)*8+ii-4];
			}
		}
		if(i==nx-2)
		{
			for( int ii = 0; ii < 4; ++ii ){
				cinext[pos(nx-1,j,k)*8+ii]=cinext[pos(nx-2,j,k)*8+ii+4];
				cnsrnext[pos(nx-1,j,k)*8+ii]=cnsrnext[pos(nx-2,j,k)*8+ii+4];
			}
		}
	#else
		if(k==1)
			for( int ii = 1; ii < 8; ii+=2 )
				cnsrnext[pos(i,j,0)*8+ii]=cnsrnext[pos(i,j,1)*8+ii-1];
		if(k==nz-2)
			for( int ii = 0; ii < 8; ii+=2 )
				cnsrnext[pos(i,j,nz-1)*8+ii]=cnsrnext[pos(i,j,nz-2)*8+ii+1];
		if(j==1)
		{
			for( int ii = 2; ii < 8; ii+=4 ){
				cnsrnext[pos(i,0,k)*8+ii]=cnsrnext[pos(i,1,k)*8+ii-2];
				cnsrnext[pos(i,0,k)*8+ii+1]=cnsrnext[pos(i,1,k)*8+ii-1];
			}
		}
		if(j==ny-2)
		{
			for( int ii = 0; ii < 8; ii+=4 ){
				cnsrnext[pos(i,ny-1,k)*8+ii]=cnsrnext[pos(i,ny-2,k)*8+ii+2];
				cnsrnext[pos(i,ny-1,k)*8+ii+1]=cnsrnext[pos(i,ny-2,k)*8+ii+3];
			}
		}
		if(i==1)
			for( int ii = 4; ii < 8; ++ii )
				cnsrnext[pos(0,j,k)*8+ii]=cnsrnext[pos(1,j,k)*8+ii-4];
		if(i==nx-2)
			for( int ii = 0; ii < 4; ++ii )
				cnsrnext[pos(nx-1,j,k)*8+ii]=cnsrnext[pos(nx-2,j,k)*8+ii+4];
	#endif


	CRU2[ac].state=localState;
}


__global__ void	setup_kernel(unsigned long long seed, cru2 *CRU2 )	///curandState *state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	curand_init(seed,pos(i,j,k),0,&(CRU2[pos(i,j,k)].state)	);
}

