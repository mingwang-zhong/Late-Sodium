// functions called by GPU

__device__ int lcc_markov(curandState *state,int i, double cpx,double v)
{
	curandState localState=*state;
	
	double dv5 = 5;//-12.6;
	double dvk = 8;//6;

	double fv5 = -22.8;
	double fvk = 9.1;//6.1;

	double alphac = 0.22; //still want to reduce this to get the right ica, cat
	double betac = 4; //3
	
	#ifdef ISO
		betac=2;
		dv5 = 0;//-12.6;
		fv5 = -28;//-36; -23; -28
		fvk = 8.5;
	#endif

	double dinf = 1.0/(1.0+exp(-(v-dv5)/dvk));
	double taudin = 1.0/((1.0-exp(-(v-dv5)/dvk))/(0.035*(v-dv5))*dinf);
	if( (v < 0.0001) && (v > -0.0001) )
		taudin = 0.035*dvk/dinf;
	
	double finf = 1.0-1.0/(1.0+exp(-(v-fv5)/fvk))/(1.+exp((v-60)/12.0));
	double taufin = (0.02-0.007*exp(-pow2(0.0337*(v+10.5))))/1.2;//(0.0197*exp(-pow2(0.0337*(v+10.5)))+0.02); //14.5
	
	

	double alphad = dinf*taudin;
	double betad = (1-dinf)*taudin;
	
	double alphaf = (finf)*taufin;
	double betaf = (1-finf)*taufin;
	
	double alphafca = 0.012/2.;
	double betafca = 0.175/(1+pow2(35./cpx));
	
	double ragg=curand_uniform_double(&localState);
	*state=localState;
	double rig = ragg/DT;
	
	if ( (i%2) )
	if ( rig < alphac )
		return i-1;
	else
		rig-=alphac;
	else
	if ( rig < betac )
		return i+1;
	else
		rig-=betac;
	
	if ( ((i/2)%2) )
	if ( rig < alphad )
		return i-2;
	else
		rig-=alphad;
	else
	if ( rig < betad )
		return i+2;
	else
		rig-=betad;
	
	
	if ( ((i/4)%2) )
	if ( rig < alphaf )
		return i-4;
	else
		rig-=alphaf;
	else
	if ( rig < betaf )
		return i+4;
	else
		rig-=betaf;
	
	
	if ( ((i/8)%2) )
	if ( rig < alphafca )
		return i-8;
	else
		rig-=alphafca;
	else
	if ( rig < betafca )
		return i+8;
	else
		rig-=betafca;
	
	return(i);

	
}




__device__ double ncx(double cs, double v,double tperiod, double xnai, double *Ancx)	//Na_Ca exchanger
{
	double za=v*Farad/xR/Temper;
	double Ka = *Ancx;
	double t1=Kmcai*pow3(xnao)*(1.0+pow3(xnai/Kmnai));
	double t2=pow3(Kmnao)*cs*(1.0+cs/Kmcai);		//i'm not sure what this is (check hund/Rudy 2004)
	double t3=(Kmcao+Cext)*pow3(xnai)+cs*pow3(xnao);
	double Inaca=Ka*vnaca*(exp(eta*za)*pow3(xnai)*Cext-exp((eta-1.0)*za)*pow3(xnao)*cs)/((t1+t2+t3)*(1.0+ksat*exp((eta-1.0)*za)));
	
	double Ancxdot = (1.0/(1.0+pow3(0.000256/cs))-*Ancx)/150.;
	*Ancx = 1.0/(1.0+pow3(0.0003/cs));

	return (Inaca);
}

__device__ double uptake(double ci, double cnsr)		//uptake
{
	double Iup;
	double vup=0.3;	//0.3 for T=400ms
	double Ki=0.123;
	double Knsr=1700.0;		//1700 for T=400ms
	double HH=1.787;
	double upfactor=1.0;	 //factor of SERCA increasing
	Iup = upfactor*vup*(pow(ci/Ki,HH)-pow(cnsr/Knsr,HH))/(1.0+pow(ci/Ki,HH)+pow(cnsr/Knsr,HH));
	return(Iup);
}

 



__device__ double lcccurrent(double v, double cp)	//Ica
{
	double za=v*Farad/xR/Temper;
	double ica; 
		
	if (fabs(za)<0.001) 
	{
		ica=2.0*Pca*Farad*gammai*(cp/1000.0*exp(2.0*za)-Cext);
	}
	else 
	{
		ica=4.0*Pca*za*Farad*gammai*(cp/1000.0*exp(2.0*za)-Cext)/(exp(2.0*za)-1.0);
	}
	if (ica > 0.0)
		ica=0.0;
	return (ica);
}




__device__ int ryrgating (curandState *state, double Ku, double Ku2, double Kb, double cp, double cjsr, int * ncu, int * nou, 
							int * ncb, int * nob, int i, int j, int k, int step)
{

	curandState localState=*state;
	int ret = 0;

	double ryrpref = 80.0;
	double cygatek = 19.0;
	double ryrhill = 2;

	double ku = ryrpref * 1.0/(1+pow2(5000/cjsr)) * 1.0/(1+pow(cygatek/cp,ryrhill));
	double kb = ryrpref * 1.0/(1+pow2(5000/cjsr)) * 1.0/(1+pow(cygatek/cp,ryrhill))/36.0 * Kb;

	double kuminus=1.0/taucu;
	double kbminus=1.0/taucb;

	if (ku*DT > 1.0) ku = 1.0/DT;
	if (kb*DT > 1.0) kb = 1.0/DT;
	if (kuminus*DT > 1.0) kuminus = 1.0/DT;
	if (kbminus*DT > 1.0) kbminus = 1.0/DT; 
 
	double kub=((1.0)/( 1.0+pow(31.*cjsr/13.3/(kbers+cjsr), 24) ))/taub;
	double kbu=1.0/tauu;

	double rate_ou_cu;//kuminus
	double rate_cu_ou;//ku
	double rate_ou_ob;//ku2b
	double rate_ob_ou;//kb2u,  should be kb2u*ku/kb
	double rate_cb_cu;//kb2u
	double rate_cu_cb;//ku2b
	double rate_ob_cb;//kbminus
	double rate_cb_ob;//kb

	int kk;
	double puu;
	double u1;
	double u2;
	double re;

	int n_ou_cu;
	int n_cu_ou;
	int n_ou_ob;
	int n_ob_ou;
	int n_cu_cb;
	int n_cb_cu;
	int n_ob_cb;
	int n_cb_ob;
 
 
	
	//////////////////////////////////// going from OU >> CU ///////////////////////////////
	rate_ou_cu = kuminus*DT;
	n_ou_cu=-1;
	if ((*nou) <=1 || rate_ou_cu < 0.2 || ((*nou) <=5 && rate_ou_cu < 0.3))
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*nou)*rate_ou_cu) && kk < 195)	//generates poisson number = fraction of closed RyR's that open
		{
			kk++;
			re=curand_uniform_double(&localState);
			puu=puu*(re);
		}
		n_ou_cu = kk-1;
	}
	else
	{
		kk = 0;
		while(n_ou_cu < 0)
		{
			//next is really a gaussian
			u1=curand_uniform_double(&localState);
			u2=curand_uniform_double(&localState);
			n_ou_cu = floor((*nou)*rate_ou_cu +sqrt((*nou)*rate_ou_cu*(1.0-rate_ou_cu))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ curand(&localState)%2;
			kk++;
			if( kk > 200)
			{
				n_ou_cu = 0;
				ret = 100000*i+1000*j+10*k+1;
			}
		}
	}
	if(n_ou_cu > nryr) {n_ou_cu = nryr;}
	
	//////////////////////////////////// going from CU >> OU ///////////////////////////////
	rate_cu_ou = ku*DT;
	n_cu_ou = -1;
	if((*ncu) <= 1 ||rate_cu_ou < 0.2 || ((*ncu) <= 5 && rate_cu_ou < 0.3))	//checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*ncu)*rate_cu_ou) && kk < 195)			//generates poisson number = fraction of closed RyR's that open
		{
			kk++;
			re=curand_uniform_double(&localState);
			puu=puu*re;
		}
		n_cu_ou = kk-1;
	}
	else
	{
		kk = 0;
		while(n_cu_ou < 0)
		{
			//next is really a gaussian
			u1=curand_uniform_double(&localState);
			u2=curand_uniform_double(&localState);
			n_cu_ou = floor((*ncu)*rate_cu_ou +sqrt((*ncu)*rate_cu_ou*(1.0-rate_cu_ou))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ curand(&localState)%2;
			kk++;
			if( kk > 200)
			{
				n_cu_ou = 0;
				ret = 100000*i+1000*j+10*k+2;
			}
		}
	}
	if(n_cu_ou > nryr) {n_cu_ou = nryr;}
	
		
	
	//////////////////////////////////// going from OU >> OB ///////////////////////////////
	rate_ou_ob = kub*DT;
	n_ou_ob = -1;
	if((*nou) <= 1 || rate_ou_ob < 0.2 || (*nou <= 5 && rate_ou_ob < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*nou)*rate_ou_ob) && kk < 195)			//generates poisson number = fraction of open RyR's that close
		{
			kk++;
			re=curand_uniform_double(&localState);
			puu=puu*re;
		}
		n_ou_ob = kk-1;
	}
	else
	{
		kk = 0;
		while(n_ou_ob < 0)
		{
			//next is really a gaussian
			u1=curand_uniform_double(&localState);
			u2=curand_uniform_double(&localState);
			n_ou_ob = floor(*nou*rate_ou_ob +sqrt(*nou*rate_ou_ob*(1.0-rate_ou_ob))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ curand(&localState)%2;
			kk++;
			if( kk > 200)
			{
				n_ou_ob = 0;
				ret = 100000*i+1000*j+10*k+3;
			}
		}
	}
	if(n_ou_ob > nryr) n_ou_ob = nryr;
		
	//////////////////////////////////// going from OB >> OU ///////////////////////////////
	rate_ob_ou = kbu*DT*(ku/kb);
	n_ob_ou = -1;
		
	if((*nob) <= 1 || rate_ob_ou < 0.2 || (*nob <= 5 && rate_ob_ou < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*nob)*rate_ob_ou) && kk < 195)			//generates poisson number = fraction of open RyR's that close
		{
			kk++;
			re=curand_uniform_double(&localState);
			puu=puu*re;
		}
		n_ob_ou = kk-1;
		
	}	
	else
	{
		kk = 0;
		while(n_ob_ou < 0)
		{		
			//next is really a gaussian
			u1=curand_uniform_double(&localState);
			u2=curand_uniform_double(&localState);
			n_ob_ou = floor(*nob*rate_ob_ou +sqrt(*nob*rate_ob_ou*(1.0-rate_ob_ou))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ curand(&localState)%2;
			kk++;
			if( kk > 200)
			{
				n_ob_ou = 0;
				ret = 100000*i+1000*j+10*k+4;
			}
		}
	}
	if(n_ob_ou > nryr) n_ob_ou = nryr;
	
		
		
	//////////////////////////////////// going from CB >> CU ///////////////////////////////
	rate_cb_cu = kbu*DT;
	n_cb_cu = -1;
	if((*ncb) <= 1 || rate_cb_cu < 0.2	|| (rate_cb_cu < 0.3 && (*ncb) <= 5))	//checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*ncb)*rate_cb_cu) && kk < 195)			//generates poisson number = fraction of open RyR's that close
		{
			kk++;
			re=curand_uniform_double(&localState);
			puu=puu*re;
		}
		n_cb_cu = kk-1;
	}
	else
	{
		kk = 0;
		while(n_cb_cu < 0)
		{		
			//next is really a gaussian
			u1=curand_uniform_double(&localState);
			u2=curand_uniform_double(&localState);
			n_cb_cu = floor((*ncb)*rate_cb_cu +sqrt((*ncb)*rate_cb_cu*(1.0-rate_cb_cu))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ curand(&localState)%2;
			kk++;
			if( kk > 200)
			{
				n_cb_cu = 0;
				ret = 100000*i+1000*j+10*k+5;
			}
		}
	}
	if(n_cb_cu > nryr) {n_cb_cu = nryr;}
	
	//////////////////////////////////// going from CU >> CB ///////////////////////////////
	rate_cu_cb = kub*DT;
	n_cu_cb = -1;
	if(*ncu <= 1 || rate_cu_cb < 0.2 || (*ncu <= 5 && rate_cu_cb < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-*ncu*rate_cu_cb) && kk < 195)			//generates poisson number = fraction of open RyR's that close
		{
			kk++;
			re=curand_uniform_double(&localState);
			puu=puu*re;
		}
		n_cu_cb = kk-1;
	}
	else
	{
		kk = 0;
		while(n_cu_cb < 0)
		{		
			//next is really a gaussian
			u1=curand_uniform_double(&localState);
			u2=curand_uniform_double(&localState);
			n_cu_cb = floor(*ncu*rate_cu_cb +sqrt(*ncu*rate_cu_cb*(1.0-rate_cu_cb))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ curand(&localState)%2;
			kk++;
			if( kk > 200)
			{
				n_cu_cb = 0;
				ret = 100000*i+1000*j+10*k+6;
			}
		}
	}
	if(n_cu_cb > nryr) {n_cu_cb = nryr;}
	

	
	//////////////////////////////////// going from OB >> CB ///////////////////////////////
	rate_ob_cb = kbminus*DT;
	n_ob_cb = -1;
	if((*nob) <= 1 || rate_ob_cb < 0.2 || ((*nob) <= 5 && rate_ob_cb < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*nob)*rate_ob_cb) && kk < 195)			//generates poisson number = fraction of closed RyR's that open
		{
			kk++;
			re=curand_uniform_double(&localState);
			puu=puu*re;
		}
		n_ob_cb = kk-1;
	}
	else
	{
		kk = 0;
		while(n_ob_cb < 0)
		{			
			//next is really a gaussian
			u1=curand_uniform_double(&localState);
			u2=curand_uniform_double(&localState);
			n_ob_cb = floor((*nob)*rate_ob_cb +sqrt((*nob)*rate_ob_cb*(1.0-rate_ob_cb))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ curand(&localState)%2;
			kk++;
			if( kk > 200)
			{
				n_ob_cb = 0;
				ret = 100000*i+1000*j+10*k+7;
			}
		}
	}
	//if(n_ob_cb > nn) {n_ob_cb = nn;}
	
	//////////////////////////////////// going from CB >> OB ///////////////////////////////
	rate_cb_ob = kb*DT;
	n_cb_ob = -1;
	if((*ncb) <= 1 || rate_cb_ob < 0.2 || ((*ncb) <= 5 && rate_cb_ob < 0.3)) //checks if we use gaussian or poisson approx
	{
		kk = 0;
		puu = 1.0;
		while(puu >= exp(-(*ncb)*rate_cb_ob) && kk < 195)			//generates poisson number = fraction of closed RyR's that open
		{
			kk++;
			re=curand_uniform_double(&localState);
			puu=puu*re;
		}
		n_cb_ob = kk-1;
	}
	else
	{
		kk = 0;
		while(n_cb_ob < 0)
		{
			//next is really a gaussian
			u1=curand_uniform_double(&localState);
			u2=curand_uniform_double(&localState);
			n_cb_ob = floor((*ncb)*rate_cb_ob +sqrt((*ncb)*rate_cb_ob*(1.0-rate_cb_ob))*sqrt(-2.0*log(1.0-u1))*cos(2.0*pi*u2))
						+ curand(&localState)%2;
			kk++;
			if( kk > 200)
			{
				n_cb_ob = 0;
				ret = 100000*i+1000*j+10*k+8;
			}
		}
	}
	if(n_cb_ob > nryr) {n_cb_ob = nryr;}
	

	/////////////////////////////////////////////////////////////////////////////////////////
		
	if(n_ou_ob	+	n_ou_cu > *nou)
	{
		if(n_ou_cu >= n_ou_ob) n_ou_cu = 0;
		else	n_ou_ob = 0;
		if (n_ou_ob > *nou) n_ou_ob = 0;
		else if(n_ou_cu > *nou) n_ou_cu = 0;
	}
		
	if(n_ob_ou	+	n_ob_cb > *nob)
	{ 
		if(n_ob_ou >= n_ob_cb) n_ob_ou = 0;
		else	n_ob_cb = 0;
		if (n_ob_cb > *nob) n_ob_cb = 0;
		else if(n_ob_ou > *nob) n_ob_ou = 0;
	}
		
	if(n_cu_ou	+	n_cu_cb > *ncu ) 
	{
		if(n_cu_cb >= n_cu_ou) n_cu_cb = 0;
		else	n_cu_ou = 0;
		if (n_cu_ou > *ncu) n_cu_ou = 0;
		else if(n_cu_cb > *ncu) n_cu_cb = 0;
	}
		
		
	*nou += - n_ou_ob - n_ou_cu	+ n_ob_ou + n_cu_ou;
	if(*nou<0)		(*nou)=0;
	if(*nou>nryr)	*nou=nryr; 
	
	*nob += - n_ob_ou - n_ob_cb + n_ou_ob + n_cb_ob;
	if(*nob<0)			*nob=0;
	if(*nob>nryr)		*nob=nryr;
		
	*ncu += - n_cu_ou - n_cu_cb + n_ou_cu + n_cb_cu;
	if(*ncu<0) 			*ncu=0;
	if(*ncu>nryr)		*ncu=nryr;

	*ncb = nryr - *nou - *nob - *ncu;
	
	if(*ncb<0) 			*ncb=0;
	if(*ncb>nryr)		*ncb=nryr;

	*state=localState;
	return ret;
}
		
