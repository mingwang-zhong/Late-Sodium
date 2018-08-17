//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////

#define ktson 0.00254
#define ktsoff 0.000033
#define Bts 134.0    //*1.5 for myosin ~ the same rate buffer, less because Mg and because myosin is blocked by unbound troponin

#define ktfon 0.0327
#define ktfoff 0.0196
#define Btf 70.0

#define kcalon 0.0543
#define kcaloff 0.238
#define Bcal 24.0

#define kdyeon 0.256 //0.1
#define kdyeoff 0.1 //0.11
#define Bdye 50.0 //25

#define ksron 0.1   //(0.1*0.5)
#define ksroff 0.06 //(0.06*0.5)
#define Bsr 19.0 //change to 19 if do myo buffer

#define ksaron 0.1
#define ksaroff 1.3
#define Bsar (42*(Vi*8.0/Vs))
#define KSAR (ksaroff/ksaron)

#define ksarhon 0.1
#define ksarhoff 0.03
#define Bsarh (15.0*(Vi*8.0/Vs))
#define KSARH (ksarhoff/ksarhon)


#define Bmyo 	140
#define konmyomg 	0.0000157
#define koffmyomg 	0.000057
#define konmyoca 	0.0138
#define koffmyoca 	0.00046
#define Mgi 	500
#define Kmyomg 	(koffmyomg/konmyomg)
#define Kmyoca 	(koffmyoca/konmyoca)


		

__device__ double myoca(double CaMyo, double MgMyo, double calciu, double dt)
{
	double Itc;
	if( konmyoca*calciu*dt > 1 )
		Itc=0.95/dt*(Bmyo-CaMyo-MgMyo)-koffmyoca*CaMyo;
	else
		Itc=konmyoca*calciu*(Bmyo-CaMyo-MgMyo)-koffmyoca*CaMyo;
	return(Itc);
}


__device__ double myomg(double CaMyo, double MgMyo, double calciu, double dt)
{
	double Itc;
	if( konmyomg*Mgi*dt > 1 )
		Itc=0.95/dt*(Bmyo-CaMyo-MgMyo)-koffmyomg*MgMyo;
	else
		Itc=konmyomg*Mgi*(Bmyo-CaMyo-MgMyo)-koffmyomg*MgMyo;
	return(Itc);
}

__device__ double tropf(double CaTf, double calciu, double dt)
{
	double Itc;
	if( ktfon*calciu*dt > 1 )
		Itc=0.95/dt*(Btf-CaTf)-ktfoff*CaTf;
	else
		Itc=ktfon*calciu*(Btf-CaTf)-ktfoff*CaTf;
	return(Itc);
}

__device__ double trops(double CaTs, double calciu, double dt)
{
	double Itc;
	if( ktson*calciu*dt > 1 )
		Itc=0.95/dt*(Bts-CaTs)-ktsoff*CaTs;
	else
		Itc=ktson*calciu*(Bts-CaTs)-ktsoff*CaTs;
	return(Itc);
}

__device__ double bucal(double CaCal, double calciu, double dt)
{
	double Itc;
	if( kcalon*calciu*dt > 1 )
		Itc=0.95/dt*(Bcal-CaCal)-kcaloff*CaCal;
	else
		Itc=kcalon*calciu*(Bcal-CaCal)-kcaloff*CaCal;
	return(Itc);
}

__device__ double budye(double CaDye, double calciu, double dt)
{
	double Itc;
	if( kdyeon*calciu*dt > 1 )
		Itc=0.95/dt*(Bdye-CaDye)-kdyeoff*CaDye;
	else
		Itc=kdyeon*calciu*(Bdye-CaDye)-kdyeoff*CaDye;
	return(Itc);
}

__device__ double busr(double CaSR, double calciu, double dt)
{
	double Itc;
	if( ksron*calciu*dt > 1 )
		Itc=0.95/dt*(Bsr-CaSR)-ksroff*CaSR;
	else
		Itc=ksron*calciu*(Bsr-CaSR)-ksroff*CaSR;
	return(Itc);
}


__device__ double busar(double CaSar, double calciu, double dt)
{
	double Itc;
	if( ksaron*calciu*dt > 1 )
		Itc=0.95/dt*(Bsar-CaSar)-ksaroff*CaSar;
	else
		Itc=ksaron*calciu*(Bsar-CaSar)-ksaroff*CaSar;
	return(Itc);
}


__device__ double busarh(double CaSarh, double calciu, double dt)
{
	double Itc;
	if( ksarhon*calciu*dt > 1 )
		Itc=0.95/dt*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	else
		Itc=ksarhon*calciu*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	return(Itc);
}


__device__ double busarj(double CaSar, double calciu, double dt)
{
	double Itc;
	if( ksaron*calciu*dt > 1 )
		Itc=0.95/dt*(Bsar-CaSar)-ksaroff*CaSar;
	else
		Itc=ksaron*calciu*(Bsar-CaSar)-ksaroff*CaSar;
	return(Itc);
}


__device__ double busarhj(double CaSarh, double calciu, double dt)
{
	double Itc;
	if( ksarhon*calciu*dt > 1 )
		Itc=0.95/dt*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	else
		Itc=ksarhon*calciu*(Bsarh-CaSarh)-ksarhoff*CaSarh;
	return(Itc);
}
