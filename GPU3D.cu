#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <stdio.h>
// #include <Windows.h>
// #include <process.h>
#include <unistd.h> // linux
#include <sys/stat.h> // linux
#include <sys/types.h> // linux

#include <iomanip>

#include <cuda_runtime.h>

using namespace std;

// #define BASELINE // original TPORd model
// #define CON      // Control for casestudy
#define LQT1     // LQT1 for casestudy
#define ISO1
#define SUPREP

// HANDLE hWriteData;

//double Gks,Gto;

const int X = 690;
const int Y = 489;
const int Z = 537;

const int DimX = Y*Z;
const int DimY = Z;
const int ElementNum = X*Y*Z;


const double D1 = 0.154;
const double D2 = 0.154;


const double dx = 0.2;
const double dt = 0.02;


// ***********state values***********
// initial values for state variables, there are 41 of them (Istim included but V excluded)
const int N = 44;

// const int v;
const int nai_idx = 1;
const int nass_idx = 2;
const int ki_idx = 3;
const int kss_idx = 4;
const int cai_idx = 5;
const int cass_idx = 6;
const int cansr_idx = 7;
const int cajsr_idx = 8;
// ORd
// const int m_idx = 9;
// const int hf_idx = 10;
// const int hs_idx = 11;
// const int j_idx = 12;
// TP06
const int sm_idx = 9;
const int sh_idx = 10;
const int sj_idx = 11;

const int hsp_idx = 13;
const int jp_idx = 14;
const int mL_idx = 15;
const int hL_idx = 16;
const int hLp_idx = 17;
const int a_idx = 18;
const int iF_idx = 19;
const int iS_idx = 20;
const int ap_idx = 21;
const int iFp_idx = 22;
const int iSp_idx = 23;
const int d_idx = 24;
const int ff_idx = 25;
const int fs_idx = 26;
#ifdef ISO
const int d_pka_idx = 27;
const int ff_pka_idx = 28;
const int fs_pka_idx = 29;
#endif
const int fcaf_idx = 30;
const int fcas_idx = 31;
const int jca_idx = 32;
const int nca_idx = 33;
const int ffp_idx = 34;
const int fcafp_idx = 35;
const int xrf_idx = 36;
const int xrs_idx = 37;
const int xs1_idx = 38;
const int xs2_idx = 39;
const int xk1_idx = 40;
const int Jrelnp_idx = 41;
const int Jrelp_idx = 42;
const int CaMKt_idx = 43;
const int Istim_idx = 0;



int *g;
double *V_data;
int curCount = 0;

const int blockSize = 512;

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in " << file << " at line " << line << endl;
		char ch;
		cin >> ch;
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__))

// test code.
void writeData()
{
	// cout << " Start writing file...";

	int file_id = curCount / int(1 / dt);
	std::ostringstream os;
	os << "Output3D/ap_" << file_id << ".vtk";
	std::ofstream out(os.str().c_str(), std::ios_base::out);

	out << "# vtk DataFile Version 3.0" << std::endl;
	out << "vtk output" << std::endl;
	out << "ASCII" << std::endl;
	out << "DATASET STRUCTURED_POINTS" << std::endl;
	out << "DIMENSIONS " << X/3  << " " << Y/3  << " " << Z/3  << std::endl;
	out << "SPACING 1 1 1" << std::endl;
	out << "ORIGIN 0 0 0" << std::endl;
	out << "POINT_DATA " << (X/3)*(Y/3)*(Z/3) << std::endl;
	out << "SCALARS ImageFile float 1" << std::endl;
	out << "LOOKUP_TABLE default" << std::endl;

	for (int z = 0; z<Z; z += 3 )
	{
		for (int y = 0; y<Y; y += 3 )
		{
			for (int x = 0; x<X; x += 3 )
			{
				int this_g = g[x*DimX + y*DimY + z];
				if (this_g > 0)
					out << V_data[this_g - 1] << " ";
				else
					out << "-100 ";
			}
			out << std::endl;
		}
		out << std::endl;
	}
	out.close();

	// cout << " Done ";

}

void writeData1()
{
	// cout << " Start writing file...";

	int file_id = curCount / int(1 / dt);
	std::ostringstream os;
	os << "ap_" << file_id << ".vtk";
	std::ofstream out(os.str().c_str(), std::ios_base::out);

	out << "# vtk DataFile Version 3.0" << std::endl;
	out << "vtk output" << std::endl;
	out << "ASCII" << std::endl;
	out << "DATASET STRUCTURED_POINTS" << std::endl;
	out << "DIMENSIONS " << X / 3 << " " << Y / 3 << " " << Z / 3 << std::endl;
	out << "SPACING 1 1 1" << std::endl;
	out << "ORIGIN 0 0 0" << std::endl;
	out << "POINT_DATA " << (X / 3)*(Y / 3)*(Z / 3) << std::endl;
	out << "SCALARS ImageFile float 1" << std::endl;
	out << "LOOKUP_TABLE default" << std::endl;

	for (int z = 0; z<Z; z = z + 3)
	{
		for (int y = 0; y<Y; y = y + 3)
		{
			for (int x = 0; x<X; x = x + 3)
			{
				int this_g = g[x*DimX + y*DimY + z];
				if (this_g > 0)
					out << V_data[this_g - 1] << " ";
				else
					out << "-100 ";
			}
			out << std::endl;
		}
		out << std::endl;
	}
	out.close();

	// cout << " Done ";

}

__global__ void init_u_v(int num, double *u_v)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < num)
	{
		u_v[id] = -86.2;
	}
}



__device__ double valid_g(int _g)
{

	if (_g > 0)
	{
		// test code
		// temp = (int)(100000*u_v[_g - 1]);
		// return temp/100000.0;

		return _g;
	}

	

	else if (_g == 0)
	{
		// test code.
		return 0;
		// return -90;
	}

	// test code.
	/**
	else 
	{
		int candidate_g = g[candidate_id];
		return u_v[candidate_g - 1]; 
	}
	**/
	
	else
	{
		return -_g;
	}
}




__device__ double get_u_v(int host_id, int candidate_id, double *u_v, int *g)
{
	int _g = g[host_id];
	int temp;
	if (_g > 0) // within tissue
	{
		// test code
		// temp = (int)(100000*u_v[_g - 1]);
		// return temp/100000.0;

		return u_v[_g - 1];
	}		
	else if (_g < 0) // boundary
	{
		return u_v[-_g - 1];
		// int i, j, k;
		// for (i = -1; i <= 1; i++)
		// {
		// 	for (j = -1; j <= 1; j++)
		// 	{
		// 		for (k = -1; k <= 1; k++)
		// 		{
		// 			if (_g == -g[host_id + i*DimX + j*DimY + k])
		// 			{
		// 				// test code
		// 				// temp = (int)(100000*u_v[-_g - 1]);
		// 				// return temp/100000.0;

		// 				return u_v[-_g - 1];
		// 			}
		// 		}
		// 	}
		// }
		// return 1000000000;//-90;
	}
	else // non-tissue
	{
		// test code.
		// return 1000000000;
		return -90;  // approach 2 
	}

	// test code.
	/**
	else 
	{
		int candidate_g = g[candidate_id];
		return u_v[candidate_g - 1]; 
	}
	**/	
}

// test code

double get_u_v_host(int host_id, int candidate_id, double *u_v, int *g)
{
	int _g = g[host_id];
	int temp;
	if (_g > 0)
	{
		// test code
		// temp = (int)(100000*u_v[_g - 1]);
		// return temp/100000.0;

		return u_v[_g - 1];
	}

	else if (_g < 0)
	{
		return u_v[-_g - 1]; 
	}

	else
	{
		// test code.
		// return 1000000000;
		return -90;
	}

	// test code.
	

	
	
	
	// else
	// {
	// 	int i, j, k;
	// 	for (i = -1; i <= 1; i++)
	// 	{
	// 		for (j = -1; j <= 1; j++)
	// 		{
	// 			for (k = -1; k <= 1; k++)
	// 			{
	// 				if (_g == -g[host_id + i*DimX + j*DimY + k])
	// 				{
	// 					// test code
	// 					// temp = (int)(100000*u_v[-_g - 1]);
	// 					// return temp/100000.0;

	// 					return u_v[-_g - 1];
	// 				}
	// 			}
	// 		}
	// 	}
	// 	return 1000000000;//-90;
	// }
	
	
}


__global__ void calc_du(int totalNum, double *u_v, double *du, int *g, int *dev_rev_g)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < totalNum)
	{
		// Glory modified. test code.
		int host_id = dev_rev_g[id+1];//map[id];

		double dudx2 = (get_u_v(host_id - DimX, host_id, u_v, g) + get_u_v(host_id + DimX, host_id, u_v, g) - 2 * get_u_v(host_id, host_id, u_v, g)) / (dx*dx);
		double dudy2 = (get_u_v(host_id - DimY, host_id, u_v, g) + get_u_v(host_id + DimY, host_id, u_v, g) - 2 * get_u_v(host_id, host_id, u_v, g)) / (dx*dx);
		double dudz2 = (get_u_v(host_id - 1, host_id, u_v, g) + get_u_v(host_id + 1, host_id, u_v, g) - 2 * get_u_v(host_id, host_id, u_v, g)) / (dx*dx);		

		du[id] = D2*(dudx2 + dudy2 + dudz2);
	}
}

__global__ void init_Istim(int num, double *u)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < num)
	{
		u[id*N + Istim_idx] = 0;
	}
}

__global__ void set_Istim(int totalNum, int b_totalNum, double timeoffset, double val, double time, double *u, float *b)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < b_totalNum)
	{
		if ((time >= (timeoffset + 1000 * b[id * 2 + 1])) && (time<(timeoffset + 1000 * b[id * 2 + 1] + 3)))
		{
			int targetIdx = (int)(b[id * 2]);
			u[targetIdx*N + Istim_idx] = val;
		}
	}
}

__global__ void set_S2Istim(int totalNum, double *u, double strength, int *is_s2)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id < totalNum)
	{
		if(is_s2[id]) // position
		{
			u[id*N + Istim_idx] = strength;
		}
	}
}


__global__ void Itot(int totalNum, double dt, double *u, short *typeArray, double *Itotr, double *u_v)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;


	if (id < totalNum)
	{
		double v = u_v[id];
		// printf("v = %d\n", v);
		double &Iion = Itotr[id];
		short type = typeArray[id];

		double &nai = u[id*N + nai_idx];
		double &nass = u[id*N + nass_idx];
		double &ki = u[id*N + ki_idx];
		double &kss = u[id*N + kss_idx];
		double &cai = u[id*N + cai_idx];
		double &cass = u[id*N + cass_idx];
		double &cansr = u[id*N + cansr_idx];
		double &cajsr = u[id*N + cajsr_idx];
		// ORd
		// double &m = u[id*N + m_idx];
		// double &hf = u[id*N + hf_idx];
		// double &hs = u[id*N + hs_idx];
		// double &j = u[id*N + j_idx];
		// TP06
		double &sm = u[id*N + sm_idx];
		double &sh = u[id*N + sh_idx];
		double &sj = u[id*N + sj_idx];

		double &hsp = u[id*N + hsp_idx];
		double &jp = u[id*N + jp_idx];
		double &mL = u[id*N + mL_idx];
		double &hL = u[id*N + hL_idx];
		double &hLp = u[id*N + hLp_idx];
		double &a = u[id*N + a_idx];
		double &iF = u[id*N + iF_idx];
		double &iS = u[id*N + iS_idx];
		double &ap = u[id*N + ap_idx];
		double &iFp = u[id*N + iFp_idx];
		double &iSp = u[id*N + iSp_idx];
		double &d = u[id*N + d_idx];
		double &ff = u[id*N + ff_idx];
		double &fs = u[id*N + fs_idx];
		#ifdef ISO
		double &d_pka = u[id*N + d_pka_idx];;
		double &ff_pka = u[id*N + ff_pka_idx];;
		double &fs_pka = u[id*N + fs_pka_idx];;
		#endif
		double &fcaf = u[id*N + fcaf_idx];
		double &fcas = u[id*N + fcas_idx];
		double &jca = u[id*N + jca_idx];
		double &nca = u[id*N + nca_idx];
		double &ffp = u[id*N + ffp_idx];
		double &fcafp = u[id*N + fcafp_idx];
		double &xrf = u[id*N + xrf_idx];
		double &xrs = u[id*N + xrs_idx];
		double &xs1 = u[id*N + xs1_idx];
		double &xs2 = u[id*N + xs2_idx];
		double &xk1 = u[id*N + xk1_idx];
		double &Jrelnp = u[id*N + Jrelnp_idx];
		double &Jrelp = u[id*N + Jrelp_idx];
		double &CaMKt = u[id*N + CaMKt_idx];
		double &Istim = u[id*N + Istim_idx];

		// ***********constants*************
		double nao=140.0;//extracellular sodium in mM
		double cao=1.8;//extracellular calcium in mM
		double ko=5.4;//extracellular potassium in mM

		//buffer paramaters
		double BSRmax=0.047;
		double KmBSR=0.00087;
		double BSLmax=1.124;
		double KmBSL=0.0087;
		double cmdnmax=0.05;
		double kmcmdn=0.00238;
		double trpnmax=0.07;
		double kmtrpn=0.0005;
		#ifdef ISO
		kmtrpn *= 1.6; // O'hara et al. 2012. Heart Rhythm
		#endif
		double csqnmax=10.0;
		double kmcsqn=0.8;

		//CaMK paramaters
		double aCaMK = 0.05;
		double bCaMK = 0.00068;
		double CaMKo = 0.05;
		double KmCaM = 0.0015;
		double KmCaMK = 0.15;

		//physical constants
		double R = 8314.0;
		double T = 310.0;
		double F = 96485.0;

		//cell geometry
		double L = 0.01;
		double rad = 0.0011;
		double vcell = 1000*3.14*rad*rad*L;
		double Ageo = 2*3.14*rad*rad+2*3.14*rad*L;
		double Acap = 2*Ageo;
		double vmyo = 0.68*vcell;
		double vmito = 0.26*vcell;
		double vsr = 0.06 * vcell;
		double vnsr = 0.0552 * vcell;
		double vjsr = 0.0048 * vcell;
		double vss = 0.02 * vcell;	

		//introduce varaibles for reversal potentials, currents, fluxes, and CaMK
		double ENa,EK,EKs;
		double INa,INaL,Ito,ICaL,ICaNa,ICaK,IKr,IKs,IK1,INaCa_i,INaCa_ss,INaCa,INaK,IKb,INab,IpCa,ICab;
		double Jrel,Jup,Jtr,Jdiff,JdiffNa,JdiffK,Jleak;
		double CaMKa,CaMKb;


		// revpots(): compute reversal potentials
		ENa=(R*T/F)*log(nao/nai);
		EK=(R*T/F)*log(ko/ki);
		EKs=(R*T/F)*log((ko+0.01833*nao)/(ki+0.01833*nai));

		// RGC(): compute rates, gates, and currents
		CaMKb=CaMKo*(1.0-CaMKt)/(1.0+KmCaM/cass);
		CaMKa=CaMKb+CaMKt;
		double vffrt=v*F*F/(R*T);
		double vfrt=v*F/(R*T);

		// -----------------------ORd INa-----------------------
		/*
		double mss=1.0/(1.0+exp((-(v+39.57+10))/9.871));   // left-shift by 10 mV!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		double tm=1.0/(6.765*exp((v+11.64)/34.77)+8.552*exp(-(v+77.42)/5.955));
		m=mss-(mss-m)*exp(-dt/tm);
		double hss=1.0/(1+exp((v+82.90)/6.086));
		#ifdef ISO
	    hss=1.0/(1+exp((v+82.90 + 5.0)/6.086));  // O'hara et al. 2012. Heart Rhythm
	    #endif
		double thf=1.0/(1.432e-5*exp(-(v+1.196)/6.285)+6.149*exp((v+0.5096)/20.27));
		double ths=1.0/(0.009794*exp(-(v+17.95)/28.05)+0.3343*exp((v+5.730)/56.66));
		double Ahf=0.99;
		double Ahs=1.0-Ahf;
		hf=hss-(hss-hf)*exp(-dt/thf);
		hs=hss-(hss-hs)*exp(-dt/ths);
		double h=Ahf*hf+Ahs*hs;
		double jss=hss;
		double tj=2.038+1.0/(0.02136*exp(-(v+100.6)/8.281)+0.3052*exp((v+0.9941)/38.45));
		j=jss-(jss-j)*exp(-dt/tj);
		double hssp=1.0/(1+exp((v+89.1)/6.086));
		#ifdef ISO
	    hssp=1.0/(1+exp((v+89.1 + 5.0)/6.086));  // O'hara et al. 2012. Heart Rhythm
	    #endif
		double thsp=3.0*ths;
		hsp=hssp-(hssp-hsp)*exp(-dt/thsp);
		double hp=Ahf*hf+Ahs*hsp;
		double tjp=1.46*tj;
		jp=jss-(jss-jp)*exp(-dt/tjp);
		double GNa=75;
		double fINap=(1.0/(1.0+KmCaMK/CaMKa));
		#ifdef ISO
		GNa *= 2.7;
		#endif
		INa = GNa*(v-ENa)*m*m*m*((1.0-fINap)*h*j+fINap*hp*jp);
		*/
		

		// ----------------TP06 INa----------------------
		
		double AM, BM, TAU_M, TAU_H, TAU_J, M_INF, H_INF, J_INF, AH_1, BH_1, AH_2, BH_2, AJ_1, AJ_2, BJ_1, BJ_2;

	    AM=1./(1.+exp((-60.-v)/5.));
	    BM=0.1/(1.+exp((v+35.)/5.))+0.10/(1.+exp((v-50.)/200.));
	    TAU_M=AM*BM;
	    M_INF=1./((1.+exp((-56.86-v)/9.03))*(1.+exp((-56.86-v)/9.03)));
	    if (v>=-40.)
	    {
			AH_1=0.; 
			BH_1=(0.77/(0.13*(1.+exp(-(v+10.66)/11.1))));
			TAU_H= 1.0/(AH_1+BH_1);
		}
	    else
	    {
			AH_2=(0.057*exp(-(v+80.)/6.8));
			BH_2=(2.7*exp(0.079*v)+(3.1e5)*exp(0.3485*v));
			TAU_H=1.0/(AH_2+BH_2);
	    }
	    H_INF=1./((1.+exp((v+71.55)/7.43))*(1.+exp((v+71.55)/7.43)));
	    #ifdef ISO
	    H_INF=1./((1.+exp((v+71.55 + 5.0)/7.43))*(1.+exp((v+71.55 + 5.0)/7.43)));  // O'hara et al. 2012. Heart Rhythm
	    #endif
	    if(v>=-40.)
	    {
			AJ_1=0.;      
			BJ_1=(0.6*exp((0.057)*v)/(1.+exp(-0.1*(v+32.))));
			TAU_J= 1.0/(AJ_1+BJ_1);
	    }
	    else
	    {
			AJ_2=(((-2.5428e4)*exp(0.2444*v)-(6.948e-6)*
				exp(-0.04391*v))*(v+37.78)/
			      (1.+exp(0.311*(v+79.23))));    
			BJ_2=(0.02424*exp(-0.01052*v)/(1.+exp(-0.1378*(v+40.14))));
			TAU_J= 1.0/(AJ_2+BJ_2);
	    }
	    J_INF=H_INF;

		sm = M_INF-(M_INF-sm)*exp(-dt/TAU_M);
		sh = H_INF-(H_INF-sh)*exp(-dt/TAU_H);
		sj = J_INF-(J_INF-sj)*exp(-dt/TAU_J);
		double GNa = 14.838;
		#ifdef ISO
		GNa *= 1.3;  // O'hara et al. 2012. Heart Rhythm
		#endif
		INa = GNa*sm*sm*sm*sh*sj*(v-ENa);
		
		//-------------------------------------------------------

		  
		double mLss=1.0/(1.0+exp((-(v+42.85))/5.264));
		double tm = 1.0/(6.765*exp((v+11.64)/34.77)+8.552*exp(-(v+77.42)/5.955)); // shared by INa and INaL in ORd
		double tmL=tm;
		mL=mLss-(mLss-mL)*exp(-dt/tmL);
		double hLss=1.0/(1.0+exp((v+87.61)/7.488));
		double thL=200.0;
		hL=hLss-(hLss-hL)*exp(-dt/thL);
		double hLssp=1.0/(1.0+exp((v+93.81)/7.488));
		double thLp=3.0*thL;
		hLp=hLssp-(hLssp-hLp)*exp(-dt/thLp);
		double GNaL=0.0075;
		// INaL from ORd model
		// 3:3:4
		if ((type == 7) || (type == 8) || (type == 9) || (type == 10) || (type == 17) || (type == 18) || (type == 19) || (type == 20))  // EPI
		{
			GNaL*=0.6;
		}

		double fINaLp=(1.0/(1.0+KmCaMK/CaMKa));
		INaL=GNaL*(v-ENa)*mL*((1.0-fINaLp)*hL+fINaLp*hLp);

		double ass=1.0/(1.0+exp((-(v-14.34))/14.82));
		double ta=1.0515/(1.0/(1.2089*(1.0+exp(-(v-18.4099)/29.3814)))+3.5/(1.0+exp((v+100.0)/29.3814)));
		a=ass-(ass-a)*exp(-dt/ta);
		double iss=1.0/(1.0+exp((v+43.94)/5.711));
		double delta_epi;
		if ((type == 7) || (type == 8) || (type == 9) || (type == 10) || (type == 17) || (type == 18) || (type == 19) || (type == 20))  // EPI
		{
			delta_epi=1.0-(0.95/(1.0+exp((v+70.0)/5.0)));
		}
		else
		{
			delta_epi=1.0;
		}
		double tiF=4.562+1/(0.3933*exp((-(v+100.0))/100.0)+0.08004*exp((v+50.0)/16.59));
		double tiS=23.62+1/(0.001416*exp((-(v+96.52))/59.05)+1.780e-8*exp((v+114.1)/8.079));
		tiF*=delta_epi;
		tiS*=delta_epi;
		double AiF=1.0/(1.0+exp((v-213.6)/151.2));
		double AiS=1.0-AiF;
		iF=iss-(iss-iF)*exp(-dt/tiF);
		iS=iss-(iss-iS)*exp(-dt/tiS);
		double i=AiF*iF+AiS*iS;
		double assp=1.0/(1.0+exp((-(v-24.34))/14.82));
		ap=assp-(assp-ap)*exp(-dt/ta);
		double dti_develop=1.354+1.0e-4/(exp((v-167.4)/15.89)+exp(-(v-12.23)/0.2154));
		double dti_recover=1.0-0.5/(1.0+exp((v+70.0)/20.0));
		double tiFp=dti_develop*dti_recover*tiF;
		double tiSp=dti_develop*dti_recover*tiS;
		iFp=iss-(iss-iFp)*exp(-dt/tiFp);
		iSp=iss-(iss-iSp)*exp(-dt/tiSp);
		double ip=AiF*iFp+AiS*iSp;
		double Gto=0.02;
		// 3：3：4
		if ((type == 7) || (type == 8) || (type == 9)|| (type == 10) || (type == 17) || (type == 18) || (type == 19) || (type == 20)) // EPI
		{
			Gto *= 4.0;
		}
		if ((type == 4) || (type == 5) || (type == 6) || (type == 14) || (type == 15) || (type == 16))
		{
			Gto *= 4.0;
		}


		double fItop = (1.0/(1.0+KmCaMK/CaMKa));
		Ito = Gto*(v-EK)*((1.0-fItop)*a*i+fItop*ap*ip);

		#ifdef BASELINE
		double dss = 1.0/(1.0+exp((-(v+3.940))/4.230));
		#endif
		#if defined CON || defined LQT1
		// double dss = 1.0/(1.0+exp((-(v+3.940 - 20))/10.230)); // case study
		double dss = 1.0/(1.0+exp((-(v+3.940))/4.230)); // same as baseline
		#endif
		#ifdef ISO1
		// dss = 1.0/(1.0+exp((-(v+3.940 - 8))/9.5)); // case study - ISO
		dss = 1.0/(1.0+exp((-(v+3.940 + 12))/4.230)); // case study - ISO
		#endif

		double td = 0.6+1.0/(exp(-0.05*(v+6.0))+exp(0.09*(v+14.0)));
		d = dss-(dss-d)*exp(-dt/td);
		double fss = 1.0/(1.0+exp((v+19.58)/3.696));

		#ifdef ISO1
		fss = 1.0/(1.0+exp((v+19.58 + 8.0)/3.696)); // O'hara et al. 2012. Heart Rhythm
		#endif

		double tff = 7.0+1.0/(0.0045*exp(-(v+20.0)/10.0)+0.0045*exp((v+20.0)/10.0));
		double tfs = 1000.0+1.0/(0.000035*exp(-(v+5.0)/4.0)+0.000035*exp((v+5.0)/6.0));
		double Aff = 0.6;
		double Afs = 1.0-Aff;
		ff = fss-(fss-ff)*exp(-dt/tff);
		fs = fss-(fss-fs)*exp(-dt/tfs);
		double f = Aff*ff+Afs*fs;
		double fcass = fss;
		double tfcaf = 7.0+1.0/(0.04*exp(-(v-4.0)/7.0)+0.04*exp((v-4.0)/7.0));
		double tfcas = 100.0+1.0/(0.00012*exp(-v/3.0)+0.00012*exp(v/7.0));
		double Afcaf = 0.3+0.6/(1.0+exp((v-10.0)/10.0));
		double Afcas = 1.0-Afcaf;
		fcaf=fcass-(fcass-fcaf)*exp(-dt/tfcaf);
		fcas=fcass-(fcass-fcas)*exp(-dt/tfcas);
		double fca=Afcaf*fcaf+Afcas*fcas;
		double tjca=75.0;
		jca=fcass-(fcass-jca)*exp(-dt/tjca);
		double tffp=2.5*tff;
		ffp=fss-(fss-ffp)*exp(-dt/tffp);
		double fp=Aff*ffp+Afs*fs;
		double tfcafp=2.5*tfcaf;
		fcafp=fcass-(fcass-fcafp)*exp(-dt/tfcafp);
		double fcap=Afcaf*fcafp+Afcas*fcas;
		double Kmn=0.002;
		double k2n=1000.0;
		double km2n=jca*1.0;
		double anca=1.0/(k2n/km2n+pow(1.0+Kmn/cass,4.0));
		nca=anca*k2n/km2n-(anca*k2n/km2n-nca)*exp(-km2n*dt);
		// #ifdef ISO
		// if (cass > 0.03 /*mM*/) cass = 0.03; // O'hara et al. 2012. Heart Rhythm
		// #endif
		double PhiCaL=4.0*vffrt*(cass*exp(2.0*vfrt)-0.341*cao)/(exp(2.0*vfrt)-1.0);
		double PhiCaNa=1.0*vffrt*(0.75*nass*exp(1.0*vfrt)-0.75*nao)/(exp(1.0*vfrt)-1.0);
		double PhiCaK=1.0*vffrt*(0.75*kss*exp(1.0*vfrt)-0.75*ko)/(exp(1.0*vfrt)-1.0);
		double zca=2.0;
		double PCa=0.0001;

		#ifdef ISO1
		PCa *= 4; // 3.5 FOR 2D;   4 For 3D(AND BCL@500)
		#endif

		// 3;3;4
		if ((type == 7) || (type == 8) || (type == 9) || (type == 10) || (type == 17) || (type == 18) || (type == 19) || (type == 20))  // EPI
		{
			PCa*=1.2;
		}
		if ((type == 4) || (type == 5) || (type == 6) || (type == 14) || (type == 15) || (type == 16))  // MCELL
		{
			PCa *= 2.5;
		}
		// #ifdef ISO
		// PCa *= 4.4; // O'hara et al. 2012. Heart Rhythm  // safe critical 3.8(LQT1000) (LQT500)  4.6(CON1000) 
		// #endif

		double PCap = 1.1*PCa;
		double PCaNa = 0.00125*PCa;
		double PCaK = 3.574e-4*PCa;
		double PCaNap = 0.00125*PCap;
		double PCaKp = 3.574e-4*PCap;
		double fICaLp = (1.0/(1.0+KmCaMK/CaMKa));
		ICaL = (1.0-fICaLp)*PCa*PhiCaL*d*(f*(1.0-nca)+jca*fca*nca)+fICaLp*PCap*PhiCaL*d*(fp*(1.0-nca)+jca*fcap*nca);

		// -------------------------------
		#ifdef ISO
		double dss_pka = 1.0/(1.0+exp((-(v+3.940 + 16.0))/4.230)); // O'hara et al. 2012. Heart Rhythm
		d_pka = dss_pka-(dss_pka - d_pka)*exp(-dt/td);
		
		double fss_pka = 1.0/(1.0+exp((v+19.58 + 8.0)/3.696)); // O'hara et al. 2012. Heart Rhythm
		ff_pka = fss_pka-(fss_pka-ff_pka)*exp(-dt/tff);
		fs_pka = fss_pka-(fss_pka-fs_pka)*exp(-dt/tfs);
		double f_pka = Aff*ff_pka+Afs*fs_pka;

		double cass_pka = cass;
		if (cass > 0.03 /*mM*/) cass_pka = 0.03; // O'hara et al. 2012. Heart Rhythm
		double PhiCaL_pka = 4.0*vffrt*(cass_pka*exp(2.0*vfrt)-0.341*cao)/(exp(2.0*vfrt)-1.0);
		double PCa_pka = 2.5*PCa; // O'hara et al. 2012. Heart Rhythm

		double ICaL_pka = PCa_pka*PhiCaL_pka*d_pka*(f_pka*(1.0-nca)+jca*fca*nca);
		// safe critical for single cell: 0.29(CON1000)  0.28(CON500)   0.25(LQT1000)  0.24(LQT500)
		// safe critical for 1-D tissue : 0.63(CON1000)  0.61(CON500)   0.61(LQT1000)  0.56(LQT500)
		//                                               0.69(CON500)
		// safe critical for 2-D idealiz: (CON1000)  0.23(CON500)   0.2(LQT1000)0.21_2366  0.19(LQT500)0.2_1900/0.21_1500
		// safe critical for 2-D idealiz: (CON1000)  (CON500)   (LQT1000)  (LQT500)0.2
		double frac_pka = 0.2;
		ICaL = (1-frac_pka)*ICaL + frac_pka*ICaL_pka;
		#endif
		// -------------------------------


		ICaNa = (1.0-fICaLp)*PCaNa*PhiCaNa*d*(f*(1.0-nca)+jca*fca*nca)+fICaLp*PCaNap*PhiCaNa*d*(fp*(1.0-nca)+jca*fcap*nca);
		ICaK = (1.0-fICaLp)*PCaK*PhiCaK*d*(f*(1.0-nca)+jca*fca*nca)+fICaLp*PCaKp*PhiCaK*d*(fp*(1.0-nca)+jca*fcap*nca);

		double xrss = 1.0/(1.0+exp((-(v+8.337))/6.789));
		double txrf = 12.98+1.0/(0.3652*exp((v-31.66)/3.869)+4.123e-5*exp((-(v-47.78))/20.38));
		double txrs = 1.865+1.0/(0.06629*exp((v-34.70)/7.355)+1.128e-5*exp((-(v-29.74))/25.94));
		double Axrf = 1.0/(1.0+exp((v+54.81)/38.21));
		double Axrs = 1.0-Axrf;
		xrf = xrss-(xrss-xrf)*exp(-dt/txrf);
		xrs = xrss-(xrss-xrs)*exp(-dt/txrs);
		double xr = Axrf*xrf+Axrs*xrs;
		double rkr = 1.0/(1.0+exp((v+55.0)/75.0))*1.0/(1.0+exp((v-10.0)/30.0));
		double GKr = 0.046;  //0.046

		// 3;3;4
		if ((type == 7) || (type == 8) || (type == 9) || (type == 10) || (type == 17) || (type == 18) || (type == 19) || (type == 20))  // EPI
		{
			GKr*=1.3;
		}
		if ((type == 4) || (type == 5) || (type == 6) || (type == 14) || (type == 15) || (type == 16))  // MCELL
		{
			GKr*=0.8;
		}

		IKr=GKr*sqrt(ko/5.4)*xr*rkr*(v-EK);

		#ifdef BASELINE
		double xs1ss=1.0/(1.0+exp((-(v+11.60))/8.932));  // orignal
		#else
		// double xs1ss=1.0/(1.0+exp((-(v+11.60 -45))/20));  // case study
		double xs1ss=1.0/(1.0+exp((-(v+11.60))/8.932));  // orignal 
		#endif

		double txs1=817.3+1.0/(2.326e-4*exp((v+48.28)/17.80)+0.001292*exp((-(v+210.0))/230.0));
		xs1 = xs1ss-(xs1ss-xs1)*exp(-dt/txs1);
		double xs2ss = xs1ss;
		double txs2=1.0/(0.01*exp((v-50.0)/20.0)+0.0193*exp((-(v+66.54))/31.0));
		xs2 = xs2ss-(xs2ss-xs2)*exp(-dt/txs2);
		double KsCa=1.0+0.6/(1.0+pow(3.8e-5/cai,1.4));		
		#ifdef BASELINE 
		double GKs = 0.0034;
		#else  // CON
		double GKs = 0.0034; // to achieve the APD shortening @ BCL 500ms
		// double GKs = 5.4*0.0034; // O'hara et al. 2012. Heart Rhythm; recabrilated according to ORd model
		#endif


		//  1  2  3      LV ENDO
		//  4  5  6      LV MID
		//  7  8  9 10   LV EPI

		// 11 12 13      RV ENDO
		// 14 15 16      RV MID
		// 17 18 19 20   RV EPI

		// 3:3:4
		bool isEpi = false;			
		if((type == 7)||(type == 8)||(type == 9)||(type == 10)||(type == 17)||(type == 18)||(type == 19)||(type == 20)) // EPI
		{
			GKs *= 1.4;
			isEpi = true;
		}	

		#ifdef LQT1
		GKs = 0.05*GKs;
		#endif

		#ifdef SUPREP
		double GKs_sup = GKs; // current GKs to be supressed
		double GKs_rep = isEpi? 0.0034*1.4 : 0.0034; // the replacing GKs
		double dose = 100; // fmol
		
		// PART I: the fraction of GKs_sup
		double frac_wt = 1.0;
		if(dose >= 25)
			frac_wt = 3.129*pow(dose,-0.560);
		else
			frac_wt = pow(0.974,dose);
		
		// PART II: the fraction of GKs_rep
		double frac_shIMM = 2.018*(1.0-exp(-2.0*dose*0.0059))/(1.0+exp(-2.0*dose*0.0059)); // the fraction of the replacing KCNQ1
		
		// PART III: final output
		GKs = frac_wt*GKs_sup + frac_shIMM*GKs_rep;

		// printf("GKs:%f\n", GKs/GKs_rep);
		#endif


		#ifdef ISO1
		GKs *= 3.2;  // O'hara et al. 2012. Heart Rhythm
		#endif

		IKs = GKs*KsCa*xs1*xs2*(v-EKs);


		double xk1ss=1.0/(1.0+exp(-(v+2.5538*ko+144.59)/(1.5692*ko+3.8115)));
		double txk1=122.2/(exp((-(v+127.2))/20.36)+exp((v+236.8)/69.33));
		xk1=xk1ss-(xk1ss-xk1)*exp(-dt/txk1);
		double rk1=1.0/(1.0+exp((v+105.8-2.6*ko)/9.493));
		double GK1=0.1908;

		if((type == 7)||(type == 8)||(type == 9)||(type == 10)||(type == 17)||(type == 18)||(type == 19)||(type == 20)) // EPI
		{	
			GK1*=1.2;
		}
		if ((type == 4) || (type == 5) || (type == 6) || (type == 14) || (type == 15) || (type == 16))  // MCELL
		{
			GK1*=1.3;
		}
		IK1=GK1*sqrt(ko)*rk1*xk1*(v-EK);

		double kna1=15.0;
		double kna2=5.0;
		double kna3=88.12;
		double kasymm=12.5;
		double wna=6.0e4;
		double wca=6.0e4;
		double wnaca=5.0e3;
		double kcaon=1.5e6;
		double kcaoff=5.0e3;
		double qna=0.5224;
		double qca=0.1670;
		double hca=exp((qca*v*F)/(R*T));
		double hna=exp((qna*v*F)/(R*T));
		double h1=1+nai/kna3*(1+hna);
		double h2=(nai*hna)/(kna3*h1);
		double h3=1.0/h1;
		double h4=1.0+nai/kna1*(1+nai/kna2);
		double h5=nai*nai/(h4*kna1*kna2);
		double h6=1.0/h4;
		double h7=1.0+nao/kna3*(1.0+1.0/hna);
		double h8=nao/(kna3*hna*h7);
		double h9=1.0/h7;
		double h10=kasymm+1.0+nao/kna1*(1.0+nao/kna2);
		double h11=nao*nao/(h10*kna1*kna2);
		double h12=1.0/h10;
		double k1=h12*cao*kcaon;
		double k2=kcaoff;
		double k3p=h9*wca;
		double k3pp=h8*wnaca;
		double k3=k3p+k3pp;
		double k4p=h3*wca/hca;
		double k4pp=h2*wnaca;
		double k4=k4p+k4pp;
		double k5=kcaoff;
		double k6=h6*cai*kcaon;
		double k7=h5*h2*wna;
		double k8=h8*h11*wna;
		double x1=k2*k4*(k7+k6)+k5*k7*(k2+k3);
		double x2=k1*k7*(k4+k5)+k4*k6*(k1+k8);
		double x3=k1*k3*(k7+k6)+k8*k6*(k2+k3);
		double x4=k2*k8*(k4+k5)+k3*k5*(k1+k8);
		double E1=x1/(x1+x2+x3+x4);
		double E2=x2/(x1+x2+x3+x4);
		double E3=x3/(x1+x2+x3+x4);
		double E4=x4/(x1+x2+x3+x4);
		double KmCaAct=150.0e-6;
		double allo=1.0/(1.0+pow(KmCaAct/cai,2.0));
		double zna=1.0;
		double JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp;
		double JncxCa=E2*k2-E1*k1;
		double Gncx=0.0008;


		if((type == 7)||(type == 8)||(type == 9)||(type == 10)||(type == 17)||(type == 18)||(type == 19)||(type == 20)) // EPI
		{	
			Gncx*=1.1;
		}
		if ((type == 4) || (type == 5) || (type == 6) || (type == 14) || (type == 15) || (type == 16))  // MCELL
		{
			Gncx*=1.4;
		}
		INaCa_i=0.8*Gncx*allo*(zna*JncxNa+zca*JncxCa);

		h1=1+nass/kna3*(1+hna);
		h2=(nass*hna)/(kna3*h1);
		h3=1.0/h1;
		h4=1.0+nass/kna1*(1+nass/kna2);
		h5=nass*nass/(h4*kna1*kna2);
		h6=1.0/h4;
		h7=1.0+nao/kna3*(1.0+1.0/hna);
		h8=nao/(kna3*hna*h7);
		h9=1.0/h7;
		h10=kasymm+1.0+nao/kna1*(1+nao/kna2);
		h11=nao*nao/(h10*kna1*kna2);
		h12=1.0/h10;
		k1=h12*cao*kcaon;
		k2=kcaoff;
		k3p=h9*wca;
		k3pp=h8*wnaca;
		k3=k3p+k3pp;
		k4p=h3*wca/hca;
		k4pp=h2*wnaca;
		k4=k4p+k4pp;
		k5=kcaoff;
		k6=h6*cass*kcaon;
		k7=h5*h2*wna;
		k8=h8*h11*wna;
		x1=k2*k4*(k7+k6)+k5*k7*(k2+k3);
		x2=k1*k7*(k4+k5)+k4*k6*(k1+k8);
		x3=k1*k3*(k7+k6)+k8*k6*(k2+k3);
		x4=k2*k8*(k4+k5)+k3*k5*(k1+k8);
		E1=x1/(x1+x2+x3+x4);
		E2=x2/(x1+x2+x3+x4);
		E3=x3/(x1+x2+x3+x4);
		E4=x4/(x1+x2+x3+x4);
		KmCaAct=150.0e-6;
		allo=1.0/(1.0+pow(KmCaAct/cass,2.0));
		JncxNa=3.0*(E4*k7-E1*k8)+E3*k4pp-E2*k3pp;
		JncxCa=E2*k2-E1*k1;
		INaCa_ss=0.2*Gncx*allo*(zna*JncxNa+zca*JncxCa);

		INaCa=INaCa_i+INaCa_ss;

		double k1p=949.5;
		double k1m=182.4;
		double k2p=687.2;
		double k2m=39.4;
		k3p=1899.0;
		double k3m=79300.0;
		k4p=639.0;
		double k4m=40.0;
		double Knai0=9.073;
		#ifdef ISO
		Knai0 *= 0.7;  // O'hara et al. 2012. Heart Rhythm
		#endif
		double Knao0=27.78;
		double delta=-0.1550;
		double Knai=Knai0*exp((delta*v*F)/(3.0*R*T));
		double Knao=Knao0*exp(((1.0-delta)*v*F)/(3.0*R*T));
		double Kki=0.5;
		double Kko=0.3582;
		double MgADP=0.05;
		double MgATP=9.8;
		double Kmgatp=1.698e-7;
		double H=1.0e-7;
		double eP=4.2;
		double Khp=1.698e-7;
		double Knap=224.0;
		double Kxkur=292.0;
		double P=eP/(1.0+H/Khp+nai/Knap+ki/Kxkur);
		double a1=(k1p*pow(nai/Knai,3.0))/(pow(1.0+nai/Knai,3.0)+pow(1.0+ki/Kki,2.0)-1.0);
		double b1=k1m*MgADP;
		double a2=k2p;
		double b2=(k2m*pow(nao/Knao,3.0))/(pow(1.0+nao/Knao,3.0)+pow(1.0+ko/Kko,2.0)-1.0);
		double a3=(k3p*pow(ko/Kko,2.0))/(pow(1.0+nao/Knao,3.0)+pow(1.0+ko/Kko,2.0)-1.0);
		double b3=(k3m*P*H)/(1.0+MgATP/Kmgatp);
		double a4=(k4p*MgATP/Kmgatp)/(1.0+MgATP/Kmgatp);
		double b4=(k4m*pow(ki/Kki,2.0))/(pow(1.0+nai/Knai,3.0)+pow(1.0+ki/Kki,2.0)-1.0);
		x1=a4*a1*a2+b2*b4*b3+a2*b4*b3+b3*a1*a2;
		x2=b2*b1*b4+a1*a2*a3+a3*b1*b4+a2*a3*b4;
		x3=a2*a3*a4+b3*b2*b1+b2*b1*a4+a3*a4*b1;
		x4=b4*b3*b2+a3*a4*a1+b2*a4*a1+b3*b2*a1;
		E1=x1/(x1+x2+x3+x4);
		E2=x2/(x1+x2+x3+x4);
		E3=x3/(x1+x2+x3+x4);
		E4=x4/(x1+x2+x3+x4);
		double zk=1.0;
		double JnakNa=3.0*(E1*a3-E2*b3);
		double JnakK=2.0*(E4*b1-E3*a1);
		double Pnak=30;

		if((type == 7)||(type == 8)||(type == 9)||(type == 10)||(type == 17)||(type == 18)||(type == 19)||(type == 20)) // EPI
		{	
		    Pnak*=0.9;
		}
		if ((type == 4) || (type == 5) || (type == 6) || (type == 14) || (type == 15) || (type == 16))  // MCELL
		{
		    Pnak*=0.7;
		}
		INaK=Pnak*(zna*JnakNa+zk*JnakK);

		double xkb=1.0/(1.0+exp(-(v-14.48)/18.34));
		double GKb=0.003;  
		if((type == 7)||(type == 8)||(type == 9)||(type == 10)||(type == 17)||(type == 18)||(type == 19)||(type == 20)) // EPI
		{	
		    GKb*=0.6;
		}
		#ifdef ISO
		GKb *= 2.5; // O'hara et al. 2012. Heart Rhythm
		#endif
		IKb=GKb*xkb*(v-EK);

		double PNab=3.75e-10;
		INab=PNab*vffrt*(nai*exp(vfrt)-nao)/(exp(vfrt)-1.0);

		double PCab=2.5e-8;
		ICab=PCab*4.0*vffrt*(cai*exp(2.0*vfrt)-0.341*cao)/(exp(2.0*vfrt)-1.0);

		double GpCa=0.0005;
		IpCa=GpCa*cai/(0.0005+cai);

		
		// voltage(): calculate the new membrane voltage
		Iion = INa+INaL+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa+INaK+INab+IKb+IpCa+ICab + Istim;
		
		
		// FBC(): calculate fluxes, buffers, and concentrations
		CaMKb = CaMKo*(1.0-CaMKt)/(1.0+KmCaM/cass);
		CaMKa = CaMKb+CaMKt;
		CaMKt += dt*(aCaMK*CaMKb*(CaMKb+CaMKt)-bCaMK*CaMKt);

		JdiffNa = (nass-nai)/2.0;
		JdiffK = (kss-ki)/2.0;
		Jdiff = (cass-cai)/0.2;


		// --------Jrel---------
		// 1. ---not phos---
		double bt = 4.75;
		double a_rel = 0.5*bt;
		#ifdef ISO
		a_rel *= 1.75; // O'hara et al. 2012. Heart Rhythm
		#endif
		double Jrel_inf = a_rel*(-ICaL)/(1.0+pow(1.5/cajsr,8.0));

		if ((type == 4) || (type == 5) || (type == 6) || (type == 14) || (type == 15) || (type == 16))  // MCELL
		{
			Jrel_inf *= 1.7;
		}

		double tau_rel = bt/(1.0+0.0123/cajsr);
		#ifdef ISO
		tau_rel *= 0.5;// O'hara et al. 2012. Heart Rhythm
		#endif
		if (tau_rel < 0.005)
		{
			tau_rel = 0.005;
		}
		Jrelnp=Jrel_inf-(Jrel_inf-Jrelnp)*exp(-dt/tau_rel);
		// 2. ---phos---
		double btp=1.25*bt;
		double a_relp = 0.5*btp;
		#ifdef ISO
		a_relp *= 1.75; // O'hara et al. 2012. Heart Rhythm
		#endif
		double Jrel_infp=a_relp*(-ICaL)/(1.0+pow(1.5/cajsr,8.0));

		if ((type == 4) || (type == 5) || (type == 6) || (type == 14) || (type == 15) || (type == 16))  // MCELL
		{
			Jrel_infp*=1.7;
		}
		double tau_relp=btp/(1.0+0.0123/cajsr);
		#ifdef ISO
		tau_relp *= 0.5;// O'hara et al. 2012. Heart Rhythm
		#endif
		if (tau_relp < 0.005)
		{
			tau_relp = 0.005;
		}
		Jrelp=Jrel_infp-(Jrel_infp-Jrelp)*exp(-dt/tau_relp); // phos
		// ratio
		double fJrelp=(1.0/(1.0+KmCaMK/CaMKa));
		Jrel=(1.0-fJrelp)*Jrelnp + fJrelp*Jrelp;

		// -------Jup-------
		double Jupnp=0.004375*cai/(cai+0.00092);		
		double Jupp=2.75*0.004375*cai/(cai+0.00092-0.00017);
		#ifdef ISO
		Jupnp = 0.004375*cai/(cai+0.54*0.00092);	// O'hara et al. 2012. Heart Rhythm	
		Jupp = 2.75*0.004375*cai/(cai+0.54*(0.00092-0.00017)); // O'hara et al. 2012. Heart Rhythm		
		#endif
		if((type == 7)||(type == 8)||(type == 9)||(type == 10)||(type == 17)||(type == 18)||(type == 19)||(type == 20)) // EPI
		{	
			Jupnp*=1.3;
			Jupp*=1.3;
		}
		double fJupp=(1.0/(1.0+KmCaMK/CaMKa));
		Jleak=0.0039375*cansr/15.0;
		Jup=(1.0-fJupp)*Jupnp + fJupp*Jupp-Jleak;

		Jtr=(cansr-cajsr)/100.0;

		nai+=dt*(-(INa+INaL+3.0*INaCa_i+3.0*INaK+INab)*Acap/(F*vmyo)+JdiffNa*vss/vmyo);
		nass+=dt*(-(ICaNa+3.0*INaCa_ss)*Acap/(F*vss)-JdiffNa);

		ki+=dt*(-(Ito+IKr+IKs+IK1+IKb+Istim-2.0*INaK)*Acap/(F*vmyo)+JdiffK*vss/vmyo);
		kss+=dt*(-(ICaK)*Acap/(F*vss)-JdiffK);

		double Bcai;
		if((type == 7)||(type == 8)||(type == 9)||(type == 10)||(type == 17)||(type == 18)||(type == 19)||(type == 20)) // EPI
		{	
			Bcai=1.0/(1.0+1.3*cmdnmax*kmcmdn/pow(kmcmdn+cai,2.0)+trpnmax*kmtrpn/pow(kmtrpn+cai,2.0));
		}
		else
		{
			Bcai=1.0/(1.0+cmdnmax*kmcmdn/pow(kmcmdn+cai,2.0)+trpnmax*kmtrpn/pow(kmtrpn+cai,2.0));
		}
		cai+=dt*(Bcai*(-(IpCa+ICab-2.0*INaCa_i)*Acap/(2.0*F*vmyo)-Jup*vnsr/vmyo+Jdiff*vss/vmyo));

		double Bcass=1.0/(1.0+BSRmax*KmBSR/pow(KmBSR+cass,2.0)+BSLmax*KmBSL/pow(KmBSL+cass,2.0));
		cass+=dt*(Bcass*(-(ICaL-2.0*INaCa_ss)*Acap/(2.0*F*vss)+Jrel*vjsr/vss-Jdiff));

		cansr+=dt*(Jup-Jtr*vjsr/vnsr);

		double Bcajsr=1.0/(1.0+csqnmax*kmcsqn/pow(kmcsqn+cajsr,2.0));
		cajsr+=dt*(Bcajsr*(Jtr-Jrel));
	
	}


}

__global__ void new_u_v(int num, double *u_v, double dt, double *du, double *itotr)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id < num)
	{
		u_v[id] += dt*(du[id] - itotr[id]);
	}
}



















int main(int argc, char **argv)
{

	HANDLE_ERROR(cudaSetDevice(1));
	int isbound;
	int gg[27];

	double root2 = sqrt(2.0);
	double root3 = sqrt(3.0);
	double ic, ir, il, imax;
	double tflt;  // 0 Mb


	g = new int[ElementNum];
	int *dev_g;
	HANDLE_ERROR(cudaMalloc((void**)&dev_g, sizeof(int)*ElementNum)); // 829 Mb


	for (int i = 0; i<ElementNum; i++)
	{
		g[i] = 0;
	}


	cout << "Reading file..." << endl;
	FILE *fp_ventricle;
	fp_ventricle = fopen("wholeventriclewithfibervector.txt", "r");
	int num = 0;

	while (!feof(fp_ventricle))
	{	// the last line in the file MUST NOT be empty.
		int t1, t2, t3, t7;
		float t4, t5, t6;
		fscanf(fp_ventricle, "%d %d %d %f %f %f %d", &t1, &t2, &t3, &t4, &t5, &t6, &t7);

		// cout << t1 << " " << t2 << " " << t3 << endl;

		g[(t1)*DimX + (t2)*DimY + t3] = 1;

		// boundary check
		if(t1 == 0 || t1 == X-1 || t2 == 0 || t2 == Y-1 || t3 == 0 || t3 == Z-1)
			if (t7 != 0)  cout << "Tissue exists in geometry boundary!" << endl;

		num = num + 1;
	}
	cout << "There are " << num << " ventricular points." << endl;
	cout.flush();
	fclose(fp_ventricle);


	// test code.
	int *rev_g = new int[num+1];
	int *dev_rev_g;
	HANDLE_ERROR(cudaMalloc((void**)&dev_rev_g, sizeof(int)*(num+1))); // 923 Mb



	int num1 = 1;
	for (int x = 0; x < X; x++)
	{
		for (int y = 0; y < Y; y++)
		{
			for (int z = 0; z < Z; z++)
			{
				if (g[x*DimX + y*DimY + z]>0)
				{

					if(num1 == 1118236+1) //22579871+1)  //646 293 9
					{
						cout<< x << " " << y << " " << z << endl;
						cout<< x << " " << y << " " << z << endl;
					}

					g[x*DimX + y*DimY + z] = num1; // start from 1 NOT 0
					rev_g[num1] = x*DimX + y*DimY + z;
					num1++;


				}
			}
		}
	}
	if (num != num1 - 1)
		cout << "Duplicated Points Found!" << endl;


	V_data = new double[num];

	short *type, *dev_type;
	type = new short[num];
	HANDLE_ERROR(cudaMalloc((void**)&dev_type, sizeof(short)*num));

	int *is_s2, *dev_is_s2;
	is_s2 = new int[num];
	HANDLE_ERROR(cudaMalloc((void**)&dev_is_s2, sizeof(int)*num));

	double *dev_u_v;
	HANDLE_ERROR(cudaMalloc((void**)&dev_u_v, sizeof(double)*num));

	double *dev_du;
	HANDLE_ERROR(cudaMalloc((void**)&dev_du, sizeof(double)*num));

	double *dev_Itotr;
	HANDLE_ERROR(cudaMalloc((void**)&dev_Itotr, sizeof(double)*num));  // 2181 Mb


	// test code. 
	double *Itotr = new double[num];
	double *du = new double[num]; 

	double *u, *dev_u;
	u = new double[num*N];


	for (int i = 0; i<num; i++)
	{
		u[i*N + nai_idx] = 7;
		u[i*N + nass_idx] = 7;
		u[i*N + ki_idx] = 145;
		u[i*N + kss_idx] = 145;
		u[i*N + cai_idx] = 1.0e-4;
		u[i*N + cass_idx] = 1.0e-4;
		u[i*N + cansr_idx] = 1.2;
		u[i*N + cajsr_idx] = 1.2;
		// ORd
		// u[i*N + m_idx] = 0;
		// u[i*N + hf_idx] = 1;
		// u[i*N + hs_idx] = 1;
		// u[i*N + j_idx] = 1;
		// TP06
		u[i*N + sm_idx] = 0;
		u[i*N + sh_idx] = 1;
		u[i*N + sj_idx] = 1;

		u[i*N + hsp_idx] = 1;
		u[i*N + jp_idx] = 1;
		u[i*N + mL_idx] = 0;
		u[i*N + hL_idx] = 1;
		u[i*N + hLp_idx] = 1;
		u[i*N + a_idx] = 0;
		u[i*N + iF_idx] = 1;
		u[i*N + iS_idx] = 1;
		u[i*N + ap_idx] = 0;
		u[i*N + iFp_idx] = 1;
		u[i*N + iSp_idx] = 1;
		u[i*N + d_idx] = 0;
		u[i*N + ff_idx] = 1;
		u[i*N + fs_idx] = 1;
		#ifdef ISO
		u[i*N + d_pka_idx] = 0;
		u[i*N + ff_pka_idx] = 1;
		u[i*N + fs_pka_idx] = 1;
		#endif
		u[i*N + fcaf_idx] = 1;
		u[i*N + fcas_idx] = 1;
		u[i*N + jca_idx] = 1;
		u[i*N + nca_idx] = 0;
		u[i*N + ffp_idx] = 1;
		u[i*N + fcafp_idx] = 1;
		u[i*N + xrf_idx] = 0;
		u[i*N + xrs_idx] = 0;
		u[i*N + xs1_idx] = 0;
		u[i*N + xs2_idx] = 0;
		u[i*N + xk1_idx] = 1;
		u[i*N + Jrelnp_idx] = 0;
		u[i*N + Jrelp_idx] = 0;
		u[i*N + CaMKt_idx] = 0;
	}
	cudaMalloc((void**)&dev_u, sizeof(double)*N*num); // 7897 Mb
	cudaMemcpy(dev_u, u, sizeof(double)*N*num, cudaMemcpyHostToDevice); // 7897 Mb


	init_u_v << <(num + blockSize - 1) / blockSize, blockSize >> >(num, dev_u_v);

	fp_ventricle = fopen("wholeventriclewithfibervector.txt", "r");
	cout << "Rescan ...";
	cout.flush();
	int first = 1; // test !!!!!
	while (!feof(fp_ventricle))
	{
		int t1, t2, t3, t7;
		double t4, t5, t6;
		fscanf(fp_ventricle, "%d %d %d %lf %lf %lf %d", &t1, &t2, &t3, &t4, &t5, &t6, &t7);

		int index = g[t1*DimX + t2*DimY + t3] - 1;

		// set s22222222222222222222222222222222222222222222
		// if((t1 >= 300 && t1 < 500) && (t2 >= 150 && t2 < 400) && (t3 >= 160 && t3 < 170) && t5 == 13)
		if((t1 >= 100*3 && t1 < 229*3) && (t2 >= 40*3 && t2 < 162*3) && (t3 >= 140*3 && t3 < 178*3)     )//&& (t7 == 9 || t7 == 10 || t7 == 19 || t7 == 20)) // ismail
		// if((t1 >= 0*3 && t1 < 150*3) && (t2 >= 110*3 && t2 < 140*3) && (t3 >= 20*3 && t3 < 80*3) && (t7 == 9 || t7 == 10 || t7 == 19 || t7 == 20)) // ismail expand
			is_s2[index] = 1;
		else
			is_s2[index] = 0;


		// ---set the M-island for case study---
		// 1. All MCELLs in RV are cancelled and assigned as ENDO or EPI in a ratio of 5:5 
		if(t7 >= 4 && t7 <= 6)
			if(t5 <= 4)
				t7 = 2;
			else
				t7 = 7;

		// 2. For LV, only a small portion of MCELLs is kept to form an "island"
		double dist = sqrt(pow((t1 - 199*3),2.0) + pow((t2 - 112*3),2.0) + pow((t3 - 112*3),2.0));
		if(t7 >= 14 && t7 <= 16) // MID cells			
			if(dist < 25*3) // within circular area with radius of 15 (visulized from paraview)   // 25 not ok; try 40
				t7 = t7;// remain the same
			else if(t7 <= 14)
				t7 = 12;// set as ENDO
			else if(t7 > 14)
				t7 = 17;






		type[index] = t7;
	}
	cout << "Done" << endl;
	cout.flush();
	fclose(fp_ventricle);

	HANDLE_ERROR(cudaMemcpy(dev_type, type, sizeof(short)*num, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_is_s2, is_s2, sizeof(int)*num, cudaMemcpyHostToDevice));


	int i, j, k;

	// test coce. following 3 lines for approach 1.
	double dislow, dis;
	double normalx, normaly, normalz;
	int nnodex, nnodey, nnodez, xcnt, ycnt, zcnt;

	// cout << "test boundary" << g[(X)*DimX+(Y)*DimY+Z] << endl;

	for (int x = 0; x<X; x++)
	{		
		for (int y = 0; y<Y; y++)
		{			
			for (int z = 0; z<Z; z++)
			{				
				if (g[x*DimX+y*DimY+z] == 0)
				{
					if(x-1 >= 0 && y-1 >= 0 && z-1 >= 0) gg[1] = g[(x - 1)*DimX + (y - 1)*DimY + (z - 1)];
					else gg[1] = 0;
					if(x-1 >= 0 && y-1 >= 0) gg[2] = g[(x - 1)*DimX + (y - 1)*DimY + (z)];
					else gg[2] = 0;
					if(x-1 >= 0 && y-1 >= 0 && z+1 < Z) gg[3] = g[(x - 1)*DimX + (y - 1)*DimY + (z + 1)];
					else gg[3] = 0;
					if(x-1 >= 0 && z-1 >= 0) gg[4] = g[(x - 1)*DimX + (y)*DimY + (z - 1)];
					else gg[4] = 0;
					if(x-1 >= 0) gg[5] = g[(x - 1)*DimX + (y)*DimY + (z)];
					else gg[5] = 0;
					if(x-1 >= 0 && z+1 < Z) gg[6] = g[(x - 1)*DimX + (y)*DimY + (z + 1)];
					else gg[6] = 0;
					if(x-1 >= 0 && y+1 < Y && z-1 >=0) gg[7] = g[(x - 1)*DimX + (y + 1)*DimY + (z - 1)];
					else gg[7] = 0;
					if(x-1 >= 0 && y+1 < Y) gg[8] = g[(x - 1)*DimX + (y + 1)*DimY + (z)];
					else gg[8] = 0;
					if(x-1 >= 0 && y+1 < Y && z+1 < Z) gg[9] = g[(x - 1)*DimX + (y + 1)*DimY + (z + 1)];
					else gg[9] = 0;

					if(y-1 >=0 && z-1 >=0) gg[10] = g[(x)*DimX + (y - 1)*DimY + (z - 1)];
					else gg[10] = 0;
					if(y-1 >=0) gg[11] = g[(x)*DimX + (y - 1)*DimY + (z)];
					else gg[11] = 0;
					if(y-1 >=0 && z+1 < Z) gg[12] = g[(x)*DimX + (y - 1)*DimY + (z + 1)];
					else gg[12] = 0;
					if(z-1 >=0) gg[13] = g[(x)*DimX + (y)*DimY + (z - 1)];
					else gg[13] = 0;
					if(z+1 < Z) gg[14] = g[(x)*DimX + (y)*DimY + (z + 1)];
					else gg[14] = 0;
					if(y+1 < Y && z-1 >=0) gg[15] = g[(x)*DimX + (y + 1)*DimY + (z - 1)];
					else gg[15] = 0;
					if(y+1 < Y) gg[16] = g[(x)*DimX + (y + 1)*DimY + (z)];
					else gg[16] = 0;
					if(y+1 < Y && z+1 < Z) gg[17] = g[(x)*DimX + (y + 1)*DimY + (z + 1)];
					else gg[17] = 0;

					if(x+1 < X && y-1 >=0 && z-1 >=0) gg[18] = g[(x + 1)*DimX + (y - 1)*DimY + (z - 1)];
					else gg[18] = 0;
					if(x+1 < X && y-1 >=0) gg[19] = g[(x + 1)*DimX + (y - 1)*DimY + (z)];
					else gg[19] = 0;
					if(x+1 < X && y-1 >=0 && z+1 < Z) gg[20] = g[(x + 1)*DimX + (y - 1)*DimY + (z + 1)];
					else gg[20] = 0;
					if(x+1 < X && z-1 >=0) gg[21] = g[(x + 1)*DimX + (y)*DimY + (z - 1)];
					else gg[21] = 0;
					if(x+1 < X) gg[22] = g[(x + 1)*DimX + (y)*DimY + (z)];
					else gg[22] = 0;
					if(x+1 < X && z+1 < Z) gg[23] = g[(x + 1)*DimX + (y)*DimY + (z + 1)];
					else gg[23] = 0;
					if(x+1 < X && y+1 < Y && z-1 >=0) gg[24] = g[(x + 1)*DimX + (y + 1)*DimY + (z - 1)];
					else gg[24] = 0;
					if(x+1 < X && y+1 < Y) gg[25] = g[(x + 1)*DimX + (y + 1)*DimY + (z)];
					else gg[25] = 0;
					if(x+1 < X && y+1 < Y && z+1 < Z) gg[26] = g[(x + 1)*DimX + (y + 1)*DimY + (z + 1)];
					else gg[26] = 0;					
				}


				isbound = 0;

				for (i = 1; i <= 26; i++)
				{
					if (gg[i]>0)
					{
						gg[i] = 1; isbound++;
					}
					else
					{
						gg[i] = 0;
					}
				}

				if (g[(x)*DimX + (y)*DimY + (z)] == 0 && isbound > 0)
				{
					// ic for z
					ic = (gg[3] / root3) - (gg[1] / root3) + (gg[6] / root2) + (gg[9] / root3) - (gg[7] / root3) - (gg[4] / root2) + (gg[12] / root2) - (gg[10] / root2) + gg[14] + (gg[17] / root2) - (gg[15] / root2) - gg[13] + (gg[20] / root3) - (gg[18] / root3) + (gg[23] / root2) + (gg[26] / root3) - (gg[24] / root3) - (gg[21] / root2);
					// ir for y
					ir = (gg[9] / root3) - (gg[2] / root2) - (gg[3] / root3) - (gg[1] / root3) + (gg[8] / root2) + (gg[7] / root3) + (gg[17] / root2) - gg[11] - (gg[12] / root2) - (gg[10] / root2) + gg[16] + (gg[15] / root2) + (gg[26] / root3) - (gg[19] / root2) - (gg[20] / root3) - (gg[18] / root3) + (gg[25] / root2) + (gg[24] / root3);
					// il for x
					il = (gg[18] / root3) + (gg[19] / root2) + (gg[20] / root3) + (gg[21] / root2) + gg[22] + (gg[23] / root2) + (gg[24] / root3) + (gg[25] / root2) + (gg[26] / root3) - (gg[1] / root3) - (gg[2] / root2) - (gg[3] / root3) - (gg[4] / root2) - gg[5] - (gg[6] / root2) - (gg[7] / root3) - (gg[8] / root2) - (gg[9] / root3);


					imax = fabs(ic);
					if (fabs(ir) > imax)
						imax = fabs(ir);
					if (fabs(il) > imax)
						imax = fabs(il);

					// approach 1 
					// test code. 
					normalx = il/imax;
					normaly = ir/imax;
					normalz = ic/imax;
					if (imax-0 < 0.0001) {normalx = 0; normaly = 0; normalz = 0; cout << "hit 000" << endl;} // imax is not int!!
					dislow = 1000; // 给一个很大的初始值然后依次找最小的
					nnodex = 0; nnodey = 0; nnodez = 0;
					for (zcnt=-1; zcnt<=1; zcnt++)
					for (ycnt=-1; ycnt<=1; ycnt++)
					for (xcnt=-1; xcnt<=1; xcnt++) 
					{
						dis = sqrt(1.0*(xcnt-normalx)*(xcnt-normalx)+(ycnt-normaly)*(ycnt-normaly)+(zcnt-normalz)*(zcnt-normalz));
						if (x+xcnt >= 0 && x+xcnt < X && y+ycnt >= 0 && y+ycnt < Y && z+zcnt >= 0 && z+zcnt < Z )
						if (g[(x+xcnt)*DimX+(y+ycnt)*DimY+(z+zcnt)] > 0 && dis < dislow) 
						{
							nnodex = xcnt;
							nnodey = ycnt;
							nnodez = zcnt;
							dislow = dis;
						}
					}
					g[x*DimX + y*DimY + z] = -g[(x+nnodex)*DimX+(y+nnodey)*DimY+(z+nnodez)]; // 把最近的那个node的index取反存在自己里面
					
					// test code.
					if(g[x*DimX + y*DimY + z] == 0)  cout << "STILL GET NEIGHBOUR = 0";					
					
					// end of approach 1 
					

					// approach 2.
					/**
					i = 0; j = 0; k = 0;

					tflt = ir / fabs(imax);

					if (tflt <= 0.5 && tflt >= -0.5)
					{
						i = 0;
					}
					else if (tflt>0.5)
					{
						i = 1;
					}
					else if (tflt<-0.5)
					{
						i = -1;
					}

					tflt = ic / fabs(imax);

					if (tflt <= 0.5&&tflt >= -0.5)
					{
						j = 0;
					}
					else if (tflt>0.5)
					{
						j = 1;
					}
					else if (tflt<-0.5)
					{
						j = -1;
					}

					tflt = il / fabs(imax);

					if (tflt <= 0.5&&tflt >= -0.5)
					{
						k = 0;
					}
					else if (tflt>0.5)
					{
						k = 1;
					}
					else if (tflt<-0.5)
					{
						k = -1;
					}

					if (imax == 0)
					{
						i = 0; j = 0; k = 0;
					}

					if (g[(x + k)*DimX + (y + i)*DimY + (z + j)]>0)
					{
						g[(x)*DimX + (y)*DimY + (z)] = -1 * g[(x + k)*DimX + (y + i)*DimY + (z + j)];
					}
					else

					{
						g[(x)*DimX + (y)*DimY + (z)] = g[(x + k)*DimX + (y + i)*DimY + (z + j)];
					}
					**/
					// end of approach 2
				}
			}
		}
	}


	HANDLE_ERROR(cudaMemcpy(dev_g, g, sizeof(int)*ElementNum, cudaMemcpyHostToDevice));
	// test code.
	HANDLE_ERROR(cudaMemcpy(dev_rev_g, rev_g, sizeof(int)*(num+1), cudaMemcpyHostToDevice));

	delete[] u;
	delete[] type;


	cout << "Reading stimulation profile... ";
	cout.flush();
	FILE *fp_stim;
	fp_stim = fopen("stimulationprofile.cnd", "r");
	int Num = 800000;
	//float (*b)[8];
	//b=new float[Num][8];
	float *new_b = new float[Num * 2];
	int num2 = 0;
	while (!feof(fp_stim))
	{
		int t1, t2, t3;
		float t4, t5, t6, t7;
		float t8;
		fscanf(fp_stim, "%d %d %d %f %f %f %f %str", &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8);


		if (g[t1*DimX + t2*DimY + t3] > 0)
		{
			new_b[num2 * 2] = (float)(g[t1*DimX + t2*DimY + t3] - 1);
			new_b[num2 * 2 + 1] = t7;
			num2++;
		}
	}
	cout << "There are " << num2 << " stimulus points." << endl << endl;
	cout.flush();
	fclose(fp_stim);

	float *dev_b;
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, sizeof(float) * 2 * num2));
	HANDLE_ERROR(cudaMemcpy(dev_b, new_b, sizeof(float) * 2 * num2, cudaMemcpyHostToDevice));

	
	double time = 0;
	int timedur = 60000;
	int count = 0;

	int BCL1 = 500;
	int BCL2 = 340;

	while (time <= timedur)
	{
		if (count%int(10 / dt) == 0)
		{
			cout << "Progress = " << 100.0*time/timedur << "%" << endl;
			// WaitForSingleObjectEx(hWriteData, INFINITE, false);
			cudaMemcpy(V_data, dev_u_v, sizeof(double)*num, cudaMemcpyDeviceToHost);
			curCount = count;
			writeData();
			// hWriteData = (HANDLE)_beginthreadex(NULL, 0, writeData, NULL, 0, NULL);
		}



		// if (count%int(1 / dt) == 0)
		// {
		// 	cout << (count / (int(1 / dt))) << " ";
		// }

		// cout << "x";

		calc_du <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dev_u_v, dev_du, dev_g, dev_rev_g);

		init_Istim <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dev_u);
		cudaDeviceSynchronize();

		if((time - floor(time/BCL1)*BCL1) >= 0 && (time - floor(time/BCL1)*BCL1) < 82)
		// if ((time >= 0) && (time <= 82)) //0-800
		{
			set_Istim <<<(num2 + blockSize - 1) / blockSize, blockSize >>>(num, num2, 0, -52, (time - floor(time/BCL1)*BCL1), dev_u, dev_b);
			cudaDeviceSynchronize();
		}


		// s2222222222222222222222222222222222222222222222222222222222222 
		// if(time >= 297 && time <= 299)  // 295 for SQT1 OK
		// if(time >= 215 && time <= 217)  //         210 late   205 early    (260 CON; 405 QUIN)   400 
		// {
		// 	set_S2Istim<<<(num + blockSize - 1) / blockSize, blockSize>>>(num, dev_u, -104, dev_is_s2);
		// 	cudaDeviceSynchronize();
		// }
		// if ((time >= BCL1) && (time <= BCL1 + 82)) //800-1600
		// {
		// 	set_Istim <<<(num2 + blockSize - 1) / blockSize, blockSize >>>(num, num2, BCL1, -52, time, dev_u, dev_b);
		// 	cudaDeviceSynchronize();
		// }
		// if ((time >= 2 * BCL1) && (time <= 2 * BCL1 + 82)) //1600
		// {
		// 	set_Istim <<<(num2 + blockSize - 1) / blockSize, blockSize >>>(num, num2, 2 * BCL1, -52, time, dev_u, dev_b);
		// 	cudaDeviceSynchronize();
		// }
		// if ((time >= 2 * BCL1 + BCL2) && (time <= 2 * BCL1 + BCL2 + 82)) //1940
		// {
		// 	set_Istim <<<(num2 + blockSize - 1) / blockSize, blockSize >>>(num, num2, 2 * BCL1 + BCL2, -104, time, dev_u, dev_b);
		// 	cudaDeviceSynchronize();
		// }

		Itot <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dt, dev_u, dev_type, dev_Itotr, dev_u_v);
		cudaDeviceSynchronize();

		new_u_v <<<(num + blockSize - 1) / blockSize, blockSize >>>(num, dev_u_v, dt, dev_du, dev_Itotr);
		cudaDeviceSynchronize();

		// test code.
		
		int oddnum = 0;
		int bignum = 0;
		int big_index = 0, small_index = 0;
		double vmin = 10000;
		double vmax = -1000;
		bool down_flag = false, up_flag = false;
		cudaMemcpy(V_data, dev_u_v, sizeof(double)*num, cudaMemcpyDeviceToHost);
		for (i = 0; i < num; i++)
		{
			if(V_data[i] < -90)
			{
				down_flag = true;
				oddnum += 1;
				if(V_data[i] < vmin) {vmin = V_data[i]; small_index = i;}
			}

			if(V_data[i] > 200)
			{
				up_flag = true;
				bignum += 1;
				if(V_data[i] > vmax) {vmax = V_data[i]; big_index = i;}
			}
		}
		if(up_flag) 
		{
			// cout << "ODD POINT NUMBER = " << oddnum << "; Vmin = " << vmin << "; time = " << time <<"; index = "<< index << endl;
			cout << "UP POINT NUMBER = " << bignum << "; Vmax = " << vmax << "; time = " << time <<"; index = "<< big_index << endl;
		}
		
		if(down_flag) 
		{
			cout << "DW POINT NUMBER = " << oddnum << "; Vmin = " << vmin << "; time = " << time <<"; index = "<< small_index << endl;
			// cout << "ODD POINT NUMBER = " << bignum << "; Vmax = " << vmax << "; time = " << time <<"; index = "<< index << endl;
		}
		









		time = time + dt;
		count++;
	}

	delete[] g;
	delete[] rev_g;

	HANDLE_ERROR(cudaFree(dev_g));
	HANDLE_ERROR(cudaFree(dev_rev_g));
	HANDLE_ERROR(cudaFree(dev_u_v));
	HANDLE_ERROR(cudaFree(dev_du));
	HANDLE_ERROR(cudaFree(dev_Itotr));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_type));
	HANDLE_ERROR(cudaFree(dev_u));
	
	// saving memory test


	return 0;
}
