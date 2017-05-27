
#include <iostream>
#include <cmath>
#include <armadillo>
#include <complex>
#include <chrono>

using namespace std;

double u = 1.660539040e-27;
double c = 299792458;
double e = 1.60217662e-19;
double pi = 4*atan(1);
double epsilon = 8.854187817e-12;
double coulombconst = pow(e,2)/(4*pi*epsilon);
double hbar = 1.054571800e-34;
double reshbar = 1.0/hbar;
double V0 = 2.14692e-11;

struct atom
{	
	double pos;
	double vel;
	int isoid;
	int protonid;
	double mass;
	double kin;

	atom(int isoid, int protonid, double mas)
		:	isoid(isoid), protonid(protonid), mass(mas)
	{
		mass = mass*u;
	}
};

double calcwellwidth (double Rscale, atom* daugther)
{
	return Rscale*pow(daugther->isoid,1.0/3.0);
}

double decayenergy (atom* parent, atom* daugther, atom* alpha)
{
	double energy = (parent->mass - (daugther->mass + alpha->mass))*pow(c,2);

	//cout << "parent mass: " << parent->mass << endl;	
	//cout << "decay energy: " << energy << endl;
	if (energy > 0) return energy;
		else cout << "error: no or negative energy yield" << endl; return 0;
}

void velocity (atom* parent, atom* daugther, atom* alpha)
{
	double energy = decayenergy(parent, daugther, alpha);

	alpha->vel = sqrt(2*energy)*daugther->mass/sqrt(daugther->mass*alpha->mass*(daugther->mass+alpha->mass));
}

void alphakinetic (atom* alpha)
{	
	alpha->kin = alpha->mass*pow(alpha->vel,2)/2;
}

double hitfreq(atom* alpha, double R)
{
	return alpha->vel/(2*R);
}

double coulomb (atom* daugther, atom* alpha, double r)
{	
	return alpha->protonid*daugther->protonid*coulombconst/r;
}

double revcoulomb (atom* daugther, atom* alpha)
{
	return alpha->protonid*daugther->protonid*coulombconst/alpha->kin;
}

double wkbapprox(int segments, double R0, double stepsize, atom* daugther, atom* alpha)
{
	vector <double> probs;

	for (int k = 1; k <= segments; k++)
	{
		double r = R0+(k-0.5)*stepsize;
		double prob = exp(-2*sqrt(2*alpha->mass*(coulomb(daugther,alpha,r)-alpha->kin))*(stepsize)*reshbar);
		probs.push_back(prob);
		//cout << "bar hight (MeV): " << coulomb(daugther,alpha,r)*6.242+12 << endl;
		//cout << "prob: " << prob << endl;
	}

	double accprob = 1;
	for (auto& el : probs)
		accprob *= el;

	return accprob;
}

void calcfactors (double segments, double R0, vector <complex<double>>* factors, double stepsize, atom* daugther, atom* alpha)
{
	vector <complex<double>>  klist;

	klist.push_back(sqrt(static_cast<complex<double>>(2*alpha->mass*(alpha->kin+V0-coulomb(daugther,alpha,R0))))*reshbar); // k left of barrier;
	
	for(int k = 1; k <= segments; k++)
	{
		klist.push_back(sqrt(static_cast<complex<double>>(2*alpha->mass*(-coulomb(daugther,alpha,R0+(k-0.5)*stepsize)+alpha->kin)))*reshbar);
		//cout << alpha->kin << endl;
		//cout << coulomb(daugther,alpha,R0+stepsize*k) << endl;
		//cout << alpha->kin - coulomb(daugther,alpha,R0+stepsize*k) << endl;
	}

	klist.push_back(sqrt(static_cast<complex<double>>(2*alpha->mass*(alpha->kin)))*reshbar); // k right of barrier

	int l = 0;
	for(int k = 1; k <= 2*(segments+1); k++)
	{
		if (k % 2)
		{
			//cout << "odd row" << endl;
			//cout << "k itterator: " << k << endl;
			//cout << "l itterator: " << l << endl;	
			(*factors).push_back(exp(1i*klist[l]*(R0+(l)*stepsize)));
			(*factors).push_back(exp(-1i*klist[l]*(R0+(l)*stepsize)));
			(*factors).push_back(-exp(1i*klist[l+1]*(R0+(l)*stepsize)));
			(*factors).push_back(-exp(-1i*klist[l+1]*(R0+(l)*stepsize)));
		}
		else
		{
			//cout << "even row" << endl;
			//cout << "k itterator: " << k << endl;
			//cout << "l itterator: " << l << endl;
			(*factors).push_back(1i*klist[l]*exp(1i*klist[l]*(R0+(l)*stepsize)));
	//		cout << "ping: " <<  klist[l] << " " << exp(1i*klist[l]*R0) << endl;
			(*factors).push_back(-1i*klist[l]*exp(-1i*klist[l]*(R0+(l)*stepsize)));
			(*factors).push_back(-1i*klist[l+1]*exp(1i*klist[l+1]*(R0+(l)*stepsize)));
			(*factors).push_back(1i*klist[l+1]*exp(-1i*klist[l+1]*(R0+(l)*stepsize)));
		}
		
		k % 2 == 0 ? l++ : false;
	}

	(*factors).back() = 0;
	(*factors)[(*factors).size()-5] = 0;

	//cout << "klist content: " << endl;
	//for (auto& el : klist)
	//	cout << el << endl;
	//cout << "klist length: " << endl << klist.size() << endl;
	
	//cout << "factor list content: " << endl;
	//for (auto& el : (*factors))
	//	cout << el << endl;
	//cout << "factor length: " << endl << (*factors).size() << endl;
	
	//cout << "first element: " << (*factors)[0] << endl;
}

double schsolve (int segments, vector <complex<double>>* factors)
{
	arma::cx_mat top(1,2*(segments+2),arma::fill::zeros);
	arma::cx_mat bot = top;

	top(0,0) = 1;
	bot(0,2*(segments+2)-1) = 1;

	arma::cx_mat B = top.t();
	
	arma::cx_mat schrodinger(2*(segments+1),2*(segments+2),arma::fill::zeros);
	//cout << "ping 1" << endl;

	//cout << "first element: " << (*factors)[0] << endl;

	int l = 0;
	for(int k = 0; k < 2*(segments+1); k++)
	{
		schrodinger(k,l) = (*factors)[4*k];
		//cout << "element added to list: " << (*factors)[4*k] << endl;
		schrodinger(k,l+1) = (*factors)[4*k+1];
		schrodinger(k,l+2) = (*factors)[4*k+2];
		schrodinger(k,l+3) = (*factors)[4*k+3];
		if (k % 2)
		   	l += 2;
	}
	//cout << "first element: " << schrodinger(0,0) << endl;

	//cout << "ping 1" << endl;
	schrodinger.insert_rows(0,top);
	//cout << "ping 1" << endl;
	schrodinger.insert_rows(2*(segments+1)+1,bot);
	//cout << "ping 1" << endl;

	//schrodinger.print();
	//B.print();

	//cout << "Determinant: " << arma::det(schrodinger) << endl;

	//cout << schrodinger.size() << endl;

	arma::cx_mat schinv = pinv(schrodinger,0.00000000001,"std");
	
	arma::cx_vec sols = schinv*B;
	//arma::cx_vec sols = arma::solve(schrodinger,B);

	//cout << schrodinger*sols-B << endl;	

	//sols.print();
	//cout << sols(2*(segments+1)) << endl;

	double prob = pow(abs(sols(2*(segments+1))),1);

	return prob;
}


int main()
{
	int segments = 1000;
	double Rscale = 1.5e-15; //~ 1.48 good for wkb

	//cout << "coulomb constant: " << coulombconst << endl;

	atom po212(212,84,211.9888421);
	atom pb208(208,82,207.9766521);
	atom alpha(4,2,4.002602);
   
	chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();       //for measuring execution time

	velocity(&po212,&pb208,&alpha);
	alphakinetic(&alpha);

	//cout << "alpha vel: " << alpha.vel << endl << "alpha kin: " << alpha.kin << endl;

	double wellwidth = calcwellwidth(Rscale,&pb208);
	//cout << "well width: " << wellwidth << endl;
	double hits = hitfreq(&alpha, wellwidth);
	//double barrierhight = coulomb(&pb208,&alpha,wellwidth);
	double barrierwidth =  revcoulomb(&pb208,&alpha)-wellwidth;
	double stepsize = barrierwidth / segments;
	//cout << "barrier hight: " << barrierhight << endl << "barrier width: " << barrierwidth << endl;

	// WKB stuff

	double wkbprob = wkbapprox(segments,wellwidth,stepsize,&pb208,&alpha);
	double wkbhalflife = log(2)/(hits*wkbprob);

	// Brute force stuff

	vector <complex<double>> factors; 
	calcfactors(segments, wellwidth, &factors, stepsize, &pb208, &alpha);
	double bruteprob = schsolve(segments, &factors);
	double brutehalflife = log(2)/(hits*bruteprob);

	chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	
	double benchtime = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

	cout << "tunneling probability using wkb: " << wkbprob << endl << "halflife using wkb: " << wkbhalflife << endl << "tunneling probability brute force: " << bruteprob << endl << "halflife using brute force: " << brutehalflife << endl << "execution time (\u03BCs): " << benchtime << endl;

}




























