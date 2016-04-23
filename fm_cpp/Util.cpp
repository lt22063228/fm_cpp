/*
 * Util.cpp
 *
 *  Created on: Dec 21, 2015
 *      Author: lin
 */

#include "Util.h"
using namespace std;
namespace primer {

Util::Util() {
	// TODO Auto-generated constructor stub

}

Util::~Util() {
	// TODO Auto-generated destructor stub
}

} /* namespace primer */

double logistic(double score) {
	if(score > 0) {
		return (1/(1+exp(-score)));
	} else {
		return (exp(score) / (1+exp(score)));
	}
}
double logloss(double score) {
	if(score > 0) {
		return -(log(1+exp(-score)));
	} else {
		return (score - log(1+exp(score)));
	}

}
double ran_uniform() {
	return rand()/((double)RAND_MAX + 1);
}
double ran_range(int range) {
	return rand() % range;
}
double ran_gaussian() {
	// Joseph L. Leva: A fast normal Random number generator
	double u,v, x, y, Q;
	do {
		do {
			u = ran_uniform();
		} while (u == 0.0);
		v = 1.7156 * (ran_uniform() - 0.5);
		x = u - 0.449871;
		y = std::abs(v) + 0.386595;
		Q = x*x + y*(0.19600*y-0.25472*x);
		if (Q < 0.27597) { break; }
	} while ((Q > 0.27846) || ((v*v) > (-4.0*u*u*std::log(u))));
	return v / u;
}

double ran_gaussian(double mean, double stdev) {
	if ((stdev == 0.0) || (std::isnan(stdev))) {
		return mean;
	} else {
		return mean + stdev*ran_gaussian();
	}
}
ivec sort_indexes(dvec &v, int topK, int u) {
//	if(u==1) {
//		cout << "0" << endl;
//	}
	std::priority_queue<std::pair<double,int>, std::vector<std::pair<double,int> >, std::greater<std::pair<double, int> > > q;
//	if(u == 1) {
//		cout << "1" << endl;
//	}
	for(int i = 0; i < topK; i++) {
		q.push(std::pair<double,int>(v[i],i));
	}
//	if(u == 1) {
//		cout << "2" << endl;
//	}
	for(int i = topK; i < v.size(); i++) {
		double top_value = q.top().first;
		if(top_value < v[i]) {
			q.pop();
			q.push(std::pair<double,int>(v[i],i));
		}
	}
//	if(u == 1) {
//		cout << "3" << endl;
//	}
	ivec res(topK);
	for(int i = topK-1; i >= 0; i--) {
		int top = q.top().second;
		q.pop();
		res[i] = top;
	}
	return res;
}
