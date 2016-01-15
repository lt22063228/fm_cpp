/*
 * Util.h
 *
 *  Created on: Dec 21, 2015
 *      Author: lin
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <vector>

namespace primer {

class Util {
public:
	Util();
	virtual ~Util();
};

} /* namespace primer */

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <typeinfo>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <queue>
typedef std::vector<std::string> svec;
typedef std::vector<double> dvec;
typedef std::vector<int> ivec;
typedef std::vector<unsigned long> lvec;
typedef std::vector<std::vector<double> > ddvec;
typedef std::map<std::string,std::string> ssmap;
typedef std::map<std::string,int> simap;
typedef std::pair<std::string,int> sipair;
typedef std::map<int,std::string> ismap;
typedef std::pair<int,std::string> ispair;
double logistic(double score);
double logloss(double score);
double ran_uniform();
double ran_range(int range);
double ran_gaussian();
double ran_gaussian(double mean, double stdev);
ivec sort_indexes(dvec &v, int topK, int u);
//template <typename T>
//std::vector<size_t> sort_indexes(const std::vector<T> &v) {
//
//	// initialize original index locations
//	std::vector<size_t> idx(v.size());
//	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
//
//	// sort indexes based on comparing values in v
//	sort(idx.begin(), idx.end(),
//	     [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
//
//	return idx;
//}
//template <typename T>
//std::vector<size_t> sort_indexes(const std::vector<T> &v) {
//
//	// initialize original index locations
//	std::vector<size_t> idx(v.size());
//	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
//
//	// sort indexes based on comparing values in v
//	sort(idx.begin(), idx.end(),
//	     [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
//
//	return idx;
//}

#endif /* UTIL_H_ */
