/*
 * Cluster.cpp
 *
 *  Created on: Dec 20, 2015
 *      Author: lin
 */

#include "Cluster.h"

namespace primer {

Cluster::Cluster() {
	// TODO Auto-generated constructor stub

}

Cluster::~Cluster() {
	// TODO Auto-generated destructor stub
}

} /* namespace primer */

using namespace std;

void loadMap(simap &user_map, ismap &map_user, simap &video_map, ismap &map_video, string userpath, string videopath, char sep) {
	ifstream readFile;
	readFile.open(userpath.c_str());
	string line;
	int cnt = 0;
	while(getline(readFile,line)) {
		if(line.empty()) break;
		stringstream linestream(line);
		int id = 0; string sid;
		linestream >> id; linestream.ignore(1); getline(linestream, sid, sep);
		user_map.insert(sipair(sid,id));
		map_user.insert(ispair(id,sid));
	}
	readFile.close();

	readFile.open(videopath.c_str());
	while(getline(readFile,line)) {
		if(line.empty()) break;
		stringstream linestream(line);
		int id = 0; string sid;
		linestream >> id; linestream.ignore(1); getline(linestream, sid, sep);
		video_map.insert(sipair(sid,id));
		map_video.insert(ispair(id,sid));
	}
	readFile.close();

}

void loadCluster(simap &user_map, simap &video_map, ivec &key, ivec &left, ivec &right, ivec &clu_dt, string path, char sep) {
	ifstream readFile;
	readFile.open(path.c_str());
	string line;
	while(getline(readFile,line)) {
		if(line.empty()) break;
		stringstream linestream(line);
		char u[100], vl[100], vr[100];
		int dt;
		linestream.getline(u,100,sep);
		linestream.getline(vl,100,sep);
		linestream.getline(vr,100,sep);
		linestream >> dt;
		key.push_back(user_map[string(u)]);
		left.push_back(video_map[string(vl)]);
		right.push_back(video_map[string(vr)]);
		clu_dt.push_back(dt);
	}
}
void sgdNormCluster(ddvec &midMatrix, ddvec &sideMatrix, ivec &mid, ivec &left, ivec &right, ivec &clu_dt,
		double learnRate, ivec &indice, double &delta, double &LL, double regv, int scale, double phi, double alpha) {
	int numTrain = mid.size();
	int dim = midMatrix[0].size();
	int numVideo = sideMatrix.size();
	LL = 0.; delta = 0.;
	random_shuffle(indice.begin(), indice.end());

	for(int i = 0; i < numTrain; i++) {
		double tmpdelta = 0.;
		int cnt = 0;

		int idx = indice[i];
		int m = mid[idx], lhs = left[idx], rhs = right[idx], dt = clu_dt[idx];
		dvec m_f = midMatrix[m], lhs_f = sideMatrix[lhs], rhs_f = sideMatrix[rhs]; // feature vector in ranking

		// normalize the feature vector
		dvec m_vec, lhs_vec, rhs_vec; // normalized vector in clustering
		double m_norm = 0., lhs_norm = 0., rhs_norm = 0.;
		for(int d = 0; d < dim; d++) {
			m_norm += exp(m_f[d]*phi);
			lhs_norm += exp(lhs_f[d]*phi);
			rhs_norm += exp(rhs_f[d]*phi);
		}
		for(int d = 0; d < dim; d++) {
			m_vec.push_back(exp(m_f[d]*phi)/m_norm);
			lhs_vec.push_back(exp(lhs_f[d]*phi)/lhs_norm);
			rhs_vec.push_back(exp(rhs_f[d]*phi)/rhs_norm);
		}

		double score = 0.;
		for(int d = 0; d < dim; d++) {
			score += m_vec[d] * lhs_vec[d] * rhs_vec[d];
		}
		double normalize = (1-logistic(score));
		dvec mvector(dim), lhsvector(dim), rhsvector(dim), randvector(dim); // the updated feature vector in ranking
		double grad = 0.;
		dt = 1 + dt/scale;
		double disc = 1./dt;


		// mid update
		dvec mul_vec(dim);
		for(int d = 0; d < dim; d++) {	mul_vec[d] = 0.; }
		double sum_ex_over_c_plus_ex_square = 0.;
		for(int d = 0; d < dim; d++) {
			double multiplier = lhs_vec[d] * rhs_vec[d];
			double C = m_norm - exp(m_f[d]*phi);
			double ex = m_vec[d] * m_norm;

			for(int dd = 0; dd < dim; dd++) {
				if(dd == d) continue;
				mul_vec[dd] += multiplier * ex;
			}
			grad = alpha * normalize * multiplier * ((C*ex) / ((ex+C)*(ex+C)));

			mvector[d] = m_f[d] + disc * learnRate * grad;
			tmpdelta += grad * grad;
			cnt ++;
		}
		for(int d = 0; d < dim; d++) {
			double multiplier = lhs_vec[d] * rhs_vec[d];
			double C = m_norm - exp(m_f[d]*phi);
			double ex = m_vec[d] * m_norm;

			double ex_over_cplusex_square = ex/((ex+C)*(ex+C));
			grad = alpha * normalize * ex_over_cplusex_square * mul_vec[d];

			mvector[d] += disc * learnRate * grad;
			tmpdelta += grad * grad;
		}

		// lhs update
		for(int d = 0; d < dim; d++) {	mul_vec[d] = 0.; }
		for(int d = 0; d < dim; d++) {
			double multiplier = m_vec[d] * rhs_vec[d];
			double C = lhs_norm - exp(lhs_f[d]*phi);
			double ex = lhs_vec[d] * lhs_norm;
			grad =  alpha * normalize * multiplier * ((C*ex) / ((ex + C)*(ex+C)));

			for(int dd = 0; dd < dim; dd++) {
				if(dd == d) continue;
				mul_vec[dd] += multiplier * ex;
			}

			lhsvector[d] = lhs_f[d] + disc * learnRate * grad;
			tmpdelta += grad * grad;
			cnt ++;
		}
		for(int d = 0; d < dim; d++) {
			double multiplier = m_vec[d] * rhs_vec[d];
			double C = lhs_norm - exp(lhs_f[d]*phi);
			double ex = lhs_vec[d] * lhs_norm;

			double ex_over_cplusex_square = ex/((ex+C)*(ex+C));
			grad = alpha * normalize * ex_over_cplusex_square * mul_vec[d];

			lhsvector[d] += disc * learnRate * grad;
			tmpdelta += grad * grad;
		}

		// rhs update
		for(int d = 0; d < dim; d++) {	mul_vec[d] = 0.; }
		for(int d = 0; d < dim; d++) {
			double multiplier = m_vec[d] * lhs_vec[d];
			double C = rhs_norm - exp(rhs_f[d]*phi);
			double ex = rhs_vec[d] * rhs_norm;
			grad = alpha * normalize * multiplier * ((C*ex) / ((ex + C)*(ex+C)));

			for(int dd = 0; dd < dim; dd++) {
				if(dd == d) continue;
				mul_vec[dd] += multiplier * ex;
			}

			rhsvector[d]  = rhs_f[d] + disc * learnRate * grad;
			tmpdelta += grad * grad;
			cnt ++;
		}
		for(int d = 0; d < dim; d++) {
			double multiplier = m_vec[d] * lhs_vec[d];
			double C = rhs_norm - exp(rhs_f[d]*phi);
			double ex = rhs_vec[d] * rhs_norm;

			double ex_over_cplusex_square = ex/((ex+C)*(ex+C));
			grad = alpha * normalize * ex_over_cplusex_square * mul_vec[d];

			rhsvector[d] += disc * learnRate * grad;
			tmpdelta += grad * grad;
		}

		// store update
		for(int d = 0; d < dim; d++) {
			midMatrix[m][d] = mvector[d];
			sideMatrix[lhs][d] = lhsvector[d];
			sideMatrix[rhs][d] = rhsvector[d];
		}

		tmpdelta /= cnt;
		delta += tmpdelta;
		LL += logloss(score);


	}
	delta /= numTrain;
}
void sgdCluster(ddvec &midMatrix, ddvec &sideMatrix, ivec &mid, ivec &left, ivec &right, ivec &clu_dt,
		double learnRate, ivec &indice, double &delta, double &LL, double regv, int scale, double alpha) {
	int numTrain = mid.size();
	int dim = midMatrix[0].size();
	int numVideo = sideMatrix.size();
	LL = 0.; delta = 0.;
	random_shuffle(indice.begin(), indice.end());

	for(int i = 0; i < numTrain; i++) {
		double tmpdelta = 0.;
		int cnt = 0;

		int idx = indice[i];
		int u = mid[idx], vl = left[idx], vr = right[idx], dt = clu_dt[idx];

		int randv = ran_range(numVideo);

		double score = 0.;
		for(int f = 0; f < dim; f++) {
			score += midMatrix[u][f] * sideMatrix[vl][f] * sideMatrix[vr][f];
			score -= midMatrix[u][f] * sideMatrix[vl][f] * sideMatrix[randv][f];
		}
		double normalizer = (1-logistic(score));
		dvec uvector(dim), vlvector(dim), vrvector(dim), randvector(dim);

		double grad = 0.;

		dt = 1 + dt/scale;
		double disc = 1./dt;
		disc = 1.;
		// user update
		for(int f = 0; f < dim; f++) {
			grad = alpha * normalizer * sideMatrix[vl][f] * sideMatrix[vr][f];
			grad -= alpha * normalizer * sideMatrix[vl][f] * sideMatrix[randv][f];
			grad -= 2 * midMatrix[u][f] * regv;
			uvector[f] = midMatrix[u][f] + disc * learnRate * grad;
			tmpdelta += abs(grad);
			cnt ++;
		}

		// left update
		for(int f = 0; f < dim; f++) {
			grad = alpha * normalizer * midMatrix[u][f] * sideMatrix[vr][f];
			grad -= alpha * normalizer * midMatrix[u][f] * sideMatrix[randv][f];
			grad -= 2 * sideMatrix[vl][f] * regv;
			vlvector[f] = sideMatrix[vl][f] + disc * learnRate * grad;
			tmpdelta += abs(grad);
			cnt ++;
		}

		// right update
		for(int f = 0; f < dim; f++) {
			grad = alpha * normalizer * midMatrix[u][f] * sideMatrix[vl][f];
			grad -= 2 * sideMatrix[vr][f] * regv;
			vrvector[f] = sideMatrix[vr][f] + disc * learnRate * grad;
			tmpdelta += abs(grad);
			cnt ++;
		}

		// random update
		for(int f = 0; f < dim; f++) {
			grad = -alpha * normalizer * midMatrix[u][f] * sideMatrix[vl][f];
			grad -= 2 * sideMatrix[randv][f] * regv;
			randvector[f] = sideMatrix[randv][f] + disc * learnRate * grad;
			tmpdelta += abs(grad);
			cnt ++;
		}

		// store update
		for(int f = 0; f < dim; f++) {
			midMatrix[u][f] = uvector[f];
			sideMatrix[vl][f] = vlvector[f];
			sideMatrix[vr][f] = vrvector[f];
			sideMatrix[randv][f] = randvector[f];
		}

		tmpdelta /= cnt;
		delta += tmpdelta;
		LL += logloss(score);

	}

	delta /= numTrain;

}

void loadTarget(simap &user_map, simap &video_map, map<int, vector<int> > &target, string path) {
	ifstream readFile;
	readFile.open(path.c_str());
	while(!readFile.eof()) {
		string u, v, dt;
//		readFile >> u >> v >> dt;
		readFile >> u; readFile.ignore(1); readFile >> v; readFile.ignore(1); readFile >> dt;
		if(u.empty()) break;
		if(target.find(user_map[u]) == target.end()) {
			target.insert(std::pair<int, vector<int> >(user_map[u], vector<int>()));
		}
		target[user_map[u]].push_back(video_map[v]);
	}
}



void evaluateCluster(ddvec &userMatrix, ddvec &videoMatrix, map<int,vector<int> > &target, ivec &utest, ivec &vltest, ivec &vrtest, double &hit, double &prec) {
//	int numTest = utest.size();
//	int numVideo = videoMatrix.size();
//	int dim = userMatrix[0].size();
//	hit = 0.;
//
//	for(int i = 0; i < numTest; i++) {
//		cout << i << endl;
//		int u = utest[i], vl = vltest[i], vr = vrtest[i];
//		ivec candidates = target[u];
//		dvec scores;
//
//		for(int j = 0; j < candidates.size(); j++) {
//			double score = 0.;
//			for(int f = 0; f < dim; f++) {
//				score += userMatrix[u][f] * videoMatrix[vl][f] * videoMatrix[candidates[j]][f];
//			}
//			scores.push_back(score);
//		}
//		vector<size_t> rank = sort_indexes(scores);
//		int pre1 = candidates[rank[0]];
//		if(pre1 == vr) {
//			hit += 1.;
//		}
//	}
//	prec = hit / (numTest*3);
}


//int main() {
//	// loading user & video map
//	string dir = "/home/lin/workspace/data/youtube";
//	char sep = '\t';
//	simap user_map, video_map;
//	ismap map_user, map_video;
//	string userpath = dir + "/block/user.map", videopath = dir + "/block/video.map";
//	std::cout << "loading map..." << std::endl;
//	loadMap(user_map, map_user, video_map, map_video, userpath, videopath, sep);
//
//	// loading cluster train data
//	ivec user, leftv, rightv;
//	string ctrain_path = dir + "/ctrain.dat";
//	std::cout << "loading ctrain..." << std::endl;
//	loadCluster(user_map, video_map, user, leftv, rightv, ctrain_path, sep);
//
//	// init features
//	ddvec userMatrix, videoMatrix;
//	int dim = 48, numUser = user_map.size(), numVideo = video_map.size();
//	double stdev = 0.1;
//	std::cout << "init cluster features..." << std::endl;
//	initMatrix(userMatrix, videoMatrix, stdev, numUser, numVideo, dim);
//
//	ivec indice;
//	for(int i = 0; i < user.size(); i++) {
//		indice.push_back(i);
//	}
//
//	// loading cluster test data
//	ivec utest, vltest, vrtest;
//	string ctest_path = dir + "/ctest.dat";
//	std::cout << "loading ctest..." << std::endl;
//	loadCluster(user_map, video_map, utest, vltest, vrtest, ctest_path, sep);
//
//	// loading target video set
//	map<int, ivec > target;
//	string target_path = dir + "/test.dat";
//	std::cout << "loading target video..." << std::endl;
//	loadTarget(user_map, video_map, target, target_path);
//
//
//	int numIter = 3000;
//	double learnRate = 0.05;
//	std::cout << "start stochastic gradient descent..." << std::endl;
//	for(int i = 0; i < numIter; i++) {
////		std::random_shuffle(indice.begin(), indice.end());
//
//		double delta = 0., LL = 0., regv = 0.000;
//		sgdCluster(userMatrix, videoMatrix, user, leftv, rightv, learnRate, indice, delta, LL, regv);
//
//		double hit = 0., prec = 0.;
//		if((i+1) % 50 == 0) {
//			evaluateCluster(userMatrix, videoMatrix, target, utest, vltest, vrtest, hit, prec);
//			std::cout << "iterNum: " << i << "\tLL: " << LL << "\tdelta: " << delta << "\thit: "
//				  << hit << "\tprec: " << prec << std::endl;
//		} else {
//			std::cout << "iterNum: " << i << "\tLL: " << LL << "\tdelta: " << delta << std::endl;
//		}
//
//	}
//}

