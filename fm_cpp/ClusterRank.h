/*
 * ClusterRank.h
 *
 *  Created on: Dec 21, 2015
 *      Author: lin
 */

#ifndef CLUSTERRANK_H_
#define CLUSTERRANK_H_
#include "Util.h"
#include "Recommend.h"
#include "Cluster.h"

namespace primer {

class ClusterRank {
public:
	ClusterRank();
	virtual ~ClusterRank();
};

} /* namespace primer */

#endif /* CLUSTERRANK_H_ */

using namespace std;

void sgdRCluster(ddvec &userMatrix, ddvec &videoMatrix, ivec &user, ivec &leftv, ivec &rightv,
				ddvec &rank_u_matrix, ddvec &rank_v_matrix, double rank_regv,
				double learnRate, ivec &indice, double &delta, double &LL, double regv) {
	int numTrain = user.size();
	int dim = userMatrix[0].size();
	int numVideo = videoMatrix.size();
	std::random_shuffle(indice.begin(), indice.end());
	LL = 0.; delta = 0.;

	for(int i = 0; i < numTrain; i++) {
		double tmpdelta = 0.;
		int cnt = 0;

		int idx = indice[i];
		int u = user[idx], vl = leftv[idx], vr = rightv[idx];

		int randv = ran_range(numVideo);

		double score = 0.;
		for(int f = 0; f < dim; f++) {
			score += userMatrix[u][f] * videoMatrix[vl][f] * videoMatrix[vr][f];
//			score -= userMatrix[u][f] * videoMatrix[vl][f] * videoMatrix[randv][f];
		}
		double normalizer = (1-logistic(score));
		dvec uvector(dim), vlvector(dim), vrvector(dim), randvector(dim);

		double grad = 0.;

		// user update
		for(int f = 0; f < dim; f++) {
			grad = normalizer * videoMatrix[vl][f] * videoMatrix[vr][f];
//			grad -= normalizer * videoMatrix[vl][f] * videoMatrix[randv][f];
			grad -= 2 * userMatrix[u][f] * regv;
//			grad += 2 * (rank_u_matrix[u][f] - userMatrix[u][f]) * rank_regv;
			uvector[f] = userMatrix[u][f] + learnRate * grad;
			tmpdelta += grad * grad;
			cnt ++;
		}

		// left update
		for(int f = 0; f < dim; f++) {
			grad = normalizer * userMatrix[u][f] * videoMatrix[vr][f];
//			grad -= normalizer * userMatrix[u][f] * videoMatrix[randv][f];
			grad -= 2 * videoMatrix[vl][f] * regv;
			grad += 2 * (rank_v_matrix[vl][f] -videoMatrix[vl][f]) * rank_regv;
			vlvector[f] = videoMatrix[vl][f] + learnRate * grad;
			tmpdelta += grad * grad;
			cnt ++;
		}

		// right update
		for(int f = 0; f < dim; f++) {
			grad = normalizer * userMatrix[u][f] * videoMatrix[vl][f];
			grad -= 2 * videoMatrix[vr][f] * regv;
			grad += 2 * (rank_v_matrix[vr][f] - videoMatrix[vr][f]) * rank_regv;
			vrvector[f] = videoMatrix[vr][f] + learnRate * grad;
			tmpdelta += grad * grad;
			cnt ++;
		}

		// random update
//		for(int f = 0; f < dim; f++) {
//			grad = -normalizer * userMatrix[u][f] * videoMatrix[vl][f];
//			grad -= 2 * videoMatrix[randv][f] * regv;
//			grad += 2 * (rank_v_matrix[randv][f] - videoMatrix[randv][f]) * rank_regv;
//			randvector[f] = videoMatrix[randv][f] + learnRate * grad;
//			tmpdelta += grad * grad;
//			cnt ++;
//		}

		// store update
		for(int f = 0; f < dim; f++) {
			userMatrix[u][f] = uvector[f];
			videoMatrix[vl][f] = vlvector[f];
			videoMatrix[vr][f] = vrvector[f];
//			videoMatrix[randv][f] = randvector[f];
		}

		tmpdelta /= cnt;
		delta += tmpdelta;
		LL += logloss(score);

	}

	delta /= numTrain;
}
void sgdCRank(ivec &indice, ivec &user, ivec &video, map<int,ivec> &done,
		ddvec &clu_u_matrix, ddvec &clu_v_matrix, double clu_regv,
		dvec &videoVector, ddvec &userMatrix, ddvec &videoMatrix,
		double learnRate, double regw, double regv,
		double &LL, double &delta) {
	int dim = userMatrix[0].size();
	std::random_shuffle(indice.begin(), indice.end());

	LL = 0.; delta = 0.;
	for(int i = 0; i < indice.size(); i++) {
		double tmpdelta = 0.;
		int cnt = 0;

		int idx = indice[i];
		int u = user[idx], vp = video[idx];
		int vn = getNegative(done[u], videoVector.size());

		double vp_scalar = 0., vn_scalar = 0.;
		dvec u_vector(dim), vp_vector(dim), vn_vector(dim);

		double tmpscore = computeScore(u, vp, vn, videoVector, userMatrix, videoMatrix);
		double normalizer = (1-logistic(tmpscore));

		double g = 0.;

//		// unary postive
//		g = normalizer - 2 * regw * videoVector[vp];
//		vp_scalar = videoVector[vp] + learnRate * g;
//		tmpdelta += g * g;
//		cnt ++;
//
//		// unary negative
//		g = -normalizer - 2 * regw * videoVector[vn];
//		vn_scalar = videoVector[vn] + learnRate * g;
//		tmpdelta += g * g;
//		cnt ++;

		// user
		for(int d = 0; d < dim; d++) {
			g = normalizer * (videoMatrix[vp][d] - videoMatrix[vn][d]);
			g -= 2 * regv * userMatrix[u][d];
//			g -= 2 * regv * (userMatrix[u][d] - clu_u_matrix[u][d]);
			u_vector[d] = userMatrix[u][d] + learnRate * g;
			tmpdelta += g * g;
			cnt ++;
		}

		// pairwise positive
		for(int d = 0; d < dim; d++) {
			g = normalizer * userMatrix[u][d];
//			g -= 2 * regv * videoMatrix[vp][d];
			g -= 2 * regv * (videoMatrix[vp][d] - clu_v_matrix[vp][d]);
			vp_vector[d] = videoMatrix[vp][d] + learnRate * g;
			tmpdelta += g * g;
			cnt ++;
		}

		// pairwise negativ
		for(int d = 0; d < dim; d++) {
			g = -normalizer * userMatrix[u][d];
//			g -= 2 * regv * videoMatrix[vn][d];
			g -= 2 * regv * (videoMatrix[vn][d] - clu_v_matrix[vn][d]);
			vn_vector[d] = videoMatrix[vn][d] + learnRate * g;
			tmpdelta += g * g;
			cnt ++;
		}



		videoVector[vp] = vp_scalar; videoVector[vn] = vn_scalar;
		for(int f = 0; f < dim; f++) {
			userMatrix[u][f] = u_vector[f];
			videoMatrix[vp][f] = vp_vector[f];
			videoMatrix[vn][f] = vn_vector[f];
		}

		delta += tmpdelta/cnt;
		LL += logloss(tmpscore);

	}
	delta /= indice.size();
}

//		coFactor(rank_indice, rank_user, rank_video, done, rank_order, // params for rank input
//				rank_v_vector, rank_u_matrix, rank_v_matrix, // params for rank_feature
//				learnRate, rank_regw, rank_regv, rank_LL, rank_grad // params for model
//				clu_indice, clu_user, clu_lvideo, clu_rvideo, clu_regv); // params for cluster
void coFactor(ivec &rank_indice, ivec &rank_user, ivec &rank_video, map<int,ivec> &done, ivec &rank_order,
			  dvec &rank_v_vector, ddvec &rank_u_matrix, ddvec &rank_v_matrix,
			  ddvec &clu_u_matrix, ddvec &clu_v_matrix, // params for cluster
			  double rank_learnRate, double rank_regw, double rank_regv, double &rank_LL, double &rank_grad,
			  double clu_learnRate, double clu_regv, double &clu_LL, double &clu_grad,
			  ivec &clu_vindice, ivec &clu_video, ivec &clu_luser, ivec &clu_ruser, ivec &clu_vdt, // input for cluster by video
			  ivec &clu_uindice, ivec &clu_user, ivec &clu_lvideo, ivec &clu_rvideo, ivec &clu_udt, ivec &clu_order,
			  string method, double alpha
			  ) {
//		sgdRank(rank_indice, rank_user, rank_video, done,
//			rank_order,
//			rank_v_vector, rank_u_matrix, rank_v_matrix,
//			rank_learnRate, rank_regw, rank_regv,
//			rank_LL, rank_grad);

		double dumdelta = 0., dumLL = 0., dumregv = 0.;
		if(method.compare("byuser") == 0) {
			sgdCluster(rank_u_matrix, rank_v_matrix, clu_user, clu_lvideo, clu_rvideo, clu_udt, clu_learnRate, clu_uindice, clu_grad, clu_LL, clu_regv, 720, alpha, clu_order);
		} else if(method.compare("byvideo") == 0) {
//			sgdCluster(rank_v_matrix, rank_u_matrix, clu_video, clu_luser, clu_ruser, clu_vdt, clu_learnRate, clu_vindice, clu_grad, clu_LL, clu_regv, 720, alpha);
		} else if(method.compare("both") == 0) {
//			double clu_urate = learnRate;
//			double clu_vrate = learnRate * 0.05;
//			sgdCluster(rank_u_matrix, rank_v_matrix, clu_user, clu_lvideo, clu_rvideo, clu_udt, clu_urate, clu_uindice, dumdelta, dumLL, dumregv, 720);
//			sgdCluster(rank_v_matrix, rank_u_matrix, clu_video, clu_luser, clu_ruser, clu_vdt, clu_vrate, clu_vindice, dumdelta, dumLL, dumregv, 720);
		} else if(method.compare("norm_user") == 0) {
			double phi = 1.;
			sgdNormCluster(rank_u_matrix, rank_v_matrix, clu_user, clu_lvideo, clu_rvideo, clu_udt, clu_learnRate, clu_uindice, clu_grad, clu_LL, clu_regv, 720, phi, alpha);
		} else if(method.compare("bynorm_user") == 0) {
			double phi = 1.;
//			sgdCluster(rank_u_matrix, rank_v_matrix, clu_user, clu_lvideo, clu_rvideo, clu_udt, clu_learnRate, clu_uindice, clu_grad, clu_LL, clu_regv, 720, alpha);
			sgdNormCluster(rank_u_matrix, rank_v_matrix, clu_user, clu_lvideo, clu_rvideo, clu_udt, clu_learnRate, clu_uindice, clu_grad, clu_LL, clu_regv, 720, phi, alpha);
		}

}
int main() {

	string dataname = "youku";
//	string dataname = "youtube";
	string dir = "/home/lin/workspace/data/" + dataname;
	char sep = '\t';

	// loading user and video map
	std::cout << "loading user and video map data" << std::endl;
	simap user_map, video_map;
	ismap map_user, map_video;
	string userpath = dir + "/block/user.map", videopath = dir + "/block/video.map";
	loadMap(user_map, map_user, video_map, map_video, userpath, videopath, sep);
	int numUser = user_map.size(), numVideo = video_map.size();

	// global params
	string sparsity = "80_20";
	int dim = 15;
	double stdev = 0.1;
	string path;


	// prepare data structure for ranking
	std::cout << "preparing data structure for ranking" << std::endl;
	ivec rank_user, rank_video, rank_order, rank_utest, rank_vtest;
	lvec rank_time;
	dvec rank_v_vector;
	ddvec rank_u_matrix, rank_v_matrix;
	map<int,ivec> done; map<int, lvec> time;
	ivec rank_indice;
	int num_rank_train = 0;
	{
		// loading ranking training data
		std::cout << "loading ranking training data" << std::endl;
		path = dir + "/train_"+sparsity+".dat";
		loadRank(rank_user, rank_video, rank_time, user_map, video_map, path, sep);
		num_rank_train = rank_user.size();

		// loading ranking testing data
		std::cout << "loading ranking testing data" << std::endl;
		path = dir + "/test_"+sparsity+".dat";
		loadTest(rank_utest, rank_vtest, user_map, video_map, path, sep);

		// loading done data
		std::cout << "loading done data" << std::endl;
		loadDone(done, rank_user, rank_video);

		// loading done time
		std::cout << "loading done time" << std::endl;
		loadTime(time, rank_user, rank_time);
		loadOrder(rank_order, time, rank_user, rank_time);

		// initialize ranking features
		std::cout << "initializing ranking features" << std::endl;
		initVector(rank_v_vector, numVideo);
		initMatrix(rank_u_matrix, rank_v_matrix, stdev, numUser, numVideo, dim);

		// init random shuffling indice
		for(int i = 0; i < num_rank_train; i++) rank_indice.push_back(i);
	}

	// prepare data structure for clustering, cluster by user
	std::cout << "preparing data structure for clustering" << std::endl;
	ivec clu_user, clu_lvideo, clu_rvideo, clu_udt, clu_utest, clu_vltest, clu_vrtest, clu_dtutest, clu_order;
	ddvec clu_u_matrix, clu_v_matrix;
	map<int,ivec> target;
	ivec clu_uindice;
	int num_cluster_utrain = 0;
	{
		// loading cluster train data
		std::cout << "loading cluster training data" << std::endl;
		path = dir + "/cutrain_"+sparsity+".dat";
		loadCluster(user_map, video_map, clu_user, clu_lvideo, clu_rvideo, clu_udt, path, sep);
		loadCorder(clu_user, clu_lvideo, done, clu_order);
		num_cluster_utrain = clu_user.size();

		// loading cluster testing data
		std::cout << "loading cluster testing data" << std::endl;
		path = dir + "/cutest_"+sparsity+".dat";
		loadCluster(user_map, video_map, clu_utest, clu_vltest, clu_vrtest, clu_dtutest, path, sep);

		// loading target video
		std::cout << "loading target video..." << std::endl;
		path = dir + "/test_.dat";
		loadTarget(user_map, video_map, target, path);

		// initialize cluster features
		std::cout << "initializing cluster features..." << std::endl;
		initMatrix(clu_u_matrix, clu_v_matrix, stdev, numUser, numVideo, dim);

		// init random shuffling
		for(int i = 0; i < num_cluster_utrain; i++) clu_uindice.push_back(i);
	}
	// prepare data structure for clustering, cluster by video
	ivec clu_video, clu_luser, clu_ruser, clu_vdt, clu_vtest, clu_ultest, clu_urtest, clu_dtvtest;
	ivec clu_vindice;
	int num_cluster_vtrain = 0;
	{
		// loading cluster train data
		std::cout << "loading cluster training data" << std::endl;
		path = dir + "/cvtrain_"+sparsity+".dat";
		loadCluster(user_map, video_map, clu_video, clu_luser, clu_ruser, clu_vdt, path, sep);
		num_cluster_vtrain = clu_video.size();

		// loading cluster testing data
		std::cout << "loading cluster testing data" << std::endl;
		path = dir + "/cvtest_"+sparsity+".dat";
		loadCluster(user_map, video_map, clu_vtest, clu_ultest, clu_urtest, clu_dtvtest, path, sep);

		// init random shuffling
		for(int i = 0; i < num_cluster_vtrain; i++) clu_vindice.push_back(i);
	}

	// start the training
	std::cout << "start the ranking and clustering learning" << std::endl;
	int numIter = 20000;
	double rank_learnRate = 0.0001, clu_learnRate = 0.0001;
	double alpha = 1;
	// learning params for ranking
	int topK = 50;
	double reg = 0.1;
	double clu_regv = 0.1;
	double rank_regw = reg, rank_regv = reg;
	double rank_LL = 0., rank_grad = 0.;
	// learning params for cluster
//	stringstream reg_ss;
	double clu_LL = 0., clu_grad = 0.;

	string method = "clu_user";
	// convergence file
	vector<ofstream> writeFiles(topK/5);
	string split = "80_20";
//	string tmp = "tmp";
//	string tmp = "disc";
//	string tmp = "avg";
	string tmp = "only";
	for(int i = 0; i < topK/5; i++) {
		stringstream reg_ss; reg_ss << "_rankreg" << reg; reg_ss << "_clureg" << clu_regv; reg_ss << "_dim" << dim << "_learnRate" << rank_learnRate << "_clearnRate" << clu_learnRate << "_top" << (5*(i+1));
		writeFiles[i].open((dir + "/" + tmp + method + "_convergence_" + split + reg_ss.str() + ".out").c_str(), ios::out);
//		cout << (dir + "/" + tmp + method + "_convergence_" + split + reg_ss.str() + ".out") << endl;
		writeFiles[i] << "method: " << method << "\tLearnRate: " << rank_learnRate << "\tClearnRate: " << clu_learnRate << "\tdim: " << dim << "\trank_regw: " << rank_regw << "\trank_regv: " << rank_regv << "\tclu_regv: " << clu_regv << "\n";
	}
	for(int i = 0; i < numIter; i++) {
		std::cout << "iter..." << i << std::endl;
		if(method.compare("sgd") == 0) {
			sgdRank(rank_indice, rank_user, rank_video, done,
				rank_order,
				rank_v_vector, rank_u_matrix, rank_v_matrix,
				rank_learnRate, rank_regw, rank_regv,
				rank_LL, rank_grad);

		} else if(method.compare("clu_user") == 0) {
			coFactor(rank_indice, rank_user, rank_video, done, rank_order, // input for rank
					rank_v_vector, rank_u_matrix, rank_v_matrix, // params for rank_feature
					clu_u_matrix, clu_v_matrix,
					rank_learnRate, rank_regw, rank_regv, rank_LL, rank_grad, // params for model
					clu_learnRate, clu_regv, clu_LL, clu_grad,
					clu_vindice, clu_video, clu_luser, clu_ruser, clu_vdt, // input for cluster by video
					clu_uindice, clu_user, clu_lvideo, clu_rvideo, clu_udt, clu_order, // input for cluster by user
					"byuser", alpha
					); // input for cluster by user
		}
//		 else if(method.compare("clu_video") == 0) {
//			coFactor(rank_indice, rank_user, rank_video, done, rank_order, // input for rank
//					rank_v_vector, rank_u_matrix, rank_v_matrix, // params for rank_feature
//					clu_u_matrix, clu_v_matrix,
//					rank_learnRate, rank_regw, rank_regv, rank_LL, rank_grad, // params for model
//					clu_learnRate, clu_regv, clu_LL, clu_grad,
//					clu_vindice, clu_video, clu_luser, clu_ruser, clu_vdt, // input for cluster by video
//					clu_uindice, clu_user, clu_lvideo, clu_rvideo, clu_udt, // input for cluster by user
//					"byvideo", alpha
//					); // input for cluster by user
//		} else if(method.compare("clu_both") == 0) {
//			coFactor(rank_indice, rank_user, rank_video, done, rank_order, // input for rank			coFactor(rank_indice, rank_user, rank_video, done, rank_order, // input for rank
//					rank_v_vector, rank_u_matrix, rank_v_matrix, // params for rank_feature
//					clu_u_matrix, clu_v_matrix,
//					rank_learnRate, rank_regw, rank_regv, rank_LL, rank_grad, // params for model
//					clu_learnRate, clu_regv, clu_LL, clu_grad,
//					clu_vindice, clu_video, clu_luser, clu_ruser, clu_vdt, // input for cluster by video
//					clu_uindice, clu_user, clu_lvideo, clu_rvideo, clu_udt, // input for cluster by user
//					"norm_user", alpha
//					); // input for cluster by user
//		} else if (method.compare("clu_norm_user") == 0) {
//			coFactor(rank_indice, rank_user, rank_video, done, rank_order, // input for rank
//					rank_v_vector, rank_u_matrix, rank_v_matrix, // params for rank_feature
//					clu_u_matrix, clu_v_matrix,
//					rank_learnRate, rank_regw, rank_regv, rank_LL, rank_grad, // params for model
//					clu_learnRate, clu_regv, clu_LL, clu_grad,
//					clu_vindice, clu_video, clu_luser, clu_ruser, clu_vdt, // input for cluster by video
//					clu_uindice, clu_user, clu_lvideo, clu_rvideo, clu_udt, // input for cluster by user
//					"norm_user", alpha
//					); // input for cluster by user
//		} else if(method.compare("clu_bynorm_user") == 0) {
//			coFactor(rank_indice, rank_user, rank_video, done, rank_order, // input for rank
//					rank_v_vector, rank_u_matrix, rank_v_matrix, // params for rank_feature
//					clu_u_matrix, clu_v_matrix,
//					rank_learnRate, rank_regw, rank_regv, rank_LL, rank_grad, // params for model
//					clu_learnRate, clu_regv, clu_LL, clu_grad,
//					clu_vindice, clu_video, clu_luser, clu_ruser, clu_vdt, // input for cluster by video
//					clu_uindice, clu_user, clu_lvideo, clu_rvideo, clu_udt, // input for cluster by user
//					"bynorm_user", alpha
//					); // input for cluster by user
//		}

		// learning for ranking
		dvec prec(topK/5), recall(topK/5), mean_ap(topK/5), rankscore(topK/5);
		bool toFile = false;
		if(i >= 1 && (i+1) % 1 == 0) {
			if(i % 1000 == 0) toFile = true;
			evaluateRank(rank_utest, rank_vtest, done, rank_v_vector, rank_u_matrix, rank_v_matrix, topK, prec, recall, mean_ap, rankscore, map_video, map_user, dir+"/block", toFile, method);
			toFile = false;
//			cout << method << "\titerNum: " << i << "\trank_LL: " << rank_LL << "\trank_delta: " << rank_grad << "\tclu_LL: " << clu_LL << "\tclu_delta: " << clu_grad << "\tprec: " << prec << "\ttrainHit: " << trainHit << "\ttestHit: " << hit
//					<< "\ttestAll: " << testAll << "\trankscore: " << rankscore << endl;
			for(int j = 0; j < topK/5; j++) {
				writeFiles[j] << i << "\t" << rank_LL << "\t" << prec[j] << "\t" << recall[j] << "\t" << mean_ap[j] << "\t" << rank_grad << '\t' << clu_grad << "\t" << rankscore[j] << "\n";
			}
		} else {
//			cout << method << "\titerNum: " << i << "\trank_LL: " << rank_LL << "\trank_delta: " << rank_grad  << "\tclu_LL: " << clu_LL << "\tclu_delta: " << clu_grad << endl;
//			writeFile << i << "\t" << rank_LL << "\n";
		}
		for(int i = 0; i < topK/5; i++) {
			writeFiles[i].flush();
		}



	}

}

















































