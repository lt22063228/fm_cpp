/*
 * Cluster.h
 *
 *  Created on: Dec 20, 2015
 *      Author: lin
 */

#ifndef CLUSTER_H_
#define CLUSTER_H_
#include "Util.h"
#include "Recommend.h"

namespace primer {

class Cluster {
public:
	Cluster();
	virtual ~Cluster();
};

} /* namespace primer */
void loadMap(simap &user_map, ismap &map_user, simap &video_map, ismap &map_video, std::string userpath, std::string videopath, char sep);
void loadCluster(simap &user_map, simap &video_map, ivec &key, ivec &left, ivec &right, ivec &clu_dt, std::string path, char sep);
void loadCorder(ivec &user, ivec &video, std::map<int,ivec> &done, ivec &clu_order);
void sgdCluster(ddvec &midMatrix, ddvec &sideMatrix, ivec &key, ivec &left, ivec &right, ivec &clu_dt,
		double learnRate, ivec &indice, double &delta, double &LL, double regv, int scale, double alpha, ivec &rank_order);
void sgdNormCluster(ddvec &midMatrix, ddvec &sideMatrix, ivec &mid, ivec &left, ivec &right, ivec &clu_dt,
		double learnRate, ivec &indice, double &delta, double &LL, double regv, int scale, double phi, double alpha);
void loadTarget(simap &user_map, simap &video_map, std::map<int, ivec > &target, std::string path);
void evaluateCluster(ddvec &userMatrix, ddvec &videoMatrix, std::map<int,ivec> &target, ivec &utest, ivec &vltest, ivec &vrtest, double &hit, double &prec);

#endif /* CLUSTER_H_ */






























