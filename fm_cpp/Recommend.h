/*
 * Recommend.h
 *
 *  Created on: Dec 17, 2015
 *      Author: lin
 */

#ifndef RECOMMEND_H_
#define RECOMMEND_H_
#include "Util.h"
#include "Cluster.h"
#include "string"


namespace primer {

class Recommend {
public:
	Recommend();
	virtual ~Recommend();
};

} /* namespace primer */

void loadTrain(svec &container, std::string filePath);
void loadDone(std::map<int, ivec > &container, const ivec &users, const ivec &videos);
void loadTime(std::map<int, lvec> &time, ivec &rank_user, lvec &rank_time);
void loadOrder(ivec &order, std::map<int,lvec> &time, ivec &rank_user, lvec &rank_time);
int getNegative(const ivec &vlist, int numVideos);
void initFeatures(dvec &videoVector, ddvec &userMatrix, ddvec &videoMatrix, double stdev, int numUsers, int numVideos, int dim);
double computeScore(int u, int vp, int vn, const dvec &videoVector, const ddvec &userMatrix, const ddvec &videoMatrix);
void sgdRank(const ivec &indice, const ivec &user, const ivec &video, std::map<int, ivec > &done,
		ivec &order,
		dvec &videoVector, ddvec &userMatrix, ddvec &videoMatrix,
		double learnRate, double regw, double regv,
		double &LL, double &tmpdelt);
void scoring(dvec &scores, int dim, int u,
		const dvec &videoVector, const ddvec &userMatrix, const ddvec &videoMatrix);
void topping(ivec &tops, std::vector<size_t> &rank, std::map<int, ivec> done,
			int u, int topK, double &trainHit);
void evaluateRank(ivec &utest, ivec &vtest, std::map<int, ivec> &done,
		dvec videoVector, ddvec userMatrix, ddvec videoMatrix,
		int topK, dvec &prec, dvec &recall, dvec &mean_ap, dvec &rankscore, ismap& map_video, ismap &map_user, std::string dir, bool toFile, std::string method);
void loadRank(ivec &user, ivec &video, lvec &time, simap &user_map, simap &video_map, std::string datpath, char sep);
void loadTest(ivec &utest, ivec &vtest, simap &user_map, simap &video_map, std::string datpath, char sep);
void initVector(dvec &videoVector, int numVideo);
void initMatrix(ddvec &userMatrix, ddvec &videoMatrix, double stdev, int numUser, int numVideo, int dim);
#endif /* RECOMMEND_H_ */
















