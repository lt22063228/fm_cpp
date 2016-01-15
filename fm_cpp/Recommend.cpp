/*
 * Recommend.cpp
 *
 *  Created on: Dec 17, 2015
 *      Author: lin
 */

#include "Recommend.h"

namespace primer {

Recommend::Recommend() {
	// TODO Auto-generated constructor stub

}

Recommend::~Recommend() {
	// TODO Auto-generated destructor stub
}

} /* namespace primer */

using namespace std;
void loadTrain(svec &container, string filePath) {
	ifstream readFile;
	readFile.open(filePath.c_str());
	while(!readFile.eof()) {
		string data;
		readFile >> data;
		if(data.empty()) continue;
		container.push_back(data);
	}
}

void loadDone(map<int, ivec > &container, const ivec &users, const ivec &videos) {
	int len = users.size();
	for(int i = 0; i < len; i++) {
		if(container.find(users[i]) == container.end()) {
			container.insert(std::pair<int, ivec >(users[i], ivec()));
		}
		container[users[i]].push_back(videos[i]);
	}
}


int getNegative(const ivec &vlist, int numVideos) {
	int vn;
	while (true) {
		vn = rand() % numVideos;
		if(std::find(vlist.begin(), vlist.end(), vn) == vlist.end()) break;
	}
	return vn;
}
//	initFeatures(videoVector, userMatrix, videoMatrix, stdev, numUsers, numVideos, dim);
void initFeatures(dvec &videoVector, ddvec &userMatrix, ddvec &videoMatrix, double stdev, int numUsers, int numVideos, int dim) {
	for(int i = 0; i < numVideos; i++) videoVector.push_back(0);
	for(int i = 0; i < numUsers; i++) {
		userMatrix.push_back(dvec());
		for(int f = 0; f < dim; f++) {
			userMatrix[i].push_back(ran_gaussian(0,stdev));
		}
	}
	for(int i = 0; i < numVideos; i++) {
		videoMatrix.push_back(dvec());
		for(int f = 0; f < dim; f++) {
			videoMatrix[i].push_back(ran_gaussian(0, stdev));
		}
	}
}
double computeScore(int u, int vp, int vn, const dvec &videoVector, const ddvec &userMatrix, const ddvec &videoMatrix) {
	double total = 0.;
	int dim = videoMatrix[0].size();
	total += videoVector[vp];
	total -= videoVector[vn];
	for(int i = 0; i < dim; i++) {
		total += userMatrix[u][i] * videoMatrix[vp][i];
		total -= userMatrix[u][i] * videoMatrix[vn][i];
	}

	return total;
}

void sgdRank(const ivec &indice, const ivec &user, const ivec &video, map<int, ivec > &done,
		ivec &order,
		dvec &videoVector, ddvec &userMatrix, ddvec &videoMatrix,
		double learnRate, double regw, double regv,
		double &LL, double &tmpdelt) {
	int dim = userMatrix[0].size();


	LL = 0.; tmpdelt = 0.;
	double total = 0.;
	for(int i = 0; i < indice.size(); i++) {
//		cout << i << endl;
		double delta = 0.;
		int cnt = 0;
		int idx = indice[i];
		int u = user[idx], vp = video[idx]; int ord = order[idx];
		int vn = getNegative(done[u], videoVector.size());
		double dicount = 1./done[u].size();
//		double tmplr = learnRate;
//		learnRate *= dicount;
//		std::cout << "start..." << std::endl;
		double vp_scalar, vn_scalar;
		dvec u_vector(dim), vp_vector(dim), vn_vector(dim);
		double tmpscore = computeScore(u, vp, vn, videoVector, userMatrix, videoMatrix);
		double normalizer = (1-logistic(tmpscore));

		double g = 0.;

		// unary postive
		g = normalizer - 2 * regw * videoVector[vp];
		vp_scalar = videoVector[vp] + learnRate * g;
		delta += abs(g);
		cnt ++;

		// unary negative
		g = -normalizer - 2 * regw * videoVector[vn];
		vn_scalar = videoVector[vn] + learnRate * g;
		delta += abs(g);
		cnt ++;

		double disc = 1./ord;
		// user
		for(int d = 0; d < dim; d++) {
			g = normalizer * (videoMatrix[vp][d] - videoMatrix[vn][d]);
			g -= 2 * regv * userMatrix[u][d];
			u_vector[d] = userMatrix[u][d] + learnRate * g;
			total += u_vector[d];
			delta += abs(g);
			cnt ++;
		}

		// pairwise positive
		for(int d = 0; d < dim; d++) {
			g = normalizer * userMatrix[u][d];
			g -= 2 * regv * videoMatrix[vp][d];
			vp_vector[d] = videoMatrix[vp][d] + learnRate * g;
			total += vp_vector[d];
			delta += abs(g);
			cnt ++;
		}

		// pairwise negativ
		for(int d = 0; d < dim; d++) {
			g = -normalizer * userMatrix[u][d];
			g -= 2 * regv * videoMatrix[vn][d];
			vn_vector[d] = videoMatrix[vn][d] + learnRate * g;
			total += vn_vector[d];
			delta += abs(g);
			cnt ++;
		}


//		learnRate = tmplr;

		videoVector[vp] = vp_scalar; videoVector[vn] = vn_scalar;
		for(int f = 0; f < dim; f++) {
			userMatrix[u][f] = u_vector[f];
			videoMatrix[vp][f] = vp_vector[f];
			videoMatrix[vn][f] = vn_vector[f];
		}


		tmpdelt += delta/cnt;
		total /= cnt;
		LL += std::log(1 / (1 + exp(-tmpscore)));

	}
	tmpdelt /= indice.size();
//	cout << (total/indice.size()) << endl;

}



void scoring(dvec &scores, int dim, int u,
		const dvec &videoVector, const ddvec &userMatrix, const ddvec &videoMatrix, map<int,ivec> done) {
	for(int v = 0; v < videoVector.size(); v++) {
		double score = videoVector[v];
//		double score = 0.;
		for(int d = 0; d < dim; d++) {
			score += userMatrix[u][d] * videoMatrix[v][d];
		}
		scores.push_back(score);
	}
	for(int i = 0; i < done[u].size(); i++) {
		scores[done[u][i]] = -1000.;
	}

}

void topping(ivec &tops, vector<size_t> &rank, map<int, ivec> done,
			int u, int topK, double &trainHit) {

	for(int r = 0; r < rank.size(); r ++) {
		if(tops.size() == topK) break;
		if(std::find(done[u].begin(), done[u].end(), rank[r]) != done[u].end()) {
			trainHit ++;
			continue;
		}
		tops.push_back(rank[r]);
	}

}

void evaluateRank(ivec &utest, ivec &vtest, map<int, ivec> &done,
		dvec videoVector, ddvec userMatrix, ddvec videoMatrix,
		int topK, double &prec, double &recall, double &mean_ap, double &trainHit, double &testHit, double &testAll, double &rankscore,
		ismap &video_map, ismap &user_map, string dir, bool toFile) {

	// get ground truth for each user
	int dim = userMatrix[0].size();
	int numTest = utest.size();
	map<int, ivec> test;
	for(int i = 0; i < numTest; i++) {
		int u = utest[i], v = vtest[i];
		if(test.find(u) == test.end()) {
			test.insert(std::pair<int, ivec>(u, ivec()));
		}
		test[u].push_back(v);
	}

	// get precision for each user
	testAll = 0.; rankscore = 0.;
	prec = 0.; trainHit = 0.; testHit = 0.;
	set<string> vs;
	set<string> us;
	double ill = 0.;
	for(int u = 0; u < done.size(); u++) {


		// compute score for each video
		dvec scores;
		scoring(scores, dim, u, videoVector, userMatrix, videoMatrix, done);

		int magic = 10;
		// rank the video and got topK
		vector<int> tops = sort_indexes(scores, magic, u);
//		topping(tops, rank, done, u, topK, trainHit);


		// calculate precision
		double all = 0., hit = 0.;
		double ap = 0.;
		double recall_denom = test[u].size();
		for(int r = 0; r < magic; r++) {
			all += 1.;
			testAll += 1.;
			if(std::find(test[u].begin(), test[u].end(), tops[r]) != test[u].end()) {
				hit += 1.;
				rankscore += 1./(1+r);
				testHit += 1;
				vs.insert(video_map[tops[r]]);
				ap += hit/all;
			}
		}

		if(all != 0) {
			recall += hit/recall_denom;
			prec += hit/all;
			mean_ap += ap/all;
		} else {
			ill += 1.;
		}
	}
	prec /= (done.size()-ill);
	recall /= (done.size()-ill);
	mean_ap /= (done.size()-ill);
	cout << "vl size: " << vs.size() << endl;
//	for(set<string>::iterator iter = vs.begin(); iter != vs.end(); iter ++) cout << *iter << endl;
//	prec = testHit/testAll;

	//
}
void loadTime(map<int,lvec> &time, ivec &rank_user, lvec &rank_time) {
	int len = rank_user.size();
	for(int i = 0; i < len; i++) {
		if(time.find(rank_user[i]) == time.end()) {
			time.insert(std::pair<int, lvec >(rank_user[i], lvec()));
		}
		time[rank_user[i]].push_back(rank_time[i]);
	}
}
//		loadOrder(rank_order, time, rank_user, rank_time);
void loadOrder(ivec &order, map<int,lvec> &time, ivec &rank_user, lvec &rank_time) {
	// sort a
	for(map<int,lvec>::iterator it = time.begin(); it != time.end(); it++) {
		lvec &v = time[it->first];
		sort(v.begin(), v.end(), greater<unsigned long>());
	}
	for(int i = 0; i < rank_user.size(); i++) {
		int u = rank_user[i]; unsigned long t = rank_time[i];
		lvec &v = time[u];
		int seq = 0;
		for(int j = 0; j < v.size(); j++) {
			if(v[j] == t) {
				seq = j+1;
			}
		}
		order.push_back(seq);
	}
}
void loadRank(ivec &user, ivec &video, lvec &time, simap &user_map, simap &video_map, string datpath, char sep) {
	ifstream readFile;
	readFile.open(datpath.c_str());
	string line;
	while(getline(readFile,line)) {
		if(line.empty()) break;
		stringstream linestream(line);
		char u[100], v[100], t[100];
		linestream.getline(u,100,sep);
		linestream.getline(v,100,sep);
		linestream.getline(t,100, sep);
		user.push_back(user_map[string(u)]);
		video.push_back(video_map[string(v)]);
		unsigned long tt;
		stringstream(string(t)) >> tt;
		time.push_back(tt);
//		cout << string(u) << " " << string(v) << endl;
//		cout << user_map[string(u)] << " " << video_map[string(v)] << endl;
	}
}
void loadTest(ivec &utest, ivec &vtest, simap &user_map, simap &video_map, string datpath, char sep) {
	ifstream readFile;
	readFile.open(datpath.c_str());
	string line;
	while(getline(readFile,line)) {
		if(line.empty()) break;
		stringstream linestream(line);
		char u[100],v[100],t[100];
		linestream.getline(u,100,sep);
		linestream.getline(v,100,sep);
		linestream.getline(t,100,sep);
		utest.push_back(user_map[string(u)]);
		vtest.push_back(video_map[string(v)]);
	}
}
//	loadTest(utest, vtest, user_map, video_map, testpath);
void initVector(dvec &videoVector, int numVideo) {
	for(int i = 0; i < numVideo; i++) videoVector.push_back(0);
}
void initMatrix(ddvec &userMatrix, ddvec &videoMatrix, double stdev, int numUser, int numVideo, int dim) {
	for(int i = 0; i < numUser; i++) {
		userMatrix.push_back(dvec());
		for(int f = 0; f < dim; f++) {
			userMatrix[i].push_back(ran_gaussian(0,stdev));
		}
	}
	for(int i = 0; i < numVideo; i++) {
		videoMatrix.push_back(dvec());
		for(int f = 0; f < dim; f++) {
			videoMatrix[i].push_back(ran_gaussian(0, stdev));
		}
	}

}

//int main() {
//	// load data
//	string dir = "/home/lin/workspace/data/youku";
//
//	// loading user and video map
//	std::cout << "loading user and video map data" << std::endl;
//	simap user_map, video_map;
//	string userpath = dir + "/block/user.map", videopath = dir + "/block/video.map";
//	loadMap(user_map, video_map, userpath, videopath);
//
//	// loading training rank data
//	std::cout << "loading train data" << std::endl;
//	ivec user, video;
//	string trainpath = dir + "/train.dat";
//	loadRank(user, video, user_map, video_map, trainpath);
//
//	// loading done data
//	std::cout << "loading done data" << std::endl;
//	map<int, ivec > done;
//	loadDone(done, user, video);
//
//	// loading testing rank data
//	std::cout << "loading test data" << std::endl;
//	ivec utest = ivec(); ivec vtest = ivec();
//	string testpath = dir + "/test.dat";
//	loadTest(utest, vtest, user_map, video_map, testpath);
//
//
//	std::cout << "initializing the features" << std::endl;
//	int numUser = user_map.size(), numVideo = video_map.size();
//	int dim = 48;
//	double stdev = 0.1;
//	dvec videoVector = dvec();	ddvec userMatrix = ddvec();	ddvec videoMatrix = ddvec();
//	initVector(videoVector, numVideo);
//	initMatrix(userMatrix, videoMatrix, stdev, numUser, numVideo, dim);
//
//	// learning params
//	int numIter = 100000, topK = 100;
//	double learnRate = 0.01, regw = 0.01, regv = 0.01;
//
//	std::cout << "dim: " << dim << "\tlearRate: " << learnRate << "\tegw: " << regw << "\tegv: " << regv << std::endl;
//
//	std::cout << "shuffling the data" << std::endl;
//	// random indice
//	int numTrain = user.size();
//	ivec indices;
//	for(int i = 0; i < numTrain; i++) {
//		indices.push_back(i);
//	}
//
//
//	std::cout << "start the gradient descent" << std::endl;
//	// iterate to do stochastic gradient descent
//	for(int i = 0; i < numIter; i++) {
//		std::random_shuffle(indices.begin(), indices.end());
//		double LL = 0., delta = 0.;
//		sgdRank(indices, user, video, done,
//				videoVector, userMatrix, videoMatrix,
//				learnRate, regw, regv,
//				LL, delta);
//		double prec = 0., trainHit = 0., testHit = 0., testAll = 0.;
//		if(i % 50 == 0) {
//			evaluateRank(utest, vtest, done, videoVector, userMatrix, videoMatrix, topK, prec, trainHit, testHit, testAll);
//			cout << "iter: " << i << "\tLL: " << LL << "\tdelta: " << delta << "\tprec: " << prec << "\ttrainHit: " << trainHit << "\ttestHit: " << testHit
//					<< "\ttestAll: " << testAll << endl;
//		} else {
//			cout << "iter: " << i << "\tLL: " << LL << "\tdelta: " << delta << endl;
//		}
//	}
//
//
//
//
//}




