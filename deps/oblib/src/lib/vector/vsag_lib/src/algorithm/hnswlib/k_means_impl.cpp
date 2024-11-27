#pragma once

#include "k_means.h"
#include <random>


KMeans::KMeans(int _k, int _N, int _dim, hnswlib::SpaceInterface<float> *_space, 
    DataLoader *_data_loader, vector<int> &ids)
    : k(_k), N(_N), dim(_dim), space(_space), data_loader(_data_loader) {

    // Constructor implementation

    centroids.resize(k);
    for (int i = 0; i < k; i++)
        centroids[i].resize(dim);
    assignments.resize(N, -1); // 初始化点分配为-1
    cluster_nums.resize(k, 0);
    diameters.resize(k, 0);
    globalIDS = ids;
    clusters.resize(k);

    center_dis.resize(k, vector<float>(k));
    dis2centroid.resize(N, 1e9);
}

static std::random_device rd; // 用于获取随机数种子
static std::mt19937 gen(rd()); // 标准 mersenne_twister_engine
int rand_int(int l, int r) {
    std::uniform_int_distribution<> dis(l, r); // 定义在[min, max]范围内的均匀分布
    return dis(gen);
}

// 使用的均为[0, N)的local id

void KMeans::initializeCenters() {
    std::mt19937 gen(random_device{}()); // 使用设备随机数作为种子
    std::uniform_real_distribution<float> dist(0.0, 1.0); // 定义分布范围

    vector<int> centers;
    centers.resize(k, -1); // 初始化中心索引为-1

    // first random center
    centers[0] = rand_int(0, N-1);
    vector<double> weight(N);

    for (int i = 0; i < N; i++)
        weight[i] = MAXFLOAT;


    for (int j = 1; j < k; j++) {
        double sum = 0;
        double choice = 0;

        int lst_center = centers[j-1];

        for (int i = 0; i < N; i++) {
            weight[i] = min(weight[i], (double)this->dis(i, lst_center));
            sum += weight[i];
        }
        
        choice = dist(gen) * sum;
        for (int i = 0; i < N; i++) {
            choice -= weight[i];
            if (weight[i] > 0 && choice <= 0) {
                centers[j] = i;
                break;
            }
        }
    }

    // sort and check
    sort(centers.begin(), centers.end());
    for (int i = 0; i+1 < centers.size(); i++) {
        if (centers[i] == centers[i+1]) {
            cerr << "error: duplicate center\n";
            assert(centers[i] != centers[i+1]);
        }
    }
    
    for (int i = 0; i < k; i++) {
        memcpy(centroids[i].data(), data_loader->point_data(centers[i]), sizeof(float) * dim);
    }
    
}


void KMeans::assignPointsToClosestCenter() {
    for (int j = 0; j < k; j++) {
        cluster_nums[j] = 0;
    }
    for (int i = 0; i < N; ++i) {
        float min_dist = 1e9;
        int closest_center = assignments[i];
        if (closest_center != -1) min_dist = dis(i, centroids[closest_center].data());
        for (int j = 0; j < k; ++j) {
            if (closest_center != -1 && min_dist * 2 < center_dis[closest_center][j]) continue;
            float dist = dis(i, centroids[j].data());
            if (dist < min_dist) {
                min_dist = dist;
                closest_center = j;
            }
        }
        assignments[i] = closest_center;
        dis2centroid[i] = min_dist;
        cluster_nums[closest_center] ++ ;
    }
}


void KMeans::updateCenters() {
    vector<vector<float> > tem(k);
    for (int i = 0; i < k; i++)
        tem[i].resize(dim);
    for (int i = 0; i < N; i++) {
        const float *point = reinterpret_cast<const float*>(data_loader->point_data(i));
        int centroid_id = assignments[i];
        for (int j = 0; j < dim; j++)
            tem[centroid_id][j] += point[j];
    }
    for (int i = 0; i < k; i++)
        for (int j = 0; j < dim; j++)
            centroids[i][j] = tem[i][j] / cluster_nums[i];
    
    return ;
}


void KMeans::updateCenterDis() {
    for (int i = 0; i < k; i ++)
        for (int j = i + 1; j < k; j++)
            center_dis[i][j] = center_dis[j][i] = dis(centroids[i].data(), centroids[j].data());
    for (int i = 0; i < k; i++)
        center_dis[i][i] = 0;  
}


void KMeans::run(int maxIterations) {

    initializeCenters();
    updateCenterDis();

    for (int iter = 0; iter < maxIterations; ++iter) {
        assignPointsToClosestCenter();

        auto lst_centers = centroids;

        updateCenters();
        updateCenterDis();

        cout << "iter = " << iter << ' ' << this->tot_dist_recalc() << '\n';
        bool changed = 0;
        for (int j = 0; j < k; j++)
            for (int d = 0; d < dim; d++) {
                if (fabs(centroids[j][d] - lst_centers[j][d]) > 1e-3) {
                    changed = 1;
                    break;
                }
            }
        if (!changed) break;

    }
    for (int i = 0; i < N; i++) {
        clusters[assignments[i]].push_back(i);
    }
    calc_diameter();
}

float KMeans::dis(int i, int j) {
    // dis 实现...
    i = globalIDS[i];
    j = globalIDS[j];
    return space->get_dist_func()(data_loader->point_data(i), data_loader->point_data(j), space->get_dist_func_param());
}

t
float KMeans::dis(int i, const void* data_point) {
    // dis 实现...
    i = globalIDS[i];
    return space->get_dist_func()(data_loader->point_data(i), data_point, space->get_dist_func_param());
}


float KMeans::dis(const void* data_point, const void* data_point2) {
    // dis 实现...
    return space->get_dist_func()(data_point, data_point2, space->get_dist_func_param());
}


float KMeans::tot_dist() {
    // tot_dist 实现...
    float sum = 0;
    return sum;
}


float KMeans::tot_dist_recalc() {
    // tot_dist_recalc 实现...
    float ans = 0;
    for (int i = 0; i < N; i++)
        // ans += dis(i, centers[assignments[i]]);
        ans += dis(i, centroids[assignments[i]].data());
    return ans;
}


void KMeans::calc_diameter() {
    for (int i = 0; i < k; i++) {
        diameters[i] = 0;
        for (auto p: clusters[i]) {
            // float d = dis(centers[i], p);
            float d = dis(p, centroids[i].data());
            diameters[i] = max(diameters[i], d);
        }
    }
}

void KMeans::output() {
    // output 实现...
    cout << "centers:\n";
    for (int i = 0; i < k; i++) {
        assert(clusters[i].size() == cluster_nums[i]);
        cout << i << " nums = " << cluster_nums[i] << " diameter = " << diameters[i] << '\n';
    }
}


vector<vector<float> > KMeans::get_centers_global() {
    return centroids;
}

int searchCalc = 0;


priority_queue<pair<float, int> > KMeans::find_nearest_centers_id(const void* data_point, int nprobe) {
    priority_queue<pair<float, int> > near_centers;
    for (int i = 0; i < k; i++) {
        float d = dis(centroids[i].data(), data_point);
        near_centers.push(make_pair(d, i));
        if (near_centers.size() > nprobe) near_centers.pop();
    }
    return near_centers;
}


vector<vector<float> > KMeans::find_nearest_centers(const void* data_point, int nprobe) {
    auto nn_ids = find_nearest_centers_id(data_point, nprobe);
    vector<vector<float> > ans;
    while (nn_ids.size()) {
        ans.push_back(centroids[nn_ids.top().second]);
        nn_ids.pop();
    }
    reverse(ans.begin(), ans.end());
    return ans;
}
// ivf-flat

vector<tableint> KMeans::searchKnn(const void* data_point, int knn, int nprobe) {

    // centers id: [0, k-1]
    auto near_centers = find_nearest_centers_id(data_point, nprobe);

    priority_queue<pair<float, int> > nns;
    while(near_centers.size()) {
        int center_id = near_centers.top().second;
        // // statis search cluster nums
        // cout << clusters[center_id].size() << "\n";
        searchCalc += clusters[center_id].size();
        for (auto p: clusters[center_id]) {
            float d = dis(p, data_point);
            nns.push(make_pair(d, globalIDS[p]));
            if (nns.size() > knn) nns.pop();
        }
        near_centers.pop();
    }

    vector<tableint> result;
    while(nns.size()) {
        result.push_back(nns.top().second);
        nns.pop();
    }
    reverse(result.begin(), result.end());
    return result;
}

inline float * KMeans::get_assign(int id) {
    return centroids[assignments[id]].data();
}

int prune_search_calc = 0;
int prune_search_probes = 0;

vector<tableint> KMeans::searchKnn_prune(const void* data_point, int knn, int nprobe, float epsilon) {
    auto near_centers = find_nearest_centers_id(data_point, nprobe);
    vector<pair<float, int> > centers;
    while(near_centers.size()) {
        centers.push_back(near_centers.top());
        near_centers.pop();
    }
    reverse(centers.begin(), centers.end());

    priority_queue<pair<float, int> > nns;
    float min_dis = 1e9;  
    for (auto pr: centers) {
        if (pr.first < min_dis) min_dis = pr.first;
        else if (pr.first > min_dis * (1+epsilon)) break;
        int center_id = pr.second;
        prune_search_calc += clusters[center_id].size();
        prune_search_probes ++;
        for (auto p: clusters[center_id]) {
            float d = dis(p, data_point);
            nns.push(make_pair(d, globalIDS[p]));
            if (nns.size() > knn) nns.pop();
        }
    }

    vector<tableint> result;
    while(nns.size()) {
        result.push_back(nns.top().second);
        nns.pop();
    }
    reverse(result.begin(), result.end());
    return result;
}


vector<int> KMeans::find_center_point_global_id() {
    vector<int> center_points(this->k);
    vector<float> center_dis(this->k, 1e9);
    for (int i = 0; i < N; i++) {
        int belong = assignments[i];
        auto dist = dis(i, get_assign(i));
        if (dist < center_dis[belong]);
        center_points[belong] = i;
    }
    for (auto &c: center_points)
        c = globalIDS[c];
    return center_points;
}