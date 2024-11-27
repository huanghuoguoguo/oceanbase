// k_means.h
#ifndef K_MEANS_H
#define K_MEANS_H

#include "../hnswlib/hnswlib.h"
#include <vector>
#include <limits>
#include "../../default_allocator.h"
#include "../../simd/simd.h"
#include "../../utils.h"
#include <functional>

namespace hnswlib{
using tableint = unsigned int;
class KMeans : public AlgorithmInterface<float>{
public:
    KMeans(int _k, int _N, int _dim);
    ~Kmeans()=default;

    void initializeCenters();
    void assignPointsToClosestCenter();
    void updateCenters();
    void updateCenterDis();

    // 最主要的计算K个质心的函数
    void run(int maxIterations = 100);

    // i,j均为局部坐标
    float dis(int i, int j);
    float dis(int i, const void* data_point);
    float dis(const void* data_point, const void* data_point2);
    float tot_dist();
    float tot_dist_recalc();
    void output();
    // 返回
    vector<vector<float> > get_centers_global();
    void calc_diameter();
    vector<tableint> searchKnn(const void* data_point, int knn, int nprobe);
    vector<tableint> searchKnn_prune(const void* data_point, int knn, int nprobe, float epsilon);
    priority_queue<pair<float, int> > find_nearest_centers_id(const void* data_point, int nprobe);
    vector<vector<float> > find_nearest_centers(const void* data_point, int nprobe);
    inline float *get_assign(int id);

    // 找到每个类中距离centroids最近的那个点
    vector<int> find_center_point_global_id();


    // agf
    bool
    addPoint(const void* datapoint, labeltype label);

    std::priority_queue<std::pair<float, labeltype>>
    searchKnn(const void*, size_t, size_t, BaseFilterFunctor* isIdAllowed = nullptr);

    std::priority_queue<std::pair<float, labeltype>>
    searchRange(const void*, float, size_t, BaseFilterFunctor* isIdAllowed = nullptr);

    // Return k nearest neighbor in the order of closer fist
    std::vector<std::pair<float, labeltype>>
    searchKnnCloserFirst(const void* query_data,
                         size_t k,
                         size_t ef,
                         BaseFilterFunctor* isIdAllowed = nullptr) const;

    void
    saveIndex(const std::string& location);

    void
    saveIndex(void* d);

    void
    saveIndex(std::ostream& out_stream);

    size_t
    getMaxElements();

    float
    getDistanceByLabel(labeltype label, const void* data_point);

    virtual const float*
    getDataByLabel(labeltype label) const;

    virtual std::priority_queue<std::pair<float, labeltype>>
    bruteForce(const void* data_point, int64_t k);

    virtual void
    resizeIndex(size_t new_max_elements);

    virtual size_t
    calcSerializeSize();

    virtual void
    loadIndex(std::function<void(uint64_t, uint64_t, void*)> read_func,
              SpaceInterface* s,
              size_t max_elements_i = 0);

    virtual void
    loadIndex(std::istream& in_stream, SpaceInterface* s, size_t max_elements_i = 0);

    virtual size_t
    getCurrentElementCount();

    virtual size_t
    getDeletedCount();

    virtual bool
    isValidLabel(labeltype label);

    virtual bool
    init_memory_space();

protected:
    int k, N, dim;
    hnswlib::SpaceInterface<float> *space;
    vector<int> cluster_nums, globalIDS;
    vector<vector<float> > center_dis;
    vector<float> dis2centroid;
    // vector<int> centers;

    // 质心
    vector<vector<float> > centroids;

    // assignments: [0, k)
    vector<int> assignments;
    vector<vector<int> > clusters;
    vector<float> diameters;

};
}




#include "k_means_impl.cpp"
#endif // K_MEANS_H