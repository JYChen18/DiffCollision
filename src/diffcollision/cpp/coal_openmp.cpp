#include <coal/collision_object.h>
#include <coal/distance.h>
#include <coal/shape/convex.h>
#include <coal/shape/geometric_shapes.h>
#include <coal/BVH/BVH_model.h>
#include <coal/mesh_loader/loader.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include <random>

namespace py = pybind11;

std::shared_ptr<const coal::CollisionGeometry> get_convex_from_file(const std::string& file_name, const py::array_t<double> scale) {
    coal::NODE_TYPE bv_type = coal::BV_AABB;
    coal::MeshLoader loader(bv_type);
    auto scale_ptr = scale.data();
    Eigen::Vector3d eigen_scale(scale_ptr[0], scale_ptr[1], scale_ptr[2]);
    coal::BVHModelPtr_t bvh = loader.load(file_name, eigen_scale);
    bvh->buildConvexHull(true, "Qt");
    return bvh->convex; 
}

std::shared_ptr<const coal::CollisionGeometry> get_convex_from_data(const py::array_t<double> verts) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> eigen_verts(
        verts.data(), verts.shape(0), 3);
    auto bvh = std::make_shared<coal::BVHModel<coal::AABB>>();
    bvh->beginModel();
    bvh->addVertices(eigen_verts);
    bvh->endModel();
    bvh->buildConvexHull(true, "Qt");
    return bvh->convex; 
}

void batched_coal_distance(
    const std::vector<std::shared_ptr<const coal::CollisionGeometry>> shape_lst,  // Use CollisionGeometry here
    const py::array_t<size_t> shape1_idx_lst, 
    const py::array_t<double> pose1_lst, 
    const py::array_t<size_t> shape2_idx_lst, 
    const py::array_t<double> pose2_lst, 
    const py::array_t<size_t> group_idx_lst, 
    const py::array_t<size_t> valid_idx_lst, 
    const size_t n_batch,
    const size_t n_pair,     // convex piece pairs
    const size_t n_group,    // mesh pairs
    const size_t n_valid,
    const int n_thread,
    py::array_t<double> dist_result, 
    py::array_t<double> normal_result, 
    py::array_t<double> wp1_result, 
    py::array_t<double> wp2_result,
    py::array_t<size_t> min_idx_result) {
    
    const double* pose1_ptr = pose1_lst.data();
    const size_t* shape1_idx_ptr = shape1_idx_lst.data();
    const double* pose2_ptr = pose2_lst.data();
    const size_t* shape2_idx_ptr = shape2_idx_lst.data();
    const size_t* select_idx_ptr = valid_idx_lst.data();
    const size_t* group_idx_ptr = group_idx_lst.data();

    double* dist_result_ptr = dist_result.mutable_data();
    double* normal_result_ptr = normal_result.mutable_data();
    double* wp1_result_ptr = wp1_result.mutable_data();
    double* wp2_result_ptr = wp2_result.mutable_data();
    size_t* min_idx_result_ptr = min_idx_result.mutable_data();

    size_t total_groups = n_batch * n_group;

    omp_set_num_threads(n_thread);

    // Check the real number of threads
    int T = 0;
    #pragma omp parallel
    {
        #pragma omp single
        T = omp_get_num_threads();
    }

    // Allocate and initialize
    std::vector<size_t> local_idx(T * total_groups, 0);
    std::vector<coal::DistanceResult> local_res;
    local_res.resize((size_t)T * total_groups);
    coal::DistanceResult inf_dr;
    inf_dr.min_distance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < local_res.size(); ++i) local_res[i] = inf_dr;

    // Main parallel loop: compute distances and update thread-local per-group minima
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t* my_local_idx = &local_idx[(size_t)tid * total_groups];
        coal::DistanceResult* my_local_res = &local_res[(size_t)tid * total_groups];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n_valid; ++i) {
            size_t idx = select_idx_ptr[i];
            size_t pair_idx = idx % n_pair;
            size_t batch_idx = idx / n_pair;
            size_t group_idx = batch_idx * n_group + group_idx_ptr[pair_idx];

            // Build transforms (same as your code)
            Eigen::Matrix3d R1;
            R1 << pose1_ptr[16*group_idx],   pose1_ptr[16*group_idx + 1], pose1_ptr[16*group_idx + 2],
                  pose1_ptr[16*group_idx + 4], pose1_ptr[16*group_idx + 5], pose1_ptr[16*group_idx + 6],
                  pose1_ptr[16*group_idx + 8], pose1_ptr[16*group_idx + 9], pose1_ptr[16*group_idx + 10];
            Eigen::Vector3d T1(pose1_ptr[16*group_idx + 3], pose1_ptr[16*group_idx + 7], pose1_ptr[16*group_idx + 11]);
            coal::Transform3s transform1(R1, T1);

            Eigen::Matrix3d R2;
            R2 << pose2_ptr[16*group_idx],   pose2_ptr[16*group_idx + 1], pose2_ptr[16*group_idx + 2],
                  pose2_ptr[16*group_idx + 4], pose2_ptr[16*group_idx + 5], pose2_ptr[16*group_idx + 6],
                  pose2_ptr[16*group_idx + 8], pose2_ptr[16*group_idx + 9], pose2_ptr[16*group_idx + 10];
            Eigen::Vector3d T2(pose2_ptr[16*group_idx + 3], pose2_ptr[16*group_idx + 7], pose2_ptr[16*group_idx + 11]);
            coal::Transform3s transform2(R2, T2);

            // Call COAL distance (CPU)
            coal::DistanceRequest dist_req;
            coal::DistanceResult dist_res;
            coal::distance(shape_lst[shape1_idx_ptr[pair_idx]].get(), transform1,
                           shape_lst[shape2_idx_ptr[pair_idx]].get(), transform2,
                           dist_req, dist_res);

            // Update thread-local best
            if (dist_res.min_distance < my_local_res[group_idx].min_distance) {
                my_local_res[group_idx] = dist_res;
                my_local_idx[group_idx] = pair_idx;
            }
        } // end for
    } // end parallel

    // Parallel reduction across threads per group -> write to output arrays
    #pragma omp parallel for schedule(static)
    for (size_t g = 0; g < total_groups; ++g) {
        size_t best_idx;
        coal::DistanceResult best = inf_dr;
        for (int t = 0; t < T; ++t) {
            const coal::DistanceResult& cand = local_res[(size_t)t * total_groups + g];
            if (cand.min_distance < best.min_distance) {
                best = cand;
                best_idx = local_idx[(size_t)t * total_groups + g];
            }
        }
        // Write outputs (if no sample, min_distance remains +inf and points are zero)
        dist_result_ptr[g] = best.min_distance;
        normal_result_ptr[3 * g + 0] = best.normal[0];
        normal_result_ptr[3 * g + 1] = best.normal[1];
        normal_result_ptr[3 * g + 2] = best.normal[2];
        wp1_result_ptr[3 * g + 0] = best.nearest_points[0][0];
        wp1_result_ptr[3 * g + 1] = best.nearest_points[0][1];
        wp1_result_ptr[3 * g + 2] = best.nearest_points[0][2];
        wp2_result_ptr[3 * g + 0] = best.nearest_points[1][0];
        wp2_result_ptr[3 * g + 1] = best.nearest_points[1][1];
        wp2_result_ptr[3 * g + 2] = best.nearest_points[1][2];
        min_idx_result_ptr[g] = best_idx;
    }
}

void batched_get_neighbor(
    const std::vector<std::shared_ptr<const coal::CollisionGeometry>>& shape_lst, // pass by const ref
    const py::array_t<size_t> valid_idx_lst, 
    const py::array_t<double> sep_vec_lst,    // assumed shape (n_valid, 3)
    const size_t n_valid,
    const size_t n_level,
    const size_t n_nbr,
    const int n_thread,
    py::array_t<double> neighbor_result) {

    const double* sep_vec_ptr = sep_vec_lst.data();
    const size_t* select_idx_ptr = valid_idx_lst.data();
    double* neighbor_result_ptr = neighbor_result.mutable_data();

    omp_set_num_threads(n_thread);

    #pragma omp parallel for
    for (size_t i = 0; i < n_valid; ++i) {
        const size_t sel_idx = select_idx_ptr[i];
        const std::shared_ptr<const coal::CollisionGeometry>& geom_ptr = shape_lst[sel_idx];
        const coal::ConvexBase* convex_base = dynamic_cast<const coal::ConvexBase*>(geom_ptr.get());

        Eigen::Matrix<double,3,1> sep_vec;
        sep_vec(0) = sep_vec_ptr[3 * i + 0];
        sep_vec(1) = sep_vec_ptr[3 * i + 1];
        sep_vec(2) = sep_vec_ptr[3 * i + 2];
       
        const std::vector<coal::Vec3s>& pts = *(convex_base->points);

        // find vertex with maximum dot
        double maxdot = -std::numeric_limits<double>::infinity();
        size_t max_id = 0;
        for (size_t j = 0; j < pts.size(); ++j) {
            const double dot = pts[j].dot(sep_vec);
            if (dot > maxdot) {
                maxdot = dot;
                max_id = j;
            }
        }
        std::vector<size_t> neighbor_lst;
        std::vector<size_t> level_lst;
        neighbor_lst.push_back(max_id);
        level_lst.push_back(0);

        // BFS 
        size_t curr_idx = 0;
        const std::vector<coal::ConvexBase::Neighbors>& neighbors = *(convex_base->neighbors);
        while (curr_idx < neighbor_lst.size()) {
            size_t vertex_idx = neighbor_lst[curr_idx];
            size_t curr_level = level_lst[curr_idx];
            if (curr_level < n_level) {
                const coal::ConvexBase::Neighbors& point_neighbors = neighbors[vertex_idx];
                const size_t cnt = static_cast<size_t>(point_neighbors.count());
                for (size_t jj = 0; jj < cnt; ++jj) {
                    size_t neighbor_index = point_neighbors[jj];
                    auto it = std::find(neighbor_lst.begin(), neighbor_lst.end(), neighbor_index);
                    if (it == neighbor_lst.end()) {
                        level_lst.push_back(curr_level + 1);
                        neighbor_lst.push_back(neighbor_index);
                    }
                }
            }
            ++curr_idx;
        }

        // per-iteration random generator (seed depends on i to avoid identical shuffles across threads)
        std::mt19937 gen(static_cast<size_t>(i + 123456));
        std::shuffle(neighbor_lst.begin(), neighbor_lst.end(), gen);

        // fill results (store as double)
        for (size_t j = 0; j < n_nbr; ++j) {
            size_t idx = neighbor_lst[j % neighbor_lst.size()];
            neighbor_result_ptr[3 * (n_nbr * i + j) + 0] = pts[idx][0];
            neighbor_result_ptr[3 * (n_nbr * i + j) + 1] = pts[idx][1];
            neighbor_result_ptr[3 * (n_nbr * i + j) + 2] = pts[idx][2];
        }
    } // end omp for
}


// Python bindings using pybind11
PYBIND11_MODULE(_coal_openmp, m) {
    py::class_<coal::CollisionGeometry, std::shared_ptr<coal::CollisionGeometry>>(m, "CollisionGeometry");

    // Register the get_convex_from_file function
    m.def("get_convex_from_file", &get_convex_from_file, "Load a convex mesh from file and return as CollisionGeometry pointer",
          py::arg("file_name"), py::arg("scale"));

    // Register the get_convex_from_data function
    m.def("get_convex_from_data", &get_convex_from_data, "Build a convex mesh from vertices and return as CollisionGeometry pointer",
          py::arg("vertices"));

    // Register the batched_coal_distance function
    m.def("batched_coal_distance", &batched_coal_distance, "Compute batched distances using COAL",
          py::arg("shape_lst"), py::arg("shape1_idx_lst"), py::arg("pose1_lst"), 
          py::arg("shape2_idx_lst"), py::arg("pose2_lst"), py::arg("group_idx_lst"), 
          py::arg("valid_idx_lst"), py::arg("n_batch"), py::arg("n_pair"), 
          py::arg("n_group"), py::arg("n_valid"), py::arg("n_thread"), 
          py::arg("dist_result").noconvert(), py::arg("normal_result").noconvert(), 
          py::arg("wp1_result").noconvert(), py::arg("wp2_result").noconvert(),
          py::arg("min_idx_result").noconvert());

    m.def("batched_get_neighbor", &batched_get_neighbor, "Get neighbors",
          py::arg("shape_lst"), py::arg("valid_idx_lst"), py::arg("sep_vec_lst"),
          py::arg("n_valid"), py::arg("n_level"), py::arg("n_nbr"), py::arg("n_thread"), 
          py::arg("neighbor_result").noconvert());
}