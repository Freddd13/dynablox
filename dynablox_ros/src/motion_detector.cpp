#include "dynablox_ros/motion_detector.h"

#include <math.h>

#include <future>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <minkindr_conversions/kindr_tf.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/impl/transforms.hpp>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>

namespace dynablox {

using Timer = voxblox::timing::Timer;

void MotionDetector::Config::checkParams() const {
  checkParamCond(!global_frame_name.empty(),
                 "'global_frame_name' may not be empty.");
  checkParamGE(num_threads, 1, "num_threads");
  checkParamGE(queue_size, 0, "queue_size");
}

void MotionDetector::Config::setupParamsAndPrinting() {
  setupParam("global_frame_name", &global_frame_name);
  setupParam("sensor_frame_name", &sensor_frame_name);
  setupParam("queue_size", &queue_size);
  setupParam("evaluate", &evaluate);
  setupParam("visualize", &visualize);
  setupParam("verbose", &verbose);
  setupParam("num_threads", &num_threads);
  setupParam("shutdown_after", &shutdown_after);
}

MotionDetector::MotionDetector(const ros::NodeHandle& nh,
                               const ros::NodeHandle& nh_private)
    : config_(
          config_utilities::getConfigFromRos<MotionDetector::Config>(nh_private)
              .checkValid()),
      nh_(nh),
      nh_private_(nh_private) {
  setupMembers();

  // Cache frequently used constants.
  voxels_per_side_ = tsdf_layer_->voxels_per_side();
  voxels_per_block_ = voxels_per_side_ * voxels_per_side_ * voxels_per_side_;

  // Advertise and subscribe to topics.
  setupRos();

  // Print current configuration of all components.
  LOG_IF(INFO, config_.verbose) << "Configuration:\n"
                                << config_utilities::Global::printAllConfigs();
}

void MotionDetector::setupMembers() {
  // Voxblox. Overwrite dependent config parts. Note that this TSDF layer is
  // shared with all other processing components and is mutable for processing.
  ros::NodeHandle nh_voxblox(nh_private_, "voxblox");
  nh_voxblox.setParam("world_frame", config_.global_frame_name);
  nh_voxblox.setParam("update_mesh_every_n_sec", 0);
  nh_voxblox.setParam("voxel_carving_enabled",
                      true);  // Integrate whole ray not only truncation band.
  nh_voxblox.setParam("allow_clear",
                      true);  // Integrate rays up to max_ray_length.
  nh_voxblox.setParam("integrator_threads", config_.num_threads);

  tsdf_server_ = std::make_shared<voxblox::TsdfServer>(nh_voxblox, nh_voxblox);
  tsdf_layer_.reset(tsdf_server_->getTsdfMapPtr()->getTsdfLayerPtr());

  // Preprocessing.
  preprocessing_ = std::make_shared<Preprocessing>(
      config_utilities::getConfigFromRos<Preprocessing::Config>(
          ros::NodeHandle(nh_private_, "preprocessing")));

  // Clustering.
  clustering_ = std::make_shared<Clustering>(
      config_utilities::getConfigFromRos<Clustering::Config>(
          ros::NodeHandle(nh_private_, "clustering")),
      tsdf_layer_);

  // Tracking.
  tracking_ = std::make_shared<Tracking>(
      config_utilities::getConfigFromRos<Tracking::Config>(
          ros::NodeHandle(nh_private_, "tracking")));

  // Ever-Free Integrator.
  ros::NodeHandle nh_ever_free(nh_private_, "ever_free_integrator");
  nh_ever_free.setParam("num_threads", config_.num_threads);
  ever_free_integrator_ = std::make_shared<EverFreeIntegrator>(
      config_utilities::getConfigFromRos<EverFreeIntegrator::Config>(
          nh_ever_free),
      tsdf_layer_);

  // Evaluation.
  if (config_.evaluate) {
    // NOTE(schmluk): These will be uninitialized if not requested, but then no
    // config files need to be set.
    evaluator_ = std::make_shared<Evaluator>(
        config_utilities::getConfigFromRos<Evaluator::Config>(
            ros::NodeHandle(nh_private_, "evaluation")));
  }

  // Visualization.
  visualizer_ = std::make_shared<MotionVisualizer>(
      ros::NodeHandle(nh_private_, "visualization"), tsdf_layer_);
}

void MotionDetector::setupRos() {
  lidar_pcl_sub_ = nh_.subscribe("pointcloud", config_.queue_size,
                                 &MotionDetector::pointcloudCallback, this);
}



// 入口。整个代码虽然复杂，但风格规范、结构清晰
void MotionDetector::pointcloudCallback(
    const sensor_msgs::PointCloud2::Ptr& msg) {
  Timer frame_timer("frame");
  Timer detection_timer("motion_detection");

  // Lookup cloud transform T_M_S of sensor (S) to map (M).
  // If different sensor frame is required, update the message.
  Timer tf_lookup_timer("motion_detection/tf_lookup");
  const std::string sensor_frame_name = config_.sensor_frame_name.empty()
                                            ? msg->header.frame_id
                                            : config_.sensor_frame_name;

  tf::StampedTransform T_M_S;
  if (!lookupTransform(config_.global_frame_name, sensor_frame_name,
                       msg->header.stamp.toNSec(), T_M_S)) {
    // Getting transform failed, need to skip.
    return;
  }
  tf_lookup_timer.Stop();

  // Preprocessing.
  Timer preprocessing_timer("motion_detection/preprocessing");
  frame_counter_++;
  CloudInfo cloud_info;
  Cloud cloud;
  preprocessing_->processPointcloud(msg, T_M_S, cloud, cloud_info);
  preprocessing_timer.Stop();


  // Build a mapping of all blocks to voxels to points for the scan.
  // Hatori: 通过将本帧点云投影到tsdf体素中，得到占据的block->voxel->point的映射地图pointmap，
  // 和由free-->occ的各block->voxel映射
  Timer setup_timer("motion_detection/indexing_setup");
  BlockToPointMap point_map;
  std::vector<voxblox::VoxelKey> occupied_ever_free_voxel_indices;
  setUpPointMap(cloud, point_map, occupied_ever_free_voxel_indices, cloud_info);
  setup_timer.Stop();


  // Clustering.
  // Hatori: 以voxel为单位对动态voxel进行聚类
  Timer clustering_timer("motion_detection/clustering");
  Clusters clusters = clustering_->performClustering(
      point_map, occupied_ever_free_voxel_indices, frame_counter_, cloud,
      cloud_info);
  clustering_timer.Stop();


  // Tracking.
  Timer tracking_timer("motion_detection/tracking");
  tracking_->track(cloud, clusters, cloud_info);
  tracking_timer.Stop();


  // Integrate ever-free information.
  Timer update_ever_free_timer("motion_detection/update_ever_free");
  // Hatori：主要做1.去除稳定占据voxel的free状态 2.更新新变为free的voxel状态
  ever_free_integrator_->updateEverFreeVoxels(frame_counter_);
  update_ever_free_timer.Stop();

  // Integrate the pointcloud into the voxblox TSDF map.
  Timer tsdf_timer("motion_detection/tsdf_integration");
  voxblox::Transformation T_G_C;
  tf::transformTFToKindr(T_M_S, &T_G_C);
  tsdf_server_->processPointCloudMessageAndInsert(msg, T_G_C, false);
  tsdf_timer.Stop();
  detection_timer.Stop();


  // 似乎是测试用函数和可视化部分，先不看了
  // Evaluation if requested.
  if (config_.evaluate) {
    Timer eval_timer("evaluation");
    evaluator_->evaluateFrame(cloud, cloud_info, clusters);
    eval_timer.Stop();
    if (config_.shutdown_after > 0 &&
        evaluator_->getNumberOfEvaluatedFrames() >= config_.shutdown_after) {
      LOG(INFO) << "Evaluated " << config_.shutdown_after
                << " frames, shutting down";
      ros::shutdown();
    }
  }

  // Visualization if requested.
  if (config_.visualize) {
    Timer vis_timer("visualizations");
    visualizer_->visualizeAll(cloud, cloud_info, clusters);
    vis_timer.Stop();
  }
}

bool MotionDetector::lookupTransform(const std::string& target_frame,
                                     const std::string& source_frame,
                                     uint64_t timestamp,
                                     tf::StampedTransform& result) const {
  ros::Time timestamp_ros;
  timestamp_ros.fromNSec(timestamp);

  // Note(schmluk): We could also wait for transforms here but this is easier
  // and faster atm.
  try {
    tf_listener_.lookupTransform(target_frame, source_frame, timestamp_ros,
                                 result);
  } catch (tf::TransformException& ex) {
    LOG(WARNING) << "Could not get sensor transform, skipping pointcloud: "
                 << ex.what();
    return false;
  }
  return true;
}


/////// Hatori
// 1.BlockToPointMap为自定义的voxblox::AnyIndexHashMapType<VoxelToPointMap>::type;
// AnyIndexHashMapType则为自定义的std::unordered_map<IndexType, ValueType>;
// key为Eigen3d(XYZ)，voxel,block等都是一个E3d,值就是模板参数ValueType

// BlockToPointMap实际为std::unordered_map<voxblox::BlockIndex, VoxelToPointMap>
// 将block index映射到VoxelToPointMap，VoxelToPointMap为std::unordered_map<voxblox::VoxelIndex, std::vector<voxblox::Point>>
// 将voxel index映射到std::vector<voxblox::Point>
// 也就是说，一个block对应多个voxel，一个voxel对应多个点，一个block对应多个点

// example:
// 比如索引一个Block，可以得到其中voxel的map，再对voxel索引，可以得到voxel中的vector
// 这个模板参数VoxelToPointMap为将voxel index 映射到其包含点的 std::vector<pointindice>，pointindice为点的索引，即size_t

// 2. voxblox::VoxelKey为std::pair<BlockIndex, VoxelIndex>;
// 将两种体素索引放到一个vector里面，作为occupied_ever_free_voxel_indices，这样可以同时找到block和voxel

// 3. HierarchicalIndexIntMap也是一个map，将索引映射到点(size_t)

// 4. cloudinfo：   bool has_labels = false; std::uint64_t timestamp; 
//    Point sensor_position; std::vector<PointInfo> points;
// 5. PointInfo
// Additional information stored for every point in the cloud.
// struct PointInfo {
//   // Include this point when computing performance metrics.
//   bool ready_for_evaluation = false;

//   // Set to true if the point falls into a voxel labeled ever-free.
//   bool ever_free_level_dynamic = false;

//   // Set to true if the point belongs to a cluster labeled dynamic.
//   bool cluster_level_dynamic = false;

//   // Set to true if the point belongs to a tracked object.
//   bool object_level_dynamic = false;

//   // Distance of the point to the sensor.
//   double distance_to_sensor = -1.0;

//   // Ground truth label if available.
//   bool ground_truth_dynamic = false;
// };

void MotionDetector::setUpPointMap(
    const Cloud& cloud, BlockToPointMap& point_map,
    std::vector<voxblox::VoxelKey>& occupied_ever_free_voxel_indices,
    CloudInfo& cloud_info) const {
  // Identifies for any LiDAR point the block it falls in and constructs the
  // hash-map block2points_map mapping each block to the LiDAR points that
  // fall into the block.

  // Hatori: 先将点云中的点映射到*tsdf*_block中 得到(map<tsdf_block_idx, vec<indices>>)
  const voxblox::HierarchicalIndexIntMap block2points_map =
      buildBlockToPointsMap(cloud);

  // Builds the voxel2point-map in parallel blockwise.
  std::vector<BlockIndex> block_indices(block2points_map.size());
  size_t i = 0;
  // 只存有点落入的block index到vector中
  for (const auto& block : block2points_map) {
    block_indices[i] = block.first;
    ++i;
  }

  // 对每个有点的block，向更细致的voxel构建map
  IndexGetter<BlockIndex> index_getter(block_indices);
  std::vector<std::future<void>> threads;
  std::mutex aggregate_results_mutex;
  /*
    这里可以多线程是因为每个线程处理的 block 是不同的，所以不会出现两个线程同时操作一个 block 的情况。
    具体来说，这里使用了一个 IndexGetter 对 block 进行了分配，每个线程从 IndexGetter 中获取一个 block 进行处理，
    直到所有 block 都被处理完毕。因此，每个线程处理的 block 是不同的，不会出现冲突。
  */
  for (int i = 0; i < config_.num_threads; ++i) {
    threads.emplace_back(std::async(std::launch::async, [&]() {
      // Data to store results.
      BlockIndex block_index;
      std::vector<voxblox::VoxelKey> local_occupied_indices;
      BlockToPointMap local_point_map;

      // Process until no more blocks.
      while (index_getter.getNextIndex(&block_index)) {
        VoxelToPointMap result;
        // Hatori: 得到result，即tsdf_voxel_index到点索引vector的map
        this->blockwiseBuildPointMap(cloud, block_index,
                                     block2points_map.at(block_index), result,
                                     local_occupied_indices, cloud_info);
        local_point_map.insert(std::pair(block_index, result));
        // Hatori: 所以local_point_map是 map<tsdf_block_index, map<tsdf_voxel_index, point_indices>>
      }

      // After processing is done add data to the output map.
      std::lock_guard<std::mutex> lock(aggregate_results_mutex);
      //Hatori: vec<pair<block_idx, voxel_indices> > ok
      occupied_ever_free_voxel_indices.insert(
          occupied_ever_free_voxel_indices.end(),
          local_occupied_indices.begin(), local_occupied_indices.end());
      point_map.merge(local_point_map); 
      // Hatori: 多线程结束后，得到各local_point_map合并的global map
      // global: --curr_occ_block_index--> a map<> --curr_occ_voxel_index--> point_indices 
    }));
  }

  for (auto& thread : threads) {
    thread.get();
  }
}

voxblox::HierarchicalIndexIntMap MotionDetector::buildBlockToPointsMap(
    const Cloud& cloud) const {
  voxblox::HierarchicalIndexIntMap result;

  int i = 0;
  for (const Point& point : cloud) {
    voxblox::Point coord(point.x, point.y, point.z);
    const BlockIndex blockindex =
        tsdf_layer_->computeBlockIndexFromCoordinates(coord);
    result[blockindex].push_back(i);
    i++;
  }
  return result;
}


// TODO tsdf到底是个啥东西
void MotionDetector::blockwiseBuildPointMap(
    const Cloud& cloud, const BlockIndex& block_index,
    const voxblox::AlignedVector<size_t>& points_in_block,
    VoxelToPointMap& voxel_map,
    std::vector<voxblox::VoxelKey>& occupied_ever_free_voxel_indices,
    CloudInfo& cloud_info) const {

  /* Hatori
    根据输入的tsdf block index获取相应的block指针，然后对该block中的每个点，找到其voxel_index(并确认voxel有效)。
    如果这个tsdf voxel是ever_free的，那么就将这个点标记为ever_free_level_dynamic
    并将tsdf voxel index作为key,将这个点的index作为value的一个元素存入voxel_map中
    所以voxel_map是 map<key:tsdf_voxel_index, value:point_indices>
  */

  // Get the block.
  TsdfBlock::Ptr tsdf_block = tsdf_layer_->getBlockPtrByIndex(block_index);
  if (!tsdf_block) {
    return;
  }

  // Create a mapping of each voxel index to the points it contains.
  for (size_t i : points_in_block) {
    const Point& point = cloud[i];
    const voxblox::Point coords(point.x, point.y, point.z);
    const VoxelIndex voxel_index =
        tsdf_block->computeVoxelIndexFromCoordinates(coords);
    if (!tsdf_block->isValidVoxelIndex(voxel_index)) {
      continue;
    }
    voxel_map[voxel_index].push_back(i);

    // EverFree detection flag at the same time, since we anyways lookup
    // voxels.
    if (tsdf_block->getVoxelByVoxelIndex(voxel_index).ever_free) {
      cloud_info.points.at(i).ever_free_level_dynamic = true;
    }
  }

  /*  Hatori
    对整理完的tsdf_block的 voxel_map，记录占据voxel的frame_counter_, 初始化聚类标志
    如果voxel曾是free的，则将其加入occupied_ever_free_voxel_indices，因为free-->occ，这个voxel可能是动态的
  */

  // Update the voxel status of the currently occupied voxels.
  for (const auto& voxel_points_pair : voxel_map) {
    TsdfVoxel& tsdf_voxel =
        tsdf_block->getVoxelByVoxelIndex(voxel_points_pair.first);
    tsdf_voxel.last_lidar_occupied = frame_counter_;  //Hatroi: 似乎是对应论文中记录最近一次占据，这里用frame number记录

    // This voxel attribute is used in the voxel clustering method: it
    // signalizes that a currently occupied voxel has not yet been clustered
    tsdf_voxel.clustering_processed = false;  //Hatori: 此时还没有聚类

    // The set of occupied_ever_free_voxel_indices allows for fast access of
    // the seed voxels in the voxel clustering
    if (tsdf_voxel.ever_free) { //Hatori: free-->occ的voxel可能是动态的, 记录下来大小index
      occupied_ever_free_voxel_indices.push_back(
          std::make_pair(block_index, voxel_points_pair.first));
    }
  }
}

}  // namespace dynablox
