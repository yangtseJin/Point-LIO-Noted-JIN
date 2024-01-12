#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>

/// *************Preconfiguration

#define MAX_INI_COUNT (100)

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void Process(const MeasureGroup &meas, PointCloudXYZI::Ptr pcl_un_);
  void Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot);

  ofstream fout_imu;
  // double first_lidar_time;
  int    lidar_type;
  bool   imu_en;
  V3D mean_acc,     //加速度均值,用于计算方差
        gravity_;
  bool   imu_need_init_ = true;     // 是否需要初始化imu
  bool   b_first_frame_ = true;
  bool   gravity_align_ = false;      // 是否重力对齐

 private:
  void IMU_init(const MeasureGroup &meas, int &N);
  V3D mean_gyr;     //角速度均值，用于计算方差
  int    init_iter_num = 1;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), gravity_align_(false)
{
  imu_en = true;
  init_iter_num = 1;
  mean_acc      = V3D(0, 0, 0.0);
  mean_gyr      = V3D(0, 0, 0);
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, 0.0);
  mean_gyr      = V3D(0, 0, 0);
  imu_need_init_    = true;
  init_iter_num     = 1;
}

// IMU初始化：利用开始的IMU帧的平均值初始化状态量x
void ImuProcess::IMU_init(const MeasureGroup &meas, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  /** 1. 初始化重力、陀螺偏差、acc和陀螺仪协方差
   ** 2. 将加速度测量值标准化为单位重力**/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)   // 判断是否为第一帧
  {
    Reset(); // 重置参数
    N = 1;  // 将迭代次数置1
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;    // 从common_lib.h中拿到imu初始时刻的加速度
    const auto &gyr_acc = meas.imu.front()->angular_velocity;       // 从common_lib.h中拿到imu初始时刻的角速度
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;     // 加速度测量作为初始化均值
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;     // 角速度测量作为初始化均值
  }

  // 计算均值
  for (const auto &imu : meas.imu)  // 拿到所有的imu帧
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    // 根据当前帧和均值差作为均值的更新
    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    N ++;
  }
}

void ImuProcess::Process(const MeasureGroup &meas, PointCloudXYZI::Ptr cur_pcl_un_)
{  
  if (imu_en)   // 如果IMU可用，默认true
  {
    if(meas.imu.empty())  return;   // 拿到的当前帧的imu测量为空，则直接返回
    ROS_ASSERT(meas.lidar != nullptr);

    if (imu_need_init_)
    {
      /// The very first lidar frame
      // 第一个激光雷达帧
      IMU_init(meas, init_iter_num);

      imu_need_init_ = true;

      // 需要100个IMU数据初始化，在FAST-LIO中是20
      if (init_iter_num > MAX_INI_COUNT)
      {
        ROS_INFO("IMU Initializing: %.1f %%", 100.0);
        imu_need_init_ = false;
        *cur_pcl_un_ = *(meas.lidar);   // 指向当前雷达数据
      }
      return;
    }
    // if (!gravity_align_) gravity_align_ = true;
    *cur_pcl_un_ = *(meas.lidar);   // 指向当前雷达数据
    return;
  }
  else  // 如果IMU不可用
  {
    // if (!b_first_frame_) 
    // {if (!gravity_align_) gravity_align_ = true;}
    // else
    // {b_first_frame_ = false;
    // }
    if (imu_need_init_)
    {
      imu_need_init_ = false;
    }
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
}

// 输入测得的加速度，输出对其到正确重力方向的旋转矩阵
void ImuProcess::Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  // V3D tmp_gravity = - mean_acc / mean_acc.norm() * G_m_s2; // state_gravity;
  M3D hat_grav;
  hat_grav << 0.0, gravity_(2), -gravity_(1),
              -gravity_(2), 0.0, gravity_(0),
              gravity_(1), -gravity_(0), 0.0;
  double align_norm = (hat_grav * tmp_gravity).norm() / tmp_gravity.norm() / gravity_.norm();   // 重力真值叉乘加速度均值的结果求模，再除以重力真值的模和加速度均值的模，结果为sin theta
  double align_cos = gravity_.transpose() * tmp_gravity;    // 点乘再除以模场，结果是cos theta
  align_cos = align_cos / gravity_.norm() / tmp_gravity.norm();
  if (align_norm < 1e-6)    // 如果偏移角度很小
  {
    if (align_cos > 1e-6)   // 如果cos theta 大于 0
    {
      rot = Eye3d;  // 直接赋值为单位阵
    }
    else
    {
      rot = -Eye3d; // 否则为负的单位阵
    }
  }
  else  // 如果偏移角不可忽略
  {
    V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos);   // 构造旋转的角轴，叉乘结果除以模场，即为单位旋转轴，再乘以旋转角，结果即为角轴
    rot = Exp(align_angle(0), align_angle(1), align_angle(2));  // 映射到SO3上，得到旋转矩阵
  }
}