#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "parameters.h"
#include "Estimator.h"
// #include <ros/console.h>


#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;

int feats_down_size = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;

int frame_ct = 0;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0, t_last = 0.0;

shared_ptr<ImuProcess> p_imu(new ImuProcess()); // 定义指向IMU数据预处理类ImuProcess的智能指针
bool init_map = false, flg_first_scan = true;
PointCloudXYZI::Ptr  ptr_con(new PointCloudXYZI());

// Time Log Variables
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;

bool   lidar_pushed = false, flg_reset = false, flg_exit = false;

vector<BoxPointType> cub_needrm;    // ikd-tree中，地图需要移除的包围盒序列

deque<PointCloudXYZI::Ptr>  lidar_buffer;       //记录特征提取或间隔采样后的lidar（特征）数据
deque<double>               time_buffer;        // 激光雷达数据时间戳缓存队列
deque<sensor_msgs::Imu::ConstPtr> imu_deque;    // IMU数据缓存队列

//surf feature in map
//一些点云变量
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());          //去畸变的特征
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());    //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());         //畸变纠正后降采样的单帧点云，w系

//下采样的体素点云
pcl::VoxelGrid<PointType> downSizeFilterSurf;      // 单帧内降采样使用voxel grid
pcl::VoxelGrid<PointType> downSizeFilterMap;       // 未使用

V3D euler_cur;   // 当前的欧拉角

MeasureGroup Measures;

sensor_msgs::Imu imu_last, imu_next;
sensor_msgs::Imu::ConstPtr imu_last_ptr;
//输出的路径参数
nav_msgs::Path path;    //包含了一系列位姿
nav_msgs::Odometry odomAftMapped;   //只包含了一个位姿
geometry_msgs::PoseStamped msg_body_pose;   //位姿

//按下ctrl+c后唤醒所有线程
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();    //  会唤醒所有等待队列中阻塞的线程 线程被唤醒后，会通过轮询方式获得锁，获得锁前也一直处理运行状态，不会被再次阻塞。
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang;
    if (!use_imu_as_input)
    {
        rot_ang = SO3ToEuler(kf_output.x_.rot);
    }
    else
    {
        rot_ang = SO3ToEuler(kf_input.x_.rot);
    }
    
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    if (use_imu_as_input)
    {
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1), kf_input.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1), kf_input.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1), kf_input.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1), kf_input.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1), kf_input.x_.gravity(2)); // Bias_a  
    }
    else
    {
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1), kf_output.x_.gravity(2)); // Bias_a  
    }
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            p_body_imu = kf_output.x_.offset_R_L_I * p_body_lidar + kf_output.x_.offset_T_L_I;
        }
        else
        {
            p_body_imu = kf_input.x_.offset_R_L_I * p_body_lidar + kf_input.x_.offset_T_L_I;
        }
    }
    else
    {
        p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
    }
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

int points_cache_size = 0;

//得到被剔除的点
void points_cache_collect() // seems for debug
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);     // 返回被剔除的点
    points_cache_size = points_history.size();
}

// 动态调整地图区域，防止地图过大而内存溢出，类似LOAM中提取局部地图的方法
// ikd-tree地图立方体的2个角点
BoxPointType LocalMap_Points;       // ikd-tree中,局部地图的包围盒角点
bool Localmap_Initialized = false;  // 局部地图是否初始化
void lasermap_fov_segment()
{
    cub_needrm.shrink_to_fit(); // 释放需要移除的区域的内存，不会改变 vector 中的元素数量，只会影响其内部的容量

    V3D pos_LiD;
    if (use_imu_as_input)   // use_imu_as_input 默认为true
    {
        pos_LiD = kf_input.x_.pos + kf_input.x_.rot * Lidar_T_wrt_IMU;  // 雷达在W系下位置，式子的意义是 W^p_L = W^p_I + W^R_I * I^t_L
    }
    else
    {
        pos_LiD = kf_output.x_.pos + kf_output.x_.rot * Lidar_T_wrt_IMU;
    }
    // 初始化局部地图范围，以pos_LiD为中心,长宽高均为cube_len
    // 初始化局部地图包围盒角点，以为w系下lidar位置为中心,得到长宽高200*200*200的局部地图
    if (!Localmap_Initialized){
        // 系统起始需要初始化局部地图的大小和位置
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // 各个方向上pos_LiD与局部地图边界的距离
    // 各个方向上Lidar与局部地图边界的距离，或者说是lidar与立方体盒子六个面的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    // 当前雷达系中心到各个地图边缘的距离
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（1.5*300m）太小，标记需要移除need_move(FAST-LIO2论文Fig.3)
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;     //如果不需要挪动，直接返回，不更改局部地图
    // 否则需要计算移动的距离
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    // 新的局部地图盒子边界点
    New_LocalMap_Points = LocalMap_Points;
    // 需要移动的距离
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        // 与包围盒最小值边界点距离
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);     // 移除较远包围盒
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    // 使用Boxs删除指定盒内的点
    if(cub_needrm.size() > 0) int kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        // lidar_buffer.shrink_to_fit();

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    last_timestamp_lidar = msg->header.stamp.toSec();

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr  ptr_div(new PointCloudXYZI());
    double time_div = msg->header.stamp.toSec();
    p_pre->process(msg, ptr);
    if (cut_frame)
    {
        sort(ptr->points.begin(), ptr->points.end(), time_list);

        for (int i = 0; i < ptr->size(); i++)
        {
            ptr_div->push_back(ptr->points[i]);
            // cout << "check time:" << ptr->points[i].curvature << endl;
            if (ptr->points[i].curvature / double(1000) + msg->header.stamp.toSec() - time_div > cut_frame_time_interval)
            {
                if(ptr_div->size() < 1) continue;
                PointCloudXYZI::Ptr  ptr_div_i(new PointCloudXYZI());
                *ptr_div_i = *ptr_div;
                lidar_buffer.push_back(ptr_div_i);
                time_buffer.push_back(time_div);
                time_div += ptr->points[i].curvature / double(1000);
                ptr_div->clear();
            }
        }
        if (!ptr_div->empty())
        {
            lidar_buffer.push_back(ptr_div);
            // ptr_div->clear();
            time_buffer.push_back(time_div);
        }
    }
    else if (con_frame)
    {
        if (frame_ct == 0)
        {
            time_con = last_timestamp_lidar; //msg->header.stamp.toSec();
        }
        if (frame_ct < con_frame_num)
        {
            for (int i = 0; i < ptr->size(); i++)
            {
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con->push_back(ptr->points[i]);
            }
            frame_ct ++;
        }
        else
        {
            PointCloudXYZI::Ptr  ptr_con_i(new PointCloudXYZI());
            *ptr_con_i = *ptr_con;
            lidar_buffer.push_back(ptr_con_i);
            double time_con_i = time_con;
            time_buffer.push_back(time_con_i);
            ptr_con->clear();
            frame_ct = 0;
        }
    }
    else
    { 
        lidar_buffer.emplace_back(ptr);
        time_buffer.emplace_back(msg->header.stamp.toSec());
    }
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    last_timestamp_lidar = msg->header.stamp.toSec();    
    
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr  ptr_div(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    double time_div = msg->header.stamp.toSec();
    if (cut_frame)
    {
        sort(ptr->points.begin(), ptr->points.end(), time_list);

        for (int i = 0; i < ptr->size(); i++)
        {
            ptr_div->push_back(ptr->points[i]);
            // use curvature as time of each laser points
            // 这里curvature存储的不是曲率，而是相对第一个点的时间戳，再乘1000，单位就成了毫秒ms，这里再除以1000，单位就变成了纳秒ns，stamp的单位也是纳秒ns
            // 1s(秒)=10^3ms(毫秒)=10^6μs(微秒)=10^9ns(纳秒)=10^12ps(皮秒)=10^15fs(飞秒)=10^18as(阿秒)=10^21zm(仄秒)=10^24ym(幺秒)
            if (ptr->points[i].curvature / double(1000) + msg->header.stamp.toSec() - time_div > cut_frame_time_interval)
            {
                if(ptr_div->size() < 1) continue;
                PointCloudXYZI::Ptr  ptr_div_i(new PointCloudXYZI());
                // cout << "ptr div num:" << ptr_div->size() << endl;
                *ptr_div_i = *ptr_div;
                // cout << "ptr div i num:" << ptr_div_i->size() << endl;
                lidar_buffer.push_back(ptr_div_i);
                time_buffer.push_back(time_div);
                time_div += ptr->points[i].curvature / double(1000);
                ptr_div->clear();
            }
        }
        if (!ptr_div->empty())
        {
            lidar_buffer.push_back(ptr_div);
            // ptr_div->clear();
            time_buffer.push_back(time_div);
        }
    }
    else if (con_frame)
    {
        if (frame_ct == 0)
        {
            time_con = last_timestamp_lidar; //msg->header.stamp.toSec();
        }
        if (frame_ct < con_frame_num)
        {
            for (int i = 0; i < ptr->size(); i++)
            {
                // use curvature as time of each laser points
                // 这里curvature存储的不是曲率，而是相对第一个点的时间戳，再乘1000，单位就成了毫秒ms，这里再除以1000，单位就变成了纳秒ns，stamp的单位也是纳秒ns
                // 1s(秒)=10^3ms(毫秒)=10^6μs(微秒)=10^9ns(纳秒)=10^12ps(皮秒)=10^15fs(飞秒)=10^18as(阿秒)=10^21zm(仄秒)=10^24ym(幺秒)
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con->push_back(ptr->points[i]);
            }
            frame_ct ++;
        }
        else
        {
            PointCloudXYZI::Ptr  ptr_con_i(new PointCloudXYZI());
            *ptr_con_i = *ptr_con;
            double time_con_i = time_con;
            lidar_buffer.push_back(ptr_con_i);
            time_buffer.push_back(time_con_i);
            ptr_con->clear();
            frame_ct = 0;
        }
    }
    else
    {
        lidar_buffer.emplace_back(ptr);
        time_buffer.emplace_back(msg->header.stamp.toSec());
    }
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_lag_imu_to_lidar);
    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    // 如果当前IMU的时间戳小于上一个时刻IMU的时间戳，则IMU数据有误
    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear deque");
        // imu_deque.shrink_to_fit();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }
    
    imu_deque.emplace_back(msg);
    last_timestamp_imu = timestamp;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// 把当前要处理的LIDAR和IMU数据打包到meas
bool sync_packages(MeasureGroup &meas)
{
    if (!imu_en)    // 如果IMU不可用，imu_en默认设置为true，这段不执行
    {
        if (!lidar_buffer.empty())
        {
            meas.lidar = lidar_buffer.front();  // 从激光雷达点云缓存队列中取出点云数据，放到meas中
            meas.lidar_beg_time = time_buffer.front();   // 该lidar测量起始的时间戳
            time_buffer.pop_front();
            lidar_buffer.pop_front();
            if(meas.lidar->points.size() < 1)   // 如果该lidar没有点云，则返回false
            {
                cout << "lose lidar" << std::endl;
                return false;
            }
            double end_time = meas.lidar->points.back().curvature;  // 此次雷达点云结束时刻，单位为ns
            for (auto pt: meas.lidar->points)
            {
                if (pt.curvature > end_time)
                {
                    end_time = pt.curvature;     // 点云的结束时间可能不是最后一个点的时间，找到真正的结束时间
                }
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);     // 注意点的时间戳是相对于一次扫描中的第一个点的时间
            meas.lidar_last_time = lidar_end_time;
            return true;
        }
        return false;
    }

    if (lidar_buffer.empty() || imu_deque.empty())  // 如果缓存队列中没有数据，则返回false
    {
        return false;
    }

    /*** push a lidar scan ***/
    // 如果还没有把雷达数据放到meas中的话，就执行一下操作
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();  // 从激光雷达点云缓存队列中取出点云数据，放到meas中
        if(meas.lidar->points.size() < 1)   // 如果该lidar没有点云，则返回false
        {
            cout << "lose lidar" << endl;
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }
        meas.lidar_beg_time = time_buffer.front();      // 该lidar测量起始的时间戳
        double end_time = meas.lidar->points.back().curvature;  // 此次雷达点云中最后一个点的时刻，单位为ns
        for (auto pt: meas.lidar->points)
        {
            if (pt.curvature > end_time)
            {
                end_time = pt.curvature;         // 点云的结束时间可能不是最后一个点的时间，找到真正的结束时间
            }
        }
        lidar_end_time = meas.lidar_beg_time + end_time / double(1000);     // 注意点的时间戳是相对于一次扫描中的第一个点的时间
        
        meas.lidar_last_time = lidar_end_time;
        lidar_pushed = true;        // 成功提取到lidar测量的标志
    }

    // 最新的IMU时间戳(也就是队尾的)不能早于雷达的end时间戳
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }
    /*** push imu data, and pop from imu buffer ***/
    if (p_imu->imu_need_init_)
    {
        // 压入imu数据，并从imu缓冲区弹出
        double imu_time = imu_deque.front()->header.stamp.toSec();
        meas.imu.shrink_to_fit();
        // 拿出lidar_beg_time到lidar_end_time之间的所有IMU数据
        // 如果imu缓存队列中的数据时间戳小于雷达结束时间戳，则将该数据放到meas中,代表了这一帧中的imu数据
        while ((!imu_deque.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = imu_deque.front()->header.stamp.toSec(); 
            if(imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());   //将imu数据放到meas中
            imu_last = imu_next;    // 记录上一个IMU数据
            imu_last_ptr = imu_deque.front();   // imu_last_ptr 是一个指针，它直接指向队列中的IMU数据对象，和队列中的IMU数据是同一个对象，它们指向同一块内存区域
            imu_next = *(imu_deque.front());    // 解引用操作符`*`的作用是获取指针指向的值，创建了队列中IMU数据对象的一个副本，并将这个副本赋值给`imu_next
                                                // imu_next 是队列中IMU数据的一个副本，而不是指向原始数据的指针。imu_next 和队列中的数据是两个不同的对象，它们在内存中占据不同的位置
            imu_deque.pop_front();
        }
    }
    else if(!init_map)
    {
        double imu_time = imu_deque.front()->header.stamp.toSec();
        meas.imu.shrink_to_fit();
        meas.imu.emplace_back(imu_last_ptr);

        while ((!imu_deque.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = imu_deque.front()->header.stamp.toSec(); 
            if(imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    }

    lidar_buffer.pop_front();   // 将lidar数据弹出
    time_buffer.pop_front();    // 将时间戳弹出
    lidar_pushed = false;       // 将lidar_pushed置为false，代表lidar数据已经被放到meas中了
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    
        for(int i = 0; i < feats_down_size; i++)
        {            
            if (!Nearest_Points[i].empty())
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
                /* If the nearest points is definitely outside the downsample box */
                if (fabs(points_near[0].x - mid_point.x) > 0.866 * filter_size_map_min || fabs(points_near[0].y - mid_point.y) > 0.866 * filter_size_map_min || fabs(points_near[0].z - mid_point.z) > 0.866 * filter_size_map_min){
                    PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
                continue;
            }
                /* Check if there is a point already in the downsample box */
                float dist  = calc_dist<float>(feats_down_world->points[i],mid_point);
                for (int readd_i = 0; readd_i < points_near.size(); readd_i ++)
            {
                    /* Those points which are outside the downsample box should not be considered. */
                    if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * filter_size_map_min && fabs(points_near[readd_i].y - mid_point.y) < 0.5 * filter_size_map_min && fabs(points_near[readd_i].z - mid_point.z) < 0.5 * filter_size_map_min) {
                    need_add = false;
                    break;
                }
            }
                if (need_add) PointToAdd.emplace_back(feats_down_world->points[i]);
        }
        else
        {
                // PointToAdd.emplace_back(feats_down_world->points[i]);
                PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
        }
        }
    int add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
}

void publish_init_kdtree(const ros::Publisher & pubLaserCloudFullRes)
{
    int size_init_ikdtree = ikdtree.size();
    PointCloudXYZI::Ptr   laserCloudInit(new PointCloudXYZI(size_init_ikdtree, 1));

    sensor_msgs::PointCloud2 laserCloudmsg;
    PointVector ().swap(ikdtree.PCL_Storage);
    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                
    laserCloudInit->points = ikdtree.PCL_Storage;
    pcl::toROSMsg(*laserCloudInit, laserCloudmsg);
        
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);

}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFullRes)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));
        
        for (int i = 0; i < size; i++)
        {
            // if (i % 3 == 0)
            // {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity; // feats_down_world->points[i].y; // 
            // }
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
    
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
        }
        
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        pointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

template<typename T>
void set_posestamp(T & out)
{
    if (!use_imu_as_input)
    {
        out.position.x = kf_output.x_.pos(0);
        out.position.y = kf_output.x_.pos(1);
        out.position.z = kf_output.x_.pos(2);
        Eigen::Quaterniond q(kf_output.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
    else
    {
        out.position.x = kf_input.x_.pos(0);
        out.position.y = kf_input.x_.pos(1);
        out.position.z = kf_input.x_.pos(2);
        Eigen::Quaterniond q(kf_input.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    if (publish_odometry_without_downsample)
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(time_current);
    }
    else
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    }
    set_posestamp(odomAftMapped.pose.pose);
    
    pubOdomAftMapped.publish(odomAftMapped);

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose.pose);
    // msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";
    static int jjj = 0;
    jjj++;
    // if (jjj % 2 == 0) // if path is too large, the rvis will crash
    {
        path.poses.emplace_back(msg_body_pose);
        pubPath.publish(path);
    }
}        

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");  // 初始化ros节点，节点名为laserMapping
    ros::NodeHandle nh("~");
    readParameters(nh);     // 从参数服务器读取参数值赋给变量（包括launch文件和launch读取的yaml文件中的参数）
    cout<<"lidar_type: "<<lidar_type<<endl;
    
    path.header.stamp    = ros::Time().fromSec(lidar_end_time);
    path.header.frame_id ="camera_init";

    /*** variables definition for counting ***/
    int frame_num = 0;
    double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;
    std::time_t startTime, endTime;
    
    /*** initialize variables ***/
    double FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    double HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    // 将数组point_selected_surf内元素的值全部设为true，数组point_selected_surf用于选择平面点
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    // VoxelGrid滤波器参数，即进行滤波时的创建的体素边长为 filter_size_surf_min
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    // VoxelGrid滤波器参数，即进行滤波时的创建的体素边长为 filter_size_map_min
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    // 从雷达到IMU的外参R和T
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
        else
        {
            kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
    }
    p_imu->lidar_type = p_pre->lidar_type = lidar_type;
    p_imu->imu_en = imu_en;

    kf_input.init_dyn_share_modified(get_f_input, df_dx_input, h_model_input);
    kf_output.init_dyn_share_modified_2h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
    Eigen::Matrix<double, 24, 24> P_init = MD(24,24)::Identity() * 0.01;
    P_init.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
    P_init.block<6, 6>(15, 15) = MD(6,6)::Identity() * 0.001;
    P_init.block<6, 6>(6, 6) = MD(6,6)::Identity() * 0.0001;
    kf_input.change_P(P_init);
    Eigen::Matrix<double, 30, 30> P_init_output = MD(30,30)::Identity() * 0.01;
    P_init_output.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
    P_init_output.block<6, 6>(6, 6) = MD(6,6)::Identity() * 0.0001;
    P_init_output.block<6, 6>(24, 24) = MD(6,6)::Identity() * 0.001;
    kf_input.change_P(P_init);
    kf_output.change_P(P_init_output);
    Eigen::Matrix<double, 24, 24> Q_input = process_noise_cov_input();
    Eigen::Matrix<double, 30, 30> Q_output = process_noise_cov_output();
    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_out, fout_imu_pbp;
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_imu_pbp.open(DEBUG_FILE_DIR("imu_pbp.txt"),ios::out);
    if (fout_out && fout_imu_pbp)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFullRes_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/aft_mapped_to_init", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
    ros::Publisher plane_pub = nh.advertise<visualization_msgs::Marker>
            ("/planner_normal", 1000);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    // 设置ROS程序主循环每次运行的时间至少为0.0002秒（5000Hz）
    ros::Rate rate(5000);
    bool status = ros::ok();
    // 程序主循环
    while (status)
    {
        // 如果有中断产生，则结束主循环
        if (flg_exit) break;
        // ROS消息回调处理函数，放在ROS的主循环中
        ros::spinOnce();
        if(sync_packages(Measures))    // 把一次的IMU和LIDAR数据打包到Measures
        {
            if (flg_first_scan)     // 激光雷达第一次扫描
            {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                cout << "first lidar time" << first_lidar_time << endl;
            }

            if (flg_reset)       // 判断状态，并把imu的数据清空，flg_reset 默认是 false
            {
                ROS_WARN("reset when rosbag play back");
                p_imu->Reset();
                flg_reset = false;
                continue;
            }
            double t0,t1,t2,t3,t4,t5,match_start, solve_start;
            match_time = 0;
            solve_time = 0;
            propag_time = 0;
            update_time = 0;
            t0 = omp_get_wtime();

            // 对IMU数据进行预处理，不同于FAST-LIO，这里没有UndistortPcl畸变处理，前向传播，反向传播
            p_imu->Process(Measures, feats_undistort);

            // if (feats_undistort->empty() || feats_undistort == NULL)
            if (p_imu->imu_need_init_)  // IMU如果没有完成初始化，就continue，直到初始化完成
                                        // IMU初始化完成后，imu_need_init_置为false，这里就不会执行
            {
                continue;
            }
            if(imu_en)
            {
                if (!p_imu->gravity_align_) // gravity_align_ 的初始值为false，这里会执行一次
                                            // gravity_align_ 后面被置为 true 后，下次循环就不再执行这里，直接跳过
                {
                    // imu_next 存的是最后一帧IMU数据
                    // 当雷达扫描的开始时间大于imu_next的时间戳时，即IMU的数据在雷达前时
                    while (Measures.lidar_beg_time > imu_next.header.stamp.toSec())
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());    // imu_next 存的是最后一帧IMU数据
                        imu_deque.pop_front();  // 将IMU数据拿出来丢掉
                        // imu_deque.pop();
                    }
                    if (non_station_start)  // non_station_start 在配置文件中设为 false
                    {
                        state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc *= -1;
                    }
                    else
                    {
                        // acc_norm 在 avia.yaml 中是 1.0 G，IMU加速度的单位是G
                        state_in.gravity =  -1 * p_imu->mean_acc * G_m_s2 / acc_norm;   // 根据IMU初始化时的均值计算重力
                        state_out.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm; 
                        state_out.acc = p_imu->mean_acc * G_m_s2 / acc_norm;
                    }
                    if (gravity_align)  // 如果配置文件中 gravity_align 的为true，重力对齐
                    {
                        Eigen::Matrix3d rot_init;
                        p_imu->gravity_ << VEC_FROM_ARRAY(gravity);     // 配置文件中的重力为 gravity: [0.0, 0.0, -9.810]
                        p_imu->Set_init(state_in.gravity, rot_init);    // 输入测得的加速度，输出对齐到正确重力方向的旋转矩阵，也即初始姿态
                        state_in.gravity = p_imu->gravity_;
                        state_out.gravity = p_imu->gravity_;
                        state_in.rot = rot_init;
                        state_out.rot = rot_init;
                        // state_in.rot.normalize();
                        // state_out.rot.normalize();
                        //TODO 为什么这里要带负号？
                        state_out.acc = -rot_init.transpose() * state_out.gravity;  // g = R * a，所以有 a = R^T * g
                    }
                    kf_input.change_x(state_in);
                    kf_output.change_x(state_out);
                    p_imu->gravity_align_ = true;
                }
            }
            else
            {
                if (!p_imu->gravity_align_)
                {
                    state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                    if (gravity_align)
                    {
                        Eigen::Matrix3d rot_init;
                        p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
                        p_imu->Set_init(state_in.gravity, rot_init);
                        state_out.gravity = p_imu->gravity_;
                        state_out.rot = rot_init;
                        // state_in.rot.normalize();
                        // state_out.rot.normalize();
                        state_out.acc = -rot_init.transpose() * state_out.gravity;
                    }
                    else
                    {
                        state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc *= -1;
                    }
                    // kf_input.change_x(state_in);
                    kf_output.change_x(state_out);
                    p_imu->gravity_align_ = true;
                }
            }
            // gravity_align_ 在重力对齐后被置为 true，后面的循环就直接执行后面的代码
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();     // 更新localmap边界，然后降采样当前帧点云
            /*** downsample the feature points in a scan ***/
            // 点云下采样
            t1 = omp_get_wtime();       // 记录时间
            if(space_down_sample)   // space_down_sample 默认为true
            {
                downSizeFilterSurf.setInputCloud(feats_undistort);      //获得去畸变后的点云数据
                downSizeFilterSurf.filter(*feats_down_body);        // 滤波降采样后的点云数据
                // 将一次扫描中的点按时间从小到大排序
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list); 
            }
            else
            {
                feats_down_body = Measures.lidar;
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list); 
            }
            time_seq = time_compressing<int>(feats_down_body);
            feats_down_size = feats_down_body->points.size();   // 记录滤波后的点云数量
            
            /*** initialize the map kdtree ***/
            // 构建kd树
            if(!init_map)
            {
                // 初始化ikdtree(ikdtree为空时)
                if(ikdtree.Root_Node == nullptr) //
                // if(feats_down_size > 5)
                {
                    // 设置ikd tree的降采样参数
                    ikdtree.set_downsample_param(filter_size_map_min);
                }
                    
                feats_down_world->resize(feats_down_size);  // 将下采样得到的地图点大小设为与body系大小一致
                for(int i = 0; i < feats_down_size; i++)
                {
                    // lidar坐标系转到世界坐标系
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }
                for (size_t i = 0; i < feats_down_world->size(); i++) {
                init_feats_world->points.emplace_back(feats_down_world->points[i]);}
                if(init_feats_world->size() < init_map_size) continue;
                // 根据世界坐标系下的点构建ikdtree
                ikdtree.Build(init_feats_world->points); 
                init_map = true;
                publish_init_kdtree(pubLaserCloudMap); //(pubLaserCloudFullRes);
                continue;
            }        
            /*** ICP and Kalman filter update ***/
            // ICP和迭代卡尔曼滤波更新
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            Nearest_Points.resize(feats_down_size);     // 存储近邻点的vector，将降采样处理后的点云用于搜索最近点

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            crossmat_list.reserve(feats_down_size);
            pbody_list.reserve(feats_down_size);
            // pbody_ext_list.reserve(feats_down_size);
                          
            for (size_t i = 0; i < feats_down_body->size(); i++)
            {
                V3D point_this(feats_down_body->points[i].x,
                            feats_down_body->points[i].y,
                            feats_down_body->points[i].z);
                pbody_list[i]=point_this;
                if (extrinsic_est_en)   // 是否估计外参，avia.yaml中是false
                                        // 对于激进的运动，将此变量设置为 false
                {
                    if (!use_imu_as_input)
                    {
                        point_this = kf_output.x_.offset_R_L_I * point_this + kf_output.x_.offset_T_L_I;
                    }
                    else
                    {
                        // offset_R_L_I表示雷达到IMU的旋转平移外参
                        // 将雷达系下的点转到imu系下
                        point_this = kf_input.x_.offset_R_L_I * point_this + kf_input.x_.offset_T_L_I;
                    }
                }
                else    // avia.yaml 直接执行此处
                {
                    point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;    // 将雷达系下的点转到imu系下
                }
                M3D point_crossmat;
                point_crossmat << SKEW_SYM_MATRX(point_this);   // 将点的坐标变为反对称矩阵
                crossmat_list[i]=point_crossmat;            // 反对称矩阵存入crossmat_list中
            }
            
            if (!use_imu_as_input)  // use_imu_as_input 默认为true
            {     
                // bool imu_upda_cov = false;
                effct_feat_num = 0;
                /**** point by point update ****/

                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++)
                {
                    PointType &point_body  = feats_down_body->points[idx+time_seq[k]];

                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;    // 计算雷达点的时刻，毫秒除1000变为纳秒，雷达开始时刻的时间戳单位为纳秒ns

                    if (is_first_frame) // 如果是第一帧雷达扫描
                    {
                        if(imu_en)  // imu_en默认设置为true
                        {
                            // imu_next 存的是最后一帧IMU数据
                            while (time_current > imu_next.header.stamp.toSec())
                            {
                                // 将IMU数据出队，直到雷达开始时刻在IMU时间戳之前
                                imu_last = imu_next;    // imu_last记录最后一帧IMU的上一个IMU数据
                                imu_next = *(imu_deque.front());
                                imu_deque.pop_front();
                                // imu_deque.pop();
                            }

                            angvel_avr<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            acc_avr   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        }
                        is_first_frame = false;
                        // imu_upda_cov = true;
                        time_update_last = time_current;
                        time_predict_last_const = time_current;
                    }
                    if(imu_en)  // imu_en默认设置为true
                    {
                        bool imu_comes = time_current > imu_next.header.stamp.toSec();
                        while (imu_comes)
                        {
                            // imu_upda_cov = true;
                            angvel_avr<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            acc_avr   <<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                            /*** covariance update ***/
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_deque.pop_front();
                            double dt = imu_last.header.stamp.toSec() - time_predict_last_const;
                                kf_output.predict(dt, Q_output, input_in, true, false);
                            time_predict_last_const = imu_last.header.stamp.toSec(); // big problem
                            imu_comes = time_current > imu_next.header.stamp.toSec();
                            // if (!imu_comes)
                            {
                                double dt_cov = imu_last.header.stamp.toSec() - time_update_last; 

                                if (dt_cov > 0.0)
                                {
                                    time_update_last = imu_last.header.stamp.toSec();
                                    double propag_imu_start = omp_get_wtime();

                                    kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                    propag_time += omp_get_wtime() - propag_imu_start;
                                    double solve_imu_start = omp_get_wtime();
                                    kf_output.update_iterated_dyn_share_IMU();
                                    solve_time += omp_get_wtime() - solve_imu_start;
                                }
                            }
                        }
                    }

                    double dt = time_current - time_predict_last_const;
                    double propag_state_start = omp_get_wtime();
                    if(!prop_at_freq_of_imu)
                    {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            kf_output.predict(dt_cov, Q_output, input_in, false, true);
                            time_update_last = time_current;   
                        }
                    }
                        kf_output.predict(dt, Q_output, input_in, true, false);
                    propag_time += omp_get_wtime() - propag_state_start;
                    time_predict_last_const = time_current;
                    // if(k == 0)
                    // {
                    //     fout_imu_pbp << Measures.lidar_last_time - first_lidar_time << " " << imu_last.angular_velocity.x << " " << imu_last.angular_velocity.y << " " << imu_last.angular_velocity.z \
                    //             << " " << imu_last.linear_acceleration.x << " " << imu_last.linear_acceleration.y << " " << imu_last.linear_acceleration.z << endl;
                    // }

                    double t_update_start = omp_get_wtime();

                    if (feats_down_size < 1)
                    {
                        ROS_WARN("No point, skip this scan!\n");
                        idx += time_seq[k];
                        continue;
                    }
                    if (!kf_output.update_iterated_dyn_share_modified()) 
                    {
                        idx = idx+time_seq[k];
                        continue;
                    }

                    // if(prop_at_freq_of_imu)
                    // {
                        // double dt_cov = time_current - time_update_last;
                        // if ((dt_cov >= imu_time_inte)) // (point_cov_not_prop && imu_prop_cov)
                        // {
                        //     double propag_cov_start = omp_get_wtime();
                        //     kf_output.predict(dt_cov, Q_output, input_in, false, true);
                        //     // imu_upda_cov = false;
                        //     time_update_last = time_current;
                        //     propag_time += omp_get_wtime() - propag_cov_start;
                        // }
                    // }

                    solve_start = omp_get_wtime();
                        
                    if (publish_odometry_without_downsample)
                    {
                        /******* Publish odometry *******/

                        publish_odometry(pubOdomAftMapped);
                        if (runtime_pos_log)
                        {
                            state_out = kf_output.x_;
                            euler_cur = SO3ToEuler(state_out.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose() << " " << state_out.vel.transpose() \
                            <<" "<<state_out.omg.transpose()<<" "<<state_out.acc.transpose()<<" "<<state_out.gravity.transpose()<<" "<<state_out.bg.transpose()<<" "<<state_out.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
                        }
                    }

                    for (int j = 0; j < time_seq[k]; j++)
                    {
                        PointType &point_body_j  = feats_down_body->points[idx+j+1];
                        PointType &point_world_j = feats_down_world->points[idx+j+1];
                        pointBodyToWorld(&point_body_j, &point_world_j);
                    }
                
                    solve_time += omp_get_wtime() - solve_start;
    
                    update_time += omp_get_wtime() - t_update_start;
                    idx += time_seq[k];
                    // cout << "pbp output effect feat num:" << effct_feat_num << endl;
                }
            }
            else    // 直接到这里执行
            {
                bool imu_prop_cov = false;
                effct_feat_num = 0;

                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++)
                {
                    PointType &point_body  = feats_down_body->points[idx+time_seq[k]];
                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;    // 计算雷达点的时刻，毫秒除1000变为纳秒，雷达开始时刻的时间戳单位为纳秒ns
                    if (is_first_frame)     // 如果是第一帧雷达扫描
                    {
                        while (time_current > imu_next.header.stamp.toSec()) 
                        {
                            // imu_next 存的是最后一帧IMU数据
                            // 将IMU数据出队，直到雷达开始时刻在IMU时间戳之前
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());    // imu_last记录最后一帧IMU的上一个IMU数据
                            imu_deque.pop_front();
                            // imu_deque.pop();
                        }
                        imu_prop_cov = true;
                        // imu_upda_cov = true;

                        is_first_frame = false;
                        t_last = time_current;
                        time_update_last = time_current; 
                        // if(prop_at_freq_of_imu)
                        {
                            input_in.gyro<<imu_last.angular_velocity.x,
                                        imu_last.angular_velocity.y,
                                        imu_last.angular_velocity.z;
                                            
                            input_in.acc<<imu_last.linear_acceleration.x,
                                        imu_last.linear_acceleration.y,
                                        imu_last.linear_acceleration.z;
                            // angvel_avr<<0.5 * (imu_last.angular_velocity.x + imu_next.angular_velocity.x),
                            //             0.5 * (imu_last.angular_velocity.y + imu_next.angular_velocity.y),
                            //             0.5 * (imu_last.angular_velocity.z + imu_next.angular_velocity.z);
                                            
                            // acc_avr   <<0.5 * (imu_last.linear_acceleration.x + imu_next.linear_acceleration.x),
                            //             0.5 * (imu_last.linear_acceleration.y + imu_next.linear_acceleration.y),
                                        // 0.5 * (imu_last.linear_acceleration.z + imu_next.linear_acceleration.z);

                            // angvel_avr -= state.bias_g;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;    // 将加速度换算成 m/s^2 的单位
                        }
                    }
                    
                    while (time_current > imu_next.header.stamp.toSec()) // && !imu_deque.empty())
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                        input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        input_in.acc <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z; 

                        // angvel_avr<<0.5 * (imu_last.angular_velocity.x + imu_next.angular_velocity.x),
                        //             0.5 * (imu_last.angular_velocity.y + imu_next.angular_velocity.y),
                        //             0.5 * (imu_last.angular_velocity.z + imu_next.angular_velocity.z);
                                        
                        // acc_avr   <<0.5 * (imu_last.linear_acceleration.x + imu_next.linear_acceleration.x),
                        //             0.5 * (imu_last.linear_acceleration.y + imu_next.linear_acceleration.y),
                        //             0.5 * (imu_last.linear_acceleration.z + imu_next.linear_acceleration.z);
                        input_in.acc    = input_in.acc * G_m_s2 / acc_norm; 
                        double dt = imu_last.header.stamp.toSec() - t_last;

                        // if(!prop_at_freq_of_imu)
                        // {       
                        double dt_cov = imu_last.header.stamp.toSec() - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                            time_update_last = imu_last.header.stamp.toSec(); //time_current;
                        }
                            kf_input.predict(dt, Q_input, input_in, true, false); 
                        t_last = imu_last.header.stamp.toSec();
                        imu_prop_cov = true;
                        // imu_upda_cov = true;
                    }      

                    double dt = time_current - t_last;
                    t_last = time_current;
                    double propag_start = omp_get_wtime();
                    
                    if(!prop_at_freq_of_imu)
                    {   
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {    
                            kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                            time_update_last = time_current; 
                        }
                    }
                        kf_input.predict(dt, Q_input, input_in, true, false); 

                    propag_time += omp_get_wtime() - propag_start;

                    // if(k == 0)
                    // {
                    //     fout_imu_pbp << Measures.lidar_last_time - first_lidar_time << " " << imu_last.angular_velocity.x << " " << imu_last.angular_velocity.y << " " << imu_last.angular_velocity.z \
                    //             << " " << imu_last.linear_acceleration.x << " " << imu_last.linear_acceleration.y << " " << imu_last.linear_acceleration.z << endl;
                    // }

                    double t_update_start = omp_get_wtime();
                    
                    if (feats_down_size < 1)
                    {
                        ROS_WARN("No point, skip this scan!\n");

                        idx += time_seq[k];
                        continue;
                    }
                    if (!kf_input.update_iterated_dyn_share_modified()) 
                    {
                        idx = idx+time_seq[k];
                        continue;
                    }

                    solve_start = omp_get_wtime();

                    // if(prop_at_freq_of_imu)
                    // {
                    //     double dt_cov = time_current - time_update_last;
                    //     if ((imu_prop_cov && dt_cov > 0.0) || (dt_cov >= imu_time_inte * 1.2)) 
                    //     {
                    //         double propag_cov_start = omp_get_wtime();
                    //         kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                    //         propag_time += omp_get_wtime() - propag_cov_start;
                    //         time_update_last = time_current;
                    //         imu_prop_cov = false;
                    //     }
                    // }
                    if (publish_odometry_without_downsample)     // 开关高频里程计，配置文件中设为 true 则打开高频里程计
                    {
                        /******* Publish odometry *******/

                        publish_odometry(pubOdomAftMapped);
                        if (runtime_pos_log)
                        {
                            state_in = kf_input.x_;
                            euler_cur = SO3ToEuler(state_in.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_in.pos.transpose() << " " << state_in.vel.transpose() \
                            <<" "<<state_in.bg.transpose()<<" "<<state_in.ba.transpose()<<" "<<state_in.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
                        }
                    }

                    for (int j = 0; j < time_seq[k]; j++)
                    {
                        PointType &point_body_j  = feats_down_body->points[idx+j+1];
                        PointType &point_world_j = feats_down_world->points[idx+j+1];
                        pointBodyToWorld(&point_body_j, &point_world_j); 
                    }
                    solve_time += omp_get_wtime() - solve_start;
                
                    update_time += omp_get_wtime() - t_update_start;
                    idx = idx + time_seq[k];
                }  
            }

            /******* Publish odometry downsample *******/
            if (!publish_odometry_without_downsample)   // 开关高频里程计，配置文件中设为 true 则打开高频里程计
            {
                publish_odometry(pubOdomAftMapped);
            }

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            
            if(feats_down_size > 4)
            {
                map_incremental();
            }

            t5 = omp_get_wtime();
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFullRes);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);
            
            /*** Debug variables Logging ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                {aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + update_time/frame_num;}
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + solve_time/frame_num;
                aver_time_propag = aver_time_propag * (frame_num - 1)/frame_num + propag_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = aver_time_consu;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu, aver_time_icp, aver_time_propag); 
                if (!publish_odometry_without_downsample)
                {
                    if (!use_imu_as_input)
                    {
                        state_out = kf_output.x_;
                        euler_cur = SO3ToEuler(state_out.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_out.pos.transpose() << " " << state_out.vel.transpose() \
                        <<" "<<state_out.omg.transpose()<<" "<<state_out.acc.transpose()<<" "<<state_out.gravity.transpose()<<" "<<state_out.bg.transpose()<<" "<<state_out.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                    else
                    {
                        state_in = kf_input.x_;
                        euler_cur = SO3ToEuler(state_in.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_in.pos.transpose() << " " << state_in.vel.transpose() \
                        <<" "<<state_in.bg.transpose()<<" "<<state_in.ba.transpose()<<" "<<state_in.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                }
                dump_lio_state_to_log(fp);
            }
        }
        status = ros::ok();
        rate.sleep();
    }
    //--------------------------save map-----------------------------------
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
    fout_out.close();
    fout_imu_pbp.close();

    return 0;
}
