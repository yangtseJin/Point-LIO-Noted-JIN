// #include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "Estimator.h"

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));     //特征点在地图中对应点的，局部平面参数,w系
std::vector<int> time_seq;
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());      //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());     //畸变纠正后降采样的单帧点云，w系
std::vector<V3D> pbody_list;
std::vector<PointVector> Nearest_Points;    //每个点的最近点序列
KD_TREE<PointType> ikdtree;
std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
bool   point_selected_surf[100000] = {0};   // 判断是否是有效特征点
std::vector<M3D> crossmat_list;
int effct_feat_num = 0;
int k;
int idx;
esekfom::esekf<state_input, 24, input_ikfom> kf_input;
esekfom::esekf<state_output, 30, input_ikfom> kf_output;
state_input state_in;
state_output state_out;
input_ikfom input_in;
V3D angvel_avr, acc_avr;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

Eigen::Matrix<double, 24, 24> process_noise_cov_input()
{
	Eigen::Matrix<double, 24, 24> cov;
    //创建一个24x24的矩阵cov，并将其所有元素初始化为0
	cov.setZero();
    //从(3,3)开始的一个3x3的块,表示角速度的协方差
	cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
    //表示加速度计的噪声协方差
	cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
    // bg的噪声协方差
	cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
    // ba的噪声协方差
	cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	// MTK::get_cov<process_noise_input>::type cov = MTK::get_cov<process_noise_input>::type::Zero();
	// MTK::setDiagonal<process_noise_input, vect3, 0>(cov, &process_noise_input::ng, gyr_cov_input);// 0.03
	// MTK::setDiagonal<process_noise_input, vect3, 3>(cov, &process_noise_input::na, acc_cov_input); // *dt 0.01 0.01 * dt * dt 0.05
	// MTK::setDiagonal<process_noise_input, vect3, 6>(cov, &process_noise_input::nbg, b_gyr_cov); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	// MTK::setDiagonal<process_noise_input, vect3, 9>(cov, &process_noise_input::nba, b_acc_cov);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

// 该代码定义了一个函数process_noise_cov_output()，用于生成一个30x30的矩阵cov，表示系统的噪声协方差矩阵。以及获得输入的f矩阵值
Eigen::Matrix<double, 30, 30> process_noise_cov_output()
{
	Eigen::Matrix<double, 30, 30> cov;
    //创建一个30x30的矩阵cov，并将其所有元素初始化为0
	cov.setZero();
    //从(12,12)开始的一个3x3的块,表示速度的协方差
	cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;
    //表示角速度的噪声协方差
	cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
    //表示加速度计的噪声协方差
	cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
    // bg的噪声协方差
	cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
    // ba的噪声协方差
	cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	// MTK::get_cov<process_noise_output>::type cov = MTK::get_cov<process_noise_output>::type::Zero();
	// MTK::setDiagonal<process_noise_output, vect3, 0>(cov, &process_noise_output::vel, vel_cov);// 0.03
	// MTK::setDiagonal<process_noise_output, vect3, 3>(cov, &process_noise_output::ng, gyr_cov_output); // *dt 0.01 0.01 * dt * dt 0.05
	// MTK::setDiagonal<process_noise_output, vect3, 6>(cov, &process_noise_output::na, acc_cov_output); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	// MTK::setDiagonal<process_noise_output, vect3, 9>(cov, &process_noise_output::nbg, b_gyr_cov);   //0.001 0.05 0.0001/out 0.01
	// MTK::setDiagonal<process_noise_output, vect3, 12>(cov, &process_noise_output::nba, b_acc_cov);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

// 这段代码的作用是将输入的IMU数据转化为状态转移方程中的输入向量f。
// 其中状态向量s包含了位置、速度、姿态、陀螺仪bias和加速度计bias等信息，in包含了IMU测量得到的角速度和线性加速度数据。
// fast_lio2论文公式(2), 起始这里的f就是将imu的积分方程组成矩阵形式然后再去计算，来自FAST-LIO代码中use-ikfom.hpp中的get_f()函数
// Point-LIO论文中的公式(3)
Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
    // 通过in.gyro.boxminus函数将in.gyro和s.bg进行运算，得到角速度向量omega
	in.gyro.boxminus(omega, s.bg);  // 得到imu的角速度
    // 计算惯性系下的加速度向量a_inertial,首先通过s.rot.normalized()将s.rot单位化，再通过乘法得到加速度向量，最后减去s.ba得到相对于惯性系的加速度。
    // 加速度转到世界坐标系
    vect3 a_inertial = s.rot * (in.acc-s.ba);
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];      //更新的速度
		res(i + 3) =  omega[i]; //更新的角速度
		res(i + 12) = a_inertial[i] + s.gravity[i];   //更新的加速度
	}
	return res;
}

//这段代码的作用是将输入的单点激光数据转化为状态转移方程中的输入向量f。
// 其中状态向量s包含了位置、速度、姿态、陀螺仪bias和加速度计bias等信息，in包含了IMU测量得到的角速度和线性加速度数据，没用到
Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
    // 计算惯性系下的加速度向量a_inertial,首先通过s.rot.normalized()将s.rot单位化，再通过乘法得到加速度向量。
	vect3 a_inertial = s.rot * s.acc; 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) = s.omg[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i]; 
	}
	return res;
}

// 注意该矩阵没乘dt，没加单位阵
// 这里实际上参考对应fast_lio论文公式(7)，因为df_dx_input中state_input的变量和Point-LIO 论文公式(11)的变量对不上
Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in)
{
    // 当中的24个对应了status的维度计算，为 pos(3), rot(3),offset_R_L_I(3),offset_T_L_I(3), vel(3), bg(3), ba(3), grav(3);
    // 状态量的顺序和论文中不同，要注意，论文中的顺序为 R, p, v ,bg, ba, g, omega, acc
    Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
    //一开始是一个R3的单位阵，代表速度转移
    // 对应 Point-LIO 论文公式(11)的Fx 第2行第3列，或者说fast_lio论文公式(7)第2行第3列
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);     //测量加速度 = a_m - bias
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);  //拿到角速度
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(acc_);    //fast_lio论文公式(7)第3行第1列
    // 将角度转到存入的矩阵中
	cov.template block<3, 3>(12, 18) = -s.rot;  //fast_lio论文公式(7)第3行第5列
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
    //对应fast_lio论文公式(7)第3行第6列
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix;
    //对应fast_lio论文公式(7)第1行第4列 (简化为-I)
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity();
	return cov;
}

// Eigen::Matrix<double, 24, 12> df_dw_input(state_input &s, const input_ikfom &in)
// {
// 	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
// 	cov.template block<3, 3>(12, 3) = -s.rot.normalized().toRotationMatrix();
// 	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
// 	return cov;
// }

// 对应 Point-LIO 论文公式(11)的Fx
// 注意该矩阵没乘dt，没加单位阵
Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in)
{
    // 当中的30个对应了status的维度计算，为 pos(3), rot(3), offset_R_L_I(3), offset_T_L_I(3), vel(3), ome(3), acc(3), grav(3), bg(3), ba(3);
    // 状态量的顺序和论文中不同，要注意，论文中的顺序为 R, p, v ,bg, ba, g, omega, acc
	Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
    // cov 矩阵中第0行和位移有关，对应论文中第2行
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();  // 对应 Point-LIO 论文公式(11)Fx 第2行第3列
    // cov 矩阵中第12行和速度有关，对应论文中第3行
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(s.acc);   // 对应 Point-LIO 论文公式(11)Fx的第3行1列，即 F31
	cov.template block<3, 3>(12, 18) = s.rot;   // 对应 Point-LIO 论文公式(11)Fx的第3行8列，即 F38
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; // 对应 Point-LIO 论文公式(11)Fx的第3行6列
    // cov 矩阵中第3行和旋转有关，对应论文中第1行
	cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity(); // 对应 Point-LIO 论文公式(11)Fx的第1行7列
	return cov;
}

// Eigen::Matrix<double, 30, 15> df_dw_output(state_output &s)
// {
// 	Eigen::Matrix<double, 30, 15> cov = Eigen::Matrix<double, 30, 15>::Zero();
// 	cov.template block<3, 3>(12, 0) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(15, 3) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(18, 6) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(24, 9) = Eigen::Matrix3d::Identity();
// 	cov.template block<3, 3>(27, 12) = Eigen::Matrix3d::Identity();
// 	return cov;
// }

vect3 SO3ToEuler(const SO3 &rot) 
{
	// Eigen::Matrix<double, 3, 1> _ang;
	// Eigen::Vector4d q_data = orient.coeffs().transpose();
	// //scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	// double sqw = q_data[3]*q_data[3];
	// double sqx = q_data[0]*q_data[0];
	// double sqy = q_data[1]*q_data[1];
	// double sqz = q_data[2]*q_data[2];
	// double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	// double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	// if (test > 0.49999*unit) { // singularity at north pole
	
	// 	_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
	// 	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	// 	vect3 euler_ang(temp, 3);
	// 	return euler_ang;
	// }
	// if (test < -0.49999*unit) { // singularity at south pole
	// 	_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
	// 	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	// 	vect3 euler_ang(temp, 3);
	// 	return euler_ang;
	// }
		
	// _ang <<
	// 		std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
	// 		std::asin (2*test/unit),
	// 		std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	// double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	// vect3 euler_ang(temp, 3);
	// return euler_ang;
	double sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if(!singular)
    {
        x = atan2(rot(2, 1), rot(2, 2));
        y = atan2(-rot(2, 0), sy);   
        z = atan2(rot(1, 0), rot(0, 0));  
    }
    else
    {    
        x = atan2(-rot(1, 2), rot(1, 1));    
        y = atan2(-rot(2, 0), sy);    
        z = 0;
    }
    Eigen::Matrix<double, 3, 1> ang(x, y, z);
    return ang;
}

// 计算残差信息
// 计算每个特征点的残差及H矩阵
void h_model_input(state_input &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;    // 平面点信息
	pabcd.setZero();
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;
    // 对降采样后的每个特征点进行残差计算
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];    // 获取降采样后的每个特征点
		PointType &point_world_j = feats_down_world->points[idx+j+1];   // 获取降采样后的每个特征点的世界坐标
        /* transform to world frame */
        //将点转换至世界坐标系下
		pointBodyToWorld(&point_body_j, &point_world_j);
		V3D p_body = pbody_list[idx+j+1];
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		
		{
            // Nearest_Points[i]打印出来发现是按照离 point_world 距离，从小到大的顺序的vector
			auto &points_near = Nearest_Points[idx+j+1];

            // 寻找 point_world_j 的最近邻的平面点
			ikdtree.Nearest_Search(point_world_j, NUM_MATCH_POINTS, points_near, pointSearchSqDis, 2.236); //1.0); //, 3.0); // 2.236;

            //判断是否是有效匹配点，与loam系列类似，要求特征点最近邻的地图点数量>阈值，距离<阈值  满足条件的才置为true
			if ((points_near.size() < NUM_MATCH_POINTS) || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) // 5)
			{
				point_selected_surf[idx+j+1] = false;
			}
			else
			{
				point_selected_surf[idx+j+1] = false;
                // 拟合平面方程ax+by+cz+d=0并求解点到平面距离
                // plane_thr 在配置文件avia.yaml中是0.1
				if (esti_plane(pabcd, points_near, plane_thr)) //(planeValid)   // 找平面点法向量寻找，common_lib.h中的函数
				{
                    // 计算点到平面的距离
					float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);

                    // 判断其是否满足匹配条件（即p_body.norm() > match_s * pd2 * pd2），
                    // 如果满足，则将point_selected_surf[idx + j + 1]标记为true，并将该特征点的法向量normvec->points[j]保存到normvec中。
					if (p_body.norm() > match_s * pd2 * pd2)    // match_s 在 avia.yaml 中是81
					{
						point_selected_surf[idx+j+1] = true;        // 再次回复为有效点
						normvec->points[j].x = pabcd(0);       //将法向量存储至normvec
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
                        // 为什么这里存的是平面方程中的常数d?
                        // 因为返回的结果将此时的平面方程变为：1/n * (a/d * x+ b/d * y + c/d * z + 1) = 0
                        // 其中 n 为 (a/d , b/d, c/d)^T 的模长
						normvec->points[j].intensity = pabcd(3);    // 将点到平面的距离存储至normvec的intensit中

						effect_num_k ++;    // 有效特征点数加1
					}
				}  
			}
		}
	}
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}
	ekfom_data.M_Noise = laser_point_cov;   // 来自配置文件中的 lidar_meas_cov，值为 0.001
    // 测量雅可比矩阵H和测量向量的计算 H=J*P*J^T
    // h_x是观测h相对于状态x的jacobian，见fatliov1的论文公式(14)
    // h_x 为观测相对于（姿态、位置、imu和雷达间的变换）的雅克比，尺寸为 特征点数x12
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);   // 测量雅可比矩阵H
	ekfom_data.z.resize(effect_num_k);  // 测量向量h
	int m = 0;
    // 求观测值与误差的雅克比矩阵，如Point-LIO论文式12、14
	for (int j = 0; j < time_seq[k]; j++)
	{
		if(point_selected_surf[idx+j+1])    // 对于满足要求的点
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
			
			if (extrinsic_est_en)   // 配置文件中不估计外参，false，直接执行else
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);    // 旋转矩阵的转置与法向量相乘得到C
				V3D A(p_imu_crossmat * C);      // 对imu的差距真与C相乘得到A
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C); // 对点的差距真与C相乘得到B
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{
                // crossmat_list中存的是imu系下的点生成的反对称矩阵
				M3D point_crossmat = crossmat_list[idx+j+1];
                // 计算测量雅可比矩阵H，见fatlio v1的论文公式(14)，求导这部分和LINS相同：https://zhuanlan.zhihu.com/p/258972164
                //FAST-LIO2的特别之处
                // 1.在IESKF中，状态更新可以看成是一个优化问题，即对位姿状态先验 x_bk_bk+1 的偏差，以及基于观测模型引入的残差函数f  的优化问题。
                // 2.LINS的特别之处在于，将LOAM的后端优化放在了IESKF的更新过程中实现，也就是用IESKF的迭代更新过程代替了LOAM的高斯牛顿法。
                // 这里见Point-LIO中对应公式(6)
                // 旋转矩阵的转置与法向量相乘得到C
				V3D C(s.rot.transpose() * norm_vec);    // R^-1 * 法向量,  s.rot.transpose（）是旋转矩阵转置，即旋转求逆
                // imu坐标系下的点坐标的反对称 点乘 C
                V3D A(point_crossmat * C);
                // 从第m行输入，即第m个观测值
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
            // 测量:到最近表面的距离
            // 残差存到观测z里
            // 由于这里的平面方程的系数是归一化后的，所以点到平面的距离还可以表示为： n^T*p+d，其中 n=(a,b,c)，平面方程为 ax+by+cz+d=0
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			m++;
		}
	}
	effct_feat_num += effect_num_k; // 增加有效特征点数的计数
}

void h_model_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;
    // 定义一个4维向量pabcd，将其所有元素初始化为0
	pabcd.setZero();

    // 用于存储当前时刻下所有特征点的法向量
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;
    //遍历当前时刻下所有的特征点，依次计算每个特征点的法向量和对应的观测值
	for (int j = 0; j < time_seq[k]; j++)
	{
        //对于每个特征点，首先将其从雷达系转换到世界系，并计算其在世界系下的位置p_world和在机体系下的位置p_body。
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		V3D p_body = pbody_list[idx+j+1];
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		{
			auto &points_near = Nearest_Points[idx+j+1];

            // 通过 ikdtree.Nearest_Search 函数在地图点云中搜索与当前特征点最近的NUM_MATCH_POINTS个点，并计算它们之间的平面方程pabcd。
			ikdtree.Nearest_Search(point_world_j, NUM_MATCH_POINTS, points_near, pointSearchSqDis, 2.236);

            //如果搜索到的点数量小于NUM_MATCH_POINTS或者最远点距离大于5，则认为该特征点不在地图中，将 point_selected_surf[idx + j + 1]标记为false
			if ((points_near.size() < NUM_MATCH_POINTS) || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5)
			{
				point_selected_surf[idx+j+1] = false;
			}
			else
			{
                // 如果搜索到的点数量大于等于NUM_MATCH_POINTS且最远点距离小于等于5，则认为该特征点在地图中，计算其与地图平面的距离pd2
				point_selected_surf[idx+j+1] = false;
				if (esti_plane(pabcd, points_near, plane_thr)) //(planeValid)
				{
					float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);

                    // 判断其是否满足匹配条件（即p_body.norm() > match_s * pd2 * pd2），
                    // 如果满足，则将point_selected_surf[idx + j + 1]标记为true，并将该特征点的法向量normvec->points[j]保存到normvec中。
					if (p_body.norm() > match_s * pd2 * pd2)
					{
						// point_selected_surf[i] = true;
						point_selected_surf[idx+j+1] = true;
						normvec->points[j].x = pabcd(0);
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
						normvec->points[j].intensity = pabcd(3);
						effect_num_k ++;
					}
				}  
			}
		}
	}
    // 如果当前时刻下没有满足匹配条件的特征点，则将ekfom_data.valid设为false，并直接返回。
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}
    // 否则，对于每个满足匹配条件的特征点，计算其对应的观测值z和雅可比矩阵H
	ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);

            // 如果开启了外参估计（extrinsic_est_en为true），则需要先计算特征点在IMU系下的位置，然后计算C、A、B三个矩阵，并将它们放入雅可比矩阵H中。
			if (extrinsic_est_en)
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{
                // 如果没有开启外参估计，则只需要计算C、A两个矩阵，并将它们放入雅可比矩阵H中。
				M3D point_crossmat = crossmat_list[idx+j+1];
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(point_crossmat * C);
				// V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
            // 将所有满足匹配条件的特征点的观测值z和雅可比矩阵H保存到ekfom_data中。
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			m++;
		}
	}
    // 最后将effect_num_k加到effct_feat_num中，表示当前时刻下满足匹配条件的特征点的数量
	effct_feat_num += effect_num_k;
}

// 计算IMU的残差
void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
    std::memset(ekfom_data.satu_check, false, 6);
	ekfom_data.z_IMU.block<3,1>(0, 0) = angvel_avr - s.omg - s.bg;
	ekfom_data.z_IMU.block<3,1>(3, 0) = acc_avr * G_m_s2 / acc_norm - s.acc - s.ba;
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
	if(check_satu)
	{
		if(fabs(angvel_avr(0)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[0] = true; 
			ekfom_data.z_IMU(0) = 0.0;
		}
		
		if(fabs(angvel_avr(1)) >= 0.99 * satu_gyro) 
		{
			ekfom_data.satu_check[1] = true;
			ekfom_data.z_IMU(1) = 0.0;
		}
		
		if(fabs(angvel_avr(2)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[2] = true;
			ekfom_data.z_IMU(2) = 0.0;
		}
		
		if(fabs(acc_avr(0)) >= 0.99 * satu_acc)
		{
			ekfom_data.satu_check[3] = true;
			ekfom_data.z_IMU(3) = 0.0;
		}

		if(fabs(acc_avr(1)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[4] = true;
			ekfom_data.z_IMU(4) = 0.0;
		}

		if(fabs(acc_avr(2)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[5] = true;
			ekfom_data.z_IMU(5) = 0.0;
		}
	}
}

// 把点从body系转到world系
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{    
    V3D p_body(pi->x, pi->y, pi->z);
    
    V3D p_global;
	if (extrinsic_est_en)
	{	
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (kf_output.x_.offset_R_L_I * p_body + kf_output.x_.offset_T_L_I) + kf_output.x_.pos;
		}
		else
		{
			p_global = kf_input.x_.rot * (kf_input.x_.offset_R_L_I * p_body + kf_input.x_.offset_T_L_I) + kf_input.x_.pos;
		}
	}
	else
	{
		if (!use_imu_as_input)  // use_imu_as_input 在 launch 文件中默认为 false
		{
			p_global = kf_output.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_output.x_.pos;
		}
		else
		{
			p_global = kf_input.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_input.x_.pos;
		}
	}

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};