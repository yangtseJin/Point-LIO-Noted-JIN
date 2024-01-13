/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP


#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/types/SEn.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"

namespace esekfom {

using namespace Eigen;

// 用于迭代ESKF更新
// 目的是通过一个函数同时计算测量值（z）、偏微分矩阵（h_x）、z_IMU、R_IMU、噪声协方差（R）。
// 作为维度变化的特征矩阵应用于测量
template<typename T>
struct dyn_share_modified
{
	bool valid;
	bool converge;
	T M_Noise;
	Eigen::Matrix<T, Eigen::Dynamic, 1> z;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
	Eigen::Matrix<T, 6, 1> z_IMU;
	Eigen::Matrix<T, 6, 1> R_IMU;
	bool satu_check[6];
};

//状态量，噪声维度，输入状态量这三个参数输入
template<typename state, int process_noise_dof, typename input = state, typename measurement=state, int measurement_noise_dof=0>
class esekf{

	typedef esekf self;
	enum{
		n = state::DOF,         //状态量自由度，一般代表x的维度
        m = state::DIM,         //状态量自由度，一般代表res的维度
        l = measurement::DOF    //测量噪声维度
	};

public:
	
	typedef typename state::scalar scalar_type;
	typedef Matrix<scalar_type, n, n> cov;
	typedef Matrix<scalar_type, m, n> cov_;
	typedef SparseMatrix<scalar_type> spMt;
	typedef Matrix<scalar_type, n, 1> vectorized_state;
	typedef Matrix<scalar_type, m, 1> flatted_state;
	typedef flatted_state processModel(state &, const input &);     // 声明一个函数指针
	typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
	typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
	typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;

	typedef void measurementModel_dyn_share_modified(state &, dyn_share_modified<scalar_type> &);
	typedef Eigen::Matrix<scalar_type ,l, n> measurementMatrix1(state &);
	typedef Eigen::Matrix<scalar_type , Eigen::Dynamic, n> measurementMatrix1_dyn(state &);
	typedef Eigen::Matrix<scalar_type ,l, measurement_noise_dof> measurementMatrix2(state &);
	typedef Eigen::Matrix<scalar_type ,Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &);
	typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
	typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

	esekf(const state &x = state(),
		const cov  &P = cov::Identity()): x_(x), P_(P){};   // esekf初始化，主要是初始化了x_和P_

    //接收系统特定的模型及其差异
    //作为特征矩阵的测量，其维数是变化的。
    //通过一个函数(init_dyn_share_modified)完成了测量(z)，估计测量(h)，偏微分矩阵(h_x)和噪声协方差(R)的同时计算。
	void init_dyn_share_modified(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified h_dyn_share_in)
	{
		f = f_in;
		f_x = f_x_in;
		// f_w = f_w_in;
		h_dyn_share_modified_1 = h_dyn_share_in;
		maximum_iter = 1;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
		x_.build_SEN_state();
	}
	
	void init_dyn_share_modified_2h(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified h_dyn_share_in1, measurementModel_dyn_share_modified h_dyn_share_in2)
	{
		f = f_in;
		f_x = f_x_in;
		// f_w = f_w_in;
		h_dyn_share_modified_1 = h_dyn_share_in1;
		h_dyn_share_modified_2 = h_dyn_share_in2;
		maximum_iter = 1;
		x_.build_S2_state();
		x_.build_SO3_state();
		x_.build_vect_state();
		x_.build_SEN_state();
	}

	// iterated error state EKF propogation
    // 迭代误差状态EKF传播
    // 前向传播
	void predict(double &dt, processnoisecovariance &Q, const input &i_in, bool predict_state, bool prop_cov){
		if (predict_state)  // 如果更新状态
		{
            // 来自FAST-LIO代码中use-ikfom.hpp中的 get_f函数，对应fast_lio2论文公式(2)
            // 这里对应 Point-LIO论文中的公式(3)，f() 函数对应 Estimator.cpp 中的 get_f_input 函数
			flatted_state f_ = f(x_, i_in);
			x_.oplus(f_, dt);
		}

		if (prop_cov)   // 如果更新协方差
		{
			flatted_state f_ = f(x_, i_in);
			// state x_before = x_;

            // fast_lio2论文公式(7)
            // 这里对应Point-LIO中的公式（11）
			cov_ f_x_ = f_x(x_, i_in);  // m*n维
			cov f_x_final;  // n*n 维
			F_x1 = cov::Identity(); // n*n 维，状态转移矩阵
            // 更新f_x
			for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++) {
				int idx = (*it).first.first;        //状态变量的索引
				int dim = (*it).first.second;       //状态变量的维数
				int dof = (*it).second;             //状态变量的自由度
				for(int i = 0; i < n; i++){
					for(int j=0; j<dof; j++)
					{f_x_final(idx+j, i) = f_x_(dim+j, i);}	    //更新f_x_final，形成n*n阵，用于更新
				}
			}

			Matrix<scalar_type, 3, 3> res_temp_SO3;
			MTK::vect<3, scalar_type> seg_SO3;
			for (std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
				int idx = (*it).first;      //状态变量的索引
				int dim = (*it).second;     //状态变量的维数
				for(int i = 0; i < 3; i++){
					seg_SO3(i) = -1 * f_(dim + i) * dt;     // 拿到S03更新值
				}
				// MTK::SO3<scalar_type> res;
				// res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
                //更新f_x_1
				F_x1.template block<3, 3>(idx, idx) = MTK::SO3<scalar_type>::exp(seg_SO3); // res.normalized().toRotationMatrix();		
				res_temp_SO3 = MTK::A_matrix(seg_SO3);  //转为矩阵形式
				for(int i = 0; i < n; i++){
                    // F矩阵
					f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
				}
			}
	
			F_x1 += f_x_final * dt;
			P_ = F_x1 * P_ * (F_x1).transpose() + Q * (dt * dt);
		}
	}

    //iterated error state EKF update modified for one specific system.
    // 针对一个特定系统修改了迭代错误状态EKF更新。
    // ESKF
	bool update_iterated_dyn_share_modified() {
		dyn_share_modified<scalar_type> dyn_share;  //定义一个动态共享变量
        //这里的x_是经过正向传播后的状态量和协方差矩阵，因为会先调用predict函数再调用这个函数
		state x_propagated = x_;    //用当前状态变量初始化一个新的状态变量
		int dof_Measurement;    //定义一个整数变量，用于存储测量维度
		double m_noise;     //用于存储测量噪声的方差
        // 最多进行maximum_iter次迭代优化
		for(int i=0; i<maximum_iter; i++)
		{
			dyn_share.valid = true;
            // h_dyn_share_modified_1 指向的是 h_model_input() 函数
            // 计算测量模型方程的雅克比，也就是点面残差的导数 H(代码里是h_x)
            // 这个函数只计算雷达点的残差
			h_dyn_share_modified_1(x_, dyn_share);  //不带IMU的优化
			if(! dyn_share.valid)
			{
				return false;
				// continue;
			}
            // 从动态共享变量中获取测量值
			Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
			// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R = dyn_share.R;
            // 从动态共享变量中获取测量H
			Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
			// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_v = dyn_share.h_v;
            // 获取测量函数的行数，即测量维度
			dof_Measurement = h_x.rows();
            // 获取测量噪声的方差
			m_noise = dyn_share.M_Noise;
			// dof_Measurement_noise = dyn_share.R.rows();
			// vectorized_state dx, dx_new;
			// x_.boxminus(dx, x_propagated);
			// dx_new = dx;
			// P_ = P_propagated;

			Matrix<scalar_type, n, Eigen::Dynamic> PHT;     // 用于存储协方差矩阵和测量函数的乘积
			Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> HPHT;   // 用于存储测量函数和协方差矩阵的乘积
			Matrix<scalar_type, n, Eigen::Dynamic> K_;      // 存储卡尔曼增益
			// if(n > dof_Measurement)
			{
                // 每一次迭代将重新计算增益K，即论文中公式（20）
                // 将协方差矩阵的前12行和测量函数的转置相乘，得到一个n\timesn测量维度的矩阵。
				PHT = P_. template block<n, 12>(0, 0) * h_x.transpose();
                // 将测量函数和协方差矩阵的前12列相乘，得到一个测量维度×\times×测量维度的矩阵
				HPHT = h_x * PHT.topRows(12);
				for (int m = 0; m < dof_Measurement; m++)
				{
                    //将测量噪声的方差加到矩阵的对角线上
					HPHT(m, m) += m_noise;
				}
				K_= PHT*HPHT.inverse();  //计算卡尔曼增益
			}
            // 计算状态变量的更新量
            // 由于是误差迭代KF，得到的是误差的最优估计！
			Matrix<scalar_type, n, 1> dx_ = K_ * z; // - h) + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new; 
			// state x_before = x_;

			x_.boxplus(dx_);    // 将状态变量和更新量相加，得到更新后的状态变量
			dyn_share.converge = true;
			
			// L_ = P_;
			// Matrix<scalar_type, 3, 3> res_temp_SO3;
			// MTK::vect<3, scalar_type> seg_SO3;
			// for(typename std::vector<std::pair<int, int> >::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++) {
			// 	int idx = (*it).first;
			// 	for(int i = 0; i < 3; i++){
			// 		seg_SO3(i) = dx_(i + idx);
			// 	}
			// 	res_temp_SO3 = A_matrix(seg_SO3).transpose();
			// 	for(int i = 0; i < n; i++){
			// 		L_. template block<3, 1>(idx, i) = res_temp_SO3 * (P_. template block<3, 1>(idx, i)); 
			// 	}
			// 	{
			// 		for(int i = 0; i < dof_Measurement; i++){
			// 			K_. template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
			// 		}
			// 	}
			// 	for(int i = 0; i < n; i++){
			// 		L_. template block<1, 3>(i, idx) = (L_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
			// 		// P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
			// 	}
			// 	for(int i = 0; i < n; i++){
			// 		P_. template block<1, 3>(i, idx) = (P_. template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
			// 	}
			// }
			// Matrix<scalar_type, 2, 2> res_temp_S2;
			// MTK::vect<2, scalar_type> seg_S2;
			// for(typename std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
			// 	int idx = (*it).first;
		
			// 	for(int i = 0; i < 2; i++){
			// 		seg_S2(i) = dx_(i + idx);
			// 	}
		
			// 	Eigen::Matrix<scalar_type, 2, 3> Nx;
			// 	Eigen::Matrix<scalar_type, 3, 2> Mx;
			// 	x_.S2_Nx_yy(Nx, idx);
			// 	x_propagated.S2_Mx(Mx, seg_S2, idx);
			// 	res_temp_S2 = Nx * Mx; 
	
			// 	for(int i = 0; i < n; i++){
			// 		L_. template block<2, 1>(idx, i) = res_temp_S2 * (P_. template block<2, 1>(idx, i)); 
			// 	}
				
			// 	{
			// 		for(int i = 0; i < dof_Measurement; i++){
			// 			K_. template block<2, 1>(idx, i) = res_temp_S2 * (K_. template block<2, 1>(idx, i));
			// 		}
			// 	}
			// 	for(int i = 0; i < n; i++){
			// 		L_. template block<1, 2>(i, idx) = (L_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
			// 	}
			// 	for(int i = 0; i < n; i++){
			// 		P_. template block<1, 2>(i, idx) = (P_. template block<1, 2>(i, idx)) * res_temp_S2.transpose();
			// 	}
			// }
			// if(n > dof_Measurement)
			{
                // 迭代完成后更新误差状态协方差矩阵
                // 结束迭代后，更新协方差矩阵的后验值，大致上是P=(I-K*H)*P，如 FAST-LIOv1 论文式19
                // Point-LIO 论文式20
				P_ = P_ - K_*h_x*P_. template block<12, n>(0, 0);   //更新协方差矩阵
			}
		}
		return true;
	}
	
	void update_iterated_dyn_share_IMU() {
		
		dyn_share_modified<scalar_type> dyn_share;
		for(int i=0; i<maximum_iter; i++)
		{
			dyn_share.valid = true;
            //调用函数h_dyn_share_modified_2()，这个是带IMU的优化
			h_dyn_share_modified_2(x_, dyn_share);

            // 将状态向量x_和dyn_share作为参数传入，计算出一个包含IMU测量值的向量z。
			Matrix<scalar_type, 6, 1> z = dyn_share.z_IMU;

			Matrix<double, 30, 6> PHT;
            Matrix<double, 6, 30> HP;
            Matrix<double, 6, 6> HPHT;
			PHT.setZero();
			HP.setZero();
			HPHT.setZero();
            //将P_中的第15到20列和第24到29列分别与PHT和HP中的第0到5行相加
			for (int l_ = 0; l_ < 6; l_++)
			{
				if (!dyn_share.satu_check[l_])
				{
					PHT.col(l_) = P_.col(15+l_) + P_.col(24+l_);    // P和H相乘的简化表示
					HP.row(l_) = P_.row(15+l_) + P_.row(24+l_);
				}
			}
            // 将HP的第15到20列和第24到29列分别与HPHT的第0到5行相加，如果dyn_share.satu_check[l_]为false。
            // 同时，将dyn_share.R_IMU的第l_行（l_从0到5）加到HPHT的第l_行和第l_列上。
			for (int l_ = 0; l_ < 6; l_++)
			{
				if (!dyn_share.satu_check[l_])
				{
                    // 根据HP算出的HPHT的值
					HPHT.col(l_) = HP.col(15+l_) + HP.col(24+l_);
				}
                // 加了R后的值
				HPHT(l_, l_) += dyn_share.R_IMU(l_); //, l);
			}
        	Eigen::Matrix<double, 30, 6> K = PHT * HPHT.inverse(); 
                                    
            Matrix<scalar_type, n, 1> dx_ = K * z; 

            P_ -= K * HP;
			x_.boxplus(dx_);
		}
		return;
	}
	
	void change_x(state &input_state)
	{
		x_ = input_state;

		if((!x_.vect_state.size())&&(!x_.SO3_state.size())&&(!x_.S2_state.size())&&(!x_.SEN_state.size()))
		{
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
			x_.build_SEN_state();
		}
	}

	void change_P(cov &input_cov)
	{
		P_ = input_cov;
	}

	const state& get_x() const {
		return x_;
	}
	const cov& get_P() const {
		return P_;
	}
	state x_;
private:
	measurement m_;
	cov P_;
	spMt l_;
	spMt f_x_1;
	spMt f_x_2;
	cov F_x1 = cov::Identity(); // n*n 维
	cov F_x2 = cov::Identity();
	cov L_ = cov::Identity();

	processModel *f;        // m*1 维
	processMatrix1 *f_x;    // m*n 维
	processMatrix2 *f_w;    // m * process_noise_dof，没用到

	measurementMatrix1 *h_x;
	measurementMatrix2 *h_v;

	measurementMatrix1_dyn *h_x_dyn;
	measurementMatrix2_dyn *h_v_dyn;

	measurementModel_dyn_share_modified *h_dyn_share_modified_1;

	measurementModel_dyn_share_modified *h_dyn_share_modified_2;

	int maximum_iter = 0;
	scalar_type limit[n];
	
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
