from numpy import matrix
from math import cos, sin, atan2, sqrt

import sim_env 



class GS_CI():


	def __init__(self, _index, _initial_s, _theta=0.0):
		self.index = _index

		self.s = _initial_s.copy()

		self.sigma = sim_env.initial_cov.copy()
		self.th_sigma = sim_env.initial_cov.copy()

		self.theta = _theta


	def motion_propagation_update(self, odometry_input, dt):

		[v, omega] = odometry_input

		ii = 2*self.index

		# estimation update
		self.s[ii,0] = self.s[ii,0] + cos(self.theta)*v*dt
		self.s[ii+1,0] = self.s[ii+1,0] + sin(self.theta)*v*dt

		# covariance update
		for j in range(sim_env.N):
			jj = 2*j

			if j==self.index:
				rot_mtx_theta = sim_env.rot_mtx(self.theta)
				self.sigma[jj:jj+2, jj:jj+2] += (dt**2)*rot_mtx_theta*matrix([[sim_env.var_u_v, 0],[0, 0]])*rot_mtx_theta.T
				self.th_sigma[jj:jj+2, jj:jj+2] += 2*(dt**2)*matrix([[sim_env.var_u_v, 0],[0, 0]])

			else:
				self.sigma[jj:jj+2, jj:jj+2] += (dt**2)*sim_env.var_v*sim_env.i_mtx_2.copy()
				self.th_sigma[jj:jj+2, jj:jj+2] += (dt**2)*sim_env.var_v*sim_env.i_mtx_2.copy()


	def ablt_obsv_update(self, obs_value, landmark):
		ii = 2*self.index

		H_i = matrix([[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]], dtype=float)
		H_i[0, ii] = -1
		H_i[1, ii+1] = -1

		H = sim_env.rot_mtx(self.theta).getT()*H_i
		
		dis = obs_value[0]
		phi = obs_value[1]

		hat_z = sim_env.rot_mtx(self.theta).getT() * (landmark.position + H_i*self.s)
		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, (dis**2)*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_invention = H * self.sigma * H.getT()  + sigma_z
		kalman_gain = self.sigma * H.getT() * sigma_invention.getI()

		self.s = self.s + kalman_gain*(z - hat_z)
		self.sigma = self.sigma - kalman_gain*H*self.sigma

		sigma_th_z = max(sim_env.var_dis, (sim_env.d_max**2)*sim_env.var_phi)* sim_env.i_mtx_2.copy() 
		self.th_sigma = (self.th_sigma.getI() + H_i.getT() * sigma_th_z.getI() * H_i).getI()


	def rela_obsv_update(self, obs_idx, obs_value):
		ii = 2*self.index
		jj = 2*obs_idx

		H_ij = matrix([[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]], dtype=float)
		H_ij[0, ii] = -1
		H_ij[1, ii+1] = -1
		H_ij[0, jj] = 1
		H_ij[1, jj+1] = 1

		H = sim_env.rot_mtx(self.theta).getT()*H_ij

		dis = obs_value[0]
		phi = obs_value[1]

		hat_z = H * self.s
		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, (dis**2)*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_invention = H * self.sigma * H.getT() + sigma_z
		kalman_gain = self.sigma * H.getT() * sigma_invention.getI()

		# update
		sigma_th_z = max(sim_env.var_dis, (sim_env.d_max**2)*sim_env.var_phi)* sim_env.i_mtx_2.copy() 
		self.th_sigma = (self.th_sigma.getI() + H_ij.getT() * sigma_th_z.getI() * H_ij).getI()
		self.s = self.s + kalman_gain*(z - hat_z)
		self.sigma = self.sigma - kalman_gain*H*self.sigma



	def comm_update(self, comm_robot_s, comm_robot_sigma, comm_robot_th_sigma):

		ci_coeff = 0.8

		sig_inv = ci_coeff*self.sigma.getI() + (1-ci_coeff)*comm_robot_sigma.getI()

		self.s = sig_inv.getI() * (ci_coeff*self.sigma.getI()*self.s + (1-ci_coeff)*comm_robot_sigma.getI()*comm_robot_s)
		self.sigma = sig_inv.getI()

		self.th_sigma = (ci_coeff*self.th_sigma.getI() + (1-ci_coeff)*comm_robot_th_sigma.getI()).getI()		


