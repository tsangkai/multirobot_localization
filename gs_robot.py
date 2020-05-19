from numpy import matrix
from numpy import random
from math import cos, sin, atan2, sqrt

import sim_env 

dt = sim_env.dt


class GS_Robot:


	def __init__(self, index, initial_s):
		self.index = index

		self.s = initial_s.copy()
		self.position = [initial_s[2*index,0], initial_s[2*index+1,0]]

		self.sigma = sim_env.initial_cov.copy()
		self.th_sigma = sim_env.initial_cov.copy()

		self.theta = 0.0


	def prop_update(self):


		# select valid motion input
		[v, a_v] = [sim_env.max_v*random.uniform(-1,1), sim_env.max_omega*random.uniform(-1,1)]
		v_star = v + random.normal(0, sqrt(sim_env.var_u_v))
		pre_update_position = [self.position[0]+cos(self.theta)*v_star*dt, self.position[1]+sin(self.theta)*v_star*dt]


		while(not sim_env.inRange(pre_update_position, sim_env.origin)):

			[v, a_v] = [sim_env.max_v*random.uniform(-1,1), sim_env.max_omega*random.uniform(-1,1)]
			v_star = v + random.normal(0, sqrt(sim_env.var_u_v))
			pre_update_position = [self.position[0]+cos(self.theta)*v_star*dt, self.position[1]+sin(self.theta)*v_star*dt]


		self.theta = self.theta + a_v * dt

		# real position update
		self.position[0] = self.position[0] + cos(self.theta)*v_star*dt
		self.position[1] = self.position[1] + sin(self.theta)*v_star*dt



		i = 2*self.index

		# estimation update
		self.s[i,0] = self.s[i,0] + cos(self.theta)*v*dt
		self.s[i+1,0] = self.s[i+1,0] + sin(self.theta)*v*dt

		# covariance update
		for j in range(sim_env.N):
			idx = 2*j

			if j==self.index:
				rot_mtx_theta = sim_env.rot_mtx(self.theta)
				self.sigma[idx:idx+2, idx:idx+2] += (dt**2)*rot_mtx_theta*matrix([[sim_env.var_u_v, 0],[0, 0]])*rot_mtx_theta.T
				self.th_sigma[idx:idx+2, idx:idx+2] += 2*(dt**2)*matrix([[sim_env.var_u_v, 0],[0, 0]])

			else:
				self.sigma[idx:idx+2, idx:idx+2] += (dt**2)*sim_env.var_v*sim_env.i_mtx_2.copy()
				self.th_sigma[idx:idx+2, idx:idx+2] += (dt**2)*sim_env.var_v*sim_env.i_mtx_2.copy()





	def ablt_obsv(self, obs_value, landmark):
		i = 2*self.index


		H_i = matrix([[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]], dtype=float)
		H_i[0, i] = -1
		H_i[1, i+1] = -1


		H = sim_env.rot_mtx(self.theta).getT()*H_i
		
		dis = obs_value[0]
		phi = obs_value[1]

		#z = [dis*cos(phi), dis*sin(phi)]
		hat_z = sim_env.rot_mtx(self.theta).getT() * (landmark.position + H_i*self.s)
		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, (dis**2)*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_invention = H * self.sigma * H.getT()  + sigma_z
		kalman_gain = self.sigma*H.getT()*sigma_invention.getI()

		sigma_th_z = max(sim_env.var_dis, (sim_env.d_max**2)*sim_env.var_phi)* sim_env.i_mtx_2.copy() 

		self.th_sigma = (self.th_sigma.getI() + H_i.getT() * sigma_th_z.getI() * H_i).getI()
		self.s = self.s + kalman_gain*(z - hat_z)

		self.sigma = self.sigma - kalman_gain*H*self.sigma



	def rela_obsv(self, obs_idx, obs_value):
		i = 2*self.index
		j = 2*obs_idx


		H_ij = matrix([[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]], dtype=float)
		H_ij[0, i] = -1
		H_ij[1, i+1] = -1
		H_ij[0, j] = 1
		H_ij[1, j+1] = 1


		H = sim_env.rot_mtx(self.theta).getT()*H_ij

		dis = obs_value[0]
		phi = obs_value[1]

		#z = [dis*cos(phi), dis*sin(phi)]

		hat_z = H * self.s
		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, (dis**2)*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_invention = H * self.sigma * H.getT()  + sigma_z
		kalman_gain = self.sigma*H.getT()*sigma_invention.getI()


		# update
		sigma_th_z = max(sim_env.var_dis, (sim_env.d_max**2)*sim_env.var_phi)* sim_env.i_mtx_2.copy() 
		self.th_sigma = (self.th_sigma.getI() + H_ij.getT() * sigma_th_z.getI() * H_ij).getI()
		self.s = self.s + kalman_gain*(z - hat_z)
		self.sigma = self.sigma - kalman_gain*H*self.sigma

