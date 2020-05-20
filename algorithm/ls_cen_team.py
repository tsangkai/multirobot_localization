from numpy import matrix
from numpy import random
from math import cos, sin, atan2, sqrt

import sim_env

dt = sim_env.dt


class LS_Cen_Team:


	def __init__(self , initial_s):

		self.s = initial_s.copy()
		self.sigma = sim_env.initial_cov.copy()

		self.position = initial_s.copy()
		self.theta = [0.0,0,0,0,0]


	def prop_update(self):

		for i in range(sim_env.N):
			ii = 2*i

			# select valid motion input
			[v, omega] = [0,0]
			v_star = 0
			pre_update_position = [100, 100]

			while(not sim_env.inRange(pre_update_position, sim_env.origin)):
				[v, omega] = [sim_env.max_v*random.uniform(-1,1), sim_env.max_omega*random.uniform(-1,1)]
				v_star = v + random.normal(0, sqrt(sim_env.var_u_v))
				pre_update_position = [self.position[ii] + cos(self.theta[i])*v_star*dt, self.position[ii+1] + sin(self.theta[i])*v_star*dt]

			# real position update
			self.position[ii,0] = self.position[ii,0] + cos(self.theta[i])*v_star*dt
			self.position[ii+1,0] = self.position[ii+1,0] + sin(self.theta[i])*v_star*dt

			self.theta[i] = self.theta[i] + omega*dt

			# estimation update
			self.s[ii,0] += cos(self.theta[i]) * v * dt
			self.s[ii+1,0] += sin(self.theta[i]) * v * dt

			# covariance update
			rot_mtx_theta_i = sim_env.rot_mtx(self.theta[i])
			self.sigma[ii:ii+2, ii:ii+2] += dt*dt*rot_mtx_theta_i*matrix([[sim_env.var_u_v, 0],[0, 0]])*rot_mtx_theta_i.T




	def ablt_obsv(self, idx, obs_value, landmark):
		ii = 2*idx

		H_i = matrix([[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]], dtype=float)
		H_i[0, ii] = -1
		H_i[1, ii+1] = -1

		H = sim_env.rot_mtx(self.theta[idx]).getT()*H_i
		
		dis = obs_value[0]
		phi = obs_value[1]

		hat_z = sim_env.rot_mtx(self.theta[idx]).getT() * (landmark.position + H_i*self.s)
		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, dis*dis*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_invention = H * self.sigma * H.getT()  + sigma_z
		kalman_gain = self.sigma * H.getT() * sigma_invention.getI()


		self.s = self.s + kalman_gain * (z-hat_z)

		self.sigma = self.sigma - kalman_gain * H * self.sigma




	def rela_obsv(self, idx, obs_idx, obs_value):
		i = 2*idx
		j = 2*obs_idx


		H_ij = matrix([[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]], dtype=float)
		H_ij[0, i] = -1
		H_ij[1, i+1] = -1
		H_ij[0, j] = 1
		H_ij[1, j+1] = 1


		H = sim_env.rot_mtx(self.theta[idx]).getT()*H_ij

		dis = obs_value[0]
		phi = obs_value[1]

		hat_z = H * self.s
		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, dis*dis*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_invention = H * self.sigma * H.getT()  + sigma_z
		kalman_gain = self.sigma*H.getT()*sigma_invention.getI()

		self.s = self.s + kalman_gain*(z - hat_z)

		self.sigma = self.sigma - kalman_gain*H*self.sigma






