from numpy import matrix
from math import cos, sin, atan2, sqrt

import sim_env

i_mtx_2 = sim_env.i_mtx_2
i_mtx_10 = sim_env.i_mtx_10



class LS_SCI:


	def __init__(self , initial_s):

		self.s = initial_s.copy()
		self.sigma_i = 0.99 * sim_env.initial_cov.copy()
		self.sigma_d = 0.01 * sim_env.initial_cov.copy()

		# self.position = initial_s.copy()
		self.theta = [0.0,0,0,0,0]


	def motion_propagation_update(self, odometry_input, dt):

		for i in range(sim_env.N):
			[v, omega] = odometry_input[i]

			ii = 2*i


			self.theta[i] = self.theta[i] + omega * dt

			# estimation update
			self.s[ii,0] += cos(self.theta[i]) * v * dt
			self.s[ii+1,0] += sin(self.theta[i]) * v * dt

			# covariance update
			rot_mtx_theta_i = sim_env.rot_mtx(self.theta[i])
			self.sigma_i[ii:ii+2, ii:ii+2] += dt*dt*rot_mtx_theta_i*matrix([[sim_env.var_u_v, 0],[0, 0]])*rot_mtx_theta_i.T


	def ablt_obsv_update(self, idx, obs_value, landmark):
		ii = 2*idx
		total_sigma = self.sigma_i + self.sigma_d

		H_i = matrix([[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]], dtype=float)
		H_i[0, ii] = -1
		H_i[1, ii+1] = -1

		H = sim_env.rot_mtx(self.theta[idx]).getT()*H_i
		
		dis = obs_value[0]
		phi = obs_value[1]

		hat_z = sim_env.rot_mtx(self.theta[idx]).getT() * (landmark.position + H_i*self.s)
		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, dis*dis*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_invention = H * total_sigma * H.getT()  + sigma_z
		kalman_gain = total_sigma * H.getT() * sigma_invention.getI()


		self.s = self.s + kalman_gain * (z-hat_z)

		total_sigma = total_sigma - kalman_gain * H * total_sigma
		self.sigma_i = (i_mtx_10.copy() - kalman_gain*H) * self.sigma_i * (i_mtx_10.copy() - kalman_gain*H).getT() + kalman_gain * sigma_z * kalman_gain.getT()
		self.sigma_d = total_sigma - self.sigma_i





	def rela_obsv_update(self, idx, obs_idx, obs_value):
		ii = 2*idx
		jj = 2*obs_idx

		ci_coeff = 0.83

		sigma_j = (1/ci_coeff) * self.sigma_d[jj:jj+2,jj:jj+2] + self.sigma_i[jj:jj+2,jj:jj+2]
		sigma_i = (1/(1-ci_coeff)) * self.sigma_d[ii:ii+2,ii:ii+2] + self.sigma_i[ii:ii+2,ii:ii+2]

		dis = obs_value[0]
		phi = obs_value[1]

		z = matrix([[dis*cos(phi)],[dis*sin(phi)]])

		hat_j = self.s[ii:ii+2] + sim_env.rot_mtx(self.theta[idx]) * z

		kalman_gain = sigma_j * (sigma_i + sigma_j).getI()
		self.s[jj:jj+2] = self.s[jj:jj+2] + kalman_gain * (hat_j - self.s[jj:jj+2])

		total_sigma = (i_mtx_2 - kalman_gain) * sigma_j
		self.sigma_i[jj:jj+2,jj:jj+2] = (i_mtx_2 - kalman_gain) * self.sigma_i[jj:jj+2,jj:jj+2] * (i_mtx_2 - kalman_gain).getT() + kalman_gain * self.sigma_i[ii:ii+2,ii:ii+2] * kalman_gain.getT()
		self.sigma_d[jj:jj+2,jj:jj+2] = total_sigma - self.sigma_i[jj:jj+2,jj:jj+2]


	def getSigma(self):
		return self.sigma_i + self.sigma_d

