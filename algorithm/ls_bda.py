from numpy import matrix
from math import cos, sin, atan2, sqrt

from ls_cen import LS_Cen

import sim_env


class LS_BDA(LS_Cen):


	def ablt_obsv_update(self, idx, obs_value, landmark):
		ii = 2*idx

		local_s = self.s[ii:ii+2].copy()
		local_sigma = self.sigma[ii:ii+2,ii:ii+2]

		H = sim_env.rot_mtx(self.theta[idx]).getT()*matrix([[-1,0],[0,-1]], dtype=float)

		dis = obs_value[0]
		phi = obs_value[1]

		hat_z = sim_env.rot_mtx(self.theta[idx]).getT() * (landmark.position + matrix([[-1,0],[0,-1]], dtype=float) * local_s)
		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, dis*dis*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_invention = H * local_sigma * H.getT()  + sigma_z
		kalman_gain = local_sigma * H.getT() * sigma_invention.getI()


		self.s[ii:ii+2]	= local_s + kalman_gain * (z - hat_z)

		self.sigma[ii:ii+2,ii:ii+2] = local_sigma - kalman_gain * H * local_sigma

		multi_i = sim_env.i_mtx_2.copy() - kalman_gain * H
		for k in range(5):
			if( k!=idx ):
				kk = 2*k
				self.sigma[ii:ii+2,kk:kk+2] = multi_i * self.sigma[ii:ii+2,kk:kk+2]
				self.sigma[kk:kk+2,ii:ii+2] = self.sigma[kk:kk+2,ii:ii+2] * multi_i.getT()


	def rela_obsv_update(self, idx, obs_idx, obs_value):
		ii = 2*idx
		jj = 2*obs_idx


		H_ij = matrix([[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]], dtype=float)
		H_ij[0, ii] = -1
		H_ij[1, ii+1] = -1
		H_ij[0, jj] = 1
		H_ij[1, jj+1] = 1


		H = sim_env.rot_mtx(self.theta[idx]).getT()*H_ij

		dis = obs_value[0]
		phi = obs_value[1]

		hat_z = H * self.s
		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		reduced_sigma = sim_env.z_mtx_10.copy()
		reduced_sigma[ii:ii+2,ii:ii+2] = self.sigma[ii:ii+2,ii:ii+2]
		reduced_sigma[jj:jj+2,jj:jj+2] = self.sigma[jj:jj+2,jj:jj+2]
		reduced_sigma[ii:ii+2,jj:jj+2] = self.sigma[ii:ii+2,jj:jj+2]
		reduced_sigma[jj:jj+2,ii:ii+2] = self.sigma[jj:jj+2,ii:ii+2] 

		multi_i = self.sigma[ii:ii+2,ii:ii+2].getI()
		multi_j = self.sigma[jj:jj+2,jj:jj+2].getI()


		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, dis*dis*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_invention = H * reduced_sigma * H.getT()  + sigma_z
		kalman_gain = reduced_sigma * H.getT() * sigma_invention.getI()


		self.s[ii:ii+2] = self.s[ii:ii+2] + kalman_gain[ii:ii+2] * (z-hat_z)
		self.s[jj:jj+2] = self.s[jj:jj+2] + kalman_gain[jj:jj+2] * (z-hat_z)

		self.sigma = self.sigma - kalman_gain*H*reduced_sigma


		multi_i = self.sigma[ii:ii+2,ii:ii+2] * multi_i
		multi_j = self.sigma[jj:jj+2,jj:jj+2] * multi_j


		for k in range(sim_env.N):
			if( k != idx):
				if( k != obs_idx):
					kk = 2*k
					self.sigma[ii:ii+2,kk:kk+2] = multi_i * self.sigma[ii:ii+2,kk:kk+2]
					self.sigma[jj:jj+2,kk:kk+2] = multi_j * self.sigma[jj:jj+2,kk:kk+2]

					self.sigma[kk:kk+2,ii:ii+2] =  self.sigma[kk:kk+2,ii:ii+2] * multi_i.getT()
					self.sigma[kk:kk+2,jj:jj+2] =  self.sigma[kk:kk+2,jj:jj+2] * multi_j.getT()

