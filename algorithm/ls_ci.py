from numpy import matrix
from math import cos, sin, atan2, sqrt

from ls_cen import LS_Cen

import sim_env


class LS_CI(LS_Cen):


	def rela_obsv_update(self, idx, obs_idx, obs_value):
		ii = 2*idx
		jj = 2*obs_idx

		sigma_i = self.sigma[ii:ii+2,ii:ii+2]
		sigma_j = self.sigma[jj:jj+2,jj:jj+2]

		dis = obs_value[0]
		phi = obs_value[1]

		z = matrix([[dis*cos(phi)],[dis*sin(phi)]])

		hat_j = self.s[ii:ii+2] + sim_env.rot_mtx(self.theta[idx]) * z

		sigma_z = sim_env.rot_mtx(phi) * matrix([[sim_env.var_dis, 0],[0, dis*dis*sim_env.var_phi]]) * sim_env.rot_mtx(phi).getT() 
		sigma_hat_j = sigma_i + sim_env.rot_mtx(self.theta[idx]) * sigma_z * sim_env.rot_mtx(self.theta[idx]).getT()

		ci_coeff = 0.83

		sigma_j_next_inv = ci_coeff * sigma_j.getI() + (1-ci_coeff) * sigma_hat_j.getI()

		self.s[jj:jj+2] = sigma_j_next_inv.getI()*(ci_coeff*sigma_j.getI()*self.s[jj:jj+2] + (1-ci_coeff)*sigma_hat_j.getI()*hat_j)
		self.sigma[jj:jj+2,jj:jj+2] = sigma_j_next_inv.getI()


