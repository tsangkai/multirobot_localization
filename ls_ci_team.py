from numpy import matrix
from numpy import random
from numpy import linalg
from math import cos, sin, atan2, sqrt

from sim_env import *



########################################################


class LS_CI_Team:


	def __init__(self , initial_s):

		self.s = initial_s.copy()
		self.sigma = initial_cov

		#self.th_sigma = z_mtx_10.copy()

		self.position = initial_s.copy()
		self.theta = [0.0,0,0,0,0]




	def prop_update(self):


		for i in range(5):
			ii = 2*i

			# select valid motion input
			[v, a_v] = [random.uniform(-max_v,max_v), random.uniform(-max_omega,max_omega)]
			v_star = v + random.normal(0, sqrt(var_u_v))
			pre_update_position = [self.position[ii] + cos(self.theta[i])*v_star*dt, self.position[ii+1] + sin(self.theta[i])*v_star*dt]


			while(not inRange(pre_update_position, origin)):

				[v, a_v] = [random.uniform(-max_v,max_v), random.uniform(-max_oemga,max_oemga)]
				v_star = v + random.normal(0, sqrt(var_u_v))
				pre_update_position = [self.position[ii] + cos(self.theta[i])*v_star*dt, self.position[ii+1] + sin(self.theta[i])*v_star*dt]




			# real position update
			self.position[ii] = self.position[ii] + cos(self.theta[i])*v_star*dt
			self.position[ii+1] = self.position[ii+1] + sin(self.theta[i])*v_star*dt

			self.theta[i] = self.theta[i] + a_v*dt



			# estimation update
			self.s[ii] = self.s[ii] + cos(self.theta[i])*v*dt
			self.s[ii+1] = self.s[ii+1] + sin(self.theta[i])*v*dt


			self.sigma[ii:ii+2, ii:ii+2] = self.sigma[ii:ii+2, ii:ii+2]+ dt*dt*rot_mtx(self.theta[i])*matrix([[var_u_v, 0],[0, 0]])*rot_mtx(self.theta[i]).T





	def ablt_obsv(self, idx, obs_value, landmark):

		ii = 2*idx


		local_s = self.s[ii:ii+2]	
		local_sigma = self.sigma[ii:ii+2,ii:ii+2]

		dis = obs_value[0]
		phi = obs_value[1]

		H = rot_mtx(self.theta[idx]).getT()*matrix([[-1,0],[0,-1]], dtype=float)

		# hat_z = rot_mtx(self.theta[idx]).getT() * (landmark.position + H * local_s)
		hat_z = rot_mtx(self.theta[idx]).getT() * (landmark.position + matrix([[-1,0],[0,-1]], dtype=float) * local_s)

		z = matrix([dis*cos(phi), dis*sin(phi)]).getT()

		# print(hat_z)
		# print(z)


		sigma_z = rot_mtx(phi) * matrix([[var_dis, 0],[0, dis*dis*var_phi]]) * rot_mtx(phi).getT() 
		sigma_invention = H * local_sigma * H.getT() + sigma_z
		kalman_gain = local_sigma * H.getT() * sigma_invention.getI()

		# print(self.sigma[ii:ii+2,ii:ii+2])

		self.s[ii:ii+2]	= local_s + kalman_gain * (z - hat_z)
		self.sigma[ii:ii+2,ii:ii+2] = local_sigma - kalman_gain*H*local_sigma

		# print(self.sigma[ii:ii+2,ii:ii+2])
		# raw_input()

	def rela_obsv(self, idx, obs_idx, obs_value):
		i = 2*idx
		j = 2*obs_idx


		H_ij = matrix([[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]], dtype=float)
		H_ij[0, i] = -1
		H_ij[1, i+1] = -1
		H_ij[0, j] = 1
		H_ij[1, j+1] = 1


		H = rot_mtx(self.theta[idx]).getT()*H_ij

		dis = obs_value[0]
		phi = obs_value[1]

		#z = [dis*cos(phi), dis*sin(phi)].getT()
		#hat_z = H * self.s
		z = matrix([[dis*cos(phi)],[dis*sin(phi)]])


		
		hat_j = z + self.s[i:i+2]

		#matrix([[z[0]+self.s[i]], [z[1],self.s[i+1]]], dtype=float)


		sigma_i = self.sigma[i:i+2,i:i+2]
		sigma_j = self.sigma[j:j+2,j:j+2]


		e = 0.83


		#print(self.s)

		sigma_j_next_inv = e*sigma_j.getI() + (1-e)*sigma_i.getI()


		self.s[j:j+2] = sigma_j_next_inv.getI()*(e*sigma_j.getI()*self.s[j:j+2]  + (1-e)*sigma_i.getI()*hat_j)
		self.sigma[j:j+2,j:j+2] = sigma_j_next_inv.getI()


