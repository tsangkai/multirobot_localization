# The class of GS-CI robot
# The whole class is inherited from GS-CI
# Only the communication update is overwritten

from gs_robot import GS_Robot



class GS_CI_Robot(GS_Robot):

	def comm(self, comm_robot_s, comm_robot_sigma, comm_robot_th_sigma):

		ci_coeff = 0.8

		sig_inv = ci_coeff*self.sigma.getI() + (1-ci_coeff)*comm_robot_sigma.getI()

		self.s = sig_inv.getI() * (ci_coeff*self.sigma.getI()*self.s + (1-ci_coeff)*comm_robot_sigma.getI()*comm_robot_s)
		self.sigma = sig_inv.getI()

		self.th_sigma = (ci_coeff*self.th_sigma.getI() + (1-ci_coeff)*comm_robot_th_sigma.getI()).getI()		


