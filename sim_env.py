import numpy as np

from math import cos, sin, atan2, sqrt




##### Simulation Parameter setup

num_of_trial = 2
total_T = 2000

##### Simulation Environment Setup

# number of robot
N = 5

# number of landmark
M = 1


dt = 0.5


max_v = 0.09#25
max_oemga = 0.05


var_u_v = pow(0.05, 2)* pow(max_v,2)
var_u_theta = 0.0001


var_v = 2.0 * 4*max_v*max_v/12

var_dis = pow(0.05,2)
var_phi = pow(2.0 / 180, 2)

d_max = 25


##### Constants


J = np.matrix([[0, -1],[1, 0]])

z_mtx_10 = np.matrix([
	[0,0,0,0,0,0,0,0,0,0], 
	[0,0,0,0,0,0,0,0,0,0], 
	[0,0,0,0,0,0,0,0,0,0], 
	[0,0,0,0,0,0,0,0,0,0], 
	[0,0,0,0,0,0,0,0,0,0], 
	[0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0],
	[0,0,0,0,0,0,0,0,0,0]], dtype=float)

i_mtx_10 = np.matrix([
	[1,0,0,0,0,0,0,0,0,0], 
	[0,1,0,0,0,0,0,0,0,0], 
	[0,0,1,0,0,0,0,0,0,0], 
	[0,0,0,1,0,0,0,0,0,0], 
	[0,0,0,0,1,0,0,0,0,0], 
	[0,0,0,0,0,1,0,0,0,0],
	[0,0,0,0,0,0,1,0,0,0],
	[0,0,0,0,0,0,0,1,0,0],
	[0,0,0,0,0,0,0,0,1,0],
	[0,0,0,0,0,0,0,0,0,1]], dtype=float)


i_mtx_2 = np.matrix([
	[1, 0],
	[0, 1]], dtype=float)



########################################################


origin = [0.0, 0.0]

def inRange(a, b):
	if sqrt(pow(a[0]-b[0], 2)+pow(a[1]-b[1], 2)) > d_max:
		return False
	else:
		return True

class Landmark:

	def __init__(self, index, position):
		self.index = index
		self.position = position




def rot_mtx(theta):
	return np.matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])



def relative_measurement(pos_1, theta_1, pos_2):

	delta_x = pos_2[0] - pos_1[0]
	delta_y = pos_2[1] - pos_1[1]
	dis = sqrt(delta_y*delta_y + delta_x*delta_x) + np.random.normal(0, sqrt(var_dis))
	phi = atan2(delta_y, delta_x) + np.random.normal(0, sqrt(var_phi)) - theta_1

	return [dis, phi]