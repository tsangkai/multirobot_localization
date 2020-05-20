import sys
sys.path.append("algorithm/")


import numpy as np
import math

import sim_env


algorithm = sys.argv[1]
output_file_name = 'ls_' + algorithm + '_output.txt'

if algorithm == 'cen':	
	from ls_cen_team import LS_Cen_Team as LS_Team

elif algorithm == 'bda':
	from ls_bda_team import LS_BDA_Team as LS_Team

elif algorithm == 'ci':
	from ls_ci_team import LS_CI_Team as LS_Team

print("run " + algorithm + ":")


num_of_trial = sim_env.num_of_trial
total_T = sim_env.total_T

# number of robot
N = sim_env.N 

# number of landmark
M = sim_env.M 

sigma_tr_arr = [0] * total_T
error_arr = [0] * total_T


for i in range(num_of_trial):

	print(i)
	initial = sim_env.initial_position

	robots = LS_Team(initial)

	landmarks = [None] * M
	for m in range(M):
		landmarks[m] = sim_env.Landmark(m, np.matrix(sim_env.landmark_position, dtype=float).getT())

	for t in range(total_T):

		# motion propagation
		robots.prop_update()


		# observation update
		#robot 0
		[dis, phi] = sim_env.relative_measurement(robots.position[0:2], robots.theta[0], landmarks[0].position)
		robots.ablt_obsv(0, [dis, phi], landmarks[0])

		# robot 2
		# [dis, phi] = sim_env.relative_measurement(robots.position[4:6], robots.theta[2], robots.position[0:2])
		# robots.rela_obsv(2, 0, [dis, phi])

		# [dis, phi] = sim_env.relative_measurement(robots.position[4:6], robots.theta[2], robots.position[2:4])
		# robots.rela_obsv(2, 1, [dis, phi])

		# robot 3
		[dis, phi] = sim_env.relative_measurement(robots.position[6:8], robots.theta[3], landmarks[0].position)
		robots.ablt_obsv(3, [dis, phi], landmarks[0])

		# [dis, phi] = sim_env.relative_measurement(robots.position[6:8], robots.theta[3], robots.position[8:10])
		# robots.rela_obsv(3, 4, [dis, phi])
		

		# real error
		s = 0
		for j in range(N):
			jj = 2*j
			s += ((robots.s[jj,0] - robots.position[jj,0])**2 + (robots.s[jj+1,0] - robots.position[jj+1,0])**2)
		
		error_arr[t] += math.sqrt(s/float(N)) / float(num_of_trial)

		# covariance error
		sigma_tr_arr[t] += math.sqrt(robots.sigma.trace()[0,0]/float(N)) / float(num_of_trial)


# output performance

file = open(output_file_name, 'w')

for t in range(total_T):

	file.write(str(sigma_tr_arr[t]) + ', ' + str(error_arr[t]) + '\n')

file.close()
