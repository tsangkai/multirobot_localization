import sys
sys.path.append("algorithm/")

import numpy as np
import math

from gs_sci_robot import GS_SCI_Robot
import sim_env



# number of robot
N = sim_env.N 

# number of landmark
M = sim_env.M 



num_of_trial = sim_env.num_of_trial
total_T = sim_env.total_T


###

sigma_tr_arr = [0] * total_T
error_arr = [0] * total_T




for i in range(num_of_trial):

	print(i)

	initial = sim_env.initial_position



	robots = [None] * N
	for n in range(N):
		robots[n] = GS_SCI_Robot(n, initial.copy())


	landmarks = [None] * M
	for m in range(M):
		landmarks[m] = sim_env.Landmark(m, np.matrix(sim_env.landmark_position, dtype=float).getT())


	for t in range(total_T):

		# motion propagation
		robots[0].prop_update()
		robots[1].prop_update()
		robots[2].prop_update()
		robots[3].prop_update()
		robots[4].prop_update()

		#robot 0
		[dis, phi] = sim_env.relative_measurement(robots[0].position, robots[0].theta, landmarks[0].position)
		robots[0].ablt_obsv([dis, phi], landmarks[0])


		# robot 2
		[dis, phi] = sim_env.relative_measurement(robots[2].position, robots[2].theta, robots[0].position)
		robots[2].rela_obsv(0, [dis, phi])

		[dis, phi] = sim_env.relative_measurement(robots[2].position, robots[2].theta, robots[1].position)
		robots[2].rela_obsv(1, [dis, phi])


		# observation - robot 3
		[dis, phi] = sim_env.relative_measurement(robots[3].position, robots[3].theta, landmarks[0].position)
		robots[3].ablt_obsv([dis, phi], landmarks[0])

		[dis, phi] = sim_env.relative_measurement(robots[3].position, robots[3].theta, robots[4].position)
		robots[3].rela_obsv(4, [dis, phi])


		# communication
		robots[2].comm(robots[3].s, robots[3].sigma_i, robots[3].sigma_d)
		robots[0].comm(robots[2].s, robots[2].sigma_i, robots[2].sigma_d)



		# real error
		s = 0
		for j in range(N):
			s += pow(robots[0].s[2*j,0] - robots[j].position[0],2) + pow(robots[0].s[2*j+1,0] - robots[j].position[1],2)

		error_arr[t] += math.sqrt(s/float(N)) /float(num_of_trial)

		# covariance error
		total_sigma = robots[0].getSigma()
		sigma_tr_arr[t] += math.sqrt(total_sigma.trace()[0,0]/float(N)) /float(num_of_trial)


#for k in range (5):
#	robots[k].status()




file = open('gs_sci_output.txt', 'w')

for t in range(total_T):

	file.write( str(sigma_tr_arr[t]) + ', ' + str(error_arr[t]) + '\n')

file.close()


