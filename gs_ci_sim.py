import numpy as np
import math

import sim_env
import gs_ci_robot



# number of robot
N = sim_env.N 

# number of landmark
M = sim_env.M 





iteration = sim_env.num_of_trial
time_end = sim_env.total_T

sigma_tr_arr = [0] * time_end
sigma_th_tr_arr = [0] * time_end
error_arr = [0] * time_end




for i in range(iteration):

	print(i)

	initial = np.matrix([1, 1, 1, 2, 2, 1, -1, -1, 1, 3], dtype=float).T


	# initialization
	robots = [None] * N
	for n in range(N):
		robots[n] = robot_gs_ci.Robot_GS_CI(n, initial.copy())

	landmarks = [None] * M
	for m in range(M):
		landmarks[m] = sim_env.Landmark(m, np.matrix([0.01, 0.02], dtype=float).getT())


	# simulation body
	for t in range(time_end):
 
		### motion propagation 
		robots[0].prop_update()
		robots[1].prop_update()
		robots[2].prop_update()
		robots[3].prop_update()
		robots[4].prop_update()


		### observation update
		# robot 2
		[dis, phi] = sim_env.relative_measurement(robots[2].position, robots[2].theta, robots[0].position)
		robots[2].rela_obsv(0, [dis, phi])

		[dis, phi] = sim_env.relative_measurement(robots[2].position, robots[2].theta, robots[1].position)
		robots[2].rela_obsv(1, [dis, phi])


		# observation - robot 3
		[dis, phi] = sim_env.relative_measurement(robots[3].position, robots[3].theta, landmarks[0].position)
		robots[3].ablt_obsv([dis, phi], landmarks[0])

		[dis, phi] = sim_env.relative_measurement(robots[3].position, robots[3].theta, robots[0].position)
		robots[3].rela_obsv(0, [dis, phi])		

		[dis, phi] = sim_env.relative_measurement(robots[3].position, robots[3].theta, robots[4].position)
		robots[3].rela_obsv(4, [dis, phi])


		### communication update
		robots[2].comm(robots[3].s, robots[3].sigma, robots[3].th_sigma)
		#robots[3].comm(robots[2].s, robots[2].sigma, robots[2].th_sigma)


		### error calculation
		# real error
		focus_index = 2
		s = 0
		for j in range(5):
			s += pow(robots[focus_index].s[2*j,0] - robots[j].position[0],2) + pow(robots[focus_index].s[2*j+1,0] - robots[j].position[1],2)
		s = math.sqrt(s*0.2)

		error_arr[t] = error_arr[t] + s*(1/float(iteration))

		# covariance error
		sigma_tr_arr[t] = sigma_tr_arr[t] + math.sqrt(0.2*robots[focus_index].sigma.trace()[0,0] )*(1/float(iteration))

		sigma_th_tr_arr[t] = sigma_th_tr_arr[t] + math.sqrt(0.2*robots[focus_index].th_sigma.trace()[0,0])*(1/float(iteration))



# output
file = open('output_gs_ci.txt', 'w')

for t in range(time_end):

	file.write( str(sigma_tr_arr[t]) + ' ' + str(sigma_th_tr_arr[t]) + ' ' + str(error_arr[t]) + '\n')

file.close()


