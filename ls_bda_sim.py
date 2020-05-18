import numpy as np
import math


from ls_bda_team import LS_BDA_Team
import sim_env




num_of_trial = sim_env.num_of_trial
total_T = sim_env.total_T

# number of robot
N = sim_env.N 

# number of landmark
M = sim_env.M 




###


iteration = sim_env.num_of_trial
time_end = sim_env.total_T

sigma_tr_arr = [0] * time_end
error_arr = [0] * time_end


for i in range(iteration):

	print(i)

	initial = sim_env.initial_position


	robots = LS_BDA_Team(initial)


	landmarks = [None] * M
	for m in range(M):
		landmarks[m] = sim_env.Landmark(m, np.matrix(sim_env.landmark_position, dtype=float).getT())


	for t in range(time_end):

		# motion propagation
		robots.prop_update()

		#robot 0
		[dis, phi] = sim_env.relative_measurement(robots.position[0:2], robots.theta[0], landmarks[0].position)
		robots.ablt_obsv(0, [dis, phi], landmarks[0])


		# robot 2
		[dis, phi] = sim_env.relative_measurement(robots.position[4:6], robots.theta[2], robots.position[0:2])
		robots.rela_obsv(2, 0, [dis, phi])

		[dis, phi] = sim_env.relative_measurement(robots.position[4:6], robots.theta[2], robots.position[2:4])
		robots.rela_obsv(2, 1, [dis, phi])


		# observation - robot 3
		[dis, phi] = sim_env.relative_measurement(robots.position[6:8], robots.theta[3], landmarks[0].position)
		robots.ablt_obsv(3, [dis, phi], landmarks[0])

		[dis, phi] = sim_env.relative_measurement(robots.position[6:8], robots.theta[3], robots.position[8:10])
		robots.rela_obsv(3, 4, [dis, phi])





		# real error
		s = 0
		for j in range(N):
			s += pow(robots.s[2*j,0] - robots.position[2*j,0],2) + pow(robots.s[2*j+1,0] - robots.position[2*j+1,0],2)
		
		error_arr[t] += math.sqrt(s/float(N)) /float(iteration)


		# covariance error
		sigma_tr_arr[t] += math.sqrt(robots.sigma.trace()[0,0]/float(N)) /float(iteration)





file = open('ls_bda_output.txt', 'w')

for t in range(time_end):

	file.write( str(sigma_tr_arr[t]) + ', ' + str(error_arr[t]) + '\n')

file.close()

