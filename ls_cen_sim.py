import numpy as np
import math

from ls_cen_team import LS_Cen_Team
import sim_env



num_of_trial = sim_env.num_of_trial
total_T = sim_env.total_T

# number of robot
N = sim_env.N 

# number of landmark
M = sim_env.M 

sigma_tr_arr = [0] * total_T
sigma_th_tr_arr = [0] * total_T
error_arr = [0] * total_T


for i in range(num_of_trial):

	print(i)

	initial = sim_env.initial_position


	robots = LS_Cen_Team(initial)


	landmarks = [None] * M
	for m in range(M):
		landmarks[m] = sim_env.Landmark(m, np.matrix(sim_env.landmark_position, dtype=float).getT())


	for t in range(total_T):

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
		
		# s = math.sqrt(s / float(N))
		error_arr[t] += math.sqrt(s/float(N)) / float(num_of_trial)


		# covariance error
		sigma_tr_arr[t] += math.sqrt(robots.sigma.trace()[0,0]/float(N)) /float(num_of_trial)

		sigma_th_tr_arr[t] += math.sqrt(robots.th_sigma.trace()[0,0]/float(N))/float(num_of_trial)




file = open('ls_cen_output.txt', 'w')

for t in range(total_T):

	file.write(str(sigma_tr_arr[t]) + ', ' + str(sigma_th_tr_arr[t]) + ', ' + str(error_arr[t]) + '\n')

file.close()

