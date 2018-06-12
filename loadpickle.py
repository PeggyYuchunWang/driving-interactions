#Peggy's code

import pickle
#import numpy as np
import subprocess
from pylab import *
import matplotlib.pyplot as plt

dt = 0.1

#file = open('data/world0-1527286477.pickle', 'r')
#file = open('data/roadfeatureremove.pickle', 'r')
#file = open('data/simple_simulation.pickle', 'r')
file = open('data/world10-1528091681.pickle', 'r') #human
file2 = open('data/world11-1528096528.pickle', 'r') #after IRL
file3 = open('data/world12-1528096387.pickle', 'r') #baseline
file4 = open('data/world_sample_5_traffic_test-1528493389.pickle')
file5 = open('data/world_sample_5_traffic-1528491902.pickle')
file6 = open('data/world3_traffic_test-1528495859.pickle')
file7 = open('data/world3_traffic-1528494667.pickle')
file8 = open('data/world3_traffic_baseline-1528528212.pickle')
file9 = open('data/world5_traffic_baseline-1528527719.pickle')

u, x = pickle.load(file5)
uh, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10 = u
xh, x1, x2, x3, x4, x5, x6, x8, x8, x9, x10 = x
t0 = arange(len(xh))*dt

u, x = pickle.load(file4)
ur, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10 = u
xr, x1, x2, x3, x4, x5, x6, x8, x8, x9, x10 = x
t1 = arange(len(xr))*dt

u, x = pickle.load(file9)
ub, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10 = u
xb, x1, x2, x3, x4, x5, x6, x8, x8, x9, x10 = x
t2 = arange(len(xb))*dt

xh = asarray(xh).T
xr = asarray(xr).T
uh = asarray(uh).T
ur = asarray(ur).T
xb = asarray(xb).T
ub = asarray(ub).T

human_x = xh[0]
human_y = xh[1]
human_heading = xh[2]
human_speed = xh[3]
human_steer = uh[0]
human_acceleration = uh[1]

robot_x = xr[0]
robot_y = xr[1]
robot_heading = xr[2]
robot_speed = xr[3]
robot_steer = ur[0]
robot_acceleration = ur[1]

base_x = xb[0]
base_y = xb[1]
base_heading = xb[2]
base_speed = xb[3]
base_steer = ub[0]
base_acceleration = ub[1]

#plt.plot(t, human_acceleration)
#plt.plot(t, human_x, color='blue', label = 'human_x')
# plt.plot(t, human_y, color='blue', label = 'human_y')
#plt.plot(t0, human_heading, color='blue', label = 'human_heading')
#plt.plot(t0, human_speed, color='blue', label = 'human_speed')
plt.plot(t0, human_steer, color='blue', label = 'human_steer')
#plt.plot(t0, human_acceleration, color='blue', label = 'human_acceleration')
#plt.plot(t, robot_x, color='green', label = 'robot_x')
# plt.plot(t, robot_y, color='green', label = 'robot_y')
# plt.plot(t1, robot_acceleration, color='green', label = 'robot_acceleration')
# plt.plot(t2, base_acceleration, color='red', label = 'base_acceleration')
# plt.plot(t1, robot_heading, color='green', label = 'robot_heading')
# plt.plot(t2, base_heading, color='red', label = 'base_heading')
# plt.plot(t1, robot_speed, color='green', label = 'robot_speed')
# plt.plot(t2, base_speed, color='red', label = 'baseline_speed')
plt.plot(t1, robot_steer, color='green', label = 'robot_steer')
plt.plot(t2, base_steer, color='red', label = 'base_steer')
# plt.plot(human_x, human_y, color = 'blue', label = 'human_trajectory')
# plt.plot(robot_x, robot_y, color = 'green', label = 'robot_trajectory')
# plt.plot(base_x, base_y, color = 'red', label = 'base_trajectory')

plt.legend()
#plt.xlabel('time step')
plt.show()
