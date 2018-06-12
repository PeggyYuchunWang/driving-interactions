import pickle
#import numpy as np
import subprocess
from pylab import *
import matplotlib.pyplot as plt

dt = 0.1

file = open('data/world0-1527286477.pickle', 'r')
#file = open('data/simple_simulation.pickle', 'r')

u, x = pickle.load(file)
uh, ur = u
xh, xr = x
t = arange(len(xh))*dt

xh = asarray(xh).T
xr = asarray(xr).T
uh = asarray(uh).T
ur = asarray(ur).T

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

#plt.plot(t, human_acceleration)
plt.plot(t, human_x)
plt.plot(t, robot_x)
plt.show()
print t.shape
print human_x.shape
