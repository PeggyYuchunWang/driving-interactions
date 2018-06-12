import lane
import car
import math
import feature
import dynamics
import visualize
import utils
import sys
import theano as th
import theano.tensor as tt
import numpy as np
import shelve

th.config.optimizer_verbose = False
th.config.allow_gc = False
th.config.optimizer = 'fast_compile'

class Object(object):
    def __init__(self, name, x):
        self.name = name
        self.x = np.asarray(x)

class World(object):
    def __init__(self):
        self.cars = []
        self.lanes = []
        self.roads = []
        self.fences = []
        self.objects = []
    def simple_reward(self, trajs=None, lanes=None, roads=None, fences=None, speed=1., speed_import=1.):
        if lanes is None:
            lanes = self.lanes
        if roads is None:
            roads = self.roads
        if fences is None:
            fences = self.fences
        if trajs is None:
            trajs = [c.linear for c in self.cars]
        elif isinstance(trajs, car.Car):
            trajs = [c.linear for c in self.cars if c!=trajs]
        r = 0.1*feature.control()
        theta = [1., 50., 10., 10., 60.] # Simple model
        for lane in lanes:
            r = r+theta[0]*lane.gaussian()
        for fence in fences:
            r = r+theta[1]*fence.gaussian()
        for road in roads:
            r = r+theta[2]*road.gaussian(10.)
        if speed is not None:
            r = r+speed_import*theta[3]*feature.speed(speed)
        for traj in trajs:
            r = r+theta[4]*traj.gaussian()
        return r

    def test_reward(self, trajs=None, lanes=None, roads=None, fences=None, speed=1., speed_import=1.):
        if lanes is None:
            lanes = self.lanes
        if roads is None:
            roads = self.roads
        if fences is None:
            fences = self.fences
        if trajs is None:
            trajs = [c.linear for c in self.cars]
        elif isinstance(trajs, car.Car):
            trajs = [c.linear for c in self.cars if c!=trajs]
        r = 0.1*feature.control()
        theta = [1., 50., 10., 500., -60.]
        # for lane in lanes:
        #     r = r+theta[0]*lane.gaussian()
        # for fence in fences:
        #     r = r+theta[1]*fence.gaussian()
        # for road in roads:
        #     r = r+theta[2]*road.gaussian(10.)
        # if speed is not None:
        #     r = r+speed_import*theta[3]*feature.speed(speed)
        # for traj in trajs:
        #     r = r+theta[4]*traj.gaussian()
        # return r

        for lane in lanes:
            r = r+theta[0]*lane.gaussian()
        for fence in fences:
            r = r+theta[1]*fence.gaussian()
        r = r+theta[2]*feature.distanceTravelled()
        if speed is not None:
            r = r+speed_import*theta[3]*feature.speed(speed)
        for traj in trajs:
            r = r+theta[4]*traj.gaussian()
        return r

    def human_reward(self, trajs=None, lanes=None, roads=None, fences=None, speed=1., speed_import=1.):
        if lanes is None:
            lanes = self.lanes
        if roads is None:
            roads = self.roads
        if fences is None:
            fences = self.fences
        if trajs is None:
            trajs = [c.linear for c in self.cars]
        elif isinstance(trajs, car.Car):
            trajs = [c.linear for c in self.cars if c!=trajs]
        r = 0.1*feature.control()
        # theta = [207.45344712, -43.22037785, 33.0465513, 0.85747694, -60.04716053] # Learned Weights - A1
        # theta = [9.24206007e+05, -5.00007549e+01, 5.15248892e+04, 3.70347822e+03, -2.11667599e+05] # Learned Weights - A2
        # theta = [3.92719120e+08, -6.71421199e+01, 1.81421557e+06, 4.05569065e+05, -2.52208233e+05] # Learned Weights - A3
        # theta = [35533.99846081, -4737.15384395, -10539.6647274, 918.3771547, -227.922436] # Learned Weights - A4
        # theta = [0.64638118, -23.84898939, 6.15886934, 0.56278226, -60.70986132] # Learned Weights - A5
        theta = [98.23497628, 8.26946595, 9.40316899, 0.56561252, -12.73674575] # Learned Weights - A6

        for lane in lanes:
            r = r+theta[0]*lane.gaussian()
        for fence in fences:
            r = r+theta[1]*fence.gaussian()
        for road in roads:
            r = r+theta[2]*road.gaussian(10.)
        if speed is not None:
            r = r+speed_import*theta[3]*feature.speed(speed)
        for traj in trajs:
            r = r+theta[4]*traj.gaussian()
        return r

def world3():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.15)

    world.lanes  += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads  += [clane, clane.shifted(1), clane.shifted(-1)]
    world.fences += [clane.shifted(3), clane.shifted(-3), clane.shifted(4), clane.shifted(-4)]

    world.cars.append(car.UserControlledCar(dyn, [0.15, 0., math.pi/2., 0.3], color='blue'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, -0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, 0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.3, math.pi/2., 0.3], color='red'))

    world.cars[0].human = world.cars[0]
    world.cars[1].reward = world.test_reward(world.cars[1], speed=0.75)
    world.cars[2].reward = world.test_reward(world.cars[2], speed=0.75)
    world.cars[3].reward = world.test_reward(world.cars[3], speed=0.75)
    world.cars[4].reward = world.test_reward(world.cars[4], speed=0.85)
    world.cars[5].reward = world.test_reward(world.cars[5], speed=0.80)
    
    return world

def world3_traffic():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.15)
    fence = lane.StraightLane([0., -1.], [0., 1.], 0.30)

    world.lanes  += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads  += [clane, clane.shifted(1), clane.shifted(-1)]
    # world.fences += [clane.shifted(3), clane.shifted(-3), clane.shifted(4), clane.shifted(-4), clane.shifted(5), clane.shifted(-5)]
    world.fences += [clane.shifted(3), clane.shifted(-3), fence.shifted(2), fence.shifted(-2), fence.shifted(3), fence.shifted(-3)]

    world.cars.append(car.UserControlledCar(dyn, [0.15, 0., math.pi/2., 0.3], color='blue'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0.6, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, 0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.3, math.pi/2., 0.3], color='red'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, -0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.6, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, -0.3, math.pi/2., 0.3], color='red'))

    world.cars[0].human = world.cars[0]
    world.cars[1].reward = world.test_reward(world.cars[1], speed=0.75)
    world.cars[2].reward = world.test_reward(world.cars[2], speed=0.75)
    world.cars[3].reward = world.test_reward(world.cars[3], speed=0.75)
    world.cars[4].reward = world.test_reward(world.cars[4], speed=0.85)
    world.cars[5].reward = world.test_reward(world.cars[5], speed=0.80)

    world.cars[6].reward = world.test_reward(world.cars[6], speed=0.80)
    world.cars[7].reward = world.test_reward(world.cars[7], speed=0.70)
    world.cars[8].reward = world.test_reward(world.cars[8], speed=0.85)
    world.cars[9].reward = world.test_reward(world.cars[9], speed=0.85)
    world.cars[10].reward = world.test_reward(world.cars[10], speed=0.80)
    
    return world

def world3_traffic_test():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.15)

    world.lanes  += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads  += [clane, clane.shifted(1), clane.shifted(-1)]
    world.fences += [clane.shifted(3), clane.shifted(-3), clane.shifted(4), clane.shifted(-4)]

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='blue'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0.6, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, 0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.3, math.pi/2., 0.3], color='red'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, -0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.6, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, -0.3, math.pi/2., 0.3], color='red'))

    world.cars[0].reward = world.human_reward(world.cars[0], speed=0.75)
    world.cars[1].reward = world.test_reward(world.cars[1], speed=0.75)
    world.cars[2].reward = world.test_reward(world.cars[2], speed=0.75)
    world.cars[3].reward = world.test_reward(world.cars[3], speed=0.75)
    world.cars[4].reward = world.test_reward(world.cars[4], speed=0.85)
    world.cars[5].reward = world.test_reward(world.cars[5], speed=0.80)

    world.cars[6].reward = world.test_reward(world.cars[6], speed=0.80)
    world.cars[7].reward = world.test_reward(world.cars[7], speed=0.70)
    world.cars[8].reward = world.test_reward(world.cars[8], speed=0.85)
    world.cars[9].reward = world.test_reward(world.cars[9], speed=0.85)
    world.cars[10].reward = world.test_reward(world.cars[10], speed=0.80)
    
    return world

def world3_traffic_baseline():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.15)

    world.lanes  += [clane, clane.shifted(1), clane.shifted(-1)]
    world.roads  += [clane, clane.shifted(1), clane.shifted(-1)]
    world.fences += [clane.shifted(3), clane.shifted(-3), clane.shifted(4), clane.shifted(-4)]

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='blue'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0.6, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, 0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.3, math.pi/2., 0.3], color='red'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, -0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.6, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, -0.3, math.pi/2., 0.3], color='red'))

    # world.cars[0].reward = world.human_reward(world.cars[0], speed=0.75)
    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.75)
    world.cars[1].reward = world.test_reward(world.cars[1], speed=0.75)
    world.cars[2].reward = world.test_reward(world.cars[2], speed=0.75)
    world.cars[3].reward = world.test_reward(world.cars[3], speed=0.75)
    world.cars[4].reward = world.test_reward(world.cars[4], speed=0.85)
    world.cars[5].reward = world.test_reward(world.cars[5], speed=0.80)

    world.cars[6].reward = world.test_reward(world.cars[6], speed=0.80)
    world.cars[7].reward = world.test_reward(world.cars[7], speed=0.70)
    world.cars[8].reward = world.test_reward(world.cars[8], speed=0.85)
    world.cars[9].reward = world.test_reward(world.cars[9], speed=0.85)
    world.cars[10].reward = world.test_reward(world.cars[10], speed=0.80)
    
    return world

def world5():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.15)

    world.lanes  += [clane, clane.shifted(1), clane.shifted(-1), clane.shifted(2), clane.shifted(-2)]
    world.roads  += [clane, clane.shifted(1), clane.shifted(-1), clane.shifted(2), clane.shifted(-2)]
    world.fences += [clane.shifted(3), clane.shifted(-3), clane.shifted(4), clane.shifted(-4)]

    world.cars.append(car.UserControlledCar(dyn, [0.15, 0., math.pi/2., 0.3], color='blue'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, -0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.3, 0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.3, -0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.3, math.pi/2., 0.3], color='red'))

    world.cars[0].human = world.cars[0]
    world.cars[1].reward = world.test_reward(world.cars[1], speed=0.75)
    world.cars[2].reward = world.test_reward(world.cars[2], speed=0.75)
    world.cars[3].reward = world.test_reward(world.cars[3], speed=0.75)
    world.cars[4].reward = world.test_reward(world.cars[4], speed=0.85)
    world.cars[5].reward = world.test_reward(world.cars[5], speed=0.80)
    
    return world

def world5_traffic():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.15)

    world.lanes  += [clane, clane.shifted(1), clane.shifted(-1), clane.shifted(2), clane.shifted(-2)]
    world.roads  += [clane, clane.shifted(1), clane.shifted(-1), clane.shifted(2), clane.shifted(-2)]
    world.fences += [clane.shifted(3), clane.shifted(-3), clane.shifted(4), clane.shifted(-4)]

    world.cars.append(car.UserControlledCar(dyn, [0.15, 0., math.pi/2., 0.3], color='blue'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, -0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, 0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.3, math.pi/2., 0.3], color='red'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.3, -0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.3, 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.6, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.3, -0.3, math.pi/2., 0.3], color='red'))

    world.cars[0].human = world.cars[0]
    world.cars[1].reward = world.test_reward(world.cars[1], speed=0.75)
    world.cars[2].reward = world.test_reward(world.cars[2], speed=0.75)
    world.cars[3].reward = world.test_reward(world.cars[3], speed=0.75)
    world.cars[4].reward = world.test_reward(world.cars[4], speed=0.85)
    world.cars[5].reward = world.test_reward(world.cars[5], speed=0.80)

    world.cars[6].reward = world.test_reward(world.cars[6], speed=0.80)
    world.cars[7].reward = world.test_reward(world.cars[7], speed=0.70)
    world.cars[8].reward = world.test_reward(world.cars[8], speed=0.85)
    world.cars[9].reward = world.test_reward(world.cars[9], speed=0.85)
    world.cars[10].reward = world.test_reward(world.cars[10], speed=0.80)
    
    return world

def world5_traffic_test():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.15)

    world.lanes  += [clane, clane.shifted(1), clane.shifted(-1), clane.shifted(2), clane.shifted(-2)]
    world.roads  += [clane, clane.shifted(1), clane.shifted(-1), clane.shifted(2), clane.shifted(-2)]
    world.fences += [clane.shifted(3), clane.shifted(-3), clane.shifted(4), clane.shifted(-4)]

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='blue'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, -0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, 0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.3, math.pi/2., 0.3], color='red'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.3, -0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.3, 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.6, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.3, -0.3, math.pi/2., 0.3], color='red'))

    world.cars[0].reward = world.human_reward(world.cars[0], speed=0.75)
    world.cars[1].reward = world.test_reward(world.cars[1], speed=0.75)
    world.cars[2].reward = world.test_reward(world.cars[2], speed=0.75)
    world.cars[3].reward = world.test_reward(world.cars[3], speed=0.75)
    world.cars[4].reward = world.test_reward(world.cars[4], speed=0.85)
    world.cars[5].reward = world.test_reward(world.cars[5], speed=0.80)

    world.cars[6].reward = world.test_reward(world.cars[6], speed=0.80)
    world.cars[7].reward = world.test_reward(world.cars[7], speed=0.70)
    world.cars[8].reward = world.test_reward(world.cars[8], speed=0.85)
    world.cars[9].reward = world.test_reward(world.cars[9], speed=0.85)
    world.cars[10].reward = world.test_reward(world.cars[10], speed=0.80)
    
    return world

def world5_traffic_baseline():
    dyn = dynamics.CarDynamics(0.1)
    world = World()
    clane = lane.StraightLane([0., -1.], [0., 1.], 0.15)

    world.lanes  += [clane, clane.shifted(1), clane.shifted(-1), clane.shifted(2), clane.shifted(-2)]
    world.roads  += [clane, clane.shifted(1), clane.shifted(-1), clane.shifted(2), clane.shifted(-2)]
    world.fences += [clane.shifted(3), clane.shifted(-3), clane.shifted(4), clane.shifted(-4)]

    # world.cars.append(car.UserControlledCar(dyn, [0.15, 0., math.pi/2., 0.3], color='blue'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='blue'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, -0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.15, 0.2, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., -0.3, math.pi/2., 0.3], color='red'))

    world.cars.append(car.SimpleOptimizerCar(dyn, [0.15, 0., math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.3, -0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [-0.3, 0.3, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.6, math.pi/2., 0.3], color='red'))
    world.cars.append(car.SimpleOptimizerCar(dyn, [0.3, -0.3, math.pi/2., 0.3], color='red'))

    # world.cars[0].reward = world.human_reward(world.cars[0], speed=0.75)
    world.cars[0].reward = world.simple_reward(world.cars[0], speed=0.75)
    world.cars[1].reward = world.test_reward(world.cars[1], speed=0.75)
    world.cars[2].reward = world.test_reward(world.cars[2], speed=0.75)
    world.cars[3].reward = world.test_reward(world.cars[3], speed=0.75)
    world.cars[4].reward = world.test_reward(world.cars[4], speed=0.85)
    world.cars[5].reward = world.test_reward(world.cars[5], speed=0.80)

    world.cars[6].reward = world.test_reward(world.cars[6], speed=0.80)
    world.cars[7].reward = world.test_reward(world.cars[7], speed=0.70)
    world.cars[8].reward = world.test_reward(world.cars[8], speed=0.85)
    world.cars[9].reward = world.test_reward(world.cars[9], speed=0.85)
    world.cars[10].reward = world.test_reward(world.cars[10], speed=0.80)
    
    return world
