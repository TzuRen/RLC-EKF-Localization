# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:09:14 2020

@author: tang
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.stats import truncnorm
import csv
import matplotlib.pyplot as plt
from itertools import cycle

def pi_2_pi(angle):
    # if angle > math.pi:
    #     return angle - 2*math.pi
    # else:
    return (angle + math.pi) % (2 * math.pi) - math.pi

def calc_innovation(lm, xEst, PEst, z):

    delta = lm - np.concatenate(xEst[0:2], 0)
    q = (delta.T @ delta)
    z_angle = math.atan2(delta[1], delta[0]) - xEst[2, 0]
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])

    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacob_h(q, delta)
    return y, H


def jacob_h(q, delta):
    sq = math.sqrt(q)
    H = np.array([[-sq * delta[0], - sq * delta[1], 0],
                  [delta[1], - delta[0], - q]])
    H = H / q

    return H

def getRandomPointInCircle(num, radius, centerx, centery, noise):
    samplePoint = np.zeros((0, 2))
    theta = 0
    d_theta = 2 * np.pi / num
    for i in range(num):
        theta += d_theta
        r = radius ** 2
        x = math.cos(theta) * (r ** 0.5) + centerx + np.random.randn() * noise[0, 0] ** 0.5
        y = math.sin(theta) * (r ** 0.5) + centery + + np.random.randn() * noise[1, 1] ** 0.5
        point = np.array([x, y])
        samplePoint = np.vstack((samplePoint, point))

    return samplePoint


class RL_estimator(gym.Env):

    def __init__(self, T):

        self.t = 0
        self.dt = 0.1
        self.total_time = T

        self.MAX_RANGE = 10.0  # maximum observation range
        self.STATE_SIZE = 3  # State size[x,y,yaw]
        self.LM_SIZE = 2  # LM state size[x,y]
        self.OBS_SIZE = 3  # fixed number of observations for RLagent
        self.show_animation = True

        # RLspace
        # Actionspace: 3x6 kalman gain K, Obsspace: 3x1 error state
        # x_k+1=f(x_k)+K(y_k+1-g(f(x_k))) where x~3x1, y~6x1, f~3x3, g~6x6
        obs_range = 1000
        action_range = 10 * 0.002
        self.action_space = spaces.Box(low=-action_range, high=action_range, shape=(18,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-obs_range, high=obs_range, shape=(2,), dtype=np.float64)

        self.seed()

    def reset(self, eval=False):

        self.t = 0
        # adjustable simulation parameters
        # initial state
        # self.initial_bias = np.random.normal(0, 1, size=(self.STATE_SIZE, 1)) * 1.5
        self.initial_bias = np.random.uniform(-1, 1, size=(self.STATE_SIZE, 1)) * 3
        # bias and noise
        # EKF state covariance
        self.Cx = np.diag([1, 1, np.deg2rad(30.0)]) ** 2
        # self.Cx = np.diag([0.1, 0.1, np.deg2rad(6.0)]) ** 2

        # EKF measurement covariance for three observations update
        self.Cx_obs = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) ** 2
        # self.Cx_obs = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) ** 2

        # Simulation parameter:
        # Q_sim - noise added on landmark measurements (observations)
        # R_sim - noise added on input (velocity and angular velocity)

        self.Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2 * 0.1
        # self.Q_sim = np.diag([0.1, np.deg2rad(0.5)]) ** 2
        self.R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2 * 0.1
        # self.R_sim = np.diag([0.1, n        p.deg2rad(1.0)]) ** 2

        # input
        v = 1.0  # [m/s]
        yaw_rate = 0.1  # [rad/s]
        self.u = np.array([[v, yaw_rate]]).T

        # State Vector[x y yaw]'
        # add offset to initial state for EKF thread
        self.xEst = np.zeros((self.STATE_SIZE, 1)) + self.initial_bias
        # self.xEst = np.array([[-3.09316017], [-3.07106977], [4.02412283]])
        # print(self.xEst)
        # self.xEst = np.array([[-9.6179116], [3.60767583], [-1.53236842]])
        # self.xEst = np.array([[6.11110985], [-3.75781535], [8.49916211]])
        # print(self.xEst)
        # self.xEst = np.array([[-10], [10], [0]])
        # self.xEst = np.array([[-20], [20], [0]])
        self.xTrue = np.zeros((self.STATE_SIZE, 1))
        self.PEst = np.eye(self.STATE_SIZE)

        self.xDR = self.xEst  # Deadreckoning
        self.xDR[2] = self.xTrue[2]
        self.x_RL = self.xEst
        self.P_RL = np.eye(self.STATE_SIZE)

        # groundtruth landmark generation
        num = 20
        radius = 10
        centerx, centery = 0, 10
        m_sim = np.diag([2.5, 2.5]) ** 2
        self.RFID = getRandomPointInCircle(num, radius, centerx, centery, m_sim)

        hat_eta = np.random.normal(0, 1, size=(self.STATE_SIZE - 1)) * 0.00001
        return hat_eta

    def step(self, action):
        action = action.reshape(3, 6)

        # data simulation
        self.xTrue, self.z, self.xDR, ud = self.observation(self.xTrue, self.xDR, self.u, self.RFID)  # update xTrue, xDR; generate z, ud with noise
        # EKF thread - for comparision
        # Predict
        G, Fx = self.jacob_motion(self.xEst, ud)
        self.xEst = self.motion_model(self.xEst, ud)
        self.xEst[2] = pi_2_pi(self.xEst[2])
        self.PEst = G.T @ self.PEst @ G + Fx.T @ self.Cx @ Fx

        # Update
        if self.z.shape[0] == self.OBS_SIZE:
            H = np.zeros((0, 3))
            y = np.zeros((0, 1))
            for iz in range(self.z.shape[0]):  # for each observation

                lm_id = int(self.z[iz, 2])
                lm = self.RFID[lm_id]
                y_part, H_part = calc_innovation(lm, self.xEst, self.PEst, self.z[iz, 0:2])
                H = np.vstack((H, H_part))
                y = np.vstack((y, y_part))

            S = H @ self.PEst @ H.T + self.Cx_obs
            K = (self.PEst @ H.T) @ np.linalg.inv(S)
            self.xEst = self.xEst + (K @ y)
            self.PEst = (np.eye(len(self.xEst)) - (K @ H)) @ self.PEst
            self.xEst[2] = pi_2_pi(self.xEst[2])

        # RL thread
        # Predict
        G, Fx = self.jacob_motion(self.x_RL, ud)
        self.x_RL = self.motion_model(self.x_RL, ud)
        self.x_RL[2] = pi_2_pi(self.x_RL[2])
        self.P_RL = G.T @ self.P_RL @ G + Fx.T @ self.Cx @ Fx

        # Update
        if self.z.shape[0] == self.OBS_SIZE:
            H = np.zeros((0, 3))
            y = np.zeros((0, 1))
            for iz in range(self.z.shape[0]):  # for each observation

                lm_id = int(self.z[iz, 2])
                lm = self.RFID[lm_id]
                y_part, H_part = calc_innovation(lm, self.x_RL, self.P_RL, self.z[iz, 0:2])
                H = np.vstack((H, H_part))
                y = np.vstack((y, y_part))

            S = H @ self.P_RL @ H.T + self.Cx_obs
            K = (self.P_RL @ H.T) @ np.linalg.inv(S)
            self.x_RL = self.x_RL + (K @ y)
            self.P_RL = (np.eye(len(self.x_RL)) - (K @ H)) @ self.P_RL
            self.x_RL[2] = pi_2_pi(self.x_RL[2])

        # combine rl action
            self.x_RL = self.x_RL + (action @ y)
            self.x_RL[2] = pi_2_pi(self.x_RL[2])

        hat_eta = K @ y
        # cost function
        cost =  -math.sqrt((self.x_RL[0] - self.xTrue[0]) ** 2 + (self.x_RL[1] - self.xTrue[1]) ** 2)
        # if cost > (10) or self.t >= self.total_time:
        if cost < -22 or self.t >= self.total_time:
            done = True
        else:
            done = False

        # update new for next round
        self.t = self.t + self.dt

        # for quantitative results
        cost_ekf = math.sqrt((self.xEst[0] - self.xTrue[0]) ** 2 + (self.xEst[1] - self.xTrue[1]) ** 2)
        cost_rl = math.sqrt((self.x_RL[0] - self.xTrue[0]) ** 2 + (self.x_RL[1] - self.xTrue[1]) ** 2)
        # print(np.array([cost_ekf, cost_rl]))
        info = dict(xTrue= self.xTrue, xDR = self.xDR, xEst=self.xEst, xRL= self.x_RL, quant= np.array([cost_ekf, cost_rl]))
        return hat_eta, cost, done, info

    def motion_model(self, x, u):
        F = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])

        B = np.array([[self.dt * math.cos(x[2, 0]), 0],
                      [self.dt * math.sin(x[2, 0]), 0],
                      [0.0, self.dt]])

        x = (F @ x) + (B @ u)
        x[2] = pi_2_pi(x[2])
        return x

    def jacob_motion(self, x, u):
        Fx = np.eye(self.STATE_SIZE)

        jF = np.array([[0.0, 0.0, -self.dt * u[0, 0] * math.sin(x[2, 0])],
                       [0.0, 0.0, self.dt * u[0, 0] * math.cos(x[2, 0])],
                       [0.0, 0.0, 0.0]], dtype=float)

        G = np.eye(self.STATE_SIZE) + Fx.T @ jF @ Fx

        return G, Fx

    def observation(self, xTrue, xd, u, RFID):
        xTrue = self.motion_model(xTrue, u)

        # add noise to gps x-y
        z = np.zeros((0, 3))
        distances = np.zeros((0, 1))
        for i in range(len(RFID[:, 0])):

            dx = RFID[i, 0] - xTrue[0, 0]
            dy = RFID[i, 1] - xTrue[1, 0]
            d = math.hypot(dx, dy)
            angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
            if d <= self.MAX_RANGE:
                dn = d + np.random.randn() * self.Q_sim[0, 0] ** 0.5  # add noise
                angle_n = angle + np.random.randn() * self.Q_sim[1, 1] ** 0.5  # add noise
                zi = np.array([dn, angle_n, i])
                z = np.vstack((z, zi))
                distances = np.vstack((distances, d))

        distance_mask = np.argsort(distances[:, 0])
        z = z[distance_mask[:3]]

        # add noise to input
        ud = np.array([[
            u[0, 0] + np.random.randn() * self.R_sim[0, 0] ** 0.5,
            u[1, 0] + np.random.randn() * self.R_sim[1, 1] ** 0.5]]).T

        xd = self.motion_model(xd, ud)
        return xTrue, z, xd, ud

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    T = 30 #s
    env = RL_estimator(T)
    show_animation = True
    path = []
    # path2=[]
    t1 = []
    s = env.reset()
    hxEst = env.xEst
    hxTrue = env.xTrue
    hxDR = env.xTrue
    hxRL = env.x_RL
    for i in range(int(T / env.dt)):
        action = env.action_space.sample()
        hat_eta, cost, done, info = env.step(action)
        xTrue = info['xTrue']
        xEst = info["xEst"]
        xRL = info['xRL']
        cost_ekf = info['quant'][0]
        cost_rl = info['quant'][1]
        RFID = env.RFID
        z = env.z
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, env.xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hxRL = np.hstack((hxRL, xRL))
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            plt.plot(xEst[0], xEst[1], marker='o', markeredgecolor='r', markersize=np.pi * 10 **2 / 2, linestyle='none', markerfacecolor='none')
            # plot landmark
            for i in range(z.shape[0]):
                plt.plot(RFID[int(z[i,2])][0],
                         RFID[int(z[i,2])][1], "xg")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b", label='Ground Truth')
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k", label='Dead Reconking')
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r", label='EKF Estimation')
            plt.plot(hxRL[0, :],
                     hxRL[1, :], "-g", label='RL Estimation')
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)

