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
import scipy.io
from load_dataset import MRCLAM

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def calc_innovation(lm, xEst, PEst, z):

    delta = lm - np.concatenate(xEst[0:2], 0)
    q = (delta.T @ delta)
    z_angle = math.atan2(delta[1], delta[0]) - xEst[2, 0]
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])

    y = (z - zp).T
    # print("landmark:", lm)
    # print("actual measurement:", z)
    # print("predicted measurement:", zp)
    # print("current state:", xEst)
    # print("innovation", y)
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

    def __init__(self, offset, duration):

        self.offset = offset  # s
        self.duration = duration # s
        self.dt = 0.2

        self.STATE_SIZE = 3  # State size[x,y,yaw]
        self.LM_SIZE = 2  # LM state size[x,y]
        self.OBS_SIZE = 2 # fixed number of observations for RLagent
        self.show_animation = True

        # RLspace
        # Actionspace: 3x6 kalman gain K, Obsspace: 3x1 error state
        # x_k+1=f(x_k)+K(y_k+1-g(f(x_k))) where x~3x1, y~6x1, f~3x3, g~6x6
        obs_range = 1000
        action_range = 10 * 0.002
        self.action_space = spaces.Box(low=-action_range, high=action_range, shape=(12,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-obs_range, high=obs_range, shape=(2,), dtype=np.float64)

        self.seed()

        dataset_path = "dataset/" + "MRCLAM0.2.mat"
        robot_label = 5
        self.dataset = MRCLAM(dataset_path, robot_label)
        self.gt, self.measurement, self.odom, self.landmark, self.barcode = self.dataset.load_dataset(self.offset, self.duration, self.dt)

    def reset(self, eval=False):

        self.t = self.offset
        self.count = 0
        # adjustable simulation parameters
        # initial state
        # self.initial_bias = np.random.normal(0, 1, size=(self.STATE_SIZE, 1)) * 1.5
        self.initial_bias = np.random.uniform(-1, 1, size=(self.STATE_SIZE, 1)) * 2.5
        # bias and noise
        # EKF state covariance
        self.Cx = np.diag([1, 1, np.deg2rad(10.0)]) ** 2
        # self.Cx = np.diag([0.1, 0.1, np.deg2rad(6.0)]) ** 2

        # EKF measurement covariance for three observations update
        self.Cx_obs = np.diag([0.5, 0.5, 0.5, 0.5]) ** 2 *4800
        # self.Cx_obs = np.diag([0.1, 0.1, 0.1, 0.1]) ** 2 * 200

        # Simulation parameter:
        # Q_sim - noise added on landmark measurements (observations)
        # R_sim - noise added on input (velocity 1and angular velocity)

        self.Q_sim = np.diag([0.5, 0.3]) ** 2
        # self.Q_sim = np.diag([0.5, 0.3]) ** 2

        # input
        self.u = np.array([[self.odom[self.count][1], self.odom[self.count][2]]]).T
        # State Vector[x y yaw]'
        # add offset to initial state for EKF thread
        self.xTrue = np.array([[self.gt[self.count][1]], [self.gt[self.count][2]], [self.gt[self.count][3]]])
        self.xEst = self.xTrue + self.initial_bias

        # self.xEst = np.array([[3.59194869], [-6.44289717], [1.39450001]]) #for models/20210621_0008/PPO2_4.zip
        self.xEst = np.array([[-0.28552622], [-4.31925742], [3.05818529]])

        self.PEst = np.eye(self.STATE_SIZE)

        self.xDR = self.xEst  # Deadreckoning
        self.xDR[2] = self.xTrue[2]
        self.x_RL = self.xEst
        self.P_RL = np.eye(self.STATE_SIZE)

        hat_eta = np.random.normal(0, 1, size=(self.STATE_SIZE - 1)) * 0.00001
        return hat_eta

    def step(self, action):

        action = action.reshape(3, 4)
        # update new for next round
        self.t = round(self.t + self.dt, 1)
        self.count = self.count + 1
        # forward ground truth
        self.xTrue = np.array([[self.gt[self.count][1]], [self.gt[self.count][2]], [self.gt[self.count][3]]])

        # sample measurements with identity, forward dead reconking
        self.z, self.xDR = self.observation(self.xDR, self.u, self.measurement, self.t)  # update xTrue, xDR; generate z, ud with noise


        # EKF thread - for comparision
        # Predict
        G, Fx = self.jacob_motion(self.xEst, self.u)
        self.xEst = self.motion_model(self.xEst, self.u)
        self.xEst[2] = pi_2_pi(self.xEst[2])
        self.PEst = G.T @ self.PEst @ G + Fx.T @ self.Cx @ Fx

        # Update
        if self.z.shape[0] == self.OBS_SIZE:
            H = np.zeros((0, 3))
            y = np.zeros((0, 1))
            for iz in range(self.z.shape[0]):  # for each observation
                lm_id = int(self.z[iz, 2])
                # print(lm_id)
                lm = self.landmark[lm_id]
                # print("true state:", self.xTrue)
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
        G, Fx = self.jacob_motion(self.x_RL, self.u)
        self.x_RL = self.motion_model(self.x_RL, self.u)
        self.x_RL[2] = pi_2_pi(self.x_RL[2])
        self.P_RL = G.T @ self.P_RL @ G + Fx.T @ self.Cx @ Fx

        # Update
        if self.z.shape[0] == self.OBS_SIZE:
            H = np.zeros((0, 3))
            y = np.zeros((0, 1))
            for iz in range(self.z.shape[0]):  # for each observation

                lm_id = int(self.z[iz, 2])
                lm = self.landmark[lm_id]
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

        hat_eta = (self.x_RL - self.xTrue)[:2, 0]
        # cost function
        cost =  -math.sqrt((self.x_RL[0] - self.xTrue[0]) ** 2 + (self.x_RL[1] - self.xTrue[1]) ** 2)
        # print(cost)
        # if cost > (10) or self.t >= self.total_time:
        if cost < -15 or self.t >= (self.duration + self.offset - self.dt):
            done = True
        else:
            done = False


        # forward input
        self.u = np.array([[self.odom[self.count][1], self.odom[self.count][2]]]).T

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
        return x

    def jacob_motion(self, x, u):
        Fx = np.eye(self.STATE_SIZE)

        jF = np.array([[0.0, 0.0, -self.dt * u[0, 0] * math.sin(x[2, 0])],
                       [0.0, 0.0, self.dt * u[0, 0] * math.cos(x[2, 0])],
                       [0.0, 0.0, 0.0]], dtype=float)

        G = np.eye(self.STATE_SIZE) + Fx.T @ jF @ Fx

        return G, Fx

    def observation(self, xd, u, measurement, t):

        xd = self.motion_model(xd, u)
        z = np.zeros((0, 3))
        z_sim = np.zeros((0, 3))
        distances = np.zeros((0, 1))

        for i in range(len(self.landmark[:, 0])):
            dx = self.landmark[i, 0] - self.xTrue[0, 0]
            dy = self.landmark[i, 1] - self.xTrue[1, 0]
            d = math.hypot(dx, dy)
            angle = pi_2_pi(math.atan2(dy, dx) - self.xTrue[2, 0])
            zi = np.array([d, angle, i])
            z_sim = np.vstack((z_sim, zi))

        idx = np.where(measurement[:,0] == t)[0]
        if len(idx) >= self.OBS_SIZE:
            for i in idx:
                bc = int(self.measurement[i, 1])
                lm_id = [k for k, v in self.barcode.items() if v == bc][0] - 6
                if lm_id >=0 and lm_id != 10 and lm_id != 11 and lm_id != 5:
                    # if lm_id == 11:
                    #     lm_id = 5
                    # if lm_id == 5:
                    #     lm_id = 11
                    zi = np.array([self.measurement[i, 2], self.measurement[i, 3], lm_id])
                    z = np.vstack((z, zi))
                    diff = (zi[0] - z_sim[lm_id][0]) **2 + (zi[1] - z_sim[lm_id][1]) ** 2
                    distances = np.vstack((distances, diff))
                    # print(t)
                    # print(z_sim[lm_id])
                    # print(zi)
                    # print(diff)
            distance_mask = np.argsort(distances[:, 0])
            z = z[distance_mask[:self.OBS_SIZE]]

        else:
            if self.t % 2000 == 0:
            # if self.t < (self.offset + 30):
                for i in range(len(self.landmark[:, 0])):

                    dx = self.landmark[i, 0] - self.xTrue[0, 0]
                    dy = self.landmark[i, 1] - self.xTrue[1, 0]
                    d = math.hypot(dx, dy)
                    angle = pi_2_pi(math.atan2(dy, dx) - self.xTrue[2, 0])
                    dn = d + np.random.randn() * self.Q_sim[0, 0] ** 0.5  # add noise
                    angle_n = angle + np.random.randn() * self.Q_sim[1, 1] ** 0.5  # add noise
                    zi = np.array([dn, angle_n, i])
                    z = np.vstack((z, zi))
                    print(t)
                    print(zi)
                    distances = np.vstack((distances, d))
                distance_mask = np.argsort(distances[:, 0])

                z = z[distance_mask[:self.OBS_SIZE]]

        return z, xd

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    offset = 100 # s
    duration = 1400 # s
    env = RL_estimator(offset, duration)
    show_animation = True
    path = []
    # path2=[]
    t1 = []
    s = env.reset()
    hxEst = env.xEst
    hxTrue = env.xTrue
    hxDR = env.xTrue
    hxRL = env.x_RL
    for i in range(duration * 5 - 1):
        action = env.action_space.sample()
        hat_eta, cost, done, info = env.step(action)
        xTrue = info['xTrue']
        xEst = info["xEst"]
        xRL = info['xRL']
        cost_ekf = info['quant'][0]
        cost_rl = info['quant'][1]
        landmark = env.landmark
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

            plt.plot(landmark[:, 0], landmark[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            # plt.plot(xEst[0], xEst[1], marker='o', markeredgecolor='r', markersize=np.pi * 10 **2 / 2, linestyle='none', markerfacecolor='none')
            # plot landmark
            for i in range(z.shape[0]):
                plt.plot(landmark[int(z[i,2])][0],
                         landmark[int(z[i,2])][1], "xg")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b", label='Ground Truth')
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k", label='Dead Reconking')
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r", label='EKF Estimation')
            # plt.plot(hxRL[0, :],
            #          hxRL[1, :], "-g", label='RL Estimation')
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)

