"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import random
import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.stats import truncnorm
import csv
# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2
# EKF state covariance for three observations update
Cx_obs = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) ** 2

#  Simulation parameter
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 10.0  # maximum observation range
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]
OBS_SIZE = 3 # fixed number of observations for RL agent
show_animation = True


def ekf_slam(xEst, PEst, u, z, RFID):
    # Predict
    S = STATE_SIZE
    G, Fx = jacob_motion(xEst, u)
    xEst = motion_model(xEst, u)
    PEst = G.T @ PEst @ G + Fx.T @ Cx @ Fx
    # Update
    if z.shape[0] == OBS_SIZE:
        H = np.zeros((0,3))
        y = np.zeros((0,1))
        for iz in range(z.shape[0]):  # for each observation

            lm_id = int(z[iz, 2])
            lm = RFID[lm_id]
            y_part, H_part = calc_innovation(lm, xEst, PEst, z[iz, 0:2])
            H = np.vstack((H, H_part))
            y = np.vstack((y, y_part))

        S = H @ PEst @ H.T + Cx_obs
        K = (PEst @ H.T) @ np.linalg.inv(S)
        xEst = xEst + (K @ y)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst
    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst


def calc_input():
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def observation(xTrue, xd, u, RFID):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = np.zeros((0, 3))
    distances = np.zeros((0, 1))
    for i in range(len(RFID[:, 0])):

        dx = RFID[i, 0] - xTrue[0, 0]
        dy = RFID[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise
            angle_n = angle + np.random.randn() * Q_sim[1, 1] ** 0.5  # add noise
            zi = np.array([dn, angle_n, i])
            z = np.vstack((z, zi))
            distances = np.vstack((distances, d))

    distance_mask = np.argsort(distances[:,0])
    z = z[distance_mask[:3]]

    # add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = (F @ x) + (B @ u)
    return x




def jacob_motion(x, u):
    Fx = np.eye(STATE_SIZE)

    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)

    G = np.eye(STATE_SIZE) + Fx.T @ jF @ Fx

    return G, Fx,


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


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

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

def main():
    print(__file__ + " start!!")

    time = 0.0

    # landmark simulation
    num = 20
    radius = 10
    centerx, centery = 0, 10
    m_sim = np.diag([2.5, 2.5]) ** 2
    RFID = getRandomPointInCircle(num, radius, centerx, centery, m_sim)

    # RFID positions [x, y]
    # RFID = np.array([[10.0, -2.0],
    #                  [15.0, 10.0],
    #                  [3.0, 15.0],
    #                  [-5.0, 20.0]])

    # State Vector [x y yaw]'
    xEst = np.zeros((STATE_SIZE, 1)) + np.random.normal(0, 1, size=(STATE_SIZE, 1)) * 1 # add offset to initial state for EKF thread
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    xDR = xEst  # Dead reckoning
    xDR[2] = xTrue[2]
    #
    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID) # update xTrue, xDR; generate z, ud with noise
        xEst, PEst = ekf_slam(xEst, PEst, ud, z, RFID) # ekf prediction, correction

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            plt.plot(xEst[0], xEst[1], marker='o', markeredgecolor='r', markersize=np.pi * radius **2 / 2, linestyle='none', markerfacecolor='none')
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
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.pause(0.001)


if __name__ == '__main__':
    main()
