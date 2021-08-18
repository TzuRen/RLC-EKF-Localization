"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import random

# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

#  Simulation parameter
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 10.0  # maximum observation range
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = True


def ekf_slam(xEst, PEst, u, z, RFID):
    # Predict
    S = STATE_SIZE
    G, Fx = jacob_motion(xEst[0:S], u)
    xEst[0:S] = motion_model(xEst[0:S], u)
    PEst[0:S, 0:S] = G.T @ PEst[0:S, 0:S] @ G + Fx.T @ Cx @ Fx
    initP = np.eye(2)
    # Update
    for iz in range(len(z[:, 0])):  # for each observation

        lm_id = int(z[iz, 2])
        lm = RFID[lm_id]
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:2])

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


def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_motion(x, u):
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
        (STATE_SIZE, LM_SIZE * calc_n_lm(x)))))

    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)

    G = np.eye(STATE_SIZE) + Fx.T @ jF @ Fx

    return G, Fx,


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))

    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

    return zp


def calc_innovation(lm, xEst, PEst, z):

    delta = lm - np.concatenate(xEst[0:2], 0)
    q = (delta.T @ delta)
    z_angle = math.atan2(delta[1], delta[0]) - xEst[2, 0]
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])

    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacob_h(q, delta)
    S = H @ PEst @ H.T + Cx[0:2, 0:2]
    return y, S, H


def jacob_h(q, delta):
    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0], - sq * delta[1], 0],
                  [delta[1], - delta[0], - q]])
    G = G / q

    return G


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
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID) # update xTrue, xDR; generate z, ud with noise
        print(z.shape[0])
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
                     hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()
