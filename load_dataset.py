import scipy.io
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


class MRCLAM:
    def __init__(self, dataset_path, robot_label):
        self.dataset_path = dataset_path
        self.robot_label = robot_label

    def load_dataset(self, offset, duration, dt):
        start_idx = int(offset / dt)
        end_idx = int((offset + duration) / dt)
        mat = scipy.io.loadmat(self.dataset_path)
        landmark = mat["Landmark_Groundtruth"][:,1:3]
        barcode = dict(mat["Barcodes"])
        gt = mat["Robot" + str(self.robot_label) + "_Groundtruth"][start_idx:end_idx]
        measurement = mat["Robot" + str(self.robot_label) + "_Measurement"]
        measurement_time = measurement[:, 0]
        mask = np.where((measurement_time >= offset) & (measurement_time <= (offset + duration)))
        measurement = measurement[mask, :][0]
        odom = mat["Robot" + str(self.robot_label) + "_Odometry"][start_idx:end_idx]
        return gt, measurement, odom, landmark, barcode

if __name__ == '__main__':
    offset = 0  # s
    dt = 0.2  # s
    duration = 1500  # s
    dataset_path = "dataset/" + "MRCLAM0.2.mat"
    robot_label = 2
    dataset = MRCLAM(dataset_path, robot_label)
    print ('******** Initialization Started ********')
    gt, measurement, odom, landmark, barcode = dataset.load_dataset(offset, duration, dt)
    print('******** Finished loading ********')

    # visualize input
    fig1 = plt.figure(figsize = (6,5))
    ax1 = plt.subplot(2,1,1)
    plt.plot(odom[:,0], odom[:,1])
    ax1.set_title("veloocity m/s")
    ax2 = plt.subplot(2,1,2)
    plt.plot(odom[:,0], odom[:,2])
    ax2.set_title("angular velocity rad/s")
    plt.show()


    fig2 = plt.figure(figsize=(9, 6), num="visualization path")
    plt.scatter(gt[0, 1], gt[1, 2], s=200, marker="*", c="y", label="Start")
    plt.plot(landmark[:, 0], landmark[:, 1], "*k", label='Landmarks')
    plt.plot(gt[:, 1],gt[:, 2], "-b", label='Ground Truth')
    # plt.plot(xDR_mean_path[0, :],
    #          xDR_mean_path[1, :], "-k", label='Dead Reconking')
    # plt.plot(xEst_mean_path[0, :],
    #          xEst_mean_path[1, :], "-r", label='EKF Estimation')
    # plt.plot(xRL_mean_path[0, :],
    #          xRL_mean_path[1, :], "-g", label='RL Estimation')
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

