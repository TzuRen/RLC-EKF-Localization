import stable_baselines
from RLestimator_ekf import *
import tensorflow as tf
from stable_baselines import PPO2, SAC
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import make_vec_env
import os
from itertools import cycle
import time

train = False
inference = True
model_num = 5
n_steps = 500
eposides = 300
total_timesteps_ = eposides * n_steps
algorithm = "PPO2"
ENV_NAME = "RLestimator"
model_dir = "./models/" + time.strftime("%Y%m%d_%H%M") + "/"
tensorboard_log_dir = "./logs/" + ENV_NAME + algorithm + time.strftime("%Y%m%d_%H%M") + "/"
# tensorboard --logdir=PPO2_Pendulum_v0_0_2 --port=6006 --host=127.0.0.1
T = 50 #s
env = RL_estimator(T)
env = make_vec_env(lambda: env, n_envs=1)
model = PPO2(MlpPolicy, env, n_steps=128, verbose=1, tensorboard_log=tensorboard_log_dir)

if train:
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    for i in range(model_num):
        model.learn(total_timesteps=total_timesteps_, tb_log_name=algorithm + "_" + str(i))
        model.save(model_dir + algorithm + "_" + str(i))

del model # remove to demonstrate saving and loading

if inference:
    model = PPO2.load("models/boost_function_initial10_low_measurement_noise_ac0.002/best2")
    # Enjoy trained agent
    num_of_paths = 50
    max_ep_steps = 500
    # save_figs = True
    show_animation = False
    LOG_PATH = "./logs"

    if show_animation is True:
        for i in range(num_of_paths):
            s = env.reset()
            RFID = env.get_attr("RFID")[0]
            hxEst = env.get_attr("xEst")[0]
            hxTrue = env.get_attr("xTrue")[0]
            hxDR = env.get_attr("xDR")[0]
            hxRL = env.get_attr("x_RL")[0]
            for j in range(max_ep_steps):
                action, _states = model.predict(s)
                s_, rewards, dones, infos = env.step(action)
                info = infos[0]
                xTrue = info['xTrue']
                xEst = info["xEst"]
                xRL = info['xRL']

                z = env.get_attr("z")[0]
                hxEst = np.hstack((hxEst, np.array(xEst)))
                hxDR = np.hstack((hxDR, env.get_attr("xDR")[0]))
                hxTrue = np.hstack((hxTrue, np.array(xTrue)))
                hxRL = np.hstack((hxRL, np.array(xRL)))

                # animate the states in real time
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                plt.plot(RFID[:, 0], RFID[:, 1], "*k")
                plt.plot(xEst[0], xEst[1], ".r")

                plt.plot(xEst[0], xEst[1], marker='o', markeredgecolor='r', markersize=np.pi * 10 ** 2 / 2,
                         linestyle='none', markerfacecolor='none')
                # plot landmark
                for i in range(z.shape[0]):
                    plt.plot(RFID[int(z[i, 2])][0],
                             RFID[int(z[i, 2])][1], "xg")

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

                # Terminate if max step has been reached
                if j == (max_ep_steps - 1):
                    dones[0] = True
                s = s_

                # Check if episode is done and break loop
                if dones[0]:
                    break

    else:
        for j in range(0, 1):
            roll_out_paths = {}
            roll_out_paths = {
                "r": [],
                "r_ekf": [],
                "xTrue": [],
                "xDR": [],
                "xEst": [],
                "xRL": [],
                "return": [],
                "cost_ekf": [],
                "cost_rl": [],
                "episode_length": [],
                "death_rate": 0.0,
            }

            for i in range(num_of_paths):

                # Path storage buckets
                episode_path = {
                    "r": [],
                    "r_ekf": [],
                    "xTrue": [],
                    "xDR": [],
                    "xEst": [],
                    "xRL": [],
                    "cost_ekf": [],
                    "cost_rl": [],
                }
                s = env.reset()
                RFID = env.get_attr("RFID")[0]
                # print(RFID)
                # store the initial state
                episode_path["xTrue"].append(np.array(env.get_attr("xTrue")[0]))
                episode_path["xDR"].append(np.array(env.get_attr("xDR")[0]))
                episode_path["xEst"].append(np.array(env.get_attr("xEst")[0]))
                episode_path["xRL"].append(np.array(env.get_attr("x_RL")[0]))
                initial_cost = -np.sqrt(np.sum((env.get_attr("xEst")[0] - env.get_attr("xTrue")[0]) ** 2, axis= 0))
                episode_path["r_ekf"].append(initial_cost)
                episode_path["r"].append(initial_cost)

                for j in range(max_ep_steps):
                    action, _states = model.predict(s)
                    s_, rewards, dones, infos = env.step(action)
                    info = infos[0]
                    xTrue = info['xTrue']
                    xDR = info['xDR']
                    xEst = info["xEst"]
                    xRL = info['xRL']
                    # Store observations
                    episode_path["r"].append(rewards)
                    episode_path["r_ekf"].append(np.array([-info["quant"][0]]))
                    if "xTrue" in info.keys():
                        episode_path["xTrue"].append(np.array(info["xTrue"]))
                    if "xDR" in info.keys():
                        episode_path["xDR"].append(np.array(info["xDR"]))
                    if "xEst" in info.keys():
                        episode_path["xEst"].append(np.array(info["xEst"]))
                    if "xRL" in info.keys():
                        episode_path["xRL"].append(np.array(info["xRL"]))
                    if "quant" in info.keys():
                        episode_path["cost_ekf"].append(np.array(info["quant"][0]))
                        episode_path["cost_rl"].append(np.array(info["quant"][1]))
                    # Terminate if max step has been reached
                    if j == (max_ep_steps - 1):
                        dones[0] = True
                    s = s_

                    # Check if episode is done and break loop
                    if dones[0]:
                        break
                # print(episode_path["r_ekf"])
                # print(episode_path["r"])
                # Append paths to paths list
                roll_out_paths["r"].append(episode_path["r"])
                roll_out_paths["r_ekf"].append(episode_path["r_ekf"])
                roll_out_paths["episode_length"].append(len(episode_path["xEst"]))
                roll_out_paths["xEst"].append(episode_path["xEst"])
                roll_out_paths["xTrue"].append(episode_path["xTrue"])
                roll_out_paths["xDR"].append(episode_path["xDR"])
                roll_out_paths["xRL"].append(episode_path["xRL"])
                roll_out_paths["return"].append(np.sum(episode_path["r"]))
                roll_out_paths["cost_ekf"].append(np.sum(episode_path["cost_ekf"]))
                roll_out_paths["cost_rl"].append(np.sum(episode_path["cost_rl"]))

            mean_return = np.mean(roll_out_paths["return"])
            mean_cost_ekf = np.mean(roll_out_paths["cost_ekf"])
            mean_cost_rl = np.mean(roll_out_paths["cost_rl"])
            print('mean_return: ', mean_return/max_ep_steps)
            print('mean_cost_ekf: ', mean_cost_ekf/max_ep_steps)
            print('mean_cost_rl: ', mean_cost_rl/max_ep_steps)

            print("Plotting states of reference...")
            print("Plotting mean path and standard deviation...")

            # Calculate mean path of reference and state_of_interest
            xEst_trimmed = [
                path
                for path in roll_out_paths["xEst"]
                if len(path) == max(roll_out_paths["episode_length"])
            ]  # Needed because unequal paths #
            xRL_trimmed = [
                path
                for path in roll_out_paths["xRL"]
                if len(path) == max(roll_out_paths["episode_length"])
            ]
            xTrue_trimmed = [
                path
                for path in roll_out_paths["xTrue"]
                if len(path) == max(roll_out_paths["episode_length"])
            ]
            xDR_trimmed = [
                path
                for path in roll_out_paths["xDR"]
                if len(path) == max(roll_out_paths["episode_length"])
            ]
            xEst_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(xEst_trimmed), axis=0))
            )
            xEst_std_path = np.transpose(
                np.squeeze(np.std(np.array(xEst_trimmed), axis=0))
            )
            xRL_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(xRL_trimmed), axis=0))
            )

            xRL_std_path = np.transpose(
                np.squeeze(np.std(np.array(xRL_trimmed), axis=0))
            )
            xTrue_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(xTrue_trimmed), axis=0))
            )
            xTrue_std_path = np.transpose(
                np.squeeze(np.std(np.array(xTrue_trimmed), axis=0))
            )
            xDR_mean_path = np.transpose(
                np.squeeze(np.mean(np.array(xDR_trimmed), axis=0))
            )

            # Make sure arrays are right dimension
            xEst_mean_path = (
                np.expand_dims(xEst_mean_path, axis=0)
                if len(xEst_mean_path.shape) == 1
                else xEst_mean_path
            )
            xEst_std_path = (
                np.expand_dims(xEst_std_path, axis=0)
                if len(xEst_std_path.shape) == 1
                else xEst_std_path
            )
            xRL_mean_path = (
                np.expand_dims(xRL_mean_path, axis=0)
                if len(xRL_mean_path.shape) == 1
                else xRL_mean_path
            )
            xRL_std_path = (
                np.expand_dims(xRL_std_path, axis=0)
                if len(xRL_std_path.shape) == 1
                else xRL_std_path
            )
            xTrue_mean_path = (
                np.expand_dims(xTrue_mean_path, axis=0)
                if len(xTrue_mean_path.shape) == 1
                else xTrue_mean_path
            )
            xTrue_std_path = (
                np.expand_dims(xTrue_std_path, axis=0)
                if len(xTrue_std_path.shape) == 1
                else xTrue_std_path
            )
            xDR_mean_path = (
                np.expand_dims(xDR_mean_path, axis=0)
                if len(xDR_mean_path.shape) == 1
                else xDR_mean_path
            )
            # Plot mean cost and std
            # Create figure
            fig_1 = plt.figure(
                figsize=(16, 9), num="cost-path"
            )
            ax1 = fig_1.add_subplot(121)

            # Calculate mean observation path and std
            cost_trimmed = [
                path
                for path in roll_out_paths["r"]
                if len(path) == max(roll_out_paths["episode_length"])
            ]
            cost_mean_path = np.squeeze(np.mean(np.array(cost_trimmed), axis=0))
            cost_std_path = np.squeeze(np.std(np.array(cost_trimmed), axis=0))
            cost_ekf_trimmed = [
                path
                for path in roll_out_paths["r_ekf"]
                if len(path) == max(roll_out_paths["episode_length"])
            ]
            cost_ekf_mean_path = np.squeeze(np.mean(np.array(cost_ekf_trimmed), axis=0))
            cost_ekf_std_path = np.squeeze(np.std(np.array(cost_ekf_trimmed), axis=0))
            t = range(max(roll_out_paths["episode_length"]))

            # Plot state paths and std
            ax1.plot(
                t, cost_mean_path, color="g", linestyle="dashed", label=("RL mean cost"),
            )
            ax1.fill_between(
                t,
                cost_mean_path - cost_std_path,
                cost_mean_path + cost_std_path,
                color="g",
                alpha=0.3,
                # label=f"state_of_interest_{i+1}_std",
            )
            ax1.plot(
                    t, cost_ekf_mean_path, color="r", linestyle="dashed", label=("EKF mean cost"),
                )
            ax1.fill_between(
                t,
                cost_ekf_mean_path - cost_ekf_std_path,
                cost_ekf_mean_path + cost_ekf_std_path,
                color="r",
                alpha=0.3,
                # label=f"state_of_interest_{i+1}_std",
            )
            ax1.set_title("Mean cost")
            handles3, labels3 = ax1.get_legend_handles_labels()
            ax1.legend(handles3, labels3, loc=2, fancybox=False, shadow=False)

            # Show figures

            ax2 = fig_1.add_subplot(122)
            # plt.cla()

            ax2.plot(RFID[:, 0], RFID[:, 1], "*k")

            ax2.plot(xTrue_mean_path[0, :],
                     xTrue_mean_path[1, :], "-b", label='Ground Truth')
            ax2.plot(xDR_mean_path[0, :],
                     xDR_mean_path[1, :], "-k", label='Dead Reconking')
            ax2.plot(xEst_mean_path[0, :],
                     xEst_mean_path[1, :], "-r", label='EKF Estimation')
            ax2.plot(xRL_mean_path[0, :],
                     xRL_mean_path[1, :], "-g", label='RL Estimation')
            ax2.set_title("Tracking trajectory")
            ax2.axis("equal")
            ax2.grid(True)
            ax2.legend()
            plt.show()

            fig_2 = plt.figure(
                figsize=(16, 9), num="state-vector"
            )
            ax3 = fig_2.add_subplot(311)
            ax3.plot(range(0, len(xTrue_mean_path[0, :])),
                     xTrue_mean_path[0, :], "b", label='x-True')
            ax3.plot(range(0, len(xRL_mean_path[0, :])),
                     xRL_mean_path[0, :], "g", label='x-RL')
            ax3.fill_between(
                range(0, len(xRL_mean_path[0, :])),
                xRL_mean_path[0, :] - xRL_std_path[0, :],
                xRL_mean_path[0, :] + xRL_std_path[0, :],
                color="g",
                alpha=0.3,
            )
            ax3.fill_between(
                range(0, len(xRL_mean_path[0, :])),
                xEst_mean_path[0, :] - xEst_std_path[0, :],
                xEst_mean_path[0, :] + xEst_std_path[0, :],
                color="r",
                alpha=0.3,
            )
            ax3.plot(range(0, len(xDR_mean_path[0, :])),
                     xDR_mean_path[0, :], "-k", label='x-Dead Reconking')
            ax3.plot(range(0, len(xEst_mean_path[0, :])),
                     xEst_mean_path[0, :], "-r", label='x-EKF')
            ax3.legend()
            ax4 = fig_2.add_subplot(312)
            ax4.plot(range(0, len(xTrue_mean_path[1, :])),
                     xTrue_mean_path[1, :], "b", label='y-True')
            ax4.plot(range(0, len(xRL_mean_path[1, :])),
                     xRL_mean_path[1, :], "g", label='y-RL')
            ax4.plot(range(0, len(xDR_mean_path[1, :])),
                     xDR_mean_path[1, :], "-k", label='y-Dead Reconking')
            ax4.plot(range(0, len(xEst_mean_path[1, :])),
                     xEst_mean_path[1, :], "-r", label='y-EKF')
            ax4.fill_between(
                range(0, len(xRL_mean_path[1, :])),
                xRL_mean_path[1, :] - xRL_std_path[1, :],
                xRL_mean_path[1, :] + xRL_std_path[1, :],
                color="g",
                alpha=0.3,
            )
            ax4.fill_between(
                range(0, len(xRL_mean_path[0, :])),
                xEst_mean_path[1, :] - xEst_std_path[1, :],
                xEst_mean_path[1, :] + xEst_std_path[1, :],
                color="r",
                alpha=0.3,
            )
            ax4.legend()
            ax5 = fig_2.add_subplot(313)
            ax5.plot(range(0, len(xTrue_mean_path[2, :])),
                     xTrue_mean_path[2, :], "b", label='theta-True')
            ax5.plot(range(0, len(xRL_mean_path[2, :])),
                     xRL_mean_path[2, :], "g", label='theta-RL')
            ax5.plot(range(0, len(xDR_mean_path[1, :])),
                     xDR_mean_path[2, :], "-k", label='theta-Dead Reconking')
            ax5.plot(range(0, len(xEst_mean_path[1, :])),
                     xEst_mean_path[2, :], "-r", label='theta-EKF')
            ax5.fill_between(
                range(0, len(xRL_mean_path[1, :])),
                xRL_mean_path[2, :] - xRL_std_path[2, :],
                xRL_mean_path[2, :] + xRL_std_path[2, :],
                color="g",
                alpha=0.3,
            )
            ax5.fill_between(
                range(0, len(xRL_mean_path[0, :])),
                xEst_mean_path[2, :] - xEst_std_path[2, :],
                xEst_mean_path[2, :] + xEst_std_path[2, :],
                color="r",
                alpha=0.3,
            )
            ax5.legend()
            plt.show()
            print("RMSE x-RL:", np.mean(np.sqrt((xRL_mean_path[0, :]-xTrue_mean_path[0, :])**2)))
            print("RMSE y-RL:", np.mean(np.sqrt((xRL_mean_path[1, :]-xTrue_mean_path[1, :])**2)))
            print("RMSE theta-RL:", np.mean(np.sqrt((xRL_mean_path[2, :]-xTrue_mean_path[2, :])**2)))
            print("RMSE x-EKF:", np.mean(np.sqrt((xEst_mean_path[0, :]-xTrue_mean_path[0, :])**2)))
            print("RMSE y-EKF:", np.mean(np.sqrt((xEst_mean_path[1, :]-xTrue_mean_path[1, :])**2)))
            print("RMSE theta-EKF:", np.mean(np.sqrt((xEst_mean_path[2, :]-xTrue_mean_path[2, :])**2)))


            # # Save figures to pdf if requested
            # if save_figs:
            #     fig_1.savefig(
            #         os.path.join(LOG_PATH, "Quatonian." + fig_file_type),
            #         bbox_inches="tight",
            #     )
            #     fig_2.savefig(
            #         os.path.join(LOG_PATH, "State." + fig_file_type),
            #         bbox_inches="tight",
            #     )
            #     fig_3.savefig(
            #         os.path.join(LOG_PATH, "Cost." + fig_file_type),
            #         bbox_inches="tight",
            #     )