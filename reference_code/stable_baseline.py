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
model_num = 1
n_steps = 500
eposides = 500
total_timesteps_ = eposides * n_steps
algorithm = "PPO2"
ENV_NAME = "RLestimator"
profile_num = "1"
model_dir  = "./models/" + time.strftime("%Y%m%d_%H%M") + "/"
tensorboard_log_dir = "./logs/" + ENV_NAME + algorithm + time.strftime("%Y%m%d_%H%M") + "/"
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
    model = PPO2.load("models/20210505_1823/PPO2_0")
    # Enjoy trained agent
    num_of_paths = 1
    max_ep_steps = 501
    save_figs = True
    show_animation = True
    LOG_PATH = "./logs"
    roll_out_paths = {}
    roll_out_paths = {
        "r": [],
        "xTrue": [],
        "xEst": [],
        "xRL": [],
        "xDR": [],
        "return": [],
        "death_rate": 0.0,
    }
    for i in range(num_of_paths):

        # Path storage buckets
        episode_path = {
            "r": [],
            "xTrue": [],
            "xEst": [],
            "xRL": [],
            "xDR": [],
        }
        s = env.reset()
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
            RFID = env.get_attr("RFID")[0]
            z = env.get_attr("z")[0]
            hxEst = np.hstack((hxEst, xEst))
            hxDR = np.hstack((hxDR, env.get_attr("xDR")[0]))
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








    #         # Store observations
    #         episode_path["r"].append(rewards)
    #         if "xTrue" in info.keys():
    #             episode_path["xTrue"].append(np.array([info["xTrue"]]))
    #         if "xEst" in info.keys():
    #             episode_path["xEst"].append(np.array(info["xEst"]))
    #         if "xRL" in info.keys():
    #             episode_path["xRL"].append(np.array(info["xRL"]))
    #
    #         # Terminate if max step has been reached
    #         if j == (max_ep_steps - 1):
    #             dones[0] = True
    #         s = s_
    #
    #         # Check if episode is done and break loop
    #         if dones[0]:
    #             break
    #
    #     # Append paths to paths list
    #     roll_out_paths["r"].append(episode_path["r"])
    #     roll_out_paths["xEst"].append(episode_path["xEst"])
    #     roll_out_paths["xRL"].append(episode_path["xRL"])
    #     roll_out_paths["return"].append(np.sum(episode_path["r"]))
    #
    # mean_return = np.mean(roll_out_paths["return"])
    # print('mean_return: ', mean_return)
    #
    # print("Plotting states of reference...")
    # print("Plotting mean path and standard deviation...")
    #
    # # Calculate mean path of reference and state_of_interest
    # xEst_trimmed = [
    #     path
    #     for path in roll_out_paths["xEst"]
    #     if len(path) == max(roll_out_paths["episode_length"])
    # ]  # Needed because unequal paths # FIXME: CLEANUP
    # xRL_trimmed = [
    #     path
    #     for path in roll_out_paths["reference"]
    #     if len(path) == max(roll_out_paths["episode_length"])
    # ]  # Needed because unequal paths # FIXME: CLEANUP
    # soi_mean_path = np.transpose(
    #     np.squeeze(np.mean(np.array(xEst_trimmed), axis=0))
    # )
    # soi_std_path = np.transpose(
    #     np.squeeze(np.std(np.array(xEst_trimmed), axis=0))
    # )
    # ref_mean_path = np.transpose(
    #     np.squeeze(np.mean(np.array(xRL_trimmed), axis=0))
    # )
    # nominal_mean_path = ref_mean_path[:, 0, :]
    # ekf_mean_path = ref_mean_path[:, 2, :]
    # ref_std_path = np.transpose(
    #     np.squeeze(np.std(np.array(xRL_trimmed), axis=0))
    # )
    #
    # # Make sure arrays are right dimension
    # soi_mean_path = (
    #     np.expand_dims(soi_mean_path, axis=0)
    #     if len(soi_mean_path.shape) == 1
    #     else soi_mean_path
    # )
    # soi_std_path = (
    #     np.expand_dims(soi_std_path, axis=0)
    #     if len(soi_std_path.shape) == 1
    #     else soi_std_path
    # )
    # nominal_mean_path = (
    #     np.expand_dims(nominal_mean_path, axis=0)
    #     if len(nominal_mean_path.shape) == 1
    #     else nominal_mean_path
    # )
    # ekf_mean_path = (
    #     np.expand_dims(ekf_mean_path, axis=0)
    #     if len(ekf_mean_path.shape) == 1
    #     else ekf_mean_path
    # )
    # ref_std_path = (
    #     np.expand_dims(ref_std_path, axis=0)
    #     if len(ref_std_path.shape) == 1
    #     else ref_std_path
    # )
    #
    # # Plot mean path of reference and state_of_interest
    # fig_1 = plt.figure(
    #     figsize=(9, 6), num=f"state-q-ppo2"
    # )
    # ax = fig_1.add_subplot(111)
    # colors = "bgrcmk"
    # cycol = cycle(colors)
    # for i in range(0, min(soi_mean_path.shape[0], nominal_mean_path.shape[0])):
    #     color1 = next(cycol)
    #     color2 = color1
    #     color3 = color2
    #     t = [i / 100.0 for i in range(0, max(roll_out_paths["episode_length"]))]
    #     if i <= (len(soi_mean_path) - 1):
    #         ax.plot(
    #             t,
    #             soi_mean_path[i],
    #             color=color1,
    #             linestyle="dashed",
    #             # label=f"state_of_interest_{i+1}_mean",
    #         )
    #         ax.fill_between(
    #             t,
    #             soi_mean_path[i] - soi_std_path[i],
    #             soi_mean_path[i] + soi_std_path[i],
    #             color=color1,
    #             alpha=0.3,
    #             # label=f"state_of_interest_{i+1}_std",
    #         )
    #     path = np.concatenate(
    #         [np.transpose(nominal_mean_path), np.transpose(soi_mean_path), np.transpose(soi_std_path)], 1)
    #     # np.savetxt('inferenceResult-52.csv', path, delimiter=',')
    #     if i <= (len(nominal_mean_path) - 1):
    #         ax.plot(
    #             t,
    #             nominal_mean_path[i],
    #             color=color2,
    #             # label=f"reference_{i+1}",
    #         )
    #     if i <= (len(ekf_mean_path) - 1):
    #         ax.plot(
    #             t,
    #             ekf_mean_path[i],
    #             color=color3,
    #             # label=f"reference_{i+1}",
    #         )
    #     if i <= (len(nominal_mean_path) - 1):
    #         plt.ylabel("Quaternion", fontsize=20)
    #         plt.xlabel("Time(s)", fontsize=20)
    #         plt.xticks(fontsize=20)
    #         plt.yticks(fontsize=20)
    #         plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    #         # ax.fill_between(
    #         #     t,
    #         #     nominal_mean_path[i] - ref_std_path[i],
    #         #     nominal_mean_path[i] + ref_std_path[i],
    #         #     color=color2,
    #         #     alpha=0.3,
    #         #     label=f"reference_{i+1}_std",
    #         # )  # FIXME: remove
    #     ax.set_rasterized(True)
    #
    # # Plot mean cost and std
    # # Create figure
    # fig_3 = plt.figure(
    #     figsize=(9, 6), num="return-ppo2"
    # )
    # ax3 = fig_3.add_subplot(111)
    #
    # # Calculate mean observation path and std
    # cost_trimmed = [
    #     path
    #     for path in roll_out_paths["r"]
    #     if len(path) == max(roll_out_paths["episode_length"])
    # ]
    # cost_mean_path = np.squeeze(np.mean(np.array(cost_trimmed), axis=2))
    # cost_std_path = np.squeeze(np.std(np.array(cost_trimmed), axis=2))
    # t = range(max(roll_out_paths["episode_length"]))
    #
    # # Plot state paths and std
    # ax3.plot(
    #     t, cost_mean_path, color="g", linestyle="dashed", label=("mean cost"),
    # )
    # ax3.fill_between(
    #     t,
    #     cost_mean_path - cost_std_path,
    #     cost_mean_path + cost_std_path,
    #     color="g",
    #     alpha=0.3,
    #     label=("mean cost std"),
    # )
    # ax3.set_title("Mean cost")
    # handles3, labels3 = ax3.get_legend_handles_labels()
    # ax3.legend(handles3, labels3, loc=2, fancybox=False, shadow=False)
    #
    # # Show figures
    # plt.show()
    #
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