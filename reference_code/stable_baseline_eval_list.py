import stable_baselines
# from Ekf_Ori_Env import *
# from RLestimator_with_measurement import *
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
n_steps = 1000
eposides = 500
total_timesteps_ = eposides * n_steps
algorithm = "PPO2"
ENV_NAME = "RLestimator"
MODEL_PATH = "models/RLestimator_PPO2_20210319_0030/"
model_dir  = "./models/" + ENV_NAME + "_" + algorithm + "_" + time.strftime("%Y%m%d_%H%M") + "/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
tensorboard_log_dir = "./logs/" + ENV_NAME + algorithm + time.strftime("%Y%m%d_%H%M") + "/"
T = 10
env = RL_estimator(T)
env = make_vec_env(lambda: env, n_envs=1)
# policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[64, 64])
# model = PPO2(MlpPolicy, env, n_steps=1000, policy_kwargs=policy_kwargs, verbose=1)
model = PPO2(MlpPolicy, env, n_steps=1000, verbose=1, tensorboard_log=tensorboard_log_dir)

if train:
    for i in range(model_num):
        model.learn(total_timesteps=total_timesteps_, tb_log_name=algorithm + "_" + str(i))
        model.save(model_dir + algorithm + "_" + str(i))

del model # remove to demonstrate saving and loading

if inference:
    num_policy = len([name for name in os.listdir(MODEL_PATH) if os.path.isfile(os.path.join(MODEL_PATH, name))])
    cost_list = []
    for n in range(0,num_policy):
        model_name = MODEL_PATH + algorithm + "_" + str(n)
        model = PPO2.load(model_name)
        # Enjoy trained agent
        num_of_paths = 1
        max_ep_steps = 1000
        save_figs = True
        LOG_PATH = "./logs"
        fig_file_type = "pdf"
        roll_out_paths = {}
        roll_out_paths = {
            "s": [],
            "r": [],
            "s_": [],
            "state_of_interest": [],
            "reference": [],
            "episode_length": [],
            "return": [],
            "death_rate": 0.0,
        }
        for i in range(num_of_paths):

            # Path storage buckets
            episode_path = {
                "s": [],
                "r": [],
                "s_": [],
                "state_of_interest": [],
                "reference": [],
            }
            # while not dones[0]:
            s = env.reset()
            for j in range(max_ep_steps):
                action, _states = model.predict(s)
                s_, rewards, dones, infos = env.step(action)
                # Store observations
                episode_path["s"].append(s)
                episode_path["r"].append(rewards)
                episode_path["s_"].append(s_)
                info = infos[0]
                if "state_of_interest" in info.keys():
                    episode_path["state_of_interest"].append(
                        np.array([info["state_of_interest"]])
                    )
                if "reference" in info.keys():
                    episode_path["reference"].append(np.array(info["reference"]))

                # Terminate if max step has been reached
                if j == (max_ep_steps - 1):
                    dones[0] = True
                s = s_

                # Check if episode is done and break loop
                if dones[0]:
                    break

            # Append paths to paths list
            roll_out_paths["s"].append(episode_path["s"])
            roll_out_paths["r"].append(episode_path["r"])
            roll_out_paths["s_"].append(episode_path["s_"])
            roll_out_paths["state_of_interest"].append(
                episode_path["state_of_interest"]
            )
            roll_out_paths["reference"].append(episode_path["reference"])
            roll_out_paths["episode_length"].append(len(episode_path["s"]))
            roll_out_paths["return"].append(np.sum(episode_path["r"]))

        # Calculate roll_out death rate
        roll_out_paths["death_rate"] = sum(
            [
                episode <= (max_ep_steps - 1)
                for episode in roll_out_paths["episode_length"]
            ]) / len(roll_out_paths["episode_length"])

        mean_return = np.mean(roll_out_paths["return"])
        cost_list.append(mean_return)
    print(cost_list)
        # mean_episode_length = np.mean(
        #     roll_out_paths["episode_length"]
        # )
        # print('mean_episode_length: ', mean_episode_length)
        # death_rate = roll_out_paths["death_rate"]
        # print('death_rate: ', death_rate)
        #
        # print("Plotting states of reference...")
        # print("Plotting mean path and standard deviation...")
        #
        # # Calculate mean path of reference and state_of_interest
        # soi_trimmed = [
        #     path
        #     for path in roll_out_paths["state_of_interest"]
        #     if len(path) == max(roll_out_paths["episode_length"])
        # ]  # Needed because unequal paths # FIXME: CLEANUP
        # ref_trimmed = [
        #     path
        #     for path in roll_out_paths["reference"]
        #     if len(path) == max(roll_out_paths["episode_length"])
        # ]  # Needed because unequal paths # FIXME: CLEANUP
        # soi_mean_path = np.transpose(
        #     np.squeeze(np.mean(np.array(soi_trimmed), axis=0))
        # )
        # soi_std_path = np.transpose(
        #     np.squeeze(np.std(np.array(soi_trimmed), axis=0))
        # )
        # ref_mean_path = np.transpose(
        #     np.squeeze(np.mean(np.array(ref_trimmed), axis=0))
        # )
        # nominal_mean_path = ref_mean_path[:, 0, :]
        # ekf_mean_path = ref_mean_path[:, 2, :]
        # ref_std_path = np.transpose(
        #     np.squeeze(np.std(np.array(ref_trimmed), axis=0))
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
        # #
        # # # Also plot mean and std of the observations
        # # print("Plotting observations...")
        # # print("Plotting mean path and standard deviation...")
        # #
        # # # Create figure
        # # fig_2 = plt.figure(
        # #     figsize=(9, 6), num="observation-ppo2"
        # # )
        # # colors = "bgrcmk"
        # # cycol = cycle(colors)
        # # ax2 = fig_2.add_subplot(111)
        # #
        # # # Calculate mean observation path and std
        # # obs_trimmed = [
        # #     path
        # #     for path in roll_out_paths["s"]
        # #     if len(path) == max(roll_out_paths["episode_length"])
        # # ]
        # # obs_mean_path = np.transpose(
        # #     np.squeeze(np.mean(np.array(obs_trimmed), axis=0))
        # # )
        # # obs_std_path = np.transpose(
        # #     np.squeeze(np.std(np.array(obs_trimmed), axis=0))
        # # )
        # # t = range(max(roll_out_paths["episode_length"]))
        # #
        # # # Plot state paths and std
        # # for i in range(0, obs_mean_path.shape[0]):
        # #     color = next(cycol)
        # #     ax2.plot(
        # #         t,
        # #         obs_mean_path[i],
        # #         color=color,
        # #         linestyle="dashed",
        # #         label=(f"s_{i + 1}"),
        # #     )
        # #     ax2.fill_between(
        # #         t,
        # #         obs_mean_path[i] - obs_std_path[i],
        # #         obs_mean_path[i] + obs_std_path[i],
        # #         color=color,
        # #         alpha=0.3,
        # #         label=(f"s_{i + 1}_std"),
        # #     )
        # # ax2.set_title("Observations")
        # # handles2, labels2 = ax2.get_legend_handles_labels()
        # # ax2.legend(handles2, labels2, loc=2, fancybox=False, shadow=False)
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