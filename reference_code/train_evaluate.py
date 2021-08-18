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
eposides = 100
total_timesteps_ = eposides * n_steps
algorithm = "PPO2"
ENV_NAME = "RLestimator"
model_dir = "./models/for_pre/" + time.strftime("%Y%m%d_%H%M") + "/"
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
    model = PPO2.load("models/for_pre/20210521_1145/PPO2_0")
    # Enjoy trained agent
    num_of_paths = 1
    max_ep_steps = 500
    # save_figs = True
    show_animation = False
    LOG_PATH = "./logs"

    if show_animation is True:
        for i in range(num_of_paths):
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

    else:
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
            # store the initial state
            episode_path["xTrue"].append(np.array(env.get_attr("xTrue")[0]))
            episode_path["xDR"].append(np.array(env.get_attr("xDR")[0]))
            episode_path["xEst"].append(np.array(env.get_attr("xEst")[0]))
            episode_path["xRL"].append(np.array(env.get_attr("x_RL")[0]))

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
                episode_path["r_ekf"].append(np.array([-20 * info["quant"][0]]))
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
        print('mean_return: ', mean_return)
        print('mean_cost_ekf: ', mean_cost_ekf)
        print('mean_cost_rl: ', mean_cost_rl)

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

        plt.cla()

        plt.plot(RFID[:, 0], RFID[:, 1], "*k")

        plt.plot(xTrue_mean_path[0, :],
                 xTrue_mean_path[1, :], "-b", label='Ground Truth')
        plt.plot(xDR_mean_path[0, :],
                 xDR_mean_path[1, :], "-k", label='Dead Reconking')
        plt.plot(xEst_mean_path[0, :],
                 xEst_mean_path[1, :], "-r", label='EKF Estimation')
        plt.plot(xRL_mean_path[0, :],
                 xRL_mean_path[1, :], "-g", label='RL Estimation')
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()


        # Plot mean cost and std
        # Create figure
        fig_3 = plt.figure(
            figsize=(9, 6), num="return-ppo2"
        )
        ax3 = fig_3.add_subplot(111)

        # Calculate mean observation path and std
        cost_trimmed = [
            path
            for path in roll_out_paths["r"]
            if len(path) == max(roll_out_paths["episode_length"]) - 1
        ]
        cost_mean_path = np.squeeze(np.mean(np.array(cost_trimmed), axis=2))
        cost_std_path = np.squeeze(np.std(np.array(cost_trimmed), axis=2))

        cost_ekf_trimmed = [
            path
            for path in roll_out_paths["r_ekf"]
            if len(path) == max(roll_out_paths["episode_length"]) - 1
        ]
        cost_ekf_mean_path = np.squeeze(np.mean(np.array(cost_ekf_trimmed), axis=2))
        cost_ekf_std_path = np.squeeze(np.std(np.array(cost_ekf_trimmed), axis=2))
        t = range(max(roll_out_paths["episode_length"]) - 1)

        # Plot state paths and std
        ax3.plot(
            t, cost_mean_path, color="r", linestyle="dashed", label=("RL mean cost"),
        )
        ax3.plot(
                t, cost_ekf_mean_path, color="g", linestyle="dashed", label=("mean cost"),
            )
        # ax3.fill_between(
        #     t,
        #     cost_mean_path - cost_std_path,
        #     cost_mean_path + cost_std_path,
        #     color="g",
        #     alpha=0.3,
        #     label=("mean cost std"),
        # )
        ax3.set_title("Mean cost")
        handles3, labels3 = ax3.get_legend_handles_labels()
        ax3.legend(handles3, labels3, loc=2, fancybox=False, shadow=False)

        # Show figures
        plt.show()

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