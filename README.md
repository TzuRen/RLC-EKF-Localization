## Reiforcement Learning-Compensated EKF for Orientation Estimation

This project is part of the master thesis credited by Hao Li from TU Delft. Below you will find the instructions on how to use this package.


### **Setup the python environment**

### Clone the project
```bash
git clone git@github.com:Mrhamsterleo/RLC-EKF-Localization.git
cd RLC-EKF-Localization
```
It's always wise to create a virtual conda environment without affacting other projects. To create a new environemnt:

```bash
conda create -n RLC-EKF python=python=3.6.12
conda activate RLC-EKF
```
### Install dependencies


After you created and activated the Conda environment, you have to install the python dependencies. This can be done using the following command:

```bash
pip install -r requirements.txt
```
### **Folder sturcture**
Here are descriptions for five scripts you are going to use to understand the code.
-   **RLestimator_ekf.py**: Gym environment for RLC-EKF orintation estimator to conduct simulation experiments. 
-   **RLestimator_real_data.py**: Gym environment for RLC-EKF orintation estimator to conduct real dataset experiments.
-   **train_evaluate.py**: Train and evaluate RL policy using stable-baseline platform. For evaluation, both qualitive and quantitive results compared to pure EKF performance can be provided.policy using stable-baseline platform.
-   **train_evaluate_eva_list.py**: Evaluate a list of different policies (usually 10 models for a single training). The best model can be selected out from results.
-   **data_simulator.py**: Generate range-bearing measurements, odometry input, ground truth of robot and landmarks, for simulation experiments.
-   **load_dataset.py**: Data loader especially made for UTIAS dataset.



### **Adjustable variables for training**

For simulation related settings, in **RLestimator_ekf.py** change the following parameters.
#### Measurement noise
```bash
#line 104 (for odometry input noise)
self.Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2 * 0.1
#line 106 (for feature measurement noise)
self.R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2 * 0.1
```
#### Initial deviation
```bash
#line 90 (range of uniform distribution)
self.initial_bias = np.random.uniform(-1, 1, size=(self.STATE_SIZE, 1)) * 3
```

#### Model covariance
```bash
#line 93 (for EKF process model covariance)
self.Cx = np.diag([1, 1, np.deg2rad(30.0)]) ** 2
#line 97 (for EKF measurement model covariance)
self.Cx_obs = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) ** 2
```

#### Tracking trajectory
```bash
#line 110-111 (defined by velocities)
v = 1.0  # [m/s]
yaw_rate = 0.1  # [rad/s]
```

#### Landmark 
Position devation for landmark placed in circular only, the landmark distribution can also be changed to other shape.
```bash
#line 136 
m_sim = np.diag([2.5, 2.5]) ** 2
```

Other settings adjustment can be done in **train_evaluate.py**.

-   **model_num**: Policy model you want to train for each script execution.
-   **episodes**: The number of episodes for agent to explore.
-   **T**: Total length of each eposide. 
-   **ENV_NAME**: The name of the gym formed enviroment.


When all settings are done, directly execute the script and wait for finishing. During the training, a tensorboard log file is generated in the folder ./logs. You can want the simutanous training loss through:

```bash
TensorBoard --logdir=logs/your_saved_model_name/PPO2_0_1 --host=localhost
```


### **Adjustable variables for evaluation**

Besides the experiment condition setting which is introduced in last section, there are few adjustments we can made for evaluation:

-   **model = PPO2.load("model_path")**: To load the best trained model
-   **num_of_paths**: The number of Monte Carlo Simulation for stastical results neglecting odd disturbance.
-   **MODEL_PATH**: Folder path of multiple models we want to select the best model from.
After setting is done, we can directly run the evaluation script, and obtain cost, trajectory and sub-items graphs. The numerical results of EKF and RLC-EKF are also given in the terminal. 
-   **show_animation**: Whether display the result statically or simutanously.

### **Evaluation with real data**
To do so, you can import RLestimator_real_data instead of RLestimator_ekf in training and evaluation scripts. The data path should be changed to the local path, with the format give by:
-   Column 1-4: Quaternion x, y, z, w.
-   Column 5-7: Angular velocity x, y, z.
-   Column 8-10: Linear acceleration x, y, z.
-   Column 11-13: Magnetic field x, y, z.
