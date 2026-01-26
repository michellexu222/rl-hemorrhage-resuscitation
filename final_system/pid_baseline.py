import sys
sys.path.insert(0, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation")
sys.path.insert(1, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating")
sys.path.insert(2, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\env")

import numpy as np
import os
from hemorrhage_env import HemorrhageEnv

class PIDBaseline:
    def __init__(self, kp, ki, kd, target_map, dt=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_map = target_map
        self.dt = dt

        self.integral = 0.0
        self.prev_error = None

    def step(self, current_map):
        error = self.target_map - current_map

        # Integral
        self.integral += error * self.dt

        # Derivative
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / self.dt

        self.prev_error = error

        # PID output
        u = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        return max(u, 0.0)  # no negative fluids

# controller = PIDBaseline(kp=0.1, ki=0.07/40, kd=0.07*0.7, target_map=75)
# env = HemorrhageEnv(state_file=r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\configs\patient_configs\patient0@0s.json")
# obs, info = env.reset(organ="liver", severity=0.3)
# print(info['hem'])
# done = False
#
# n_stable = 0
# n_dead = 0
# total = 0
#
# while not done:
#     current_map = obs[1]
#     print(obs[1])
#     fluid_rate = controller.step(current_map)
#     action = [0.0,fluid_rate, 0.0]  # only crystalloid for low severity
#     print(fluid_rate)
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
# print(info['o'])

