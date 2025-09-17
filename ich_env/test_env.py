from hemorrhage_env import HemorrhageEnv
from pulse.cdm.patient_actions import eHemorrhage_Compartment
import numpy as np
import dill as pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env_check_dir = os.path.join(parent_dir, "env_check")
os.makedirs(env_check_dir, exist_ok=True)
#env_check_file = os.path.join(parent_dir, "env_check")

bleed_rates_liver = []
bleed_rates_spleen = []
current_bvs = []

env = HemorrhageEnv()

# for severity in [0.1]:
#     env.induce_hemorrhage(eHemorrhage_Compartment.Liver, severity)
#     terminated = False
#     count = 0
#     while not terminated:
#         obs, reward, terminated, truncated, info = env.step(action_idx=5)
#         bv = obs[-4]
#         current_bvs.append(bv)
#         #print(terminated, obs)
#         count += 1
#
#     bleed_rates_liver.append(np.mean(abs(np.diff(current_bvs))))
#     print(f"Average bleed rate for liver with severity {severity}: {bleed_rates_liver[-1]}. Number of until terminal: {count}")
#     current_bvs = []
#     env.reset()
#     count = 0

# for severity in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     env.induce_hemorrhage(eHemorrhage_Compartment.Spleen, severity)
#     terminated = False
#     count = 0
#     while not terminated:
#         obs, reward, terminated, truncated, info = env.step(action_idx=5)
#         bv = obs[-4]
#         current_bvs.append(bv)
#         #print(terminated, obs)
#         count += 1
#
#     bleed_rates_spleen.append(np.mean(abs(np.diff(current_bvs))))
#     print(f"Average bleed rate for spleen with severity {severity}: {bleed_rates_spleen[-1]}. Number of until terminal: {count}")
#     current_bvs = []
#     env.reset()
#     count = 0


# # ---------- Liver, 0.5, nothing ------------ >
# env.induce_hemorrhage(eHemorrhage_Compartment.Liver, 0.5)
# # obs, reward, terminated, truncated, info = env.step(action_idx=5)
# terminated = False
# while not terminated:
#     obs, reward, terminated, truncated, info = env.step(action_idx=5)
#     print(terminated, obs)
#
# with open(os.path.join(env_check_dir, "liver_0.5_nothing.pkl"), "wb") as f:
#     pickle.dump(env.history, f)
#
# print(np.array(env.history)[:, 0], "\n", np.array(env.history)[:, 3])
# print("DONE WITH LIVER, 0.5, NOTHING")
#
# # ------------ Liver, 0.3, blood ---------- >
# env.reset()
# env.induce_hemorrhage(eHemorrhage_Compartment.Liver, 0.3)
# # obs, reward, terminated, truncated, info = env.step(action_idx=5)
# terminated = False
# while not terminated:
#     obs, reward, terminated, truncated, info = env.step(action_idx=2)
#     print(terminated, obs)
#
# # print(np.array(env.history)[:, 0], "\n", np.array(env.history)[:, 3])
# with open(os.path.join(env_check_dir, "liver_0.3_blood.pkl"), "wb") as f:
#     pickle.dump(env.history, f)
# print("DONE WITH LIVER, 0.5, BLOOD")
#
# # ------------ Liver, 0.3, lactated ringers ---------- >
# env.reset()
# env.induce_hemorrhage(eHemorrhage_Compartment.Liver, 0.3)
# # obs, reward, terminated, truncated, info = env.step(action_idx=5)
# terminated = False
# while not terminated:
#     obs, reward, terminated, truncated, info = env.step(action_idx=3)
#     print(terminated, obs)
#
# # print(np.array(env.history)[:, 0], "\n", np.array(env.history)[:, 3])
# with open(os.path.join(env_check_dir, "liver_0.3_lactatedRingers.pkl"), "wb") as f:
#     pickle.dump(env.history, f)
#
# print("COMPLETED LIVER, 0.3, LACTATED RINGERS")
#
# # ------------ Liver, 0.3, prbcs ---------- >
# env.reset()
# env.induce_hemorrhage(eHemorrhage_Compartment.Liver, 0.3)
# # obs, reward, terminated, truncated, info = env.step(action_idx=5)
# terminated = False
# while not terminated:
#     obs, reward, terminated, truncated, info = env.step(action_idx=1)
#     print(terminated, obs)
#
# # print(np.array(env.history)[:, 0], "\n", np.array(env.history)[:, 3])
# with open(os.path.join(env_check_dir, "liver_0.3_PRBC.pkl"), "wb") as f:
#     pickle.dump(env.history, f)
#
# print("COMPLETED LIVER, 0.3, PRBCs")

# ------------ spleen, 0.8, nothing ---------- >
env.reset()
env.induce_hemorrhage(eHemorrhage_Compartment.Spleen, 0.8)
# obs, reward, terminated, truncated, info = env.step(action_idx=5)
terminated = False
while not terminated:
    obs, reward, terminated, truncated, info = env.step(action_idx=5)
    print(terminated, obs)

# print(np.array(env.history)[:, 0], "\n", np.array(env.history)[:, 3])
with open(os.path.join(env_check_dir, "spleen_0.3_nothing.pkl"), "wb") as f:
    pickle.dump(env.history, f)

print("COMPLETED SPLEEN, 0.8, nothing")


# ------------ Spleen, 0.8, blood ---------- >
env.reset()
env.induce_hemorrhage(eHemorrhage_Compartment.Spleen, 0.8)
# obs, reward, terminated, truncated, info = env.step(action_idx=5)
terminated = False
while not terminated:
    obs, reward, terminated, truncated, info = env.step(action_idx=2)
    print(terminated, obs)

# print(np.array(env.history)[:, 0], "\n", np.array(env.history)[:, 3])
with open(os.path.join(env_check_dir, "spleen_0.8_blood.pkl"), "wb") as f:
    pickle.dump(env.history, f)

print("COMPLETED SPLEEN, 0.8, blood")




