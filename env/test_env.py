from hemorrhage_env import HemorrhageEnv
from pulse.cdm.patient_actions import eHemorrhage_Compartment
import numpy as np
import dill as pickle
import os
import csv
from pulse.cdm.patient import SEPatientConfiguration
from pulse.engine.PulseEngine import PulseEngine
from pulse.cdm.patient_actions import SEHemorrhage, eHemorrhage_Compartment, SESubstanceBolus, SESubstanceCompoundInfusion, eSubstance_Administration, SESubstanceInfusion
from pulse.cdm.scalars import VolumePerTimeUnit, VolumeUnit, MassPerVolumeUnit, TimeUnit, LengthUnit, MassUnit

pulse = PulseEngine()
substance = SESubstanceInfusion()
substance.set_substance("Norepinephrine")
#substance.get_bag_volume().set_value(2, VolumeUnit.mL)
#substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
pulse.process_action(substance)
# bolus = SESubstanceBolus()
# bolus.set_admin_route(eSubstance_Administration.Intramuscular)
# bolus.set_substance("Vasopressin")
# script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(script_dir)
# env_check_dir = os.path.join(parent_dir, "env_check")
# os.makedirs(env_check_dir, exist_ok=True)
# #env_check_file = os.path.join(parent_dir, "env_check")
# if not os.path.exists(os.path.join(parent_dir, "seed_check.csv")):
#     with open(os.path.join(parent_dir, "seed_check.csv"), "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["seed", "step", "obs", "patient_file"])
#
#
env = HemorrhageEnv()
_, info = env.reset()
print(info)
env.induce_hemorrhage(eHemorrhage_Compartment.Liver, 0.3)

for i in range(5):
    obs, reward, terminated, truncated, _ = env.step(action_idx=5)
    print(obs)
    print(env.history[-1])
    # with open(os.path.join(parent_dir, "seed_check.csv"), "a", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([5, i+1, obs, info["state_file"][-9]])


# bleed_rates_liver = []
# bleed_rates_spleen = []
# current_bvs = []
#
# env = HemorrhageEnv(state_file="./states/Soldier@0s.json")
#
# for severity in [0.3]:
#     env.induce_hemorrhage(eHemorrhage_Compartment.Liver, severity)
#     terminated = False
#     truncated = False
#     count = 0
#     while not terminated and not truncated:
#         obs, reward, terminated, truncated, info = env.step(action_idx=2)
#         bv = obs[-4]
#         current_bvs.append(bv)
#         #print(terminated, obs)
#         count += 1
#     print(env.history)
#     #bleed_rates_liver.append(np.mean(abs(np.diff(current_bvs))))
#     #print(f"Average bleed rate for liver with severity {severity}: {bleed_rates_liver[-1]}. Number of steps until terminal: {count}")
#     #current_bvs = []
#     count = 0
# #
# #     with open(os.path.join(env_check_dir, f"clot_liver_{severity}_blood.pkl"), "wb") as f:
# #         pickle.dump(env.history, f)
# #     print(f"DONE WITH LIVER, {severity}, BLOOD")
#     env.reset()

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
# # with open(os.path.join(env_check_dir, "liver_0.5_nothing.pkl"), "wb") as f:
# #     pickle.dump(env.history, f)
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

# # ------------ spleen, 0.8, nothing ---------- >
# env.reset()
# env.induce_hemorrhage(eHemorrhage_Compartment.Spleen, 0.8)
# # obs, reward, terminated, truncated, info = env.step(action_idx=5)
# terminated = False
# while not terminated:
#     obs, reward, terminated, truncated, info = env.step(action_idx=5)
#     print(terminated, obs)
#
# # print(np.array(env.history)[:, 0], "\n", np.array(env.history)[:, 3])
# with open(os.path.join(env_check_dir, "spleen_0.3_nothing.pkl"), "wb") as f:
#     pickle.dump(env.history, f)
#
# print("COMPLETED SPLEEN, 0.8, nothing")
#
#
# # ------------ Spleen, 0.8, blood ---------- >
# env.reset()
# env.induce_hemorrhage(eHemorrhage_Compartment.Spleen, 0.8)
# # obs, reward, terminated, truncated, info = env.step(action_idx=5)
# terminated = False
# while not terminated:
#     obs, reward, terminated, truncated, info = env.step(action_idx=2)
#     print(terminated, obs)
#
# # print(np.array(env.history)[:, 0], "\n", np.array(env.history)[:, 3])
# with open(os.path.join(env_check_dir, "spleen_0.8_blood.pkl"), "wb") as f:
#     pickle.dump(env.history, f)
#
# print("COMPLETED SPLEEN, 0.8, blood")




