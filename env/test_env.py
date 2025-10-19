from hemorrhage_env import HemorrhageEnv
from pulse.cdm.patient_actions import eHemorrhage_Compartment
import numpy as np
import dill as pickle
import os
import csv
import json
from pulse.cdm.patient import SEPatientConfiguration
import pulse.cdm.engine as cdm_engine
import pulse.cdm.scenario as cdm_scenario

import pulse.engine.PulseEngine as pulse_engine
from pulse.engine.PulseEngine import PulseEngine, eModelType

from pulse.cdm.patient_actions import SEHemorrhage, eHemorrhage_Compartment, SESubstanceBolus, SESubstanceCompoundInfusion, eSubstance_Administration, SESubstanceInfusion
from pulse.cdm.scalars import VolumePerTimeUnit, VolumeUnit, MassPerVolumeUnit, TimeUnit, LengthUnit, MassUnit
import time
import matplotlib.pyplot as plt
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env = HemorrhageEnv(state_file=os.path.join(parent_dir, "configs", "patient_configs", "Patient0@0s.json"))
num_episodes = 30
episode_rewards = []
all_rewards = []

for ep in range(num_episodes):
    obs = env.reset()
    #env.induce_hemorrhage(compartment="liver", given_severity=0.5)
    done = False
    ep_rewards = []
    while not done:
        # take a random action
        action = env.action_space.sample()
        #print(action)
        obs, reward, done, truncated, info = env.step(action)
        # print(obs)
        ep_rewards.append(reward)
        all_rewards.append(reward)
    episode_rewards.append(ep_rewards)
    print(f"Episode {ep+1}: Total reward = {np.sum(ep_rewards):.2f}, Length = {len(ep_rewards)}, outcome = {info['o']}, hemorrhage = {info['hem']}")


# plot per-step rewards of first few episodes
plt.figure(figsize=(10, 5))
for i, ep_r in enumerate(episode_rewards):
    plt.plot(range(len(ep_r)), ep_r, label=f"Ep {i+1}")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Per-step rewards (random policy)")
plt.legend()
plt.show()


total_rewards = [np.sum(ep_r) for ep_r in episode_rewards]
# Plot total reward distribution across episodes
plt.figure(figsize=(6, 4))
plt.hist(total_rewards, bins=10, edgecolor='black')
plt.xlabel("Total episode reward")
plt.ylabel("Count")
plt.title("Distribution of total rewards (random policy)")
plt.show()










# env.reset(seed=0)
# env.induce_hemorrhage("liver", 0.3)
# terminated = False
# truncated = False
# count = 0
# while not terminated and not truncated:
#     obs, reward, terminated, truncated, info = env.step(action=[-1,0.2,-1])
#     print(obs)
#     count += 1
# print(info)
# print(count)

# pulse = PulseEngine()
# pulse.log_to_console(True)
# pulse.serialize_from_file(os.path.join(parent_dir, "configs", "patient_configs", "Patient0@0s.json"))
# hemorrhage = SEHemorrhage()
# hemorrhage.set_comment("Laceration to the liver")
# hemorrhage.set_compartment(eHemorrhage_Compartment.Liver)
# hemorrhage.get_severity().set_value(0.7)
# pulse.process_action(hemorrhage)
# advanced = True
# count = 0
# while advanced:
#     advanced = pulse.advance_time_s(60)
#     count += 1
# print(count)
