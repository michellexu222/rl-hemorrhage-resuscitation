from pulse.cdm.patient import SEPatientConfiguration
from pulse.engine.PulseEngine import PulseEngine
from pulse.cdm.patient_actions import SEHemorrhage, eHemorrhage_Compartment, SESubstanceBolus, SESubstanceCompoundInfusion, eSubstance_Administration
from pulse.cdm.scalars import VolumePerTimeUnit, VolumeUnit, MassPerVolumeUnit, TimeUnit, LengthUnit, MassUnit

import json
import csv
import os
import string
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

class HemorrhageEnv (gym.Env):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, state_file=None, log_file: string="episode_log.csv"):
        super().__init__()
        self.pulse = PulseEngine()
        self.pulse.log_to_console(False)
        self.state_file = state_file

        # Discrete action space
        self.action_map = ["saline", "PRBC", "blood", "lactated_ringers", "epinephrine", "nothing"]
        self.action_space = spaces.Discrete(len(self.action_map))

        n_features = 10
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # create list of all patient state files
        patient_config_dir = os.path.join(script_dir, "..", "configs", "patient_configs")
        self.patient_files = [
            os.path.join(patient_config_dir, f)
            for f in os.listdir(patient_config_dir)
            if f.endswith(".json")
        ]
        if not self.patient_files:
            raise FileNotFoundError("No patient config files found.")

        # set up logging
        parent_dir = os.path.dirname(script_dir)
        log_dir = os.path.join(parent_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(parent_dir, "logs", log_file)

        if not os.path.exists(self.log_file):
            os.makedirs(target_dir, exist_ok=True)
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "patient_file", "reward", "length", "outcome", "loss"])

        self.episode_count = 0
        # self.episode_reward = 0  set in reset()
        # self.safety_violations = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.pulse.clear() # Clear any existing state

        # Load initial state if given
        if self.state_file:
            loaded = self.pulse.serialize_from_file(self.state_file, None)
            if not loaded:
                raise FileNotFoundError(f"State file {self.state_file} not found.")

            with open(self.state_file, "r") as f:
                self.patient_data = json.load(f)

        else:
            # load random patient file
            chosen_file = random.choice(self.patient_files)
            loaded = self.pulse.serialize_from_file(chosen_file, None)
            if not loaded:
                raise RuntimeError(f"Failed to load patient state: {chosen_file}")

            with open(chosen_file, "r") as f:
                self.patient_data = json.load(f)

        self.last_patient_file = chosen_file if chosen_file else self.state_file
        self.episode_reward = 0
        self.episode_length = 0
        self.safety_violations = 0
        self.episode_outcome = None # "death", "stabilized", "max length reached"

        self.pulse.advance_time_s(1)
        self.history = [] # list of List[action, reward, next_state]
        self.prev_obs = self.get_state()
        self.baseline_map = self.prev_obs["MeanArterialPressure"]

        obs = self._obs_to_array(self.prev_obs)
        info = {}
        return obs, info

    def get_state(self):
        data = self.pulse.pull_data()
        #self.pulse.print_results
        features = {}
        for idx, req in enumerate(self.pulse._data_request_mgr.get_data_requests()):
            # #print(f"Index {idx}: {req.get_property_name()} ({req.get_unit()})")
            # #print(f"{req.get_property_name()} ({req.get_unit()}): {data[idx+1]}")
            # if idx < 9:
            features[req.get_property_name()] = int(data[idx+1])

        features["age"] = self.patient_data["CurrentPatient"]["Age"]["ScalarTime"]["Value"]
        return features

    def induce_hemorrhage(self, compartment, severity):
        hemorrhage = SEHemorrhage()
        hemorrhage.set_comment("Induced ICH")
        hemorrhage.set_compartment(compartment)
        hemorrhage.get_severity().set_value(severity)
        self.pulse.process_action(hemorrhage)

    def give_saline(self, volume: float = 250, rate: float = 10):
        """
        volume given in mL
        rate given in mL / min
        use high rate to simulate bolus
        """
        substance = SESubstanceCompoundInfusion()
        substance.set_compound("Saline")
        substance.get_bag_volume().set_value(volume, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)

    def give_PRBCs(self, volume: float = 250, rate: float = 10):
        """
        packed red blood cells
        volume given in mL
        rate given in mL / min
        use high rate to simulate bolus
        """
        substance = SESubstanceCompoundInfusion()
        substance.set_compound("PackedRBC")
        substance.get_bag_volume().set_value(volume, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)

    def give_blood(self, volume: float = 250, rate: float = 10):
        """
        volume given in mL
        rate given in mL/min
        use high rate to simulate bolus
        """
        substance = SESubstanceCompoundInfusion()
        substance.set_compound("Blood")
        substance.get_bag_volume().set_value(volume, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)

    def give_lactated_ringers(self, volume: float = 250, rate: float = 10):
        """
        volume given in mL
        rate given in mL/min
        use high rate to simulate bolus
        """
        substance = SESubstanceCompoundInfusion()
        substance.set_compound("LactatedRingers")
        substance.get_bag_volume().set_value(volume, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)

    def give_epinephrine(self, volume = 250, concentration = 10):
        """
        volume give in mL
        concentration in mg/mL
        use high rate to simulate bolus
        """
        bolus = SESubstanceBolus()
        bolus.set_admin_route(eSubstance_Administration.Intramuscular)
        bolus.set_substance("Epinephrine")
        bolus.get_dose().set_value(volume, VolumeUnit.mL)
        bolus.get_concentration().set_value(concentration, MassPerVolumeUnit.mg_Per_mL)
        bolus.get_admin_duration().set_value(2, TimeUnit.s)
        self.pulse.process_action(bolus)

    def step(self, action_idx):
        """
        advances time by 15 s

        the action parameter is only for appending to history

        returns next state observation, reward, if next state is terminal

        """

        # if action not in ["saline", "PRBC", "blood", "lactated_ringers", "epinephrine", "nothing"]:
        #     raise ValueError(f"Invalid action {action}")
        if action_idx > len(self.action_map) - 1:
            raise IndexError("Action index out of range")

        # apply action
        if self.action_map[action_idx] == "saline": self.give_saline()
        if self.action_map[action_idx] == "lactated_ringers": self.give_lactated_ringers()
        if self.action_map[action_idx] == "PRBC": self.give_PRBCs()
        if self.action_map[action_idx] == "blood": self.give_blood()
        if self.action_map[action_idx] == "epinephrine": self.give_epinephrine()

        self.pulse.advance_time_s(60)
        new_obs_dict = self.get_state()
        print(new_obs_dict)
        terminated, cause = self.is_terminal()
        reward = self._compute_reward(new_obs_dict, cause)
        truncated = False # True if max steps reached
        obs = self._obs_to_array(new_obs_dict)

        self.episode_reward += reward
        self.episode_length += 1

        if terminated:
            self.episode_count += 1
            self._log_episode()

        self.history.append([self.prev_obs, self.action_map[action_idx], new_obs_dict, reward])
        self.prev_obs = new_obs_dict
        info = {}
        return obs, reward, terminated, truncated, info

    def _log_episode(self):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count,
                self.last_patient_file,
                self.episode_reward,
                self.episode_length,
                self.episode_outcome if self.episode_outcome else "unknown",
                self.last_loss if hasattr(self, "last_loss") else "n/a"
            ])


    def _obs_to_array(self, features: dict):
        vals = list(features.values())
        return np.array(vals, dtype=np.float32)

    def _compute_reward(self, next_obs: dict, terminal_cause: string):

        # per-step components (compute each timestep)
        # 1) MAP in-range reward (primary physiological signal)
        MAP = next_obs["MeanArterialPressure"]
        HR = next_obs["HeartRate"]
        SBP = next_obs["SystolicArterialPressure"] # systolic arterial pressure = systolic blood pressure
        shock_index = HR / max(1, SBP) # avoid division by zero
        CO = next_obs["CardiacOutput"]

        # previous state data to calculate trends
        prev_map = self.prev_obs["MeanArterialPressure"]  # or store prev MAP
        delta_map = MAP - prev_map
        if delta_map < 0:  # dropping
            r_trend = delta_map / 10.0  # e.g., -1 if drop by 10 mmHg in one step
        else:
            r_trend = 0.0

        # deviation from start
        baseline_map = self.baseline_map  # set at reset
        deviation = baseline_map - MAP
        if deviation > 0:
            r_deviation = - deviation / 30.0  # -1 if 30 mmHg drop
        else:
            r_deviation = 0.0

        # if 65 <= MAP <= 90:
        #     r_map = 0.12
        # else:
        #     # scale penalty with distance from nearest bound
        #     if MAP < 65:
        #         r_map = -0.02 * (65 - MAP)  # stronger penalty the lower MAP is
        #     else:
        #         r_map = -0.01 * (MAP - 90)  # small penalty for overshoot

        # 2) Shock-index penalty (early-warning)
        if 70 <= MAP <= 100:
            r_map = 1.0
        elif MAP < 70:
            r_map = - (70 - MAP) / (70 - 40)  # scales to -1 at 40
        else:
            r_map = - (MAP - 100) / (120 - 100)  # scales to -1 at 120
        r_map = max(-1, min(1, r_map)) # clip to [-1, 1]

        # shock_index = HR / SBP
        # if shock_index > 1.0:
        #     r_shock = -0.25
        # elif shock_index > 0.9:
        #     r_shock = -0.10
        # else:
        #     r_shock = 0.0

        if shock_index <= 0.7:
            r_shock = 0
        elif shock_index >= 1.0:
            r_shock = -1.0
        else:
            r_shock = - (shock_index - 0.7) / (1.0 - 0.7)
        r_shock = max(-1, min(0, r_shock))  # only penalty

        # 3) Cardiac output shaping (optional, mild)
        # encourage CO not to be too low (CO in L/min)
        # if CO < 3.0:
        #     r_co = -0.05 * (3.0 - CO)
        # else:
        #     r_co = 0.0

        if 4.0 <= CO <= 6.0:  # typical normal range
            r_co = 1
        elif CO < 4.0:
            r_co = - (4.0 - CO) / 3.0  # down to -1 if CO=1
        else:  # CO > 6
            r_co = - (CO - 6.0) / 4.0  # down to -1 if CO=10
        r_co = max(-1, min(0.5, r_co))  # clipped

        # # 4) Action cost (discourage wasteful interventions)
        # # fluid_vol_this_step in mL (e.g. 250 or 500)
        # r_fluid_cost = -0.02 * (fluid_vol_this_step / 250.0)
        # # blood_units_this_step: number of RBC units given this timestep
        # r_blood_cost = -0.12 * blood_units_this_step

        # 5) Small survival bonus (encourage lasting life)
        r_survive = 0.005  # per step small positive

        # Total per-step reward (clipped)
        # r_step = r_map + r_shock + r_co + r_fluid_cost + r_blood_cost + r_survive

        # ---- Weighted sum ----
        reward = (
                0.4 * r_map +
                0.3 * r_shock +
                0.1 * r_co +
                0.1 * r_trend +
                0.1 * r_deviation
                + r_survive
        )

        # Terminal reward
        # bias towards caution / avoid death
        # Death
        if terminal_cause == "death":
            reward -= 10.0

        # Stabilized
        if terminal_cause == "stabilization":
            # base stabilization reward
            reward += 6.0

            # # bonus adjustments to prefer "healthier" stabilization
            # # use averages over the final K timesteps (e.g., last 5 steps)
            # mean_MAP_final = mean(MAP over last K steps)
            # total_fluids_L = cumulative_fluid_mL / 1000.0
            # total_PRBC = cumulative_PRBC_units
            #
            # # positive bonus for higher mean MAP in final window (but small)
            # bonus_map = +0.3 * ((mean_MAP_final - 65.0) / 10.0)  # ~ -0.3..+0.9 range
            #
            # # penalties for resource use (discourage huge volumes or transfusions)
            # penalty_fluids = -0.6 * (total_fluids_L / 3.0)  # normalized to ~3L typical large resus
            # penalty_prbc = -0.5 * total_PRBC
            #
            # R_terminal = R_base + bonus_map + penalty_fluids + penalty_prbc
            #
            # # clamp terminal reward
            # R_terminal = max(-5.0, min(R_terminal, +10.0))

        return reward

    def is_terminal(self) -> tuple[bool, string]:
        """
        use arbitrary values for now
        
        death if MAP < 50 mmHg or organ perfusion <50%
        """
        death_map_threshold = 50
        stable_map_low = 65
        stable_map_high = 90
        shock_index_limit = 0.9 # shock index HR/SBP must be <= this

        n_timesteps = 5 # number of timesteps needed to determine stablization/death

        if len(self.history) < n_timesteps:
            return False, "not terminal"


        state_window = np.array(self.history[-n_timesteps:])[:, 2] # next_state of last 5 List[prev_state, action, next_state, reward]
        maps = np.array([s['MeanArterialPressure'] for s in state_window])
        saps = np.array([s['SystolicArterialPressure'] for s in state_window])
        heart_rates = np.array([s['HeartRate'] for s in state_window])
        shock_indices = heart_rates / saps
        # stabilization
        if min(maps) >= stable_map_low and max(maps) <= stable_map_high and (shock_indices <= shock_index_limit).all():
            return True, "stabilization"

        # death --> if map drops low even once
        if min(maps) < death_map_threshold:
            return True, "death"

        return False, "not terminal"

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "..", "configs", "patient_configs")
os.makedirs(target_dir, exist_ok=True)  # make sure it exists

env = HemorrhageEnv()
env.induce_hemorrhage(eHemorrhage_Compartment.Liver, 0.7)
env.give_blood(10, 0.7)
obs, reward, terminated, truncated, info = env.step(action_idx=2)
print(obs, "\n", reward, "\n", terminated)
obs, reward, terminated, truncated, info = env.step(action_idx=2)
print(obs, "\n", reward, "\n", terminated)
#check_env(env)