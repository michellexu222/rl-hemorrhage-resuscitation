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

class IntracranialHemorrhageEnv (gym.Env):
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

        obs = self._obs_to_array(self.prev_obs)
        info = {}
        return obs, info

    def get_state(self):
        data = self.pulse.pull_data()
        #self.pulse.print_results
        features = {}
        for idx, req in enumerate(self.pulse._data_request_mgr.get_data_requests()):
            #print(f"Index {idx}: {req.get_property_name()} ({req.get_unit()})")
            #print(f"{req.get_property_name()} ({req.get_unit()}): {data[idx+1]}")
            if idx < 9:
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

        self.pulse.advance_time_s(15)
        new_obs_dict = self.get_state()
        reward = self._compute_reward()
        terminated = self.is_terminal()
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

    def _compute_reward(self):
        return 1

    def is_terminal(self) -> bool:
        """
        use arbitrary values for now
        """
        death_map_threshold = 60

        if len(self.history) < 6:
            return False

        state_window = np.array(self.history[-10:])[:, 2]
        maps = [s['MeanArterialPressure'] for s in state_window]

        # return True if dead (max MAP below threshold) or if stable (all map values between 70 and 110)
        if max(maps) < death_map_threshold or (min(maps) > 70 and max(maps) < 110):
            return True

        return False

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "..", "configs", "patient_configs")
os.makedirs(target_dir, exist_ok=True)  # make sure it exists

env = IntracranialHemorrhageEnv()
env.induce_hemorrhage(eHemorrhage_Compartment.Brain, 0.7)
env.give_blood(10, 0.7)
obs, reward, terminated, truncated, info = env.step(action_idx=2)
print(obs, "\n", reward, "\n", terminated)
obs, reward, terminated, truncated, info = env.step(action_idx=2)
print(obs, "\n", reward, "\n", terminated)
check_env(env)