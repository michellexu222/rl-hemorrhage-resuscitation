from pulse.cdm.patient import SEPatientConfiguration
from pulse.engine.PulseEngine import PulseEngine
from pulse.cdm.patient_actions import SEHemorrhage, eHemorrhage_Compartment, SESubstanceBolus, SESubstanceCompoundInfusion, eSubstance_Administration
from pulse.cdm.scalars import VolumePerTimeUnit, VolumeUnit, MassPerVolumeUnit, TimeUnit, LengthUnit, MassUnit

import json
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

class IntracranialHemorrhageEnv (gym.Env):


    def __init__(self, patient_file = None, state_file=None):
        super().__init__()
        self.pulse = PulseEngine()
        self.pulse.log_to_console(False)
        self.patient_file = patient_file
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

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Load initial state if given
        if self.state_file:
            loaded = self.pulse.serialize_from_file(self.state_file, None)
            if not loaded:
                raise FileNotFoundError(f"State file {self.state_file} not found.")
        else:
            self.pulse.clear()  # Clear any existing state
        self.pulse.advance_time_s(1)
        self.history = [] # list of List[action, reward, next_state]

        with open(self.state_file, "r") as f:
            self.patient_data = json.load(f)

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

        self.history.append([self.prev_obs, self.action_map[action_idx], new_obs_dict, reward])
        self.prev_obs = new_obs_dict
        info = {}
        return obs, reward, terminated, truncated, info

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

env = IntracranialHemorrhageEnv(state_file=os.path.join(target_dir, f"Patient0@0s.json"))
env.induce_hemorrhage(eHemorrhage_Compartment.Brain, 0.7)
env.give_blood(10, 0.7)
prev_obs, action, new_obs, reward = env.step("blood")
print(prev_obs, "\n", action, "\n", new_obs, "\n", reward)
check_env(env)