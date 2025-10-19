from pulse.cdm.patient import SEPatientConfiguration
from pulse.engine.PulseEngine import PulseEngine, eModelType
from pulse.cdm.patient_actions import SEHemorrhage, eHemorrhage_Compartment, SESubstanceBolus, SESubstanceInfusion, SESubstanceCompoundInfusion, eSubstance_Administration
from pulse.cdm.scalars import VolumePerTimeUnit, VolumeUnit, MassPerVolumeUnit, TimeUnit, LengthUnit, MassUnit

import json
import csv
import os
import string
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from stable_baselines3.common.env_checker import check_env

class HemorrhageEnv (gym.Env):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, state_file=None, log_file: string="episode_log.csv"):
        super().__init__()
        self.pulse = PulseEngine()
        self.pulse.log_to_console(False)

        self.state_file = state_file

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.f = ["HeartRate", "SystolicArterialPressure", "MeanArterialPressure", "OxygenSaturation", "RespirationRate", "SkinTemperature", "EndTidalCarbonDioxidePressure", "age", "sex", "bmi"]
        n_features = len(self.f)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # create list of all patient state files
        patient_config_dir = os.path.join(self.script_dir, "..", "configs", "patient_configs")
        self.patient_files = [
            os.path.join(patient_config_dir, f)
            for f in os.listdir(patient_config_dir)
            if f.endswith(".json")
        ]
        if not self.patient_files:
            raise FileNotFoundError("No patient config files found.")

        # set up logging
        parent_dir = os.path.dirname(self.script_dir)
        log_dir = os.path.join(parent_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(parent_dir, "logs", log_file)

        if not os.path.exists(self.log_file):
            os.makedirs(target_dir, exist_ok=True)
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "patient_file", "reward", "length", "outcome", "loss"])

        self.episode_count = 0

        # self.safety_violations = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._rng, _ = gym.utils.seeding.np_random(seed)
        self.pulse.clear() # Clear any existing state
        self.load_state()

        self.episode_reward = 0
        self.episode_length = 0
        self.safety_violations = 0
        self.episode_outcome = None # "death", "stabilized", "truncated", or "failed to advance"

        # self.pulse.advance_time_s(1)
        self.history = [] # list of List[prev_state, action, reward, next_state]
        self.prev_obs = self._get_state()
        self.baseline_map = self.prev_obs["MeanArterialPressure"]
        self.baseline_bv = self.prev_obs["BloodVolume"]
        self.hemorrhage_type : tuple[string, float] = None # tuple["compartment", severity]

        self.V_blood = self.prev_obs["BloodVolume"]
        self.V_cryst = 0
        self.k_form, self.k_form_base = 0.05, 0.05 # clot formation rate
        self.k_lysis, self.k_lysis_base = 0.015, 0.015 # clot breakdown rate (fibrinolysis)
        self.C_prev = 0.15 # current fraction of clot formed
        self.S_base = 0

        self.induce_hemorrhage()

        obs = self._obs_to_array({feature: value for feature, value in self._get_state().items() if feature in self.f})
        info = {"state_file": self.last_patient_file}

        return obs, info


    def load_state(self):
        chosen_file = None
        # Load initial state if given
        if self.state_file:

            loaded = self.pulse.serialize_from_file(self.state_file, None)
            if not loaded:
                raise FileNotFoundError(f"State file {self.state_file} not found.")
            #print("loaded given state file")
            with open(self.state_file, "r") as f:
                self.patient_data = json.load(f)
            #print(self.patient_data["Configuration"]["TimeStep"]["ScalarTime"]["Value"])

        else:
            # print("loading random state file")
            # load random patient file
            chosen_file = self._rng.choice(self.patient_files)
            # chosen_file = os.path.join(self.script_dir, "configs", "patient_configs", "Patient2@0s.json")
            loaded = self.pulse.serialize_from_file(chosen_file, None)
            if not loaded:
                raise RuntimeError(f"Failed to load patient state: {chosen_file}")

            with open(chosen_file, "r") as f:
                self.patient_data = json.load(f)

        self.last_patient_file = chosen_file if chosen_file else self.state_file
        # print(self.patient_data["CurrentPatient"]["MeanArterialPressureBaseline"]["ScalarPressure"]["Value"])

    def _get_state(self):
        # check speed of getting data from data_request_mgr
        data = self.pulse.pull_data()
        f = ["HeartRate", "SystolicArterialPressure", "MeanArterialPressure",]
        #self.pulse.print_results
        features = {}
        for idx, req in enumerate(self.pulse._data_request_mgr.get_data_requests()):
            features[req.get_property_name()] = float(data[idx+1])

        features["age"] = self.patient_data["CurrentPatient"]["Age"]["ScalarTime"]["Value"]
        features["bmi"] = self.patient_data["CurrentPatient"]["BodyMassIndex"]["Value"]
        features["sex"] = 1 if "Sex" in self.patient_data["CurrentPatient"] else 0 # 0 if male, 1 if female

        return features

    def induce_hemorrhage(self, compartment=None, given_severity=None):

        if not compartment:
            compartment = self._rng.choice(["liver", "spleen"], p=[0.6, 0.4])

        self.hemorrhage = SEHemorrhage()
        self.hemorrhage.set_comment("Induced hemorrhage")

        if compartment == "liver":
            severity = given_severity if given_severity else self._rng.choice([0.2, 0.3, 0.5, 0.7, 0.9], p=[0.05, 0.25, 0.3, 0.3, 0.1])
            self.hemorrhage.set_compartment(eHemorrhage_Compartment.Liver)
            self.hemorrhage.get_severity().set_value(severity)
            self.pulse.process_action(self.hemorrhage)

        else:
            severity = given_severity if given_severity else self._rng.choice([0.5, 0.6, 0.7, 0.9, 1.0], p=[0.05, 0.2, 0.25, 0.3, 0.2])
            self.hemorrhage.set_compartment(eHemorrhage_Compartment.Spleen)
            self.hemorrhage.get_severity().set_value(severity)
            self.pulse.process_action(self.hemorrhage)

        if compartment == "liver": self.hemorrhage_type = ("liver", severity)
        if compartment == "spleen": self.hemorrhage_type = ("spleen", severity)
        # print(severity)
        self.S_base = severity

    def set_severity(self, MAP, temp, delta_t=1.0, alpha=0.05, beta=1.0, gamma=1.0):
        """
        Update clot strength and compute new severity

        Params:
        - MAP: current mean arterial pressure (mmHg)
        - temp: current skin temperature)
        - delta_t: timestep in minutes
        - alpha: scaling factor for how strongly MAP disrupts clot formation; larger = more sensitive, i.e. small rises in MAP strongly slow clotting

        Returns:
        - S_new: updated hemorrhage severity (0-1)
        """
        #self.k_form = min(self.k_form, 0.05) # upper bound for clotting rate
        #self.k_lysis = min(self.k_lysis, 0.05) # upper bound for fibrinolysis

        # f_MAP = max(0.0, 1.0 - alpha * max(0.0, (MAP - MAP_ref)) / MAP_ref)
        MAP_ref = self.baseline_map
        f_MAP = np.exp(-alpha * np.abs(MAP - MAP_ref) / MAP_ref)

        # Temperature effect
        if temp >= 36.0:
            f_temp = 1.0
        elif temp >= 34.0:
            f_temp = 0.75
        else:
            f_temp = 0.5

        # Clot formation update
        dC_dt = self.k_form * f_MAP * f_temp * self.C_prev * (1.0 - self.C_prev) - self.k_lysis * self.C_prev
        C_new = self.C_prev + delta_t * dC_dt
        C_new = np.clip(C_new, 0, 1)  # clamp
        #print(f"C: {self.C_prev} -> {C_new} | dC/dt: {dC_dt} | f_MAP: {f_MAP} | f_temp: {f_temp} | k_form: {self.k_form}")

        self.C_prev = C_new

        dilution_ratio = self.V_cryst / self.prev_obs["BloodVolume"]
        # D = 1 + 1.5 * dilution_ratio
        # beta = 1 # equal volumes of crystalloids and blood roughly halve clot effectiveness --> large volume crystalloid (>1:1 rel to BV) causes dilutional coagulopathy
        effective_clot = C_new / (1 + beta * dilution_ratio)

        if 0 <= self.S_base < 0.3: S_min = self.S_base * 0.25
        elif 0.3 <= self.S_base <= 0.5: S_min = self.S_base * 0.5
        else: S_min = self.S_base * 0.7

        # New severity
        #S_new = S_base * (1.0 - C_new) * D
        S_prev = self.hemorrhage.get_severity().get_value()
        alpha = 0.05 # smoothing factor, 0-1
        target = self.S_base * (1 - effective_clot)
        S_new = (1 - alpha) * S_prev + alpha * target
        #print(self.S_base)
        #max_delta = 0.02 * self.S_base
        #S_new = np.clip(S_new, S_prev - max_delta, S_prev + max_delta)
        S_new = np.clip(S_new, S_min, self.S_base) # hemorrhage can't get worse than initial severity and can't be lower than 0.25
        # print(f"dilution ratio {dilution_ratio}, effective clot {effective_clot}")
        self.hemorrhage.get_severity().set_value(S_new)
        self.pulse.process_action(self.hemorrhage)

        return S_new


    def give_saline(self, volume: float = 250, rate: float = 250):
        """
        unused currently
        """
        self.decay_k()

        substance = SESubstanceCompoundInfusion()
        substance.set_compound("Saline")
        substance.get_bag_volume().set_value(volume, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)

        self.V_cryst += rate

    def give_blood(self, rate):
        """
        volume given in mL
        rate given in mL/min
        use high rate to simulate bolus
        """
        # self.decay_k()
        rate = (rate + 1) / 2 * 300 # scale from [-1, 1] to [0, 750]

        substance = SESubstanceCompoundInfusion()
        substance.set_compound("Blood")
        substance.get_bag_volume().set_value(rate, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)
        # print(self.k_form)
        self.V_blood += rate
        factor = 1 + 0.001 * rate * 1
        self.k_form *= factor
        # print(f"new k_form = {self.k_form}")
        self.k_lysis *= 1 / factor

    def give_lactated_ringers(self, rate):
        """
        volume = rate, b/c 1 min timesteps
        rate given in mL/min
        use high rate to simulate bolus
        """
        # self.decay_k()
        rate = (rate + 1) / 2 * 750 # scale from [-1, 1] to [0, 750]
        substance = SESubstanceCompoundInfusion()
        substance.set_compound("LactatedRingers")
        substance.get_bag_volume().set_value(rate, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)

        self.V_cryst += rate * 1

    def give_norepinephrine(self, rate, concentration=0.016):
        """
        volume give in mL
        concentration in mg/mL
        use high rate to simulate bolus
        """

        rate = (rate + 1) / 2 * 0.04 # scale from [-1, 1] to [0, 0.04] mL/min = [0, 1] mcg/kg/min

        infusion = SESubstanceInfusion()
        infusion.set_substance("Norepinephrine")
        infusion.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        infusion.get_concentration().set_value(concentration, MassPerVolumeUnit.from_string("g/L"))
        self.pulse.process_action(infusion)

    def decay_k(self):
        # every step decay k_form and k_lysis towards base values
        LAMBDA = 0.02
        self.k_form = self.k_form + LAMBDA * (self.k_form_base - self.k_form) * 1 # 1 minute
        self.k_lysis = self.k_lysis + LAMBDA * (self.k_lysis_base - self.k_lysis) * 1  # 1 minute

    def step(self, action: list[float]):
        """
        action order: lactated ringer's rate, blood rate, norepinephrine rate
        """

        self.decay_k()
        self.give_lactated_ringers(rate=action[0])
        self.give_blood(rate=action[1])
        self.give_norepinephrine(rate=action[2])

        #t0 = time.time()
        advanced = self.pulse.advance_time_s(60)
        # t1 = time.time()
        # print(f"time to advance time: {t1 - t0} seconds")

        # t0 = time.time()
        new_obs_dict = self._get_state()
        # t1 = time.time()
        # print(f"time to get new state: {t1 - t0} seconds")
        # print(new_obs_dict)
        # t0 = time.time()

        sev = self.set_severity(new_obs_dict["MeanArterialPressure"], new_obs_dict["SkinTemperature"])

        # t1 = time.time()
        # print(f"time to set severity: {t1 - t0} seconds")

        # t0 = time.time()
        terminated, cause = self.is_terminal(new_obs_dict)
        # t1 = time.time()
        # print(f"time to get terminal: {t1 - t0} seconds")

        if not advanced:
            terminated = True
            cause = "failed to advance"

        # t0 = time.time()
        reward = self._compute_reward(new_obs_dict, cause)
        # t1 = time.time()
        # print(f"time to compute reward: {t1 - t0} seconds")
        truncated = True if cause == "truncated" else False # True if max steps reached
        # obs = self._obs_to_array(new_obs_dict)

        self.episode_reward += reward
        self.episode_length += 1

        info = {}
        if terminated or truncated:
            self.episode_count += 1
            self.episode_outcome = cause
            # t0 = time.time()
            self._log_episode()
            # t1 = time.time()
            # print(f"time to log episode: {t1 - t0} seconds")
            info = {"episode": {"r": self.episode_reward, "l": self.episode_length}, "o": self.episode_outcome, "hem": self.hemorrhage_type}

        self.history.append([self.prev_obs, action, new_obs_dict, reward])
        self.prev_obs = new_obs_dict
        # t0 = time.time()
        shown_obs = self._obs_to_array({feature: value for feature, value in new_obs_dict.items() if feature in self.f})
        # t1 = time.time()
        # print(f"time to obs_to_array(): {t1 - t0} seconds")
        return shown_obs, reward, terminated, truncated, info

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
        #print(features)
        vals = list(features.values())
        return np.array(vals, dtype=np.float32)

    def _compute_reward(self, next_obs: dict, terminal_cause: string):

        # per-step components
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

        # mean arterial pressure within range
        if 70 <= MAP <= 100:
            r_map = 1.0
        elif MAP < 70:
            r_map = - (70 - MAP) / (70 - 40)  # scales to -1 at 40
        else:
            r_map = - (MAP - 100) / (120 - 100)  # scales to -1 at 120
        r_map = max(-1, min(1, r_map)) # clip to [-1, 1]

        # shock index
        if shock_index <= 0.7:
            r_shock = 0
        elif shock_index >= 1.0:
            r_shock = -1.0
        else:
            r_shock = - (shock_index - 0.7) / (1.0 - 0.7)
        r_shock = max(-1, min(0, r_shock))  # only penalty

        # cardiac output
        if 4.0 <= CO <= 6.0:  # typical normal range
            r_co = 1
        elif CO < 4.0:
            r_co = - (4.0 - CO) / 3.0  # down to -1 if CO=1
        else:  # CO > 6
            r_co = - (CO - 6.0) / 4.0  # down to -1 if CO=10
        r_co = max(-1, min(0.5, r_co))  # clipped

        # # Action cost
        # # fluid_vol_this_step in mL (e.g. 250 or 500)
        # r_fluid_cost = -0.02 * (fluid_vol_this_step / 250.0)
        # # blood_units_this_step: number of RBC units given this timestep
        # r_blood_cost = -0.12 * blood_units_this_step

        # Small survival bonus
        r_survive = 0.005  # per step small positive

        # Weighted sum of components
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
        if terminal_cause == "death" or terminal_cause == "failed to advance":
            reward -= 10.0

        # Stabilized
        if terminal_cause == "stabilization":
            # base stabilization reward
            reward += 6.0

        return reward

    def is_terminal(self, new_obs) -> tuple[bool, string]:
        # criteria for stabilization or truncation, death determined by if engine can advance time

        death_map_threshold = 10
        stable_map_low = 65
        stable_map_high = 100
        shock_index_limit = 0.9  # shock index HR/SBP must be <= this
        stable_hr_low = 50  # bpm
        stable_hr_high = 120
        base_time = 20
        # n_timesteps = 10
        type, severity = self.hemorrhage_type
        if type == "liver" or type == "both" and 0 <= severity < 0.3: n_timesteps = 5  # number of timesteps needed to determine stablization
        if type == "liver" or type == "both" and 0.3 <= severity < 0.5: n_timesteps = 4
        if type == "liver" or type == "both" and 0.5 <= severity <= 1: n_timesteps = 3
        if type == "spleen": n_timesteps = 5

        # if new_obs["CardiacOutput"] < 1.5 or new_obs["BloodVolume"] < 2500 or new_obs["MeanArterialPressure"] < 35: return True, "death"

        if len(self.history) < base_time:
            return False, "not terminal"
        if len(self.history) > 60:
            return True, "truncated"

        # Extract next_state dicts from history
        next_states = [h[2] for h in self.history[-n_timesteps:]]
        state_window = np.array(next_states + [new_obs])

        maps = np.array([s['MeanArterialPressure'] for s in state_window])
        saps = np.array([s['SystolicArterialPressure'] for s in state_window])
        heart_rates = np.array([s['HeartRate'] for s in state_window])
        shock_indices = heart_rates / saps
        COs = np.array([s['CardiacOutput'] for s in state_window])
        BVs = np.array([s['BloodVolume'] for s in state_window])
        # stabilization
        if (min(maps) >= stable_map_low and max(maps) <= stable_map_high
                and (shock_indices <= shock_index_limit).all()
                and min(heart_rates) >= stable_hr_low and max(heart_rates) <= stable_hr_high):
            return True, "stabilization"

        if max(heart_rates) < 35:
            return True, "death (HR)"
        # if min(COs) < 1.5 or min(BVs) < 2500 or min(maps) < 45: #  or len(np.unique(maps)) == 1
        #    return True, "death"

        return False, "not terminal"



# script_dir = os.path.dirname(os.path.abspath(__file__))
# target_dir = os.path.join(script_dir, "..", "configs", "patient_configs")
# os.makedirs(target_dir, exist_ok=True)  # make sure it exists
#
# env = HemorrhageEnv()
# env.induce_hemorrhage(eHemorrhage_Compartment.Liver, 0.7)
# env.give_blood(10, 0.7)
# obs, reward, terminated, truncated, info = env.step(action_idx=2)
# print(obs, "\n", reward, "\n", terminated)
# obs, reward, terminated, truncated, info = env.step(action_idx=2)
# print(obs, "\n", reward, "\n", terminated)
# #check_env(env)