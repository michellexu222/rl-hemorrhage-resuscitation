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

    def __init__(self, seed=None, state_file=None, log_file: string="episode_log.csv"):
        super().__init__()
        self.pulse = PulseEngine()
        self.pulse.log_to_console(False)
        self.seed = seed
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
        # self.episode_reward = 0  set in reset()
        # self.safety_violations = 0
        # self.reset()

    def reset(self, options=None):
        super().reset(seed=self.seed)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.pulse.clear() # Clear any existing state
        chosen_file = None
        # Load initial state if given
        if self.state_file:

            loaded = self.pulse.serialize_from_file(self.state_file, None)
            if not loaded:
                raise FileNotFoundError(f"State file {self.state_file} not found.")
            print("loaded given state file")
            with open(self.state_file, "r") as f:
                self.patient_data = json.load(f)


        else:
            print("using random state file")
            # load random patient file
            print(self.patient_files)
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
        self.prev_obs = self._get_state()
        self.baseline_map = self.prev_obs["MeanArterialPressure"]
        self.baseline_bv = self.prev_obs["BloodVolume"]
        self.hemorrhage_type : tuple[string, float] = None # tuple["compartment", severity]

        self.V_blood = self.prev_obs["BloodVolume"]
        self.V_cryst = 0
        self.k_form, self.k_form_base = 0.05, 0.05 #k_form=0.08, k_lysis=0.02
        self.k_lysis, self.k_lysis_base = 0.015, 0.015
        self.C_prev = 0.15 # current fraction of clot formed, no clot in beginning
        self.S_base = 0

        obs = self._obs_to_array(self.prev_obs)
        info = {"state_file": self.last_patient_file}
        return obs, info

    def _get_state(self):
        data = self.pulse.pull_data()
        #self.pulse.print_results
        features = {}
        for idx, req in enumerate(self.pulse._data_request_mgr.get_data_requests()):
            features[req.get_property_name()] = float(data[idx+1])

        features["age"] = self.patient_data["CurrentPatient"]["Age"]["ScalarTime"]["Value"]
        return features

    def induce_hemorrhage(self, compartment, severity):
        self.hemorrhage = SEHemorrhage()
        self.hemorrhage.set_comment("Induced hemorrhage")
        self.hemorrhage.set_compartment(compartment)
        self.hemorrhage.get_severity().set_value(severity)
        self.pulse.process_action(self.hemorrhage)

        if compartment == eHemorrhage_Compartment.Liver: self.hemorrhage_type = ("liver", severity)
        if compartment == eHemorrhage_Compartment.Spleen: self.hemorrhage_type = ("spleen", severity)
        self.S_base = severity

    def set_severity(self, MAP, temp, delta_t=1.0, alpha=0.05, beta=1.0, gamma=1.0):
        """
        Update clot strength and compute new hemorrhage severity.

        Parameters:
        - C_prev: previous clot strength (0-1)
        - MAP: mean arterial pressure (mmHg)
        - temp: skin temperature (°C)
        - S_base: initial severity (0-1)
        - MAP_ref: baseline mean arterial pressure (mmHg)
        - delta_t: timestep in minutes

        - k_form: baseline clot formation rate (per min) larger = clot forms faster
        - k_lysis: baseline clot breakdown rate (per min) larger = clot dissolves faster (physiologically stands in for fibrinolysis)
        - alpha: scaling factor for how strongly MAP disrupts clot formation; larger = more sensitive, i.e. small rises in MAP strongly slow clotting

        Returns:
        - C_new: updated clot strength (0-1)
        - S_new: updated hemorrhage severity (0-1)
        """
        # MAP effect (higher MAP slows clot formation)
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
        volume given in mL
        rate given in mL / min
        use high rate to simulate bolus
        """
        self.decay_k()

        substance = SESubstanceCompoundInfusion()
        substance.set_compound("Saline")
        substance.get_bag_volume().set_value(volume, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)

        self.V_cryst += rate

    def give_PRBCs(self, volume: float = 250, rate: float = 250):
        """
        packed red blood cells
        volume given in mL
        rate given in mL / min
        use high rate to simulate bolus
        """
        self.decay_k()

        substance = SESubstanceCompoundInfusion()
        substance.set_compound("PackedRBC")
        substance.get_bag_volume().set_value(volume, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)

        k_form *= 1 + 0.00002 * rate * 1
        self.V_blood += rate

    def give_blood(self, volume: float = 250, rate: float = 250):
        """
        volume given in mL
        rate given in mL/min
        use high rate to simulate bolus
        """
        self.decay_k()

        substance = SESubstanceCompoundInfusion()
        substance.set_compound("Blood")
        substance.get_bag_volume().set_value(volume, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)
        # print(self.k_form)
        self.V_blood += rate
        factor = 1 + 0.001 * rate * 1
        self.k_form *= factor
        # print(f"new k_form = {self.k_form}")
        self.k_lysis *= 1 / factor

    def give_lactated_ringers(self, volume: float = 250, rate: float = 250):
        """
        volume given in mL
        rate given in mL/min
        use high rate to simulate bolus
        """
        self.decay_k()

        substance = SESubstanceCompoundInfusion()
        substance.set_compound("LactatedRingers")
        substance.get_bag_volume().set_value(volume, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)

        self.V_cryst += rate * 1

    def give_epinephrine(self, volume = 250, concentration = 10):
        """
        volume give in mL
        concentration in mg/mL
        use high rate to simulate bolus
        """
        self.decay_k()

        bolus = SESubstanceBolus()
        bolus.set_admin_route(eSubstance_Administration.Intramuscular)
        bolus.set_substance("Epinephrine")
        bolus.get_dose().set_value(volume, VolumeUnit.mL)
        bolus.get_concentration().set_value(concentration, MassPerVolumeUnit.mg_Per_mL)
        bolus.get_admin_duration().set_value(2, TimeUnit.s)
        self.pulse.process_action(bolus)

    def decay_k(self):
        LAMBDA = 0.02
        self.k_form = self.k_form + LAMBDA * (self.k_form_base - self.k_form) * 1 # 1 minute
        self.k_lysis = self.k_lysis + LAMBDA * (self.k_lysis_base - self.k_lysis) * 1  # 1 minute

    def step(self, action_idx):

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
        if self.action_map[action_idx] == "nothing": self.decay_k()

        advanced = self.pulse.advance_time_s(60)

        new_obs_dict = self._get_state()
        # print(new_obs_dict)
        sev = self.set_severity(new_obs_dict["MeanArterialPressure"], new_obs_dict["SkinTemperature"])
        # print(sev)
        terminated, cause = self.is_terminal(new_obs_dict)
        if not advanced:
            terminated = True
        reward = self._compute_reward(new_obs_dict, cause)
        truncated = True if cause == "truncated" else False # True if max steps reached
        obs = self._obs_to_array(new_obs_dict)

        self.episode_reward += reward
        self.episode_length += 1

        if terminated or truncated:
            self.episode_count += 1
            self.episode_outcome = cause
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

    def is_terminal(self, new_obs) -> tuple[bool, string]:

        death_map_threshold = 10
        stable_map_low = 65
        stable_map_high = 100
        shock_index_limit = 0.9 # shock index HR/SBP must be <= this

        # n_timesteps = 10
        type, severity = self.hemorrhage_type
        if type == "liver" and 0 <= severity < 0.3: n_timesteps = 70 # number of timesteps needed to determine stablization
        if type == "liver" and 0.3 <= severity < 0.5: n_timesteps = 50
        if type == "liver" and 0.5 <= severity <= 1: n_timesteps = 30
        if type == "spleen": n_timesteps = 60

        if new_obs["CardiacOutput"] < 1.5 or new_obs["BloodVolume"] < 2500 or new_obs["MeanArterialPressure"] < 45: return True, "death"

        if len(self.history) < n_timesteps:
            return False, "not terminal"
        if len(self.history) > 150:
            return True, "truncated"
        state_window = np.concatenate((np.array(self.history[-n_timesteps:])[:, 2], np.array([new_obs]))) # next_state of last 5 List[prev_state, action, next_state, reward]
        maps = np.array([s['MeanArterialPressure'] for s in state_window])
        saps = np.array([s['SystolicArterialPressure'] for s in state_window])
        heart_rates = np.array([s['HeartRate'] for s in state_window])
        shock_indices = heart_rates / saps
        COs = np.array([s['CardiacOutput'] for s in state_window])
        BVs = np.array([s['BloodVolume'] for s in state_window])
        # stabilization
        if min(maps) >= stable_map_low and max(maps) <= stable_map_high and (shock_indices <= shock_index_limit).all():
            return True, "stabilization"

        # death --> if map drops low even once
        if min(COs) < 1.5 or min(BVs) < 2500 or min(maps) < 45: #  or len(np.unique(maps)) == 1
            return True, "death"


        return False, "not terminal"

    def get_state(self):
        return self.prev_obs # after calling step() prev_obs becomes the new obs

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