import sys
sys.path.insert(0, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation")
sys.path.insert(1, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\gating")
sys.path.insert(2, r"C:\Users\michellexu\Pulse\engine\src\python\pulse\rl-hemorrhage-resuscitation\env")
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

    def __init__(self, state_file=None, eval=False, log_file: string="episode_log.csv"):
        super().__init__()
        self.pulse = PulseEngine()
        self.pulse.log_to_console(False)

        self.state_file = state_file

        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        self.f = ["HeartRate", "SystolicArterialPressure", "MeanArterialPressure", "OxygenSaturation", "RespirationRate", "SkinTemperature", "EndTidalCarbonDioxidePressure", "age", "sex", "bmi"]
        n_features = len(self.f)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # create list of all patient state files
        if eval:
            patient_config_dir = os.path.join(self.script_dir, "..", "configs", "test_patient_configs")
        else:
            patient_config_dir = os.path.join(self.script_dir, "..", "configs", "patient_configs")
        self.patient_files = [
            os.path.join(patient_config_dir, f)
            for f in os.listdir(patient_config_dir)
            if f.endswith(".json")
        ]
        if not self.patient_files:
            raise FileNotFoundError("No patient config files found.")

        # set up logging - not used anymore
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

        try:
            # preferred modern generator
            self._rng = np.random.default_rng()
            self._use_default_rng = True
        except Exception:
            # fallback to legacy RandomState for compatibility with gym seeding
            self._rng = np.random.RandomState()
            self._use_default_rng = False

    def reset(self, *, seed=None, options=None, organ=None, severity=None, sev=None):
        super().reset(seed=seed)
        self.sev = sev
        if seed is not None:
            self._rng, _ = gym.utils.seeding.np_random(seed)

        self.pulse.clear() # Clear any existing state
        self.load_state()

        self.episode_reward = 0
        self.episode_length = 0
        self.safety_violations = 0
        self.episode_outcome = None # "death", "stabilized", "truncated", or "failed to advance"

        self.history = [] # list of List[prev_state, action, reward, next_state]
        self.prev_obs = self._get_state()
        self.baseline_map = self.prev_obs["MeanArterialPressure"]
        self.baseline_bv = self.prev_obs["BloodVolume"]
        #self.base_hct = self.patient_data["BloodChemistry"]["Common"][["Hematocrit"]["Scalar0To1"]["Value"]

        if organ and severity:
            self.hemorrhage_type = (organ, severity)
        else:
            self.hemorrhage_type : tuple[string, float] = None # tuple["compartment", severity]

        # ------- Clotting model variables -------
        # updated 11/4/2025
        # clot formation
        self.k_form, self.k_form_base = 0.13, 0.13  # clot formation rate
        self.k_blood = 0.002 # how much giving blood improves clotting
        self.tau_blood = 5 # (mins) decay time for recent blood effect

        # clot breakdown
        self.k_lysis, self.k_lysis_base = 0.01, 0.01  # clot breakdown rate (fibrinolysis)
        self.beta_0 = 0.5 # base sensitivity of clot breakdown to MAP above threshold
        self.map_thresh = 75 # MAP threshold above which clot breakdown increases (mmHg)
        self.delta_c = 0.05 # amount clot breaks down by when it "pops"

        # MAP effects
        self.alpha = 1  # sensitivity of clot formation to MAP deviations
        self.map_opt = 67 # optimal MAP for clotting (mmHg) used 80 for low sev

        # clot / severity caps
        self.gamma = 0.2 # C_max = 1 - gamma * S_base
        self.eps_min, self.eps_max = 0.05, 0.2
        self.n = 1.5  # controls steepness of clotting effect on severity, n=1 (linear) -> bleed decrease steadily w/ clot,
        # n=1.5-2 -> bleed stays moderate, then drops faster when clot matures (respond quickly to clotting)
        # n=3-4-> clot has little effect until it almost complete (bleed resists clotting)

        # dilution effects
        self.tau_clear = 5 # time constant for redistribution of crystalloid from the bloodstream (minutes) (cryst redistr. half life)
        # k_beta_dil possibly

        # other stuff
        self.dt = 1.0
        self.V_blood_step = 0 # current blood given as action
        self.V_blood_recent = 0
        self.V_cryst = 0

        self.prev_cryst = 0
        self.prev_blood = 0
        self.fluid_dev = 50

        self.C_prev = 0.05 # current fraction of clot formed
        #self.S_base = 0
        if organ and severity:
            self.induce_hemorrhage(organ, severity)
        else:
            self.induce_hemorrhage()
        self.category = self._get_severity_category()

        # # for creating gating dataset using blood volume
        bv1 = self._get_state()["BloodVolume"]
        self.pulse.advance_time_s(15)
        # gating_obs_2 = self._obs_to_array({feature: value for feature, value in self._get_state().items() if feature in self.f})
        bv2 = self._get_state()["BloodVolume"]
        self.pulse.advance_time_s(15)
        bv3 = self._get_state()["BloodVolume"]
        #gating_obs_3 = self._obs_to_array({feature: value for feature, value in self._get_state().items() if feature in self.f})

        obs = self._obs_to_array({feature: value for feature, value in self._get_state().items() if feature in self.f})
        info = {"state_file": self.last_patient_file, "hem": self.hemorrhage_type}
        info["bv1"] = bv1
        info["bv2"] = bv2
        info["bv3"] = bv3
        info["sev"] = self.sev

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
        data = self.pulse.pull_data()
        f = ["HeartRate", "SystolicArterialPressure", "MeanArterialPressure",]
        features = {}
        for idx, req in enumerate(self.pulse._data_request_mgr.get_data_requests()):
            features[req.get_property_name()] = float(data[idx+1])

        features["age"] = self.patient_data["CurrentPatient"]["Age"]["ScalarTime"]["Value"]
        features["bmi"] = self.patient_data["CurrentPatient"]["BodyMassIndex"]["Value"]
        features["sex"] = 1 if "Sex" in self.patient_data["CurrentPatient"] else 0 # 0 if male, 1 if female

        return features

    def induce_hemorrhage(self, compartment=None, given_severity=None):

        if not compartment:
            compartment = self._rng.choice(["liver", "spleen"], p=[0.5, 0.5])
            sev = self.sev if self.sev else self._rng.choice(["low", "high"], p=[0.5, 0.5])
        else:
            assert self.category in ["low", "high"]
            if self.category == "low": sev = "low"
            else: sev = "high"
        self.sev = sev
        self.hemorrhage = SEHemorrhage()
        self.hemorrhage.set_comment("Induced hemorrhage")

        if compartment == "liver":
            if sev == "low":
                severity = given_severity if given_severity else self._rng.choice([0.1, 0.12, 0.15], p=[0.33, 0.34, 0.33]) # low severities (/moderate)
            else:
                severity = given_severity if given_severity else self._rng.choice([0.2, 0.3, 0.4], p=[0.3, 0.4, 0.3]) # high severity

            self.hemorrhage.set_compartment(eHemorrhage_Compartment.Liver)
            self.hemorrhage.get_severity().set_value(severity)
            self.pulse.process_action(self.hemorrhage)

        else:
            if sev == "low":
                severity = given_severity if given_severity else self._rng.choice([0.5, 0.6, 0.7, 0.8], p=[0.2, 0.3, 0.3, 0.2]) # low/mod severity
            else:
                severity = given_severity if given_severity else self._rng.choice([0.9, 1.0], p=[0.5, 0.5]) # high severity

            self.hemorrhage.set_compartment(eHemorrhage_Compartment.Spleen)
            self.hemorrhage.get_severity().set_value(severity)
            self.pulse.process_action(self.hemorrhage)

        if compartment == "liver": self.hemorrhage_type = ("liver", severity)
        if compartment == "spleen": self.hemorrhage_type = ("spleen", severity)

        self.S_base = severity

    def set_severity(self, map, temp):
        """
        Update clot strength and compute new severity

        Params:
        - MAP: current mean arterial pressure (mmHg)
        - temp: current skin temperature)

        Returns:
        - S_new: updated hemorrhage severity (0-1)
        """

        self.V_blood_recent = self.V_blood_recent * np.exp(-self.dt / self.tau_blood) + self.V_blood_step
        k_form = self.k_form_base * (1 + self.k_blood * self.V_blood_recent)

        f_map = np.exp(-self.alpha * np.abs(map - self.map_opt) / self.map_opt)

        # Temperature effect
        if temp >= 36.0 :
            f_temp = 1.0
        elif temp >= 34.0:
            f_temp = 0.75
        else:
            f_temp = 0.5

        self.dilution_ratio = min(self.V_cryst / self.prev_obs["BloodVolume"], 1)
        beta_eff = self.beta_0 * (1 + self.dilution_ratio) # increased dilution increases beta increase clot breakdown

        form_rate = k_form * f_map * f_temp * (1 - self.dilution_ratio) * self.C_prev * (1.0 - self.C_prev)
        lysis_rate = self.k_lysis + beta_eff * max(0, map - self.map_thresh) / self.map_thresh

        self.dC_dt = form_rate - lysis_rate * self.C_prev
        C_new = self.C_prev + self.dt * self.dC_dt

        # stochastic clot "pops" - clot "popping" more likely if lysis rate high and clot is not strong
        if np.random.rand() < lysis_rate * (1 - C_new) * self.dt:
            C_new -= self.delta_c
            C_new = max(0.05, C_new)

        C_max = 1 - self.gamma * self.S_base
        C_new = np.clip(C_new, 0.05, C_max)  # clamp

        self.C_prev = C_new
        self.V_cryst -= self.V_cryst * (1 - np.exp(-self.dt / self.tau_clear))

        # new severity
        epsilon = self.eps_min + (self.eps_max - self.eps_min) * self.S_base # residual bleeding when clot is perfect

        S_new = self.S_base * (epsilon + (1 - epsilon) * (1 - C_new/C_max) ** self.n) * (1 + 0.5 * self.dilution_ratio) # added the 0.5 factor to dilution ratio 11/12/2025
        S_new = np.clip(S_new, 0, 1)

        self.hemorrhage.get_severity().set_value(S_new)

        self.pulse.process_action(self.hemorrhage)

        return S_new, C_new


    def give_saline(self, volume: float = 250, rate: float = 250):
        """
        unused currently
        """

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
        rate = rate * 400 # scale from [0, 1] to [0, 600] increased max to 600 11/15/2025
        # rate = np.clip(rate, self.prev_blood - self.fluid_dev, self.prev_blood + self.fluid_dev)
        #rate = np.clip(rate, 0, 300)
        #self.prev_blood = rate

        substance = SESubstanceCompoundInfusion()
        substance.set_compound("Blood")
        substance.get_bag_volume().set_value(rate, VolumeUnit.mL)
        substance.get_rate().set_value(rate, VolumePerTimeUnit.mL_Per_min)
        self.pulse.process_action(substance)
        # print(self.k_form)
        #self.V_blood += rate
        self.V_blood_step = rate
        # factor = 1 + 0.001 * rate * 1
        # self.k_form *= factor
        # print(f"new k_form = {self.k_form}")
        # self.k_lysis *= 1 / factor

    def give_lactated_ringers(self, rate):
        """
        volume = rate, b/c 1 min timesteps
        rate given in mL/min
        use high rate to simulate bolus
        """
        rate = rate * 400 # scale from [0, 1] to [0, 500]

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
        rate = rate * 0.04 # scale from [0, 1] to [0, 0.04] mL/min = [0, 1] mcg/kg/min

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
        self.give_lactated_ringers(rate=action[0])
        self.give_blood(rate=action[1])
        self.give_norepinephrine(rate=action[2])

        advanced = self.pulse.advance_time_s(60)

        new_obs_dict = self._get_state()

        sev, clot_frac = self.set_severity(new_obs_dict["MeanArterialPressure"], new_obs_dict["SkinTemperature"])

        terminated, cause = self.is_terminal(new_obs_dict)

        if not advanced:
            terminated = True
            cause = "failed to advance"

        reward = self._compute_reward(new_obs_dict, cause, action[1]*400, action[0]*400)

        truncated = True if cause == "truncated" else False # True if max steps reached

        self.episode_reward += reward
        self.episode_length += 1

        info = {"br": new_obs_dict["BloodVolume"]}
        if terminated or truncated:
            self.episode_count += 1
            self.episode_outcome = cause

            info = {"episode": {"r": self.episode_reward, "l": self.episode_length}, "o": self.episode_outcome, "hem": self.hemorrhage_type, "br": new_obs_dict["BloodVolume"]}

        self.history.append([self.prev_obs, action, new_obs_dict, reward])
        self.prev_obs = new_obs_dict

        shown_obs = self._obs_to_array({feature: value for feature, value in new_obs_dict.items() if feature in self.f})

        info['clot_frac'] = clot_frac
        info['sev_new'] = sev

        return shown_obs, reward, terminated, truncated, info

    def _obs_to_array(self, features: dict):
        vals = list(features.values())
        return np.array(vals, dtype=np.float32)

    def _get_severity_category(self):
        organ, sev = self.hemorrhage_type
        if organ == "spleen" and sev < 0.9:
            return "low"
        elif organ == "spleen" and sev >= 0.9:
            return "high"
        elif organ == "liver" and sev < 0.2:
            return "low"
        else:
            return "high"

    def _compute_reward(self, next_obs: dict, terminal_cause: string, blood_step, cryst_step):

        if self.category == "low":
            MAP_low, MAP_high = self.baseline_map - 5, self.baseline_map + 5 #67, 77
            bad_low, bad_high = 10, 10
        if self.category == "moderate":
            MAP_low, MAP_high = 55, 75
            bad_low, bad_high = 10, 15
        if self.category == "high":
            MAP_low, MAP_high = 55, 65
            bad_low, bad_high = 10, 10
        if self.category == "massive":
            MAP_low, MAP_high = 50, 58
            bad_low, bad_high = 8

        # per-step components
        MAP = next_obs["MeanArterialPressure"]
        HR = next_obs["HeartRate"]
        SBP = next_obs["SystolicArterialPressure"] # systolic arterial pressure = systolic blood pressure
        shock_index = HR / max(1, SBP) # avoid division by zero
        CO = next_obs["CardiacOutput"]

        # deviation from target - don't want increasing MAP (over-resuscitation)
        baseline_map = self.baseline_map
        deviation = MAP - baseline_map
        if deviation > 0:
            r_deviation = - deviation / 5.0  # -1 if 5 mmHg increase // harsher for high severity
        else:
            r_deviation = 0.0

        if MAP_low <= MAP <= MAP_high:
            r_map = 1.0
        elif MAP < MAP_low:
            r_map = -(MAP_low - MAP) / 10.0  # down to -1 at 10 mmHg below
        else:  # MAP > MAP_high
            r_map = -(MAP - MAP_high) / 12.0  # -1 if MAP=MAP_high + 12

        # Clip and scale down (to not dominate total reward)
        r_map = max(-3, min(1, r_map))

        # ------- heart rate --------- 11/17/2025
        # Keep previous HR in state: self.hr_prev
        dHR = HR - self.prev_obs["HeartRate"]
        # Penalize big negative drops and low absolute HR
        w_hr_drop = 0.06
        w_hr_low = 0.05
        hr_low_thresh = 50

        r_hr = 0.0
        if dHR < 0:  # drop of >5 bpm in one step is bad
            r_hr -= w_hr_drop * abs(dHR)
        if HR < hr_low_thresh:
            r_hr -= w_hr_low * (hr_low_thresh - HR)

        r_hr = np.clip(r_hr, -1, 1)

        # ------- shock index --------
        if shock_index <= 0.8:
            r_shock = 0
        elif shock_index >= 1.0:
            r_shock = -1.0
        else:
            r_shock = - (shock_index - 0.8) / (1.0 - 0.8)
        r_shock = max(-1, min(0, r_shock))  # only penalty

        # ------ cardiac output -------
        if 3.0 <= CO <= 6.0:  # typical normal range - decreased minimal low from 4 to 3 for the permissive hypo model
            r_co = 1
        elif CO < 3.0:
            r_co = - (3.0 - CO) / 2.0  # down to -1 if CO=1
        else:  # CO > 6
            r_co = - (CO - 6.0) / 4.0  # down to -1 if CO=10
        r_co = max(-1, min(0.5, r_co))  # clipped

        # Action cost
        r_blood_cost = -0.005 * blood_step
        r_blood = np.clip(r_blood_cost, -1, 0)

        r_cryst_cost = -0.002 * cryst_step # -1 at 400 mL if 0.0025
        r_cryst = np.clip(r_cryst_cost, -1, 0)

        # ------- Clotting reward -------
        r_clot = (
            1.5 * self.C_prev  # reward stronger clot
            + 1.0 * self.dC_dt  # reward new clot formation
            - 1.0 * self.dilution_ratio  # penalize excessive crystalloids
        )

        if r_clot >= 0.6:
            r_clot = 1
        elif r_clot >= 0.088:
            r_clot = (r_clot - 0.088) / (0.6 - 0.088)  # scale to [0, 1]
        else:
            r_clot = (r_clot - 0.088) / (0.6 - 0.088)  # extend linearly below 0.088

        # Weighted sum of components
        reward = (
                0.4 * r_map +
                #0.1 * r_shock + # changed to 0.2 11/17/2025 changed to 0.1 for high sev model
                0.1 * r_co +
                #0.1 * r_trend +
                + 0.1 * r_deviation
                #+ r_survive
                + 0.4 * r_clot # no clot reward for low sev
                + 0.2 * r_hr # 0.4 for low sev, 0.2 for high sev
                + 0.2 * r_cryst # for high severity
                #+ 0.3 * r_blood_cost # for low severity
        )

        # Terminal reward
        # Death
        if terminal_cause == "death" or terminal_cause == "failed to advance":
            reward -= 7.0

        # Stabilized
        if terminal_cause == "stabilization":
            # base stabilization reward
            reward += 7.0

        return reward

    def is_terminal(self, new_obs) -> tuple[bool, string]:
        # criteria for stabilization or truncation, death determined by if engine can advance time
        death_map_threshold = 10
        stable_map_low = 70 if self.sev == "low" else 50 # 70 for low sev, 50 for high sev
        stable_map_high = 100 if self.sev == "low" else 80 # 100 for low sev, 80 for high sev
        shock_index_limit = 0.9  # shock index HR/SBP must be <= this
        stable_hr_low = 50  # bpm
        stable_hr_high = 130
        base_time = 20

        type, severity = self.hemorrhage_type

        if self.sev == "high": n_timesteps = 4
        if self.sev == "low": n_timesteps = 5

        if len(self.history) < base_time:
            return False, "not terminal"
        if len(self.history) > 60:
            return True, "truncated"

        # Extract next_state dictionaries from history
        next_states = [h[2] for h in self.history[-n_timesteps:]]
        state_window = np.array(next_states + [new_obs])

        maps = np.array([s['MeanArterialPressure'] for s in state_window])
        saps = np.array([s['SystolicArterialPressure'] for s in state_window])
        heart_rates = np.array([s['HeartRate'] for s in state_window])
        shock_indices = heart_rates / saps
        COs = np.array([s['CardiacOutput'] for s in state_window])
        # BVs = np.array([s['BloodVolume'] for s in state_window])
        # stabilization
        if (min(maps) >= stable_map_low and max(maps) <= stable_map_high
                # and (shock_indices <= shock_index_limit).all() # remove for permissive hypo model
                and min(heart_rates) >= stable_hr_low and max(heart_rates) <= stable_hr_high):
            return True, "stabilization"

        if max(heart_rates) < 35:
            return True, "death (HR)"

        return False, "not terminal"