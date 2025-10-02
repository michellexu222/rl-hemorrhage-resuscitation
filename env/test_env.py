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



script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
env = HemorrhageEnv(state_file=os.path.join(parent_dir, "configs", "patient_configs", "Patient0@0s.json"))

env.reset(seed=0)
env.induce_hemorrhage("liver", 0.3)
terminated = False
truncated = False
count = 0
while not terminated and not truncated:
    obs, reward, terminated, truncated, info = env.step(action=[-1,0.2,-1])
    print(obs)
    count += 1
print(info)
print(count)

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
