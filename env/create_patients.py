from pulse.cdm.patient import SEPatientConfiguration, eSex
from pulse.engine.PulseEngine import PulseEngine
from pulse.cdm.patient_actions import SEHemorrhage, eHemorrhage_Compartment, SESubstanceBolus, SESubstanceCompoundInfusion, eSubstance_Administration
from pulse.cdm.scalars import VolumePerTimeUnit, VolumeUnit, MassPerVolumeUnit, TimeUnit, LengthUnit, MassUnit, PressureUnit, FrequencyUnit

import os
import numpy as np
import random

def create_patient(sex: eSex, age, bmi, base_map, base_hr, filename):
    engine = PulseEngine()
    engine.log_to_console(False)

    # Create patient config
    pc = SEPatientConfiguration()
    p = pc.get_patient()
    p.set_sex(sex)
    p.get_age().set_value(age, TimeUnit.yr)
    #p.get_height().set_value(height, LengthUnit.inch)
    p.get_body_mass_index().set_value(bmi)
    p.get_mean_arterial_pressure_baseline().set_value(base_map, PressureUnit.mmHg)
    p.get_heart_rate_baseline().set_value(base_hr, FrequencyUnit.Per_min)
    #p.get_blood_volume_baseline().set_value(base_bv, VolumeUnit.mL)

    init = engine.initialize_engine(pc, None)
    if not init:
        raise RuntimeError("Could not initialize engine")

    x=engine.serialize_to_file(filename)
    print(x)
script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "..", "configs", "patient_configs")
os.makedirs(target_dir, exist_ok=True)  # make sure it exists

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

n = 250
i = 50
while i < n:
    if i % 2 == 0:
        try:
            create_patient(eSex.Male,
                            random.randint(20, 60),
                            random.randint(18, 35),
                            base_map=random.randint(75, 95),
                            base_hr=np.random.randint(65, 95),
                            filename=os.path.join(target_dir, f"Patient{i}@0s.json"))
            i += 1
        except RuntimeError as e:
            continue
    else:
        try:
            create_patient(eSex.Female,
                            random.randint(20, 60),
                            random.randint(18, 35),
                            base_map=random.randint(75, 95),
                            base_hr=np.random.randint(65, 95),
                            filename=os.path.join(target_dir, f"Patient{i}@0s.json"))
            i += 1
        except RuntimeError as e:
            continue

#
# file_path = os.path.join(target_dir, "MyPatient@1s.json")
#
# create_patient("Male", 30, 70, 180, file_path)
#
#
# print("CWD:", os.getcwd())