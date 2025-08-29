from pulse.engine.PulseEngine import PulseEngine
from pulse.cdm.patient_actions import SEHemorrhage, eHemorrhage_Compartment, SESubstanceBolus, SESubstanceCompoundInfusion, eSubstance_Administration
from pulse.cdm.scalars import VolumePerTimeUnit, VolumeUnit, MassPerVolumeUnit, TimeUnit

class IntracranialHemorrhageEnv:

    def __init__(self, state_file=None):
        self.pulse = PulseEngine()
        self.pulse.log_to_console(True)
        self.state_file = state_file
        self.reset()

    def reset(self):
        # Load initial state if given
        if self.state_file:
            loaded = self.pulse.serialize_from_file(self.state_file, None)
            if not loaded:
                raise FileNotFoundError(f"State file {self.state_file} not found.")
        else:
            self.pulse.clear()  # Clear any existing state
        self.pulse.advance_time_s(1)

    def get_state(self):
        data = self.pulse.pull_data()
        # Replace indices below with correct ones for your features
        features = {}
        for idx, req in enumerate(self.pulse._data_request_mgr.get_data_requests()):
            #print(f"Index {idx}: {req.get_property_name()} ({req.get_unit()})")
            print(f"{req.get_property_name()} ({req.get_unit()}): {data[idx+1]}")
            features[req.get_property_name()] = data[idx+1]

        return features

    def induce_hemorrhage(self, compartment, severity):
        hemorrhage = SEHemorrhage()
        hemorrhage.set_comment("Induced ICH")
        hemorrhage.set_compartment(compartment)
        hemorrhage.get_severity().set_value(severity)
        self.pulse.process_action(hemorrhage)

    def give_saline(self, volume: float, rate: float):
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

    def give_PRBCs(self, volume: float, rate: float):
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

    def give_blood(self, volume: float, rate: float):
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

    def give_lactated_ringers(self, volume: float, rate: float):
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

    def give_epinephrine(self, volume, concentration):
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

    # def give_fluids(self, volume_ml, substance="Saline", concentration_mg_per_ml=9):
    #     from pulse.cdm.patient_actions import SESubstanceCompoundInfusion, SESubstanceBolus
    #     from pulse.cdm.scalars import VolumeUnit, MassPerVolumeUnit, TimeUnit
    #
    #     substance = SESubstanceCompoundInfusion()
    #     substance.set_compound("LactatedRingers")
    #     substance.get_bag_volume().set_value(4, VolumeUnit.mL)
    #     substance.get_rate().set_value(100, VolumePerTimeUnit.mL_Per_min)
    #     self.pulse.process_action(substance)
    #
    #     bolus = SESubstanceBolus()
    #     bolus.set_admin_route(eSubstance_Administration.Intravenous)
    #     bolus.set_substance("Epinephrine")
    #     bolus.get_dose().set_value(volume_ml, VolumeUnit.mL)
    #     bolus.get_concentration().set_value(concentration_mg_per_ml, MassPerVolumeUnit.mg_Per_mL)
    #     bolus.get_admin_duration().set_value(60, TimeUnit.s)
    #
    #     if not self.pulse._is_ready:
    #         print("Pulse engine is not initialized.")
    #         return
    #
    #     if not bolus.is_valid():
    #         print("Bolus action is not valid")
    #         return
    #
    #     try:
    #         self.pulse.process_action(bolus)
    #     except Exception as e:
    #         print(f"Error processing bolus: {e}")

env = IntracranialHemorrhageEnv(state_file="./states/Soldier@0s.json")
env.induce_hemorrhage(eHemorrhage_Compartment.Brain, 0.7)  # Example compartment
#print(env.get_state())
env.give_saline(500, 3) # random arbitrary values
#env.step(60)
#print(env.get_state())



