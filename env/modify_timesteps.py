# modfiy the timestep for each patient state
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
patient_config_dir = os.path.join(script_dir, "..", "configs", "patient_configs")
patient_files = [os.path.join(patient_config_dir, f) for f in os.listdir(patient_config_dir)if f.endswith(".json") ]
for file in patient_files:
    with open(file, 'r') as f:
        data = json.load(f)
        if data['Configuration']['TimeStep']['ScalarTime']['Value'] == 0.02:
            data['Configuration']['TimeStep']['ScalarTime']['Value'] = 0.06
    with open(file, 'w') as f:
        json.dump(data, f, indent=1)