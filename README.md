# RL for Hemorrhage Resuscitation
Welcome to the repository for my 2025-26 project, "A Severity-Aware Reinforcement Learning Agent for Optimizing Abdominal Hemorrhage Resuscitation in Simulated Trauma Care"!

## Environment Dependencies:
### Pulse Physiology Engine (prerequisite):
1. Download Pulse from: https://gitlab.kitware.com/physiology/engine
2. Install following their documentation

### Install Using Conda
Clone repository
```bash
git clone https://github.com/michellexu222/rl-hemorrhage-resuscitation
```
cd into the repo and create conda environment with required dependencies
```bash
conda env create -f environment.yml
conda activate rl-hemorrhage
```

## Usage
### Main files
#### env
* hemorrhage_env: contains the Gymnasium-compatible hemorrhage environment 
* create_patients: used to create simulated patients for training and testing
* training/baseline.py and rppo.py: used to train agents
* training/train.py and config.yaml: used for Weights & Biases sweeps for hyperparameter tuning
#### gating
* create_dataset: creates dataset for gating model
* model.ipynb: creates binary classifer used for gating
#### final_system
* calc_final_metrics.ipynb: contains calculation of final evaluation metrics and statistical significance
* visuals.ipynb: creates figures
* test_models, test_models_moe, and test_models_pid: run evaluation of models, logging to CSVs and saving episode trajectory graphs
* demo.py: uses Steamlit to create a live web demo

### Training 
To train the system, run the following:
1. Create training patients using Pulse by running env/create_patients.py (should create a configs/patient_configs directory) and then running modify_timesteps.py
2. Run env/training/baseline.py to train a low-severity expert agent; Note: must change induce_hemorrhage function inside env/hemorrhage_env.py to only induce low severity hemorrhages
3. Run env/training/rppo.py to train a high-severity agent; similarly, must change induce_hemorrhage function to only induce high-severity hemorrhgaes
4. To train the gating model, run gating/create_dataset.py, followed by running all cells in gating/model.ipynb
5. Can use final_system/main.py or test_episode.py to run an episode
6. Can run "streamlit run demo.py" from terminal to view web demo





