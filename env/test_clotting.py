import numpy as np

k_form, k_form_base = 0.13, 0.13  # clot formation rate
k_blood = 0.002  # how much giving blood improves clotting
tau_blood = 5  # (mins) decay time for recent blood effect

# clot breakdown
k_lysis, k_lysis_base = 0.01, 0.01  # clot breakdown rate (fibrinolysis)
beta_0 = 0.5  # base sensitivity of clot breakdown to MAP above threshold
map_thresh = 70  # MAP threshold above which clot breakdown increases (mmHg)
delta_c = 0.05  # amount clot breaks down by when it "pops"

# MAP effects
alpha = 1  # sensitivity of clot formation to MAP deviations
map_opt = 70  # optimal MAP for clotting (mmHg)

# clot / severity caps
gamma = 0.2  # C_max = 1 - gamma * S_base
eps_min, eps_max = 0.05, 0.2
n = 1.5  # controls steepness of clotting effect on severity, n=1 (linear) -> bleed decrease steadily w/ clot,
# n=1.5-2 -> bleed stays moderate, then drops faster when clot matures (respond quickly to clotting)
# n=3-4-> clot has little effect until it almost complete (bleed resists clotting)

# dilution effects
tau_clear = 5  # time constant for redistribution of crystalloid from the bloodstream (minutes) (cryst redistr. half life)
# k_beta_dil possibly

# other stuff
dt = 1.0
V_blood_step = 0  # current blood given as action
V_blood_recent = 0
V_cryst = 0
V_blood = 4600  # initial blood volume
map = 70  # current MAP
temp = 37

C_prev = 0.05  # current fraction of clot formed
S_base = 0.4
for i in range(15):

    V_blood_recent = V_blood_recent * np.exp(-dt / tau_blood) + V_blood_step
    #V_blood_step = 0
    #print(f"V_blood_recent: {V_blood_recent}")
    k_form = k_form_base * (1 + k_blood * V_blood_recent)
    #print(f"k_form: {k_form}")
    # MAP_ref = self.baseline_map
    f_map = np.exp(-alpha * np.abs(map - map_opt) / map_opt)
    #print(f"f_map: {f_map}")
    # Temperature effect
    if temp >= 36.0:
        f_temp = 1.0
    elif temp >= 34.0:
        f_temp = 0.75
    else:
        f_temp = 0.5

    dilution_ratio = min(V_cryst / V_blood, 1)
    print(f"dilution_ratio: {dilution_ratio}")
    beta_eff = beta_0 * (1 + dilution_ratio)  # increased dilution increases beta increase clot breakdown
    print(f"beta_eff: {beta_eff}")

    form_rate = k_form * f_map * f_temp * (1 - dilution_ratio) * C_prev * (1.0 - C_prev)
    print(f"form_rate: {form_rate}")
    lysis_rate = k_lysis + beta_eff * max(0, map - map_thresh) / map_thresh
    print(f"lysis_rate: {lysis_rate}")

    dC_dt = form_rate - lysis_rate * C_prev
    print(f"dC_dt: {dC_dt}")
    C_new = C_prev + dt * dC_dt
    #print(f"C_new before pop / clip: {C_new}")
    # stochastic clot "pops" - clot failure more likely if lysis rate high and clot is not strong
    if np.random.rand() < lysis_rate * (1 - C_new) * dt:
        C_new -= delta_c
        C_new = max(0.05, C_new)
        print("clot popped")

    C_max = 1 - gamma * S_base
    #print(f"C_max: {C_max}")
    C_new = np.clip(C_new, 0.05, C_max)  # clamp
    print(f"C_new: {C_new}")
    C_prev = C_new
    V_cryst -= V_cryst * (1 - np.exp(-dt / tau_clear))
    print(f"V_cryst: {V_cryst}")
    # new severity
    epsilon = eps_min + (eps_max - eps_min) * S_base  # residual bleeding when clot is perfect
    #print(f"epsilon: {epsilon}")
    #print(f"S_base: {S_base}")
    S_new = S_base * (epsilon + (1 - epsilon) * (1 - C_new / C_max) ** n) * (1 + 0.5* dilution_ratio)
    print(f"S_new: {S_new}")

    r_clot = (
            1.5 * C_prev  # reward stronger clot
            + 1.0 * dC_dt  # reward new clot formation
            - 1.0 * dilution_ratio  # penalize excessive crystalloids
    )
    print(f"raw r_clot: {r_clot}")
    if r_clot >= 0.6:
        r_clot = 1
    elif r_clot >= 0.088:
        r_clot = (r_clot - 0.088) / (0.6 - 0.088)  # scale to [0, 1]
    else:
        r_clot = (r_clot - 0.088) / (0.6 - 0.088)  # extend linearly below 0.088
    print(f"r_clot: {r_clot}")

    print("-------------------------------------------------------")

# HR = 45
# dHR = -4
# # Penalize big negative drops and low absolute HR
# w_hr_drop = 0.06
# w_hr_low = 0.05
# hr_low_thresh = 60  # or tuned to your simulation
#
# r_hr = 0.0
# if dHR < 0:  # drop of >5 bpm in one step is bad
#     r_hr -= w_hr_drop * abs(dHR)
# if HR < hr_low_thresh:
#     r_hr -= w_hr_low * (hr_low_thresh - HR)
# print(r_hr)

