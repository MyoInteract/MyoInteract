from myosuite.utils import gym
import mujoco
import numpy as np
from myosuite.utils.myosim_utils import MUSCLE_FMG

### Parameters are taken from Rakshit et al. 2021 (https://doi.org/10.1016/j.jbiomech.2021.110695).
MUSCLE_FATIGUE_PARAMS = {
    "Ankle-Dorsiflexor-F": {"F": 0.00746, "R": 0.00081, "r": 4.97},
    "Ankle-Dorsiflexor-M": {"F": 0.00725, "R": 0.00096, "r": 10.36},
    "Ankle-Dorsiflexor": {"F": 0.00828, "R": 0.00204, "r": 7.07},
    "Ankle-Plantarflexor-F": {"F": 0.00702, "R": 0.00098},
    "Ankle-Plantarflexor-M": {"F": 0.00683, "R": 0.00093},
    "Ankle-Plantarflexor": {"F": 0.00695, "R": 0.00096},
    "Elbow-Extensor-F": {"F": 0.01874, "R": 0.00206, "r": 21.22},
    "Elbow-Extensor-M": {"F": 0.01269, "R": 0.00085, "r": 30.21},
    "Elbow-Extensor": {"F": 0.01559, "R": 0.00125, "r": 25.52},
    "Elbow-Flexor-F": {"F": 0.00965, "R": 0.00197, "r": 6.22},
    "Elbow-Flexor-M": {"F": 0.01302, "R": 0.00188, "r": 8.99},
    "Elbow-Flexor": {"F": 0.01703, "R": 0.00494, "r": 4.68},
    "Hand-Adductor-Pollicis-F": {"F": 0.00476, "R": 0.00093, "r": 6.62},
    "Hand-Adductor-Pollicis-M": {"F": 0.00586, "R": 0.00202, "r": 1.00},
    "Hand-Adductor-Pollicis": {"F": 0.00558, "R": 0.00283, "r": 1.00},
    "Hand-First-Dorsal-Interossei-F": {"F": 0.03999, "R": 0.03983},
    "Hand-First-Dorsal-Interossei-M": {"F": 0.01637, "R": 0.00360, "r": 3.66},
    "Hand-First-Dorsal-Interossei": {"F": 0.02686, "R": 0.00656, "r": 3.41},
    "Wrist-Flexor-F": {"F": 0.01159, "R": 0.00217, "r": 7.39},  #from Hand G/Grip group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Wrist-Flexor-M": {"F": 0.01238, "R": 0.00178, "r": 8.00},  #from Hand G/Grip group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Wrist-Flexor": {"F": 0.01235, "R": 0.00135, "r": 12.51},  #from Hand G/Grip group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Knee-Extensor-F": {"F": 0.01407, "R": 0.00185, "r": 6.32},
    "Knee-Extensor-M": {"F": 0.01420, "R": 0.00153, "r": 10.96},
    "Knee-Extensor": {"F": 0.00825, "R": 0.00076, "r": 14.85},
    
    "Ankle": {"F": 0.01485, "R": 0.00333, "r": 9.31},
    "Toe": {"F": 0.01485, "R": 0.00333, "r": 9.31},  #from Ankle group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Elbow": {"F": 0.01086, "R": 0.00225, "r": 4.93},
    "Hand": {"F": 0.01227, "R": 0.00134, "r": 9.10},
    "Wrist": {"F": 0.01227, "R": 0.00134, "r": 9.10},  #from Hand group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Finger": {"F": 0.01227, "R": 0.00134, "r": 9.10},  #from Hand group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Knee": {"F": 0.00825, "R": 0.00076, "r": 14.85},
    "Shoulder": {"F": 0.00825, "R": 0.00076, "r": 14.85},  #Shoulder values from Looft & Frey-Law 2020 (https://doi.org/10.1016/j.jbiomech.2020.109762)

    "Default": {"F": 0.00970, "R": 0.00091, "r": 15},  #default values from Looft et al. 2018 (https://doi.org/10.1016/j.jbiomech.2018.06.005); NOTE: r=30 is better for hand/wrist/finger muscles (see https://doi.org/10.1016/j.jbiomech.2020.109762)
}

class CumulativeFatigue():
    # 3CC-r model, adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701 for muscles 
    # based on the implementation from Aleksi Ikkala and Florian Fischer https://github.com/aikkala/user-in-the-box/blob/main/uitb/bm_models/effort_models.py
    def __init__(self, mj_model, frame_skip=1, sex=None, seed=None):
        # self.na = mj_model.na
        self._dt = mj_model.opt.timestep * frame_skip # dt might be different from model dt because it might include a frame skip
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.na = sum(muscle_act_ind)  #WARNING: here, self.na denotes the number of muscle actuators only!
        self._tauact = np.array([mj_model.actuator_dynprm[i][0] for i in range(len(muscle_act_ind)) if muscle_act_ind[i]])
        self._taudeact = np.array([mj_model.actuator_dynprm[i][1] for i in range(len(muscle_act_ind)) if muscle_act_ind[i]])
        self._MA = np.zeros((self.na,))  # Muscle Active
        self._MR = np.ones((self.na,))   # Muscle Resting
        self._MF = np.zeros((self.na,))  # Muscle Fatigue
        self.TL  = np.zeros((self.na,))  # Target Load
        
        # self._r = 10 * 15 # Recovery time multipslier i.e. how many times more than during rest intervals https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6092960/ (factor 10 to compensate for 0.1 below)
        # self._F = 0.00912  # Fatigue coefficients (default parameter was identified for elbow torque https://pubmed.ncbi.nlm.nih.gov/22579269/)
        # self._R = 0.1 * 0.00094  # Recovery coefficients (default parameter was identified for elbow torque https://pubmed.ncbi.nlm.nih.gov/22579269/; factor 0.1 to get an approx. 1% R/F ratio)
        
        # Get muscle functional groups (MFG) [use sex-specific muscle fatigue parameters, if provided]
        self.MFG = [MUSCLE_FMG[mj_model.actuator(i).name]+"-"+sex if sex is not None else MUSCLE_FMG[mj_model.actuator(i).name] for i in range(self.na)]
        self.MFG_r = [mfg if mfg in MUSCLE_FATIGUE_PARAMS.keys() else (mfg+" -> "+(mfg.split("-")[0]) if "r" in MUSCLE_FATIGUE_PARAMS[mfg.split("-")[0]].keys() else mfg+" -> "+(mfg.split("-")[0])+"->"+"Default") for mfg in self.MFG]
        self.MFG_F = [mfg if mfg in MUSCLE_FATIGUE_PARAMS.keys() else (mfg+" -> "+(mfg.split("-")[0]) if "F" in MUSCLE_FATIGUE_PARAMS[mfg.split("-")[0]].keys() else mfg+" -> "+(mfg.split("-")[0])+"->"+"Default") for mfg in self.MFG]
        self.MFG_R = [mfg if mfg in MUSCLE_FATIGUE_PARAMS.keys() else (mfg+" -> "+(mfg.split("-")[0]) if "R" in MUSCLE_FATIGUE_PARAMS[mfg.split("-")[0]].keys() else mfg+" -> "+(mfg.split("-")[0])+"->"+"Default") for mfg in self.MFG]

        self._r = np.zeros((self.na,))
        self._F = np.zeros((self.na,))
        self._R = np.zeros((self.na,))
        for i in range(self.na):
            self._r[i] = MUSCLE_FATIGUE_PARAMS.get(self.MFG[i], MUSCLE_FATIGUE_PARAMS[self.MFG[i].split("-")[0]]).get("r", MUSCLE_FATIGUE_PARAMS["Default"]["r"])
            self._F[i] = MUSCLE_FATIGUE_PARAMS.get(self.MFG[i], MUSCLE_FATIGUE_PARAMS[self.MFG[i].split("-")[0]]).get("F", MUSCLE_FATIGUE_PARAMS["Default"]["r"]) 
            self._R[i] = MUSCLE_FATIGUE_PARAMS.get(self.MFG[i], MUSCLE_FATIGUE_PARAMS[self.MFG[i].split("-")[0]]).get("R", MUSCLE_FATIGUE_PARAMS["Default"]["r"])

        # self._r = self._r * np.ones((self.na,))
        # self._F = self._F * np.ones((self.na,))
        # self._R = self._R * np.ones((self.na,))
        
        self.seed(seed)  # Create own Random Number Generator (RNG) used when reset is called with fatigue_reset_random=True
        ### NOTE: the seed from CumulativeFatigue is not synchronised with the seed used for the rest of MujocoEnv!

    def set_FatigueCoefficient(self, F):
        if isinstance(F, int) or isinstance(F, float):
            F = F * np.ones((self.na,))

        # Set Fatigue coefficients
        self._F = F
    
    def set_RecoveryCoefficient(self, R):
        if isinstance(R, int) or isinstance(R, float):
            R = R * np.ones((self.na,))

        # Set Recovery coefficients
        self._R = R
    
    def set_RecoveryMultiplier(self, r):
        if isinstance(r, int) or isinstance(r, float):
            r = r * np.ones((self.na,))

        # Set Recovery time multiplier
        self._r = r
        
    def compute_act(self, act):
        # Get target load (actual activation, which might be reached only with some "effort", 
        # depending on how many muscles can be activated (fast enough) and how many are in fatigue state)
        self.TL = act.copy()

        # Calculate effective time constant tau (see https://mujoco.readthedocs.io/en/stable/modeling.html#muscles)
        self._LD = 1/self._tauact*(0.5 + 1.5*self._MA)
        self._LR = (0.5 + 1.5*self._MA)/self._taudeact
        ## TODO: account for smooth transition phase of length tausmooth (tausmooth = mj_model.actuator_dynprm[i][2])?

        # Calculate C(t) -- transfer rate between MR and MA
        C = np.zeros_like(self._MA)
        idxs = (self._MA < self.TL) & (self._MR > (self.TL - self._MA))
        C[idxs] = self._LD[idxs] * (self.TL[idxs] - self._MA[idxs])
        idxs = (self._MA < self.TL) & (self._MR <= (self.TL - self._MA))
        C[idxs] = self._LD[idxs] * self._MR[idxs]
        idxs = self._MA >= self.TL
        C[idxs] = self._LR[idxs] * (self.TL[idxs] - self._MA[idxs])

        # Calculate rR
        rR = np.zeros_like(self._MA)
        idxs = self._MA >= self.TL
        rR[idxs] = (self._r*self._R)[idxs]
        idxs = self._MA < self.TL
        rR[idxs] = self._R[idxs]

        # Clip C(t) if needed, to ensure that MA, MR, and MF remain between 0 and 1
        C = np.clip(C, np.maximum(-self._MA/self._dt + self._F*self._MA, (self._MR - 1)/self._dt + rR*self._MF),
                    np.minimum((1 - self._MA)/self._dt + self._F*self._MA, self._MR/self._dt + rR*self._MF))

        # Update MA, MR, MF
        dMA = (C - self._F*self._MA)*self._dt
        dMR = (-C + rR*self._MF)*self._dt
        dMF = (self._F*self._MA - rR*self._MF)*self._dt
        self._MA += dMA
        self._MR += dMR
        self._MF += dMF

        return self._MA, self._MR, self._MF

    def get_effort(self):
        # Calculate effort
        return np.linalg.norm(self._MA - self.TL)

    def reset(self, fatigue_reset_vec=None, fatigue_reset_random=False):
        if fatigue_reset_random:
            assert fatigue_reset_vec is None, "Cannot use 'fatigue_reset_vec' if fatigue_reset_random=False."
            non_fatigued_muscles = self.np_random.random(size=(self.na,))
            active_percentage = self.np_random.random(size=(self.na,))
            self._MA = non_fatigued_muscles * active_percentage         # Muscle Active
            self._MR = non_fatigued_muscles * (1 - active_percentage)   # Muscle Resting
            self._MF = 1 - non_fatigued_muscles                         # Muscle Fatigue
        else:
            if fatigue_reset_vec is not None:
                assert len(fatigue_reset_vec) == self.na, f"Invalid length of initial/reset fatigue vector (expected {self.na}, but obtained {len(fatigue_reset_vec)})."
                self._MF = fatigue_reset_vec     # Muscle Fatigue
                self._MR = 1 - fatigue_reset_vec # Muscle Resting
                self._MA = np.zeros((self.na,))  # Muscle Active
            else:
                self._MA = np.zeros((self.na,))  # Muscle Active
                self._MR = np.ones((self.na,))   # Muscle Resting
                self._MF = np.zeros((self.na,))  # Muscle Fatigue

    def seed(self, seed=None):
        """
        Set random number seed
        """
        self.input_seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @property
    def MF(self):
        return self._MF
    
    @property
    def MR(self):
        return self._MR
    
    @property
    def MA(self):
        return self._MA
    
    @property
    def F(self):
        return self._F
    
    @property
    def R(self):
        return self._R
    
    @property
    def r(self):
        return self._r


