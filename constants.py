import numpy as np
import random
import os
import torch

def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch()

"""
List of experiments 
"""
# Baseline experiment without attacking
baseline = {
    "start": 1000,
    "defender": "none"
}
# Baseline experiment with attack starts at round 50
attacked = {
    "start": 50,
    "defender": "none"
}
# Experiment of Fang[2] AGR, attack starts at round 50, AgrAmplifier not equipped
fang = {
    "start": 50,
    "defender": "fang"
}
# Experiment of FL-Trust[1] AGR, attack starts at round 50, AgrAmplifier not equipped
fl_trust = {
    "start": 50,
    "defender": "fl_trust"
}
# Experiment of Tr-mean[3] AGR, attack starts at round 50, AgrAmplifier not equipped
tr_mean = {
    "start": 50,
    "defender": "tr_mean"
}
# Experiment of Median[3] AGR, attack starts at round 50, AgrAmplifier not equipped
median = {
    "start": 50,
    "defender": "median"
}
# Experiment of EuDen (Sec 3.4.1 in the original paper), AgrAmplifier equipped
p_dense = {
    "start": 50,
    "defender": "p-dense"
}
# Experiment of CosDen (Sec 3.4.1 in the original paper), AgrAmplifier equipped
p_cos = {
    "start": 50,
    "defender": "p-cosine"
}
# Experiment of MgDen (Sec 3.4.1 in the original paper), AgrAmplifier equipped
p_merge = {
    "start": 50,
    "defender": "p-merge"
}
# Experiment of EuDen (Sec 3.4.1 in the original paper), AgrAmplifier not equipped
np_dense = {
    "start": 50,
    "defender": "np-dense"
}
# Experiment of CosDen (Sec 3.4.1 in the original paper), AgrAmplifier not equipped
np_cos = {
    "start": 50,
    "defender": "np-cosine"
}
# Experiment of MgDen (Sec 3.4.1 in the original paper), AgrAmplifier not equipped
np_merge = {
    "start": 50,
    "defender": "np-merge"
}
# Experiment of Fang[2], AgrAmplifier equipped
p_fang = {
    "start": 50,
    "defender": "p-fang"
}
# Experiment of FL-Trust[1], AgrAmplifier equipped
p_trust = {
    "start": 50,
    "defender": "p-trust"
}

# Experiment settings for defender 
experiments = [baseline, attacked, fang, fl_trust, tr_mean, median, p_cos, p_dense, p_merge, p_trust, p_fang, np_cos, np_dense, np_merge]

# AgrAmplifer applied robust aggregators
poolings = [p_trust, p_fang, p_cos, p_dense, p_merge]

# Untargeted atacks
"""
mislead = label flipping + gradient ascent
min_max = S&H attack in original paper
"""
att_modes = ["mislead", "min_max", "label_flip", "grad_ascent"]

# Targeted attack
targeted_att = ["scale"]


rq3_1 = [baseline, attacked, p_cos, p_dense, p_trust, p_merge, p_trust, p_fang]
rq3_1_base = [baseline]
rq3_1_nd = [attacked]
rq3_1_time = [p_cos]
