import time
import pandas as pd
from FL_models import *
import constants

data = np.load("datasets/texas100.npz")

imgs = data['features']
labels = data['labels']
imgs = torch.tensor(imgs, dtype=torch.float)
labels = torch.tensor(labels)
labels = torch.max(labels, dim=1).indices
rand_idx = torch.randperm(imgs.size(0))
imgs = imgs[rand_idx]
labels = labels[rand_idx]

train_imgs = imgs[:60000]
train_labels = labels[:60000]
test_imgs = imgs[60000:]
test_labels = labels[60000:]

print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}, features {test_imgs.size(1)}")

print("Initializing...")
"""
Editable block, change below to your own experiment setting
"""
# Number of iterations (communication rounds)
num_iter = 201
# Number of participants
Ph = 50
# Size of hidden layer
hidden = 1024
# Fraction of malicious members
malicious_factor = 0.3
# Experimented attack
att_experimented = ["mislead", "min_max", "label_flip", "grad_ascent"]
# Experimented AGR, details in constants.py
agr_experimented = [constants.fang, constants.p_fang]
"""
End of editable block 
"""
for att_mode in att_experimented:
    for exp in agr_experimented:
        cgd = FL_torch(
            num_iter=num_iter,
            train_imgs=train_imgs,
            train_labels=train_labels,
            test_imgs=test_imgs,
            test_labels=test_labels,
            Ph=Ph,
            malicious_factor=malicious_factor,
            defender=exp['defender'],
            n_H=hidden,
            dataset="TEXAS",
            start_attack=exp['start'],
            attack_mode=att_mode,
            k_nearest=35,
            p_kernel=2,
            local_epoch=2
        )
        cgd.shuffle_data()
        cgd.federated_init()
        cgd.grad_reset()
        cgd.data_distribution(2000)
        print(f"Start {att_mode} attack to {exp['defender']}...")
        t1 = time.time()
        cgd.eq_train()
        t2 = time.time()
        print(f"{att_mode} attack to {exp['defender']} complete, time consumed {t2-t1}s")