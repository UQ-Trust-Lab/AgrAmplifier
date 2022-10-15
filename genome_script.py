import numpy as np
from FL_models import *
import constants

data = np.load("datasets/gnome.npz")

train_imgs = data['features']
train_labels = data['labels']

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.long)
rand_idx = torch.randperm(train_labels.size(0))
train_imgs = train_imgs[rand_idx]
train_labels = train_labels[rand_idx]

test_imgs = train_imgs[1000:]
test_labels = train_labels[1000:]
train_imgs = train_imgs[:1000]
train_labels = train_labels[:1000]

print(f"Data loaded, training images: {train_imgs.size(0)}, testing images: {test_imgs.size(0)}")

print("Initializing...")
"""
Editable block, change below to your own experiment setting
"""
# Number of iterations (communication rounds)
num_iter = 201
# Number of malicious participants
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
            k_nearest=35,
            dataset="GENOME",
            start_attack=exp['start'],
            attack_mode=att_mode,
            p_kernel=2,
            local_epoch=2
        )
        cgd.shuffle_data()
        cgd.federated_init()
        cgd.grad_reset()
        cgd.data_distribution(validation_size=80)
        print(f"Start {att_mode} attack to {exp['defender']}...")
        cgd.eq_train()
        print(f"{att_mode} attack to {exp['defender']} complete")