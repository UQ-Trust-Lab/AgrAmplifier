import numpy as np
from FL_models import *
import constants

data = np.load("datasets/purchase.npz")

imgs = data['arr_0']
labels = data['arr_1']
labels -= 1
train_imgs = imgs[:180000]
train_labels = labels[:180000]
test_imgs = imgs[180000:]
test_labels = labels[180000:]

train_imgs = torch.tensor(train_imgs, dtype=torch.float)
test_imgs = torch.tensor(test_imgs, dtype=torch.float)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

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
            dataset="PURCHASE",
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
        cgd.eq_train()
        print(f"{att_mode} attack to {exp['defender']} complete")