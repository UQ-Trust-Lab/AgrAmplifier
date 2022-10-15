import numpy as np
from FL_models import *
import constants
import time

# Load the local data
data = np.load("datasets/location.npz")

imgs = data['arr_0']
labels = data['arr_1']
train_imgs = imgs[:4000]
train_labels = labels[:4000]
test_imgs = imgs[4000:]
test_labels = labels[4000:]

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
            k_nearest=35,
            p_kernel=2,
            n_H=hidden,
            dataset="LOCATION",
            start_attack=exp['start'],
            attack_mode=att_mode,
            local_epoch=2
        )
        cgd.shuffle_data()
        cgd.federated_init()
        cgd.grad_reset()
        cgd.data_distribution()
        print(f"Start {att_mode} attack to {exp['defender']}...")
        t1 = time.time()
        cgd.eq_train()
        t2 = time.time()
        print(f"{att_mode} attack to {exp['defender']} complete， time consumed {t2 - t1}s")
        # time_recorder.loc[att_mode][exp['defender']] = t2 - t1
# time_recorder.to_csv("./output/Location_timer.csv")
