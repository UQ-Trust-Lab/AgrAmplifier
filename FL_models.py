import torch
import pandas as pd
import numpy as np
import datetime, time
import Defender

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H")


def min_max(all_updates, model_re):
    """
    S&H attack from [4] (see Reference in readme.md), the code is authored by Virat Shejwalkar and Amir Houmansadr.
    """
    deviation = torch.std(all_updates, 0)
    lamda = torch.Tensor([10.0]).float()
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    distance = torch.cdist(all_updates, all_updates)
    max_distance = torch.max(distance)
    del distance
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)
        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2
        lamda_fail = lamda_fail / 2
    mal_update = (model_re - lamda_succ * deviation)
    return mal_update


def targeted_flip(train_img: torch.Tensor, target: int, backdoor_idx=7):
    """
    Flip the label for a targeted output class, the idea is from [1] (see Reference in readme.md)
    :param train_img: the training set
    :param target: the target label to flip
    :param backdoor_idx: the index of feature to flip
    :return: the label-flipped training set
    """
    augmented_data = train_img.clone()
    augmented_data[:, backdoor_idx:backdoor_idx+2] = 0.5
    augmented_label = torch.ones(train_img.size(0), dtype=torch.long) * target
    return augmented_data, augmented_label


class ModelFC(torch.nn.Module):
    """
    The NN model representing a participant / or the global model, the model itself is a one-hidden-layer FC NN
    """
    def __init__(self, n_H: int, in_length=28 * 28, out_class=10):
        super().__init__()
        self.n_H = n_H
        self.networks = torch.nn.Sequential(
            torch.nn.Linear(in_length, n_H),
            torch.nn.ReLU(),
            torch.nn.Linear(n_H, n_H * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_H * 2, out_class)
        )
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.CrossEntropyLoss()
        self.grad = None

    def forward(self, x):
        return self.networks(x)

    def step(self):
        self.optimizer.step()

    def back_prop(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, local_epoch: int, revert=False):
        param = self.get_flatten_parameters()
        loss = 0
        acc = 0
        for epoch in range(local_epoch):
            batch_idx = 0
            while batch_idx * batch_size < X.size(0):
                lower = batch_idx * batch_size
                upper = lower + batch_size
                X_b = X[lower: upper]
                y_b = y[lower: upper]
                self.optimizer.zero_grad()
                out = self.forward(X_b)
                loss_b = self.loss(out, y_b)
                loss_b.backward()
                self.optimizer.step()
                loss += loss_b.item()
                pred_y = torch.max(out, dim=1).indices
                acc += torch.sum(pred_y == y_b).item()
                batch_idx += 1
        grad = self.get_flatten_parameters() - param
        loss /= local_epoch
        acc = acc / (local_epoch * X.size(0))
        if revert:
            self.load_parameters(param)
        return acc, loss, grad

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0)
        with torch.no_grad():
            for parameter in self.parameters():
                out = torch.cat([out, parameter.flatten()])
        return out

    def load_parameters(self, parameters: torch.Tensor, mask=None):
        """
        Load parameters to the current model using the given flatten parameters
        :param mask: only the masked value will be loaded
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.parameters():
            with torch.no_grad():
                length = len(param.flatten())
                to_load = parameters[start_index: start_index + length]
                to_load = to_load.reshape(param.size())
                if mask is not None:
                    local_mask = mask[start_index: start_index + length]
                    local_mask = local_mask.reshape(param.size())
                    param[local_mask] = to_load[local_mask]
                else:
                    param.copy_(to_load)
                start_index += length


class FL_torch:
    """
    The class handling the Federated learning process
    """
    def __init__(self,
                 num_iter,
                 train_imgs,
                 train_labels,
                 test_imgs,
                 test_labels,
                 Ph,
                 malicious_factor,
                 defender,
                 n_H,
                 dataset,
                 batch=5,
                 sampling_prob=0.5,
                 max_grad_norm=1,
                 sigma=0,
                 start_attack=30,
                 attack_mode="min_max",
                 k_nearest=20,
                 p_kernel=3,
                 local_epoch=1,
                 stride=10,
                 pipe_loss=0,
                 output_path="./output/"):
        # Number of iterations
        self.num_iter = num_iter
        # Training set features
        self.train_imgs = train_imgs
        # Training set labels
        self.train_labels = train_labels
        # Test set features
        self.test_imgs = test_imgs
        # Test set labels
        self.test_labels = test_labels
        # Validation set features (for AGRs need validation set or trusted set)
        self.validation_imgs = None
        # Validation set labels
        self.validation_labels = None
        # Number of participants
        self.Ph = Ph
        # Fraction of malicious participants
        self.malicious_factor = malicious_factor
        # Defender name, should be within the values specified in constant.py
        self.defender = defender
        # Size of hidden layer
        self.n_H = n_H
        # The index of batches in the training dataset
        self.batch = batch
        # The batch size
        self.batch_size = 0
        # The dataset name
        self.dataset = dataset
        # Not used
        self.sampling_prob = sampling_prob
        # Not used
        self.max_grad_norm = max_grad_norm
        # Not used
        self.sigma = sigma
        # The round when the attacker start attacking
        self.start_attack = start_attack
        # The epochs running locally on each participant at each round before sending gradients to aggregator
        self.local_epoch = local_epoch
        # Print the training information to console every 'stride' rounds
        self.stride = stride
        # Not used
        self.pipe_loss = pipe_loss
        # The path of output files
        self.output_path = output_path
        # The k-nearest neighbours examined in the distance-based AGR (see Sec 3.4.1 of the original paper)
        self.k = k_nearest
        # The kernel size of the pooling algorithm
        self.p_kernel = p_kernel
        # The number of output class
        self.out_class = torch.cat((torch.unique(self.test_labels), torch.unique(self.train_labels))).unique().size(0)
        self.global_model = ModelFC(self.n_H, in_length=self.train_imgs.size(1), out_class=self.out_class)
        self.participants = []
        self.loss = torch.nn.CrossEntropyLoss()
        self.sum_grad = None
        self.malicious_index = None
        self.malicious_labels = None
        self.attack_mode = attack_mode
        self.scale_target = 0

    def federated_init(self):
        """
        Initialize FL setting, identify malicious participants
        :return: None
        """
        param = self.global_model.get_flatten_parameters()
        for i in range(self.Ph):
            model = ModelFC(self.n_H, in_length=self.train_imgs.size(1), out_class=self.out_class)
            model.load_parameters(param)
            self.participants.append(model)
        self.malicious_index = torch.zeros(self.Ph, dtype=torch.bool)
        self.malicious_index.bernoulli_(self.malicious_factor)

    def data_distribution(self, validation_size=300):
        """
        Divide validation set and test set
        :param validation_size: the size of validation set
        """
        self.batch_size = self.train_imgs.size(0) // (self.Ph * self.batch)
        self.validation_imgs = self.test_imgs[:validation_size]
        self.validation_labels = self.test_labels[:validation_size]
        self.test_imgs = self.test_imgs[validation_size:]
        self.test_labels = self.test_labels[validation_size:]

    def shuffle_data(self):
        """
        Randomly shuffle data
        """
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.train_imgs = self.train_imgs[shuffled_index]
        self.train_labels = self.train_labels[shuffled_index]
        shuffled_index = torch.randperm(self.train_imgs.size(0))
        self.malicious_labels = self.train_labels[shuffled_index]

    def get_training_data(self, idx: int, malicious=False):
        """
        Get the training data for participant No. idx
        :param idx: the index of the participant
        :param malicious: True if the participant is malicious participant, False otherwise
        """
        sample_per_cap = self.train_imgs.size(0) // self.Ph
        low = idx * sample_per_cap
        high = low + sample_per_cap
        if malicious:
            return self.train_imgs[low: high], self.malicious_labels[low: high].flatten()
        return self.train_imgs[low: high], self.train_labels[low: high].flatten()

    def grad_reset(self):
        """
        Reset the globally collected gradients
        :return:
        """
        if self.sum_grad is None:
            length = self.global_model.get_flatten_parameters().size(0)
            self.sum_grad = torch.zeros(self.Ph, length)
        else:
            self.sum_grad.zero_()

    def back_prop(self, attack=False, attack_mode="min_max"):
        """
        Conduct back propagation of one specific participant
        :param attack: if the attacker starts attacking
        :param attack_mode: the type of attack conducted by the attacker
        """
        sum_acc = 0
        sum_loss = 0
        pipe_lost = torch.zeros(self.Ph, dtype=torch.bool)
        pipe_lost.bernoulli_(p=self.pipe_loss)
        # print(pipe_lost)
        for i in range(self.Ph):
            model = self.participants[i]
            if pipe_lost[i]:
                continue
            X, y = self.get_training_data(i)
            acc, loss, grad = model.back_prop(X, y, self.batch_size, self.local_epoch)
            self.collect_grad(i, grad)
            sum_acc += acc
            sum_loss += loss
        if attack and attack_mode == "min_max":
            # Call the code snip from [4] to conduct S&H attack
            all_updates = self.sum_grad.clone()
            all_updates = all_updates[~self.malicious_index]
            for i in range(self.Ph):
                if not self.malicious_index[i]:
                    continue
                local = self.sum_grad[i]
                mal_grad = min_max(all_updates, local)
                self.collect_grad(i, mal_grad)
        if attack and attack_mode in ["mislead", "label_flip", "grad_ascent"]:
            # Conduct label flipping attack or gradient ascent attack, or merged ('mislead')
            for i in range(self.Ph):
                if not self.malicious_index[i]:
                    continue
                model = self.participants[i]
                X, y = self.get_training_data(i, malicious=True)
                acc, loss, grad = model.back_prop(X, y, self.batch_size, self.local_epoch)
                local = self.sum_grad[i]
                if attack_mode == "label_flip":
                    mal_grad = grad
                elif attack_mode == "grad_ascent":
                    mal_grad = - local
                else:
                    mal_grad = grad - local
                self.collect_grad(i, mal_grad)
        if attack and attack_mode in ["scale"]:
            # Conduct T-scal attack from [1]
            for i in range(self.Ph):
                if not self.malicious_index[i]:
                    continue
                model = self.participants[i]
                X, y = self.get_training_data(i)
                X, y = targeted_flip(X, self.scale_target)
                acc, loss, grad = model.back_prop(X, y, self.batch_size, self.local_epoch)
                local = self.sum_grad[i]
                mal_grad = local + grad / self.malicious_factor
                self.collect_grad(i, mal_grad)
        return (sum_acc/self.Ph), (sum_loss/self.Ph)

    def collect_param(self, sparsify=False):
        """
        Participants collect the parameters from the global model
        :param sparsify: Not used, if apply sparsify update
        :return: None
        """
        param = self.global_model.get_flatten_parameters()
        pipe_lost = torch.zeros(self.Ph, dtype=torch.bool)
        pipe_lost.bernoulli_(p=self.pipe_loss)
        for i in range(self.Ph):
            if pipe_lost[i]:
                continue
            model = self.participants[i]
            if sparsify:
                to_load, idx = self.sparsify_update(param)
                model.load_parameters(to_load, mask=idx)
            else:
                model.load_parameters(param)

    def collect_grad(self, idx: int, local_grad: torch.Tensor, norm_clip=False, add_noise=False, sparsify=False):
        """
        AGR collect gradients from the participants
        :param idx: The index of the participant
        :param local_grad: the local gradients from the participant
        :param norm_clip: if apply norm clipping (not used)
        :param add_noise: if add noise to achieve differential privacy (not used)
        :param sparsify: if conduct sparsify update (not used)
        :return: None
        """
        if norm_clip and local_grad.norm() > self.max_grad_norm:
            local_grad = local_grad * self.max_grad_norm / local_grad.norm()
        if add_noise:
            noise = torch.randn(local_grad.size()) * self.sigma
            if noise.norm() > self.max_grad_norm:
                noise = noise * self.max_grad_norm / noise.norm()
            local_grad = local_grad + noise
        if sparsify:
            local_grad, _ = self.sparsify_update(local_grad)
        self.sum_grad[idx] = local_grad

    def apply_grad(self):
        """
        Apply the collected gradients to the global model
        :return: None
        """
        model = self.global_model
        grad = torch.mean(self.sum_grad, dim=0)
        param = model.get_flatten_parameters()
        param = param + grad
        model.load_parameters(param)

    def apply_pooling_def(self):
        """
        Apply distance-based defense (Sec 3.4.1 in the original paper), mainly call the code snips in Defender.py
        :return:
        """
        model = self.global_model
        defender = Defender.PoolingDef(self.train_imgs.size(1), self.n_H, model=model,
                                       validation_X=self.validation_imgs, validation_y=self.validation_labels, kernel=self.p_kernel)
        if self.defender in ["np-dense", "np-cosine", "np-merge"]:
            mode = self.defender[3:]
            grad = defender.filter(grad=self.sum_grad, out_class=self.out_class, k=self.k,
                                   malicious_factor=self.malicious_factor, pooling=False, mode=mode)
        if self.defender in ["p-dense", "p-cosine", "p-merge"]:
            mode = self.defender[2:]
            grad = defender.filter(grad=self.sum_grad, out_class=self.out_class, k=self.k,
                                   malicious_factor=self.malicious_factor, pooling=True, mode=mode)
        grad = torch.mean(grad, dim=0)
        self.last_grad = grad
        param = model.get_flatten_parameters()
        param = param + grad
        model.load_parameters(param)

    def apply_fang_def(self, pooling=False, mode="combined"):
        """
        Apply Fang[2] defense in Defender.py
        :param pooling: True if equip AgrAmplifier, False otherwise
        :param mode: LRR, ERR, or combined mode from the original paper
        :return: The detoxed gradients from Fang
        """
        model = self.global_model
        grad = Defender.fang_defense(self.sum_grad, self.malicious_factor, model, self.validation_imgs,
                                     self.validation_labels.flatten(), self.n_H, self.out_class, pooling, mode, kernel=self.p_kernel)
        grad = torch.mean(grad, dim=0)
        param = model.get_flatten_parameters()
        param += grad
        model.load_parameters(param)

    def apply_fl_trust(self, pooling=False):
        """
        Code snip calling FL-trust[1] defense in Defender.py
        :param pooling: True if apply AgrAmplifier, False if not apply
        """
        model = self.global_model
        grad = Defender.fl_trust(self.sum_grad, self.validation_imgs, self.validation_labels.flatten(),
                                 model, self.batch_size, self.local_epoch, self.n_H, self.out_class, pooling, kernel=self.p_kernel)
        param = model.get_flatten_parameters()
        param += grad
        model.load_parameters(param)

    def apply_other_def(self):
        """
        Apply trimmed-mean[3] (see Reference in readme.md) AGR
        :return:
        """
        if self.defender in ["tr_mean", "p-tr"]:
            grad = Defender.tr_mean(self.sum_grad, self.malicious_factor)
            grad = torch.mean(grad, dim=0)
        if self.defender == "median":
            grad = torch.median(self.sum_grad, dim=0).values
        model = self.global_model
        param = model.get_flatten_parameters()
        param = param + grad
        model.load_parameters(param)

    def sparsify_update(self, gradient, p=None):
        """
        Not used, using random sparsify update schema
        :param gradient: the collected gradients
        :param p: the sampling rate
        """
        if p is None:
            p = self.sampling_prob
        sampling_idx = torch.zeros(gradient.size(), dtype=torch.bool)
        result = torch.zeros(gradient.size())
        sampling_idx.bernoulli_(p)
        result[sampling_idx] = gradient[sampling_idx]
        return result, sampling_idx

    def evaluate_global(self):
        """
        Evaluate the global model accuracy and loss value
        :return: accuracy and loss value
        """
        test_x = self.test_imgs
        test_y = self.test_labels.flatten()
        model = self.global_model
        with torch.no_grad():
            out = model(test_x)
        loss_val = self.loss(out, test_y)
        pred_y = torch.max(out, dim=1).indices
        acc = torch.sum(pred_y == test_y)
        acc = acc / test_y.size(0)
        return acc.item(), loss_val.item()

    def evaluate_target(self):
        """
        Evaluate loss value and accuracy of the targeted label
        :return: accuracy and loss value
        """
        test_x = self.test_imgs
        test_x, _ = targeted_flip(test_x, 0)
        test_y = self.test_labels.flatten()
        model = self.global_model
        with torch.no_grad():
            out = model(test_x)
        loss_val = self.loss(out, test_y)
        pred_y = torch.max(out, dim=1).indices
        idx = test_y != self.scale_target
        pred_y = pred_y[idx]
        test_y = test_y[idx]
        acc = torch.sum(pred_y == self.scale_target)
        acc = acc / test_y.size(0)
        return acc.item(), loss_val.item()

    def grad_sampling(self):
        """
        Save the gradients into a csv file, not used
        :return:
        """
        sampling = torch.zeros(self.sum_grad.size(0), self.sum_grad.size(1) + 1)
        sampling[:, 0] = self.malicious_index
        sampling[:, 1:] = self.sum_grad
        nda = sampling.numpy()
        np.savez_compressed(self.output_path+f"grad_sample_{self.attack_mode}.npz", nda)

    def eq_train(self):
        """
        Organize the FL training process
        """
        epoch_col = []
        train_acc_col = []
        train_loss_col = []
        test_acc_col = []
        test_loss_col = []
        attacking = False
        pooling = False
        if self.defender.startswith("p"):
            pooling = True
        start_count = time.perf_counter()
        for epoch in range(self.num_iter):
            self.collect_param()
            self.grad_reset()
            if epoch == self.start_attack:
                attacking = True
                print(f'Start attacking at round {epoch}')
            acc, loss = self.back_prop(attacking, self.attack_mode)
            # Select a defender according to experiment setting
            if self.defender in ["p-dense", "p-cosine", "p-merge", "np-dense", "np-cosine", "np-merge"]:
                self.apply_pooling_def()
            elif self.defender in ["fang", "lrr", "err", "p-fang"]:
                self.apply_fang_def(pooling, self.defender)
            elif self.defender in ["fl_trust", "p-trust"]:
                self.apply_fl_trust(pooling)
            elif self.defender in ["tr_mean", "median", "p-tr"]:
                self.apply_other_def()
            else:
                self.apply_grad()
            # Print the training progress every 'stride' rounds
            if epoch % self.stride == 0:
                if self.attack_mode == "scale":
                    test_acc, test_loss = self.evaluate_target()
                    print(f'Epoch {epoch} - attack acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
                          f', train loss {loss:6.4f}')
                else:
                    test_acc, test_loss = self.evaluate_global()
                    print(f'Epoch {epoch} - test acc {test_acc:6.4f}, test loss: {test_loss:6.4f}, train acc {acc:6.4f}'
                          f', train loss {loss:6.4f}')
                epoch_col.append(epoch)
                test_acc_col.append(test_acc)
                test_loss_col.append(test_loss)
                train_acc_col.append(acc)
                train_loss_col.append(loss)
        end_count = time.perf_counter()
        recorder = pd.DataFrame({"epoch": epoch_col, "test_acc": test_acc_col, "test_loss": test_loss_col,
                                 "train_acc": train_acc_col, "train_loss": train_loss_col})
        recorder.to_csv(
            self.output_path + f"{self.dataset}_Ph_{self.Ph}_nH_{self.n_H}_MF_{self.malicious_factor}_K_{self.p_kernel}_def_{self.defender}"
                               f"_attack_{self.attack_mode}_start_{self.start_attack}" + "_second_"+ str(end_count-start_count) + time_str +".csv")