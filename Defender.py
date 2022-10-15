import torch

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """
    Calculate cosine distance of two vectors if x2 is given, otherwise calculate the pair-wise cosine distance of
    each line inside x1
    :param x1: the first given vector
    :param x2: the second given vector
    :param eps: the dummy minimal value to avoid division-by-zero
    :return: the pair-wise cosine similarity
    """
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def fang_pooling(grad: torch.Tensor, n_H, output_size, kernel_size=3):
    """
    Pooling method designed for Fang AGR and others who need to restore the input size. See Alg. 1 and Fig. 2
    in the original paper
    :param grad: the input collection of gradients
    :param n_H: the size of hidden layer
    :param output_size: The size of output (to avoid vector size issue)
    :param kernel_size: The size of the pooling kernel
    :return: The gradients with all less-activated nodes replaced with 0
    """
    # Remove the residual part of parameters, since we experiment with 1-hidden-layer fully-connected NN, the overall
    # parameters are input_features * hidden_layer_size + hidden_layer_size + hidden_layer_size * output_size +
    # output_size. If cut off output_size, then it should be able to divide by hidden_layer_size. Apply pooling on the
    # reshaped gradients is equivalent to apply layer_basis pooling
    residual = grad[:, -output_size:]
    grad = grad[:, :-output_size]
    grad = grad.reshape(grad.size(0), 1, -1, n_H)
    size1 = grad.size()
    pool = torch.nn.MaxPool2d(kernel_size=kernel_size, return_indices=True)
    unpool = torch.nn.MaxUnpool2d(kernel_size=kernel_size)
    grad, idx = pool(grad)
    grad = unpool(grad, idx, output_size=size1)
    grad = grad.reshape(grad.size(0), -1)
    grad = torch.hstack([grad, residual])
    return grad


def normal_pooling(grad: torch.Tensor, n_H, output_size, kernel_size=3):
    """
    Apply pooling to given collected gradients
    :param grad: collected gradients
    :param n_H: the size of hidden layer
    :param output_size: The size of output class (to avoid vector size issue)
    :param kernel_size: The size of the pooling kernel
    :return: The gradients with all less-activated nodes dropped
    """
    pool = torch.nn.MaxPool2d(kernel_size, stride=kernel_size)
    grad = grad[:, :-output_size]
    grad = grad.reshape(grad.size(0), 1, -1, n_H)
    grad = pool(grad)
    grad = grad.reshape(grad.size(0), -1)
    return grad


class PoolingDef:
    """
    The class to achieve the proposed distance-based AGR, including CosDen, EuDen, MgDen
    """
    def __init__(self, input_size:int, n_H: int, model, validation_X: torch.Tensor, validation_y: torch.Tensor, kernel=3):
        """
        Initialize the parameters
        :param input_size: the batch size of the current input
        :param n_H: The number of neurons in the hidden layer
        :param model: a reference instance of the FL participant model
        :param validation_X: A validation set features (Not used)
        :param validation_y: A validation set labels (Not used)
        :param kernel: the kernel size of pooling
        """
        self.n_H = n_H
        self.input_size = input_size
        self.stride = kernel
        self.kernel_size = kernel
        self.pool = torch.nn.MaxPool2d(self.kernel_size, self.stride)
        self.model = model
        self.validation_X = validation_X
        self.validation_y = validation_y

    def filter(self, grad: torch.Tensor, out_class,k=10, malicious_factor=0.2, pooling=True,
               normalize=True, mode="merge"):
        """
        The function achieves distance-based defense, implementation of Alg. 2 in the original paper APPENDIX
        :param grad: the collected gradients from the participants
        :param out_class: class of outputs
        :param k: count the k-th nearest neighbours, the number of neighbours to be considered
        :param malicious_factor:The fraction of malicious participants
        :param pooling: if conduct pooling, otherwise, if equipping AgrAmplifier, True to equip, FALSE not
        :param normalize: if conduct normalization of the collected gradients
        :param mode: if its EuDen, CosDen, or MgDen
        :return: detoxed gradients
        """
        if normalize:
            grad = torch.nn.functional.normalize(grad)
        replica = grad.clone()
        if pooling:
            # Apply pooling to the collected gradients using
            grad = normal_pooling(grad, self.n_H, out_class)
        selection_size = int(round(grad.size(0) * (1 - malicious_factor)))
        selected = torch.zeros(grad.size(0), dtype=torch.bool)
        if mode in ["merge", "dense"]:
            dist_matrix = torch.cdist(grad, grad)
            k_nearest = torch.topk(dist_matrix, k=k, largest=False, dim=1)
            neighbour_dist = torch.zeros(grad.size(0))
            for i in range(grad.size(0)):
                idx = k_nearest.indices[i]
                neighbour = dist_matrix[idx][:, idx]
                neighbour_dist[i] = neighbour.sum()
            dense_selected = torch.topk(neighbour_dist, largest=False, k=selection_size).indices
            if mode == "dense":
                return replica[dense_selected]
        if mode in ["merge", "cosine"]:
            cos_matrix = cosine_distance_torch(grad)
            k_nearest = torch.topk(cos_matrix, k=k, dim=1)
            neighbour_dist = torch.zeros(grad.size(0))
            for i in range(grad.size(0)):
                idx = k_nearest.indices[i]
                neighbour = cos_matrix[idx][:, idx]
                neighbour_dist[i] = neighbour.sum()
            cos_selected = torch.topk(neighbour_dist, k=selection_size).indices
            if mode == "cosine":
                return replica[cos_selected]
        if mode == "merge":
            union = torch.cat([dense_selected, cos_selected])
            uniques, count = union.unique(return_counts=True)
            selected = uniques[count>1]
        return replica[selected]


def fang_defense(grad: torch.Tensor, malicious_factor: float, model, test_X: torch.Tensor, test_y: torch.Tensor,
                 n_H, output_size, pooling=False,
                 mode="combined", kernel=3):
    """
    The proposed AGR by [2] (See Reference in readme.md)
    :param grad: The collected gradients by aggregator
    :param malicious_factor: the fraction of the malicious participants
    :param model: a reference NN model of participant
    :param test_X: the validation set features of Fang AGR to verity the Loss and Error
    :param test_y: the validation set labels of Fang AGR to verity the Loss and Error
    :param n_H: the hidden layer size
    :param output_size: The output class of the current dataset, used to resize the gradients to avoid gradient shape
    issue
    :param pooling: If conduct pooling, e.g. Equipping AgrAmplifer or not
    :param mode: 'err', 'lrr' or 'combined', representing the working mode of Fang's AGR
    :param kernel: the kernel size of pooling
    :return: The detoxed gradient collection
    """
    base_param = model.get_flatten_parameters()
    acc_rec = torch.zeros(grad.size(0))
    loss_rec = torch.zeros(grad.size(0))
    replica = grad.clone()
    if pooling:
        grad = fang_pooling(grad, n_H, output_size, kernel_size=kernel)
    for i in range(grad.size(0)):
        local_grad = grad[i]
        param = base_param + local_grad
        model.load_parameters(param)
        acc, loss, g = model.back_prop(X=test_X, y=test_y, batch_size=test_X.size(0), local_epoch=1)
        acc_rec[i] = acc
        loss_rec[i] = loss
    model.load_parameters(base_param)
    k_selection = int(round(grad.size(0) * (1 - malicious_factor)))
    ERR = torch.topk(acc_rec, k_selection).indices
    if mode == "err":
        return replica[ERR]
    LRR = torch.topk(loss_rec, k_selection, largest=False).indices
    if mode == "lrr":
        return replica[LRR]
    # If it's merged ERR and LRR, then compute the intersection
    union = torch.cat([ERR, LRR])
    uniques, counts = union.unique(return_counts=True)
    final_idx = uniques[counts > 1]
    return replica[final_idx]


def tr_mean(grad: torch.Tensor, malicious_factor: float):
    """
    The Trimmed-mean AGR proposed in [3] (see Reference in readme.md)
    :param grad:
    :param malicious_factor:
    :return:
    """
    m_count = int(round(grad.size(0) * malicious_factor))
    sorted_grad = torch.sort(grad, dim=0)[0]
    return sorted_grad[m_count: -m_count]


def fl_trust(grad: torch.Tensor, validation_imgs: torch.Tensor, validation_label: torch.Tensor, model, batch_size,
             local_epoch, n_H, output_size, pooling=False, kernel=3):
    """
    FL Trust AGR by [1]
    :param grad: the collected gradients by the aggregator
    :param validation_imgs: the trusted-root-set features of FL-Trust used to validate other inputs
    :param validation_label: the trusted-root-set labels of FL-Trust used to validate other inputs
    :param model: a replica of the global model, used to calculate the root set gradients
    :param batch_size: the batch size
    :param local_epoch: the epochs for local participant
    :param n_H: the size of the hidden layer
    :param output_size: the output class
    :param pooling: if equip AgrAmplifer or not
    :param kernel: the kernel size of pooling
    :return: the detoxed global gradients
    """
    replica = grad.clone()
    acc, loss, grad_zero = model.back_prop(validation_imgs, validation_label, batch_size, local_epoch, revert=True)
    grad_zero = grad_zero.unsqueeze(0)
    if pooling:
        grad = normal_pooling(grad, n_H, output_size, kernel_size=kernel)
        grad_zero = normal_pooling(grad_zero, n_H, output_size, kernel_size=kernel)
    cos = torch.nn.CosineSimilarity(eps=1e-5)
    relu = torch.nn.ReLU()
    norm = grad_zero.norm()
    scores = cos(grad, grad_zero)
    scores = relu(scores)
    grad = torch.nn.functional.normalize(replica) * norm
    grad = (grad.transpose(0, 1) * scores).transpose(0, 1)
    grad = torch.sum(grad, dim=0) / scores.sum()
    return grad



