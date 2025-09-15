from torchvision import datasets, transforms
import numpy as np


def get_dataset(args):
    print("Dataset: cifar10")
    trans_cifar10_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    trans_cifar10_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
    dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_test)
    print("Length train: ", len(dataset_train))
    print("Length test: ", len(dataset_test))

    min_size = 0
    min_require_size = 10
    K = args.num_classes
    y_train = np.array(dataset_train.targets)
    N = len(dataset_train)
    dict_users = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(args.num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(args.Drichlet_arg, args.num_users))
            proportions = np.array(
                [p * (len(idx_j) < N / args.num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(args.num_users):
        dict_users[j] = idx_batch[j]

    return dataset_train, dataset_test, dict_users


