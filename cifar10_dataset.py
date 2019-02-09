import chainer
from chainer.dataset import dataset_mixin


class Cifar10Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, split='train'):
        x_train, x_test = chainer.datasets.get_cifar10(ndim=3, withlabel=False,
                                                       scale=1.0)
        if split == 'train':
            self.imgs = x_train
        elif split == 'test':
            self.imgs = x_test
        self.imgs = self.imgs * 2 - 1.0  # [0, 1] to [-1.0, 1.0]

    def __len__(self):
        return len(self.imgs)

    def get_example(self, index):
        return self.imgs[index]
