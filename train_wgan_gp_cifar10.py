import argparse
import os
from datetime import datetime

import chainer
from chainer import training
from chainer.backends import cuda
from chainer.training import extensions
from chainerui import summary

from cifar10_dataset import Cifar10Dataset
from extensions import generate_images_generator
from models import Generator, Discriminator
from wgan_gp_updater import WGANGPUpdater


def make_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def _get_class_name(instance):
    return type(instance).__name__


def _parse_args():
    default_out_dir = os.path.join('results',
                                   datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--out_dir', '-o', type=str, default=default_out_dir)
    parser.add_argument('--snapshot_interval', type=int, default=10000)
    parser.add_argument('--evaluation_interval', type=int, default=10000)
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--n_dis', type=int, default=5,
                        help='the number of dis update per gen update.')
    parser.add_argument('--lam', type=float, default=10,
                        help='gradient penalty')
    parser.add_argument('--adam_alpha', type=float, default=0.0002,
                        help='alpha in Adam optimizer')
    parser.add_argument('--adam_beta1', type=float, default=0.0,
                        help='beta1 in Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.9,
                        help='beta2 in Adam optimizer')

    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    summary.set_out(args.out_dir)

    # Set up dataset
    train_ds = Cifar10Dataset()
    train_iter = chainer.iterators.SerialIterator(train_ds, args.batch_size)

    gen = Generator()
    dis = Discriminator()

    models = [gen, dis]
    if 0 <= args.gpu:
        cuda.get_device_from_id(args.gpu).use()
        print('use gpu {}'.format(args.gpu))
        for m in models:
            m.to_gpu()

    # Set up optimizers.
    opts = {
        'opt_gen': make_optimizer(
            gen, args.adam_alpha, args.adam_beta1, args.adam_beta2),
        'opt_dis': make_optimizer(
            dis, args.adam_alpha, args.adam_beta1, args.adam_beta2)
    }

    updater_args = {'iterator': {'main': train_iter}, 'device': args.gpu,
                    'n_dis': args.n_dis, 'lam': args.lam, 'models': models,
                    'optimizer': opts}

    # Set up updater and trainer.
    updater = WGANGPUpdater(**updater_args)
    trainer = training.Trainer(updater, (args.max_iter, 'iteration'),
                               out=args.out_dir)

    # Add extensions.
    report_keys = ['loss_dis', 'loss_gen', 'loss_gp', 'g']
    for m in models:
        trainer.extend(extensions.snapshot_object(
            m, 'iter{.updater.iteration}_' + _get_class_name(m) + '.npz'),
            trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(
        keys=report_keys, trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys),
                   trigger=(args.display_interval, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(generate_images_generator())

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
