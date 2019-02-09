import chainer
from chainer import training
from chainerui import summary


def generate_images_generator(n_rows=10, n_cols=10, trigger=(1, 'epoch'),
                              image_name_format='iter{.updater.iteration:06d}'):
    n_samples = n_rows * n_cols

    @training.make_extension(trigger=trigger)
    def generate_images(trainer):
        gen = trainer.updater.gen
        xp = gen.xp

        z = xp.asarray(gen.make_hidden(n_samples))

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                x = gen(z)
        summary.image(x, name=image_name_format.format(trainer), row=n_rows)

    return generate_images
