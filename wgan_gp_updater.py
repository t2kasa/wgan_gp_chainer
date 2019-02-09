import chainer
import chainer.functions as F


class WGANGPUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.lam = kwargs.pop('lam')
        super(WGANGPUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        opt_gen = self.get_optimizer('opt_gen')
        opt_dis = self.get_optimizer('opt_dis')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batch_size = len(batch)
            x_real = chainer.as_variable(xp.asarray(batch, dtype='f'))
            y_real = self.dis(x_real)

            # z -> x_fake -> y_fake
            z = chainer.as_variable(
                xp.asarray(self.gen.make_hidden(batch_size)))
            x_fake = self.gen(z)
            y_fake = self.dis(x_fake)

            if i == 0:
                loss_gen = F.mean(-y_fake)
                self.gen.cleargrads()
                loss_gen.backward()
                opt_gen.update()
                chainer.reporter.report({'loss_gen': loss_gen})
            # gen is not updated after here
            x_fake.unchain_backward()

            # compute adversarial loss
            loss_dis = F.mean(-y_real) + F.mean(y_fake)

            # compute gradient penalty
            eps = xp.random.uniform(0, 1, size=(batch_size, 1, 1, 1))
            eps = eps.astype('f')

            x_mid = chainer.as_variable(eps * x_real + (1.0 - eps) * x_fake)
            y_mid = self.dis(x_mid)

            grad = chainer.grad([y_mid], [x_mid],
                                enable_double_backprop=True)[0]
            grad_norm = F.sqrt(F.batch_l2_norm_squared(grad))

            loss_gp = self.lam * F.mean_squared_error(
                grad_norm, xp.ones_like(grad_norm.data))

            # update dis
            self.dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            opt_dis.update()

            chainer.reporter.report({'loss_dis': loss_dis})
            chainer.reporter.report({'loss_gp': loss_gp})
            chainer.reporter.report({'g': F.mean(grad_norm)})
