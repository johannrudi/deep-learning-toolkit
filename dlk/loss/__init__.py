from dlk.loss.hinge_gan import hinge_loss_fn
from dlk.loss.least_squares_gan import least_squares_loss_fn
from dlk.loss.wasserstein_gan import wasserstein_loss_fn
from dlk.opt.train_gan import GANLossFn

hinge_loss: GANLossFn = hinge_loss_fn
least_squares_gan_loss: GANLossFn = least_squares_loss_fn
wasserstein_loss: GANLossFn = wasserstein_loss_fn
