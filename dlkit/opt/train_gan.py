"""Training loops for generative adversarial networks.
"""

import logging, timeit
import numpy as np
from tqdm import tqdm

def _dlog_train_epoch_initialize(n_epochs):
    dlog = {}
    for key in ['g_loss_mean', 'g_loss_std', 'd_loss_mean', 'd_loss_std', 'd_reg_mean', 'd_reg_std']:
        dlog[key] = np.empty((n_epochs,))
    dlog['batch_dlog'] = []
    return dlog

def _dlog_train_epoch_update(dlog, epoch_idx, batch_dlog):
    for key in ['g_loss_mean', 'g_loss_std', 'd_loss_mean', 'd_loss_std', 'd_reg_mean', 'd_reg_std']:
        dlog[key][epoch_idx] = batch_dlog[key]
    dlog['batch_dlog'].append(batch_dlog)

def _dlog_train_epoch_finalize(dlog, time_train):
    dlog['time_train'] = time_train

def train_epochs(n_epochs, g_net, d_net, dataloader, g_optimizer, d_optimizer, loss_fn,
                 d_reg_fn=None, d_opt_multiplier=1, validation_fn=None,
                 device=None, logger=logging.getLogger('train_epochs')):
    epoch_dlog = _dlog_train_epoch_initialize(n_epochs)
    # <code id="training_loop_over_epochs">
    time_train = timeit.default_timer()
    for epoch_idx in tqdm(range(n_epochs)):
        # call validation function
        if validation_fn is not None:
            validation_fn(epoch_idx, g_net, d_net)
        # train on batches
        batch_dlog = train_batches(epoch_idx, g_net, d_net, dataloader, g_optimizer, d_optimizer, loss_fn,
                                   d_reg_fn=d_reg_fn, d_opt_multiplier=d_opt_multiplier,
                                   device=device, logger=logger)
        # log
        _dlog_train_epoch_update(epoch_dlog, epoch_idx, batch_dlog)
        logger.info("epoch {:6d}, d_loss_mean {:.6e} std {:.3e}, g_loss_mean {:.6e} std {:.3e}".format(
                epoch_idx,
                batch_dlog['d_loss_mean'], batch_dlog['d_loss_std'],
                batch_dlog['g_loss_mean'], batch_dlog['g_loss_std']))
    time_train = timeit.default_timer() - time_train
    # call validation function after training
    if validation_fn is not None:
        validation_fn(n_epochs, g_net, d_net)
    # </code>
    # finalize and return log
    _dlog_train_epoch_finalize(epoch_dlog, time_train)
    return epoch_dlog

def _dlog_train_batch_initialize(n_batches):
    dlog = {}
    for key in ['g_loss', 'd_loss', 'd_reg']:
        dlog[key] = np.empty((n_batches,))
    return dlog

def _dlog_train_batch_update(dlog, batch_idx, g_loss, d_loss, d_reg):
    dlog['g_loss'][batch_idx] = g_loss
    dlog['d_loss'][batch_idx] = d_loss
    dlog['d_reg'][batch_idx]  = d_reg

def _dlog_train_batch_finalize(dlog):
    for key in ['g_loss', 'd_loss', 'd_reg']:
        is_valid = np.logical_not(np.isnan(dlog[key]))
        dlog[key+'_mean'] = np.mean(dlog[key][is_valid])
        dlog[key+'_std'] = np.std(dlog[key][is_valid])

def _train_step_discriminator(x_data, y_data, z_data,
                              g_net, d_net, d_optimizer, loss_fn, d_reg_fn=None):
    """ Trains discriminator network. """
    # generate outputs with `g_net`
    x_gen = g_net(y_data, z_data).detach()
    # evalutate discriminator (begin AD)
    d_optimizer.zero_grad()
    d_outputs_gen  = d_net(x_gen, y_data)
    d_outputs_data = d_net(x_data, y_data)
    # evaluate discriminator loss
    d_loss = loss_fn(d_outputs_gen, d_outputs_data)  # loss must have correct sign for minimization
    if d_reg_fn is not None:
        d_reg = d_reg_fn(d_net, x_gen, x_data, y_data)
    else:
        d_reg = 0.0
    loss = d_loss + d_reg
    # calculate derivatives (end AD) and update network parameters
    loss.backward()
    d_optimizer.step()
    # return values for log
    return d_loss.item(), d_reg.item()

def _train_step_generator(y_data, z_data,
                          g_net, d_net, g_optimizer, loss_fn):
        """ Trains generator network. """
        # generate outputs with `g_net (begin AD)`
        g_optimizer.zero_grad()
        x_gen = g_net(y_data, z_data)
        # evalutate discriminator
        d_outputs_gen = d_net(x_gen, y_data)
        # evaluate discriminator loss
        g_loss = loss_fn(None, d_outputs_gen)  # pass generated outputs as data/truth
        loss = g_loss
        # calculate derivatives (end AD) and update network parameters
        loss.backward()
        g_optimizer.step()
        # return values for log
        return g_loss.item()

def train_batches(epoch_idx, g_net, d_net, dataloader, g_optimizer, d_optimizer, loss_fn,
                  d_reg_fn=None, d_opt_multiplier=1,
                  device=None, logger=logging.getLogger('train_batches')):
    train_batch_dlog = _dlog_train_batch_initialize(len(dataloader))
    # <code id="training_loop_over_batches">
    for batch_idx, data in enumerate(dataloader):
        # set networks to training mode
        g_net.train()
        d_net.train()
        # get input and target tensors
        x_data, y_data, z_data = data
        if device is not None:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            z_data = z_data.to(device)

        # train discriminator network
        d_loss_v, d_reg_v = _train_step_discriminator(
                x_data, y_data, z_data,
                g_net, d_net, d_optimizer, loss_fn, d_reg_fn=d_reg_fn)

        # if we skip training the generator for this batch
        if 0 < batch_idx % d_opt_multiplier:
            # log
            _dlog_train_batch_update(train_batch_dlog, batch_idx, np.nan, d_loss_v, d_reg_v)
            logger.debug("batch {:6d}, d_loss_reg {:.6e}, d_loss {:.6e}, g_loss N/A".format(
                    batch_idx, d_loss_v + d_reg_v, d_loss_v))
            continue

        # train generator network
        g_loss_v = _train_step_generator(
                y_data, z_data,
                g_net, d_net, g_optimizer, loss_fn)

        # repeat training of discriminator network
        d_loss_v, d_reg_v = _train_step_discriminator(
                x_data, y_data, z_data,
                g_net, d_net, d_optimizer, loss_fn, d_reg_fn=d_reg_fn)

        # log
        _dlog_train_batch_update(train_batch_dlog, batch_idx, g_loss_v, d_loss_v, d_reg_v)
        logger.debug("batch {:6d}, d_loss_reg {:.6e}, d_loss {:.6e}, g_loss {:.6e}".format(
                batch_idx, -d_loss_v+d_reg_v, d_loss_v, g_loss_v))
    # </code>

    # finalize and return log
    _dlog_train_batch_finalize(train_batch_dlog)
    return train_batch_dlog

