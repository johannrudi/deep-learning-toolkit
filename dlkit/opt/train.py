"""Training loops.
"""

import logging, timeit
import numpy as np
from tqdm import tqdm

########################################
# Training of Feed-Forward Networks
########################################

def _dlog_train_epoch_initialize(n_epochs):
    dlog = {}
    for key in ['loss_mean', 'loss_std']:
        dlog[key] = np.empty((n_epochs,))
    dlog['batch_dlog'] = []
    return dlog

def _dlog_train_epoch_update(dlog, epoch_idx, batch_dlog):
    for key in ['loss_mean', 'loss_std']:
        dlog[key][epoch_idx] = batch_dlog[key]
    dlog['batch_dlog'].append(batch_dlog)

def _dlog_train_epoch_finalize(dlog, time_train):
    dlog['time_train'] = time_train

def train_epochs(n_epochs, net, dataloader, optimizer, loss_fn,
                 device=None, logger=logging.getLogger('train_epochs')):
    epoch_dlog = _dlog_train_epoch_initialize(n_epochs)
    # loop over epochs
    time_train = timeit.default_timer()
    for epoch_idx in tqdm(range(n_epochs)):
        batch_dlog = train_batches(net, dataloader, optimizer, loss_fn,
                                   device=device, logger=logger)
        # log
        _dlog_train_epoch_update(epoch_dlog, epoch_idx, batch_dlog)
        logger.info("epoch {:6d}, loss_mean {:.6e} std {:.3e}".format(
                epoch_idx, batch_dlog['loss_mean'], batch_dlog['loss_std']))
    time_train = timeit.default_timer() - time_train
    _dlog_train_epoch_finalize(epoch_dlog, time_train)
    # return log
    return epoch_dlog

def _dlog_train_batch_initialize(n_batches):
    dlog = {}
    for key in ['loss']:
        dlog[key] = np.empty((n_batches,))
    return dlog

def _dlog_train_batch_update(dlog, batch_idx, loss):
    dlog['loss'][batch_idx] = loss

def _dlog_train_batch_finalize(dlog):
    dlog['loss_mean'] = np.mean(dlog['loss'])
    dlog['loss_std']  = np.std(dlog['loss'])

def train_batches(net, dataloader, optimizer, loss_fn,
                  device=None, logger=logging.getLogger('train_batches')):
    batch_dlog = _dlog_train_batch_initialize(len(dataloader))
    # loop over batches
    for batch_idx, data in enumerate(dataloader):
        # set network to training mode
        net.train()
        # get input and target tensors
        inputs, targets = data
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)
        # zero the gradients (begin AD)
        optimizer.zero_grad()
        # forward pass
        outputs = net(inputs)
        # calculate loss
        loss = loss_fn(outputs, targets)
        # calculate derivatives (end AD)
        loss.backward()
        # update network parameters
        optimizer.step()
        # log
        loss_v = loss.item()
        _dlog_train_batch_update(batch_dlog, batch_idx, loss_v)
        logger.debug("batch {:6d}, loss {:.6e}".format(batch_idx, loss_v))
    _dlog_train_batch_finalize(batch_dlog)
    return batch_dlog

########################################
# Training of Generative Adversarial Networks
########################################

def _dlog_train_gan_epoch_initialize(n_epochs):
    dlog = {}
    for key in ['g_loss_mean', 'g_loss_std', 'd_loss_mean', 'd_loss_std', 'd_reg_mean', 'd_reg_std']:
        dlog[key] = np.empty((n_epochs,))
    dlog['batch_dlog'] = []
    return dlog

def _dlog_train_gan_epoch_update(dlog, epoch_idx, batch_dlog):
    for key in ['g_loss_mean', 'g_loss_std', 'd_loss_mean', 'd_loss_std', 'd_reg_mean', 'd_reg_std']:
        dlog[key][epoch_idx] = batch_dlog[key]
    dlog['batch_dlog'].append(batch_dlog)

def _dlog_train_gan_epoch_finalize(dlog, time_train):
    dlog['time_train'] = time_train

def train_gan_epochs(n_epochs, g_net, d_net, dataloader, g_optimizer, d_optimizer, loss_fn,
                     d_reg_fn=None, d_opt_multiplier=1,
                     device=None, logger=logging.getLogger('train_gan_epochs')):
    epoch_dlog = _dlog_train_gan_epoch_initialize(n_epochs)
    # loop over epochs
    time_train = timeit.default_timer()
    for epoch_idx in tqdm(range(n_epochs)):
        batch_dlog = train_gan_batches(g_net, d_net, dataloader, g_optimizer, d_optimizer, loss_fn,
                                       d_reg_fn=d_reg_fn, d_opt_multiplier=d_opt_multiplier,
                                       device=device, logger=logger)
        # log
        _dlog_train_gan_epoch_update(epoch_dlog, epoch_idx, batch_dlog)
        logger.info("epoch {:6d}, d_loss_mean {:.6e} std {:.3e}, g_loss_mean {:.6e} std {:.3e}".format(
                epoch_idx,
                batch_dlog['d_loss_mean'], batch_dlog['d_loss_std'],
                batch_dlog['g_loss_mean'], batch_dlog['g_loss_std']))
    time_train = timeit.default_timer() - time_train
    _dlog_train_gan_epoch_finalize(epoch_dlog, time_train)
    # return log
    return epoch_dlog

def _dlog_train_gan_batch_initialize(n_batches):
    dlog = {}
    for key in ['g_loss', 'd_loss', 'd_reg']:
        dlog[key] = np.empty((n_batches,))
    return dlog

def _dlog_train_gan_batch_update(dlog, batch_idx, g_loss, d_loss, d_reg):
    dlog['g_loss'][batch_idx] = g_loss
    dlog['d_loss'][batch_idx] = d_loss
    dlog['d_reg'][batch_idx]  = d_reg

def _dlog_train_gan_batch_finalize(dlog):
    for key in ['g_loss', 'g_loss', 'd_reg']:
        dlog[key+'_mean'] = np.mean(dlog[key])
        dlog[key+'_std']  = np.std(dlog[key])

def train_gan_batches(g_net, d_net, dataloader, g_optimizer, d_optimizer, loss_fn,
                      d_reg_fn=None, d_opt_multiplier=1,
                      device=None, logger=logging.getLogger('train_gan_batches')):
    train_batch_dlog = _dlog_train_gan_batch_initialize(len(dataloader))
    # loop over batches
    for batch_idx, data in enumerate(dataloader):
        # set networks to training mode
        d_net.train()
        g_net.train()
        # get input and target tensors
        x_data, y_data, z_data = data
        if device is not None:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            z_data = z_data.to(device)

        ### train discriminator network ###

        # generate outputs with `g_net`
        x_gen = g_net(y_data, z_data)
        # evalutate discriminator (begin AD)
        d_optimizer.zero_grad()
        d_outputs_gen  = d_net(x_gen, y_data)
        d_outputs_data = d_net(x_data, y_data)
        # evaluate discriminator loss
        d_loss = loss_fn(d_outputs_gen, d_outputs_data)
        if d_reg_fn is not None:
            d_reg = d_reg_fn(x_gen, x_data, y_data, d_net)
        else:
            d_reg = 0.0
        loss = -d_loss + d_reg  # use negative sign, because we want to maximize the discriminator loss
        # calculate derivatives (end AD) and update network parameters
        loss.backward()
        d_optimizer.step()
        # get values for log
        d_loss_v = d_loss.item()
        d_reg_v  = d_reg.item()

        # if we skip training the generator for this batch
        if batch_idx % d_opt_multiplier:
            # log
            _dlog_train_gan_batch_update(train_batch_dlog, batch_idx, np.nan, d_loss_v, d_reg_v)
            logger.debug("batch {:6d}, d_loss_reg {:.6e}, d_loss {:.6e}, g_loss N/A".format(
                    batch_idx, -d_loss_v+d_reg_v, d_loss_v))
            continue

        ### train generator network ###

        # generate outputs with `g_net (begin AD)`
        g_optimizer.zero_grad()
        x_gen = g_net(y_data, z_data)
        # evalutate discriminator
        d_outputs_gen = d_net(x_gen, y_data)
        # evaluate discriminator loss
        g_loss = loss_fn(d_outputs_gen, None)
        loss = g_loss
        # calculate derivatives (end AD) and update network parameters
        loss.backward()
        g_optimizer.step()
        # get values for log
        g_loss_v = g_loss.item()

        # log
        _dlog_train_gan_batch_update(train_batch_dlog, batch_idx, g_loss_v, d_loss_v, d_reg_v)
        logger.debug("batch {:6d}, d_loss_reg {:.6e}, d_loss {:.6e}, g_loss {:.6e}".format(
                batch_idx, -d_loss_v+d_reg_v, d_loss_v, g_loss_v))

    _dlog_train_gan_batch_finalize(train_batch_dlog)
    return train_batch_dlog

