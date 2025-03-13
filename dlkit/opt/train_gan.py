"""Training loops for generative adversarial networks.
"""

import logging, math, os, timeit
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

def _checkpoint_path(checkpoint_dir, n_epochs, prefix, epoch):
    n_digits = int(math.ceil(math.log10(1.01 * n_epochs)))
    fmt      = '{}_e{:0'+str(n_digits)+'d}.pt'
    filename = fmt.format(prefix, epoch)
    return os.path.join(checkpoint_dir, filename)

def _checkpoint_save(model, filepath, epoch, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        filepath
    )

def _dlog_train_epoch_initialize(n_epochs, save_list=True):
    dlog = {}
    for key in ['g_loss_mean',      'g_loss_std',
                'd_loss_mean',      'd_loss_std',
                'd_loss_g_mean',    'd_loss_g_std',
                'd_reg_mean',       'd_reg_std',
                'd_grad_norm_mean', 'd_grad_norm_std']:
        if save_list:
            dlog[key] = np.empty((n_epochs,))
        else:
            dlog[key] = None
    dlog['batch_dlog'] = []
    return dlog

def _dlog_train_epoch_update(dlog, epoch_idx, batch_dlog):
    for key in ['g_loss_mean',      'g_loss_std',
                'd_loss_mean',      'd_loss_std',
                'd_loss_g_mean',    'd_loss_g_std',
                'd_reg_mean',       'd_reg_std',
                'd_grad_norm_mean', 'd_grad_norm_std']:
        if dlog[key] is not None:
            dlog[key][epoch_idx] = batch_dlog[key]
    dlog['batch_dlog'].append(batch_dlog)

def _dlog_train_epoch_finalize(dlog, time_train):
    dlog['time_train'] = time_train

def train_epochs(
        n_epochs,
        g_net,
        d_net,
        dataloader,
        z_sample_fn,
        g_optimizer,
        d_optimizer,
        loss_fn,
        d_reg_fn=None,
        d_opt_multiplier=1,
        validation_fn=None,
        g_lr_scheduler=None,
        d_lr_scheduler=None,
        device=None,
        logger=logging.getLogger('train_epochs'),
        checkpoint_epochs=None,
        checkpoint_dir='checkpoints'
    ):
    """ Runs training loop over epochs.

    Checkpointing saves networks `g_net`, `d_net` and both optimizer states at
    every epoch divisible by `checkpoint_epochs`. Setting `checkpoint_epochs=None`
    turns off checkpointing. Checkpointing will create a new dir under
    `checkpoint_dir/` followed by a directory with date and time of training start.
    """
    epoch_dlog = _dlog_train_epoch_initialize(n_epochs)
    # set checkpoint directory; create if it doesn't exist
    if checkpoint_epochs is not None:
        assert 1 <= checkpoint_epochs, checkpoint_epochs
        assert checkpoint_dir is not None
        checkpoint_time = datetime.now().strftime('%Y-%m-%d_t%H%M%S')
        checkpoint_dir  = os.path.join(checkpoint_dir, checkpoint_time)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    # <code id="training_loop_over_epochs">
    time_train = timeit.default_timer()
    for epoch_idx in tqdm(range(n_epochs)):
        # save checkpoint
        if checkpoint_epochs is not None and (epoch_idx % checkpoint_epochs == 0):
            for tag_, net_, opt_ in zip(['g', 'd'], [g_net, d_net], [g_optimizer, d_optimizer]):
                path = _checkpoint_path(checkpoint_dir, n_epochs, prefix=tag_+'-net', epoch=epoch_idx)
                logger.debug("epoch {:6d}, save checkpoint to {}".format(epoch_idx, path))
                _checkpoint_save(net_, path, epoch=epoch_idx, optimizer=opt_)
        # call validation function
        if validation_fn is not None:
            validation_fn(epoch_idx, g_net, d_net)
        # train on batches
        batch_dlog = train_batches(
                epoch_idx,
                g_net,
                d_net,
                dataloader,
                z_sample_fn,
                g_optimizer,
                d_optimizer,
                loss_fn,
                d_reg_fn=d_reg_fn,
                d_opt_multiplier=d_opt_multiplier,
                device=device, logger=logger
        )
        # update the learning rate schedulers
        if g_lr_scheduler is not None:
            g_lr_current = g_lr_scheduler.get_last_lr()
            if 1 == len(g_lr_current):
                g_lr_current = "{:.6e}".format(g_lr_current[0])
            else:
                g_lr_current = str(g_lr_current)
            g_lr_scheduler.step()
        else:
            g_lr_current = 'n/a'
        if d_lr_scheduler is not None:
            d_lr_current = d_lr_scheduler.get_last_lr()
            if 1 == len(d_lr_current):
                d_lr_current = "{:.6e}".format(d_lr_current[0])
            else:
                d_lr_current = str(d_lr_current)
            d_lr_scheduler.step()
        else:
            d_lr_current = 'n/a'
        if g_lr_scheduler is not None or d_lr_scheduler is not None:
            logger.debug("epoch {:6d}, g_learning_rate {}, d_learning_rate {}".format(epoch_idx, g_lr_current, d_lr_current))
        # log
        _dlog_train_epoch_update(epoch_dlog, epoch_idx, batch_dlog)
        logger.info("epoch {:6d}, d_loss_mean {:.6e} std {:.3e}, g_loss_mean {:.6e} std {:.3e}".format(
                epoch_idx,
                batch_dlog['d_loss_mean'], batch_dlog['d_loss_std'],
                batch_dlog['g_loss_mean'], batch_dlog['g_loss_std']))
    # save checkpoint---after training
    if checkpoint_epochs is not None:
        for tag_, net_, opt_ in zip(['g', 'd'], [g_net, d_net], [g_optimizer, d_optimizer]):
            path = _checkpoint_path(checkpoint_dir, n_epochs, prefix=tag_+'-net', epoch=n_epochs)
            logger.debug("epoch {:6d}, save checkpoint to {}".format(n_epochs, path))
            _checkpoint_save(net_, path, epoch=n_epochs, optimizer=opt_)
    # call validation function---after training
    if validation_fn is not None:
        validation_fn(n_epochs, g_net, d_net)
    time_train = timeit.default_timer() - time_train
    # </code>
    # finalize and return log
    _dlog_train_epoch_finalize(epoch_dlog, time_train)
    logger.info("training time {:g} sec, time/epoch {:g} sec".format(time_train, time_train/n_epochs))
    return epoch_dlog

def _dlog_train_batch_initialize(n_batches, save_list=False):
    dlog = {'n_batches': n_batches}
    for key in ['g_loss', 'd_loss', 'd_loss_g', 'd_reg', 'd_grad_norm']:
        if save_list:
            dlog[key] = np.empty((n_batches,))
        else:
            dlog[key] = None
        dlog[key+'_mean_n']  = 0
        dlog[key+'_mean']    = 0.0
        dlog[key+'_sq_mean'] = 0.0
        dlog[key+'_std']     = None
    return dlog

def _dlog_train_batch_update(dlog, batch_idx, values):
    for key in ['g_loss', 'd_loss', 'd_loss_g', 'd_reg', 'd_grad_norm']:
        if key in values:
            val = values[key]
        else:
            val = np.nan
        if dlog[key] is not None:
            dlog[key][batch_idx] = val
        if not np.isnan(val):
            dlog[key+'_mean_n']  += 1
            dlog[key+'_mean']    += val
            dlog[key+'_sq_mean'] += val*val

def _dlog_train_batch_finalize(dlog):
    for key in ['g_loss', 'd_loss', 'd_loss_g', 'd_reg', 'd_grad_norm']:
        assert dlog[key+'_std'] is None
        assert not np.isnan(dlog[key+'_mean'])
        assert not np.isnan(dlog[key+'_sq_mean'])
        if 0 < dlog[key+'_mean_n']:
            dlog[key+'_mean']    *= 1.0/dlog[key+'_mean_n']
            dlog[key+'_sq_mean'] *= 1.0/dlog[key+'_mean_n']
            dlog[key+'_std']      = np.sqrt(dlog[key+'_sq_mean'] - dlog[key+'_mean']**2)
        else:
            dlog[key+'_mean']     = 0.0
            dlog[key+'_sq_mean']  = 0.0
            dlog[key+'_std']      = 0.0

def _train_step_discriminator(x_data, y_data, z_sample_fn,
                              g_net, d_net, d_optimizer, loss_fn, d_reg_fn=None, dlog_item=None):
    """ Trains discriminator network. """
    # sample from latent space for generator
    batch_size = y_data.size(0)
    z = z_sample_fn(batch_size)
    # generate outputs with `g_net`
    x_gen = torch.vmap( lambda z_: g_net(y_data, z_) )(z).detach()
    # evalutate discriminator (begin AD)
    d_optimizer.zero_grad()
    d_outputs_gen  = torch.vmap( lambda x_: d_net(x_, y_data) )(x_gen).flatten(start_dim=0, end_dim=1)
    d_outputs_data = d_net(x_data, y_data)
    # evaluate discriminator loss
    d_loss, d_loss_g = loss_fn(d_outputs_gen, d_outputs_data)  # output must have correct sign for minimization
    d_reg_dlog = dict()
    if d_reg_fn is not None:
        d_reg = d_reg_fn(d_net, x_gen[np.random.randint(0, x_gen.size(0))], x_data, y_data, dlog=d_reg_dlog)
    else:
        d_reg = torch.tensor(0.0)
    loss = d_loss + d_reg
    # calculate derivatives (end AD) and update network parameters
    loss.backward()
    d_optimizer.step()
    # output values
    if dlog_item is not None:
        dlog_item['d_loss']   = d_loss.item()
        dlog_item['d_loss_g'] = d_loss_g.item()
        dlog_item['d_reg']    = d_reg.item()
        for key, val in d_reg_dlog.items():
            dlog_item['d_'+key] = val
    return loss.item()

def _train_step_generator(y_data, z_sample_fn,
                          g_net, d_net, g_optimizer, loss_fn, dlog_item=None):
    """ Trains generator network. """
    # sample from latent space for generator
    batch_size = y_data.size(0)
    z = z_sample_fn(batch_size)
    # generate outputs with `g_net` (begin AD)
    g_optimizer.zero_grad()
    x_gen = torch.vmap( lambda z_: g_net(y_data, z_) )(z)
    # evalutate discriminator
    d_outputs_gen = torch.vmap( lambda x_: d_net(x_, y_data) )(x_gen).flatten(start_dim=0, end_dim=1)
    # evaluate discriminator loss
    g_loss, _ = loss_fn(None, d_outputs_gen)  # pass generated outputs as data/truth
    loss = g_loss
    # calculate derivatives (end AD) and update network parameters
    loss.backward()
    g_optimizer.step()
    # output values
    if dlog_item is not None:
        dlog_item['g_loss'] = g_loss.item()
    return loss.item()

def train_batches(
        epoch_idx,
        g_net,
        d_net,
        dataloader,
        z_sample_fn,
        g_optimizer,
        d_optimizer,
        loss_fn,
        d_reg_fn=None,
        d_opt_multiplier=1,
        device=None,
        logger=logging.getLogger('train_batches')
    ):
    """ Runs training loop over batches. """
    train_batch_dlog = _dlog_train_batch_initialize(len(dataloader))
    # <code id="training_loop_over_batches">
    for batch_idx, data in enumerate(dataloader):
        dlog_item = dict()
        # set networks to training mode
        g_net.train()
        d_net.train()
        # get input and target tensors
        x_data, y_data = data
        if device is not None:
            x_data = x_data.to(device)
            y_data = y_data.to(device)

        # train discriminator network
        d_loss_reg_v = _train_step_discriminator(
                x_data, y_data, z_sample_fn,
                g_net, d_net, d_optimizer, loss_fn, d_reg_fn=d_reg_fn, dlog_item=dlog_item)

        # if we skip training the generator for this batch
        if 0 < batch_idx % d_opt_multiplier:
            # log
            _dlog_train_batch_update(train_batch_dlog, batch_idx, dlog_item)
            logger.debug("batch {:6d}, d_loss_reg {:.6e}, d_loss {:.6e}, g_loss N/A".format(
                    batch_idx, d_loss_reg_v, dlog_item['d_loss']
            ))
            continue

        # train generator network
        g_loss_v = _train_step_generator(
                y_data, z_sample_fn,
                g_net, d_net, g_optimizer, loss_fn, dlog_item=dlog_item)

        # repeat training of discriminator network
        d_loss_reg_v = _train_step_discriminator(
                x_data, y_data, z_sample_fn,
                g_net, d_net, d_optimizer, loss_fn, d_reg_fn=d_reg_fn, dlog_item=dlog_item)

        # log
        _dlog_train_batch_update(train_batch_dlog, batch_idx, dlog_item)
        logger.debug("batch {:6d}, d_loss_reg {:.6e}, d_loss {:.6e}, g_loss {:.6e}".format(
                batch_idx, d_loss_reg_v, dlog_item['d_loss'], g_loss_v
        ))
    # </code>

    # finalize and return log
    _dlog_train_batch_finalize(train_batch_dlog)
    return train_batch_dlog

