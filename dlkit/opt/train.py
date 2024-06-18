"""Training loops for feed-forward networks.
"""
import logging, math, os, timeit
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

def _checkpoint_save(model, filepath, epoch, loss, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        filepath
    )

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

def train_epochs(
    n_epochs, net, dataloader, optimizer, loss_fn,
    validation_fn=None, device=None, logger=logging.getLogger('train_epochs'),
    checkpoint_epochs=None, checkpoint_dir='checkpoints'
    ):
    """ Runs training loop over epochs.

    Checkpointing saves model and optimizer states at every epoch divisible
    by `checkpoint_epochs`. Setting `checkpoint_epochs=None` turns off
    checkpointing. Checkpointing will create a new dir under `checkpoint_dir/`
    followed by a directory with date and time of training start.
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
        # call validation function
        if validation_fn is not None:
            validation_fn(epoch_idx, net)
        # train on batches
        batch_dlog = train_batches(epoch_idx, net, dataloader, optimizer, loss_fn,
                                   device=device, logger=logger)
        # log
        _dlog_train_epoch_update(epoch_dlog, epoch_idx, batch_dlog)
        logger.info("epoch {:6d}, loss_mean {:.6e} std {:.3e}".format(
                epoch_idx, batch_dlog['loss_mean'], batch_dlog['loss_std']))
        # save checkpoint
        if checkpoint_epochs is not None and \
           ( (epoch_idx % checkpoint_epochs == 0) or (epoch_idx == (n_epochs-1)) ):
                fmt = '{:0'+str(math.ceil(math.log10(n_epochs)))+'d}_{:.2e}.pt'
                filename = fmt.format(epoch_idx, batch_dlog['loss_mean'])
                path = os.path.join(checkpoint_dir, filename)
                _checkpoint_save(net, path, epoch=epoch_idx, loss=loss_fn, optimizer=optimizer)
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
    for key in ['loss']:
        dlog[key] = np.empty((n_batches,))
    return dlog

def _dlog_train_batch_update(dlog, batch_idx, loss):
    dlog['loss'][batch_idx] = loss

def _dlog_train_batch_finalize(dlog):
    is_valid = np.logical_not(np.isnan(dlog['loss']))
    dlog['loss_mean'] = np.mean(dlog['loss'][is_valid])
    dlog['loss_std'] = np.std(dlog['loss'][is_valid])

def train_batches(epoch_idx, net, dataloader, optimizer, loss_fn,
                  device=None, logger=logging.getLogger('train_batches')):
    """ Runs training loop over batches. """
    batch_dlog = _dlog_train_batch_initialize(len(dataloader))
    # <code id="training_loop_over_batches">
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
    # </code>
    # finalize and return log
    _dlog_train_batch_finalize(batch_dlog)
    return batch_dlog

