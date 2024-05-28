"""Training loops for feed-forward networks.
"""

import logging, timeit
import numpy as np
from tqdm import tqdm

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

