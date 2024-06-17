"""Training loops for feed-forward networks.
"""
import os
import torch
import logging, timeit
import numpy as np
from tqdm import tqdm
from datetime import datetime

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

def checkpoint(model, filename, epoch, loss, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 
        filename
        )
    
def train_epochs(
    n_epochs, net, dataloader, optimizer, loss_fn,
    validation_fn=None, device=None, logger=logging.getLogger('train_epochs'),
    checkpoint_epochs=10, checkpoint_path=None
    ):

    epoch_dlog = _dlog_train_epoch_initialize(n_epochs)
    # <code id="training_loop_over_epochs">
    time_train = timeit.default_timer()

    train_start_time = datetime.now().strftime('d%m%d%y_t%H%M%S')

    ## Prep checkpoint directory if it doesnt exist yet
    if checkpoint_epochs is not None:
        # checking if the directory demo_folder  
        # exist or not. 
        _check_dir = f"checkpoints/{checkpoint_path}_{train_start_time}"
        if not os.path.exists(_check_dir): 
            os.makedirs(_check_dir)
        _save_checkpoints = True
    else: 
        _save_checkpoints = False

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
        ## Checkpoint
        if _save_checkpoints:
            if epoch_idx % checkpoint_epochs == 0: 
                filename = f"{_check_dir}/{epoch_idx:02d}-{batch_dlog['loss_mean']:.2f}.pt"
                checkpoint(
                    net, filename, epoch=epoch_idx, loss=loss_fn, 
                    optimizer=optimizer
                    )
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

