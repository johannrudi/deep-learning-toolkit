"""Training loops.
"""

import numpy as np
import timeit
from tqdm import tqdm

def train_batches(net, dataloader, optimizer, loss_fn, device=None, logger=None):
    train_batch_log = {}
    # loop over batches
    train_batch_log['loss'] = np.empty((len(dataloader),))
    for batch_idx, data in enumerate(dataloader):
        # set network to training mode
        net.train()
        # get input and target tensors
        inputs, targets = data
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)
        # zero the gradients
        optimizer.zero_grad()
        # forward pass
        outputs = net(inputs)
        # calculate loss and backward pass
        loss = loss_fn(outputs, targets)
        loss.backward()
        # update network parameters
        optimizer.step()
        # log
        loss_v = loss.item()
        train_batch_log['loss'][batch_idx] = loss_v
        if logger is not None:
            logger.debug("batch {:6d}, loss {:.6e}".format(batch_idx, loss_v))
    train_batch_log['loss_mean'] = np.mean(train_batch_log['loss'])
    train_batch_log['loss_std']  = np.std(train_batch_log['loss'])
    return train_batch_log

def train_epochs(n_epochs, net, dataloader, optimizer, loss_fn, device=None, logger=None):
    train_epoch_log = {}
    # loop over epochs
    train_epoch_log['loss_mean'] = np.empty((n_epochs,))
    train_epoch_log['loss_std']  = np.empty((n_epochs,))
    time_train = timeit.default_timer()
    for epoch_idx in tqdm(range(n_epochs)):
        train_batch_log = train_batches(net, dataloader, optimizer, loss_fn,
                                        device=device, logger=logger)
        # log
        if logger is not None:
            loss_mean = train_batch_log['loss_mean']
            loss_std  = train_batch_log['loss_std']
            train_epoch_log['loss_mean'][epoch_idx] = loss_mean
            train_epoch_log['loss_std'][epoch_idx]  = loss_std
            logger.info("epoch {:6d}, loss_batch_mean {:.6e}, loss_batch_std {:.3e}".format(epoch_idx, loss_mean, loss_std))
    time_train = timeit.default_timer() - time_train
    # return log
    train_epoch_log['time_train'] = time_train
    return train_epoch_log
