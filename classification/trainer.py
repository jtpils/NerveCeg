import abc
import os
import sys
import tqdm
import torch
import datetime

from torch.utils.data import DataLoader
from typing import Callable, Any
from typing import NamedTuple, List
from torchvision.utils import make_grid


def to_np(x):
    return x.data.cpu().numpy()

class BatchResult(NamedTuple):
    loss: float


class EpochResult(NamedTuple):
    losses: List[float]


class FitResult(NamedTuple):
    num_epochs: int
    train_loss: List[float]
    test_loss: List[float]


class Trainer:
    def __init__(self, model, loss_fn, optimizer, device='cuda', tensorboard_logger=None, tensorboard_log_images=True, experiment_prefix=None):
        self.tensorboard_logger = tensorboard_logger

        if experiment_prefix is None:
            now = datetime.datetime.now()
            self.experiment_prefix = now.strftime(r"%Y-%m-%d\%H:%M:%S")
        else:
            self.experiment_prefix = experiment_prefix
        self.tensorboard_log_images = tensorboard_log_images
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, **kw) -> FitResult:
        actual_num_epochs = 0
        train_loss, test_loss = [], []
        
        for epoch in range(num_epochs):
            verbose = None
            if epoch % print_every == 0 or epoch == num_epochs-1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)
            epoch_train_res = self.train_epoch(dl_train, verbose=verbose, **kw)
            train_loss.extend([float(x.item()) for x in epoch_train_res.losses])

            epoch_test_res = self.test_epoch(dl_test, verbose=verbose, **kw)
            test_loss.extend([float(x.item()) for x in epoch_test_res.losses])

            torch.save(self.model, './cls/'+'ckpt_'+str(epoch)+'.pt')

        return FitResult(actual_num_epochs,
                         train_loss, test_loss)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train()  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.eval()  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, index, batch_data) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """

        X, y = batch_data
        if self.tensorboard_logger and self.tensorboard_log_images:
            B = torch.zeros_like(X.squeeze())
            C = torch.stack([B, X.squeeze(), X.squeeze()])
            C = C.unsqueeze(dim=0)
            images = C
            grid = make_grid(images, normalize=True, scale_each=True)
            self.tensorboard_logger.add_image("exp-%s/batch/test/images" % self.experiment_prefix, grid, index)
        if isinstance(X, tuple) or isinstance(X, list):
            X = [x.to(self.device) for x in X]
        else:
            X = X.to(self.device)
        y = y.to(self.device)
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.tensorboard_logger:
            self.tensorboard_logger.add_scalar('exp-%s/batch/train/loss' % self.experiment_prefix, loss, index)
            if index % 300 == 0:
                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.tensorboard_logger.add_histogram('exp-%s/batch/train/param/%s' % (self.experiment_prefix, tag), to_np(value), index)
                    self.tensorboard_logger.add_histogram('exp-%s/batch/train/param/%s/grad' % (self.experiment_prefix, tag), to_np(value.grad), index)

        return BatchResult(loss)

    def test_batch(self, index, batch_data) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        with torch.no_grad():
            X, y = batch_data
            if isinstance(X, tuple) or isinstance(X, list):
                X = [x.to(self.device) for x in X]
            else:
                X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            if self.tensorboard_logger:
                self.tensorboard_logger.add_scalar('exp-%s/batch/test/loss' % self.experiment_prefix, loss, index)
            return BatchResult(loss)

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            overall_loss = avg_loss = counter = 0
            min_loss = 1
            max_loss = 0
            for batch_idx in range(num_batches):
                counter += 1
                data = next(dl_iter)
                batch_res = forward_fn(batch_idx, data)
                if batch_res.loss > max_loss:
                    max_loss = batch_res.loss
                if batch_res.loss < min_loss:
                    min_loss = batch_res.loss
                overall_loss += batch_res.loss
                losses.append(batch_res.loss)

                avg_loss = overall_loss / counter
                pbar.set_description(f'{pbar_name} (Avg. loss:{avg_loss:.3f})')
                pbar.update()
            
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, Min {min_loss:.3f}, Max {max_loss:.3f})')
                    
        return EpochResult(losses=losses)
