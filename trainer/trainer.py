import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, Mixup
from torch.amp import autocast, GradScaler
import csv

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 train_data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.train_data_loader = train_data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch

        if config['csv']:
            csv_path = self.checkpoint_dir + '/log.csv'
            self.csv_logger = csv.writer(open(csv_path, 'w'))
            self.csv_logger.writerow(['batch_idx', 'loss', 'acc', 'lr'])

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        # Mixed Precision Setup
        if self.config["mix_precision"]:
            self.scaler = GradScaler()

        if self.config['mixup']:
            self.mixup = Mixup()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Apply MixUp if enabled
            if self.config["mixup"]:
                old_target = target
                data, target = self.mixup(data, target)

            self.optimizer.zero_grad()

            # Use mixed precision if enabled
            if self.config["mix_precision"]:
                with autocast(device_type=self.device.type):  # Automatically uses FP16 where possible
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

            # Backward pass with gradient scaling if using mixed precision
            if self.config["mix_precision"]:
                self.scaler.scale(loss).backward()  # Scales the loss for stable gradients
                if self.config["gradient_clipping"]:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clipping"])
                self.scaler.step(self.optimizer)
                self.scaler.update()  # Updates the scale for next iteration
            else:
                loss.backward()
                if self.config["gradient_clipping"]:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clipping"])
                self.optimizer.step()

            if self.config["mixup"]:
                target = old_target

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                result = met(output, target)
                if (self.config['csv']):
                    self.csv_logger.writerow([(epoch - 1) * self.len_epoch + batch_idx, 
                                              loss.item(), 
                                              result, 
                                              self.optimizer.param_groups[0]['lr']])
                    # Flush the CSV file to ensure data is written to disk
                    if hasattr(self.csv_logger, 'file'):
                        self.csv_logger.file.flush()
                self.train_metrics.update(met.__name__, result)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            log.update({'lr': self.optimizer.param_groups[0]['lr']})
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # Add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
