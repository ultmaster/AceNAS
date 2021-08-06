import logging
import os
from datetime import datetime

import torch
import torch.distributed as dist
from apex.parallel import DistributedDataParallel
from mmcv.utils.logging import print_log
from torch.utils.tensorboard import SummaryWriter

from common.logging import AverageMeterGroup, AzureMLWriter

VERSION = 1


class Trainer:
    def __init__(self, model,
                 warmup_epochs=0,
                 val_every_n_epoch=1,
                 test_every_n_epoch=10,
                 save_ckpt_every_n_epoch=1,
                 fast_dev_run=False,
                 console_log_interval=50,
                 evaluate_only=False,
                 resume_from=None,
                 label_this_run=False,
                 tb_log_interval=10,
                 tb_log_dir=None,
                 checkpoint_dir=None,
                 num_epochs=50):
        self.warmup_epochs = warmup_epochs
        self.val_every_n_epoch = val_every_n_epoch
        self.test_every_n_epoch = test_every_n_epoch
        self.save_ckpt_every_n_epoch = save_ckpt_every_n_epoch
        self.fast_dev_run = fast_dev_run
        self.console_log_interval = console_log_interval
        self.evaluate_only = evaluate_only
        self.resume_from = resume_from
        self.num_epochs = num_epochs
        self.tb_log_interval = tb_log_interval
        self.tb_log_dir = tb_log_dir
        self.checkpoint_dir = checkpoint_dir
        self.label_this_run = label_this_run
        self.run_id = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")

        self.distributed = dist.is_available() and dist.is_initialized()

        if self.label_this_run:
            self.tb_log_dir = os.path.join(self.tb_log_dir, self.run_id)
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.run_id)

        if self.tb_log_dir is None:
            self.tb_logger = None
        else:
            self.tb_logger = SummaryWriter(log_dir=self.tb_log_dir)
        self.aml_writer = AzureMLWriter()

        self.model = model
        self.model.cuda()
        if self.distributed:
            self.model = DistributedDataParallel(self.model, delay_allreduce=True)

    def configure_optimizers(self):
        raise NotImplementedError

    def train_dataloader(self, current_epoch):
        raise NotImplementedError

    def val_dataloader(self, current_epoch):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    @property
    def model_inner(self):
        if self.distributed:
            return self.model.module
        else:
            return self.model

    def fit(self):
        self._optimizer, self._scheduler = self.configure_optimizers()

        self.current_epoch = 0
        self.training = False

        resume_epoch = -1
        # recover from latest checkpoint
        latest_ckpt_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        if self.resume_from is not None:
            self.load_checkpoint(self.resume_from)
        elif os.path.exists(latest_ckpt_path):
            self.load_checkpoint(latest_ckpt_path)
            resume_epoch = self.current_epoch
            print_log(f'Found checkpoint "{latest_ckpt_path}", resume from epoch {resume_epoch + 1}.',
                      __name__)

        if self.evaluate_only:
            self.model.eval()
            self.test(0)
            return

        for current_epoch in range(resume_epoch + 1, self.num_epochs):
            self.current_epoch = current_epoch
            self.training = True
            self.model.train()
            self.train(current_epoch)
            self.training = False

            ckpt_should_save = (current_epoch + 1) % self.val_every_n_epoch == 0 or \
                (current_epoch + 1) % self.test_every_n_epoch == 0 or \
                current_epoch + 1 == self.num_epochs
            if ckpt_should_save:
                # save checkpoint for later examination
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f'epoch_{current_epoch + 1:03d}.pt'))

            if current_epoch >= self.warmup_epochs and (current_epoch + 1) % self.val_every_n_epoch == 0:
                self.model.eval()
                self.validate(current_epoch)
            if (current_epoch + 1) % self.test_every_n_epoch == 0 or current_epoch + 1 == self.num_epochs:
                self.model.eval()
                self.test(current_epoch)
            if (current_epoch + 1) % self.save_ckpt_every_n_epoch == 0 or \
                    current_epoch + 1 == self.num_epochs:
                self.save_checkpoint(latest_ckpt_path)

    def print_console_log(self, msg, current_epoch, phase=None):
        if phase is None:
            phase = 'train' if self.training else 'val'
        print_log(f'{phase.capitalize()} Epoch [{current_epoch + 1}/{self.num_epochs}] {msg}', __name__)

    def log_batch_metrics(self, metrics, avg_meters, current_epoch, batch_idx, batch_num, phase):
        if batch_idx + 1 == batch_num or \
                batch_idx % self.console_log_interval == 0:
            self.print_console_log(f'Step [{batch_idx + 1}/{batch_num}]' +
                                   f'  {avg_meters}', current_epoch, phase)
        if self.tb_logger is not None and (
                batch_idx + 1 == batch_num or
                batch_idx % self.tb_log_interval == 0):
            for name, val in metrics.items():
                self.tb_logger.add_scalar(
                    f'{phase}/{name}', val,
                    global_step=current_epoch * batch_num + batch_idx)
        if batch_idx + 1 == batch_num:
            self.aml_writer.add_average_meter(phase, avg_meters)

    def train(self, current_epoch):
        train_dataloader = self.train_dataloader(current_epoch)
        training_meters = AverageMeterGroup()
        self.model.train()
        self.on_training_epoch_start(current_epoch)
        for batch_idx, batch in enumerate(train_dataloader):
            if self.fast_dev_run and batch_idx >= 2:
                break
            loss, metrics = self.training_step(batch, batch_idx)
            loss.backward()
            self.optimizer_step(self._optimizer)
            self._optimizer.zero_grad()
            training_meters.update({**metrics, 'lr': self._optimizer.param_groups[0]['lr'],
                                    'loss': loss.item()})
            self.log_batch_metrics(metrics, training_meters, current_epoch,
                                   batch_idx, len(train_dataloader), 'train')
            self._scheduler.step()
        self.on_training_epoch_end(current_epoch)

    def validate(self, current_epoch):
        val_dataloader = self.val_dataloader(current_epoch)
        validation_meters = AverageMeterGroup()
        self.on_validation_epoch_start(current_epoch)
        predictions = []
        for batch_idx, batch in enumerate(val_dataloader):
            if self.fast_dev_run and batch_idx >= 2:
                break
            prediction, metrics = self.validation_step(batch, batch_idx)
            predictions.append(prediction)
            validation_meters.update(metrics)
            self.log_batch_metrics(metrics, validation_meters, current_epoch,
                                   batch_idx, len(val_dataloader), 'val')
        self.on_validation_epoch_end(predictions, current_epoch)

    def test(self, current_epoch):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        if 'state_dict' in state_dict:
            print_log('A key named "state_dict" is found in state dict. Treat it as model.', __name__,
                      level=logging.WARNING)
            state_dict['model'] = state_dict.pop('state_dict')
        if 'model' not in state_dict:
            print_log('"model" not found in state dict. Automatically treat the whole dict as model.',
                      __name__, level=logging.WARNING)
            state_dict = {'model': state_dict}
        missing, unexpected = self.model_inner.load_state_dict(state_dict['model'], strict=strict)
        if missing or unexpected:
            print_log('Missing keys or unexpected keys: %s, %s.' % (missing, unexpected), __name__,
                      level=logging.WARNING)

        def try_to_load(load_fn, key):
            if key in state_dict:
                load_fn(state_dict[key])
            else:
                print_log(f'Key "{key}" not found in checkpoint. Skipped.', __name__, level=logging.WARNING)

        try_to_load(self._optimizer.load_state_dict, 'optimizer')
        try_to_load(self._scheduler.load_state_dict, 'scheduler')

        def assign_epoch(epoch):
            self.current_epoch = epoch
        try_to_load(assign_epoch, 'epoch')

    def state_dict(self):
        model_data = self.model_inner.state_dict()
        return {
            'model': model_data,
            'optimizer': self._optimizer.state_dict(),
            'scheduler': self._scheduler.state_dict(),
            'epoch': self.current_epoch,
            'version': VERSION
        }

    def load_checkpoint(self, f, strict=True):
        data = torch.load(f, map_location='cuda')
        self.load_state_dict(data, strict=strict)

    def save_checkpoint(self, f, **kwargs):
        if self.distributed and torch.distributed.get_rank() != 0:
            return
        torch.save(self.state_dict(), f)

    def optimizer_step(self, optimizer):
        optimizer.step()

    def on_training_epoch_start(self, current_epoch):
        pass

    def on_training_epoch_end(self, current_epoch):
        pass

    def on_validation_epoch_start(self, current_epoch):
        pass

    def on_validation_epoch_end(self, predictions, current_epoch):
        pass

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
