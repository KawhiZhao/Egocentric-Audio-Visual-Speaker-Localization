# Copyright 2021 Zhongyang Zhang 
# Contact: mirakuruyoo@gmai.com
# Modifications Copyright 2023 Jinzheng Zhao
# Contact: zhaojinzheng@bupt.cn
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import List, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from model.common import *
import pytorch_lightning as pl
import pdb
import numpy as np


class MInterfaceAVST(pl.LightningModule):
    def __init__(self,  **kargs):
        super().__init__()

        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.batch_size = kargs['batch_size']
        

    def forward(self, gcc, visual, bbox):
        # pdb.set_trace()
        # self.model.to(self.device)
        return self.model(gcc, visual, bbox)

    def training_step(self, batch, batch_idx):
        visual, gcc, gt, r, bbox, chunk_path = batch

      

        # image, stft, gt, active_wearer, speaker_id_matrix = image.half(), stft.half(), gt.half(), active_wearer.half(), speaker_id_matrix.half()
        # image, stft, gt, active_wearer, speaker_id_matrix = image.float(), stft.float(), gt.float(), active_wearer.float(), speaker_id_matrix.float()

        out = self(gcc, visual, bbox)
        # pdb.set_trace()
        gt = gt.long()
        gt = gt.squeeze()
        # out = out.double()
        if self.hparams.weight_ce:
            loss_global = self.loss_function(out, gt, weight=torch.tensor([1.0, 100.0], device=out.device, dtype=torch.float64))
        else:
            loss_global = self.loss_function(out, gt)

        # loss_local = self.loss_function(wearer, active_wearer)
        # loss_local = 0
        loss = loss_global
        location = torch.argmax(out, dim=1)
        gt = gt.double()
        ae_doa = torch.abs(location - gt)
        # pdb.set_trace()
        # find out the index in ae_doa whose value is larger than 180 and use 360 - value
        ae_doa[ae_doa > 180] = 360 - ae_doa[ae_doa > 180]
        ae_doa = ae_doa.mean()

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('ae_doa', ae_doa, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
   
        # self.log('global_loss', loss_global, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('local_loss', loss_local, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        return self.shared_step(batch, batch_idx)
    
    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs)
    
    def test_step(self, batch, batch_idx):

        return self.shared_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs)

    def shared_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing

        visual, gcc, gt, r, bbox, chunk_path = batch
        batch_size = visual.shape[0]
        #convert to tensor
        # bbox = bbox[0]
        # image, stft, gt, active_wearer, speaker_id_matrix = image.half(), stft.half(), gt.half(), active_wearer.half(), speaker_id_matrix.half()
        # image, stft, gt, active_wearer, speaker_id_matrix = image.float(), stft.float(), gt.float(), active_wearer.float(), speaker_id_matrix.float()

        # pdb.set_trace()
        out = self(gcc, visual, bbox)
        
        gt = gt.long()
        gt = gt.squeeze()
        # out = out.double()
        # pdb.set_trace()
        if self.hparams.weight_ce:
            loss_global = self.loss_function(out, gt, weight=torch.tensor([1.0, 100.0], device=out.device, dtype=torch.float64))
        else:
            loss_global = self.loss_function(out, gt)
        
        loss = loss_global
        location = torch.argmax(out, dim=1)
        # convert long to double
        gt = gt.double()
        ae_doa = torch.abs(location - gt)
        # find out the index in ae_doa whose value is larger than 180 and use 360 - value
        ae_doa[ae_doa > 180] = 360 - ae_doa[ae_doa > 180]
        ae_doa = ae_doa.mean()

        # self.log('batch_size', batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('test_ae_doa', ae_doa, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        # self.log('test_global_loss', loss_global, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('test_local_loss', loss_local, on_step=True, on_epoch=True, prog_bar=True)
        


        return {'test_loss': loss, 'test_ae_doa': ae_doa, 'batch_size': batch_size}

    def shared_epoch_end(self, outputs):
        

        

        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_ae_doa = torch.stack([x["test_ae_doa"] for x in outputs]).mean()
  

        self.log('test_loss_epoch', test_loss, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log('test_ae_doa_epoch', test_ae_doa, on_epoch=True, sync_dist=True, batch_size=self.batch_size)


    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
        elif loss == 'emd':
            self.loss_function = EMD_loss(sigma=3)
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)
        # if self.load_path != '':
        #     checkpoint = torch.load(self.load_path)
        #     new_state_dict = {}
        #     for key, value in checkpoint['state_dict'].items():
        #         new_state_dict[key[6:]] = value
        #     self.model.load_state_dict(new_state_dict)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
    
import math
class EMD_loss(nn.Module):
    def __init__(self, sigma) -> None:
        super().__init__()
        self.sigma = sigma
    
    def forward(self, y_pred, y_labels):

        x = np.arange(0, 360, 1)
        batch_size = y_labels.shape[0]
        y_labels = np.array(y_labels.cpu())
        y_dist = np.ones((batch_size, 360))
        for i in range(batch_size):
            y_dist[i] = np.exp(-(x - y_labels[i]) ** 2 /(2* self.sigma **2))/(math.sqrt(2*math.pi)*self.sigma)
        y_dist = torch.Tensor(y_dist).to(y_pred.device)

        return torch.mean(torch.sum(torch.square(y_dist - y_pred), dim=1), dim=0)