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


class MInterface(pl.LightningModule):
    def __init__(self,  **kargs):
        super().__init__()
        # self.load_path = load_path
        # self.d_model = d_model
        # self.dim_feedforward = dim_feedforward
        # self.num_encoder_layers = num_encoder_layers
        # self.nhead = nhead
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.distance_A_2_B = []
        self.distance_B_2_A = []
        

    def forward(self, stft, image):
        # self.device = stft.device
        # pdb.set_trace()
        # self.model.to(self.device)
        return self.model(stft, image)

    def training_step(self, batch, batch_idx):
        image, stft, gt, active_wearer, speaker_id_matrix = batch
        # image, stft, gt, active_wearer, speaker_id_matrix = image.half(), stft.half(), gt.half(), active_wearer.half(), speaker_id_matrix.half()
        # image, stft, gt, active_wearer, speaker_id_matrix = image.float(), stft.float(), gt.float(), active_wearer.float(), speaker_id_matrix.float()

        out = self(stft, image)
        # pdb.set_trace()
        gt, active_wearer = gt.long(), active_wearer.long()
        # gt = gt.float()
        # out = out.double()
        if self.hparams.weight_ce:
            loss_global = self.loss_function(out.squeeze(), gt, weight=torch.tensor([1.0, 100.0], device=out.device, dtype=torch.float64))
            # loss_global = self.loss_function(out.squeeze(), gt)
        else:
            loss_global = self.loss_function(out, gt)

        # loss_local = self.loss_function(wearer, active_wearer)
        # loss_local = 0
        loss = loss_global
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
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
        image, stft, gt, active_wearer, speaker_id_matrix = batch
        # image, stft, gt, active_wearer, speaker_id_matrix = image.half(), stft.half(), gt.half(), active_wearer.half(), speaker_id_matrix.half()
        # image, stft, gt, active_wearer, speaker_id_matrix = image.float(), stft.float(), gt.float(), active_wearer.float(), speaker_id_matrix.float()

        out = self(stft, image)
        
        gt, active_wearer = gt.long(), active_wearer.long()
        # gt = gt.float()
        
        # pdb.set_trace()
        # out = out.double()
        if self.hparams.weight_ce:
            loss_global = self.loss_function(out, gt, weight=torch.tensor([1.0, 100.0], device=out.device, dtype=torch.float64))
            # pdb.set_trace()
            # loss_global = self.loss_function(out.squeeze(), gt)
        else:
            loss_global = self.loss_function(out, gt)
        
        loss = loss_global
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('test_global_loss', loss_global, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('test_local_loss', loss_local, on_step=True, on_epoch=True, prog_bar=True)
        

        mean_distance_A_to_B_all, std_distance_A_to_B_all, mean_distance_B_to_A_all, std_distance_B_to_A_all, invalid = \
        self.evaluate(out, speaker_id_matrix)

        self.log('invalid_prediction', invalid, on_step=True, on_epoch=True, prog_bar=True)

        if not (mean_distance_A_to_B_all == None or std_distance_A_to_B_all == None or mean_distance_B_to_A_all == None or \
        std_distance_B_to_A_all == None):

            self.log('mean_distance_A_to_B_all', mean_distance_A_to_B_all, on_step=True, on_epoch=True, prog_bar=True)
            self.log('std_distance_A_to_B_all', std_distance_A_to_B_all, on_step=True, on_epoch=True, prog_bar=True)
            self.log('mean_distance_B_to_A_all', mean_distance_B_to_A_all, on_step=True, on_epoch=True, prog_bar=True)
            self.log('std_distance_B_to_A_all', std_distance_B_to_A_all, on_step=True, on_epoch=True, prog_bar=True)

        return {'mean_distance_A_to_B_all': mean_distance_A_to_B_all, 'std_distance_A_to_B_all': std_distance_A_to_B_all,
                'mean_distance_B_to_A_all': mean_distance_B_to_A_all, 'std_distance_B_to_A_all': std_distance_B_to_A_all,
                'test_loss': loss, 'invalid_prediction': invalid}

    def shared_epoch_end(self, outputs):
        # mean_distance_A_to_B_all = np.stack([x['mean_distance_A_to_B_all'] for x in outputs]).mean()
        # std_distance_A_to_B_all = np.stack([x['std_distance_A_to_B_all'] for x in outputs]).mean()
        # mean_distance_B_to_A_all = np.stack([x['mean_distance_B_to_A_all'] for x in outputs]).mean()
        # std_distance_B_to_A_all = np.stack([x['std_distance_B_to_A_all'] for x in outputs]).mean()
        invalid = np.stack([x['invalid_prediction'] for x in outputs]).mean()

        

        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        
        # test_local_loss = torch.stack([x["test_local_loss"] for x in outputs]).mean()
        
        # if np.isnan(std_distance_B_to_A_all):
        #     pdb.set_trace()
        if not len(self.distance_A_2_B) == 0:
            mean_distance_A_to_B_over_dataset = np.mean(np.array(self.distance_A_2_B))
            std_distance_A_to_B_over_dataset = np.std(np.array(self.distance_A_2_B))
            self.log('mean_distance_A_to_B_over_dataset', mean_distance_A_to_B_over_dataset, on_epoch=True, sync_dist=True)
            self.log('std_distance_A_to_B_over_dataset', std_distance_A_to_B_over_dataset, on_epoch=True, sync_dist=True)
        if not len(self.distance_B_2_A) == 0:
            mean_distance_B_to_A_over_dataset = np.mean(np.array(self.distance_B_2_A))
            std_distance_B_to_A_over_dataset = np.std(np.array(self.distance_B_2_A))
            self.log('mean_distance_B_to_A_over_dataset', mean_distance_B_to_A_over_dataset, on_epoch=True, sync_dist=True)
            self.log('std_distance_B_to_A_over_dataset', std_distance_B_to_A_over_dataset, on_epoch=True, sync_dist=True)

        self.distance_A_2_B = []
        self.distance_B_2_A = []

        mean_distance_A_to_B_all = np.stack([x['mean_distance_A_to_B_all'] for x in outputs])
        std_distance_A_to_B_all = np.stack([x['std_distance_A_to_B_all'] for x in outputs])
        mean_distance_B_to_A_all = np.stack([x['mean_distance_B_to_A_all'] for x in outputs])
        std_distance_B_to_A_all = np.stack([x['std_distance_B_to_A_all'] for x in outputs])

        if not (len(mean_distance_A_to_B_all[mean_distance_A_to_B_all != None]) == 0 
                or len(std_distance_A_to_B_all[std_distance_A_to_B_all != None]) == 0
                  or len(mean_distance_B_to_A_all[mean_distance_B_to_A_all != None]) == 0
                    or len(std_distance_B_to_A_all[std_distance_B_to_A_all != None]) == 0):
            
            mean_distance_A_to_B_all = np.mean(mean_distance_A_to_B_all[mean_distance_A_to_B_all != None])

            
            std_distance_A_to_B_all = np.mean(std_distance_A_to_B_all[std_distance_A_to_B_all != None])

            
            mean_distance_B_to_A_all = np.mean(mean_distance_B_to_A_all[mean_distance_B_to_A_all != None])

            
            std_distance_B_to_A_all = np.mean(std_distance_B_to_A_all[std_distance_B_to_A_all != None])

            self.log("mean_distance_A_to_B_all_epoch", mean_distance_A_to_B_all, on_epoch=True, sync_dist=True)
            self.log("std_distance_A_to_B_all_epoch", std_distance_A_to_B_all, on_epoch=True, sync_dist=True)
            self.log("mean_distance_B_to_A_all_epoch", mean_distance_B_to_A_all, on_epoch=True, sync_dist=True)
            self.log("std_distance_B_to_A_all_epoch", std_distance_B_to_A_all, on_epoch=True, sync_dist=True)

        self.log('test_loss_epoch', test_loss, on_epoch=True, sync_dist=True)
        
        self.log('invalid_prediction_epoch', invalid, on_epoch=True, sync_dist=True)

            
        # self.log('test_local_loss_epoch', test_local_loss, on_epoch=True)


    def evaluate(self, out, speaker_id_matrix):
        batch_size = out.shape[0]
        out, speaker_id_matrix = out.cpu().numpy(), speaker_id_matrix.cpu().numpy()
        cnt = 0
        invalid = 0.
        mean_distance_A_to_B_all, std_distance_A_to_B_all, mean_distance_B_to_A_all, std_distance_B_to_A_all = 0., 0., 0., 0.
        for i in range(batch_size):
            selected_points = modified_nms(out[i, :, :, :], threshold_distance=6, peak_confidence=0)
            speaker_id = np.argwhere(speaker_id_matrix[i, :, :] == 1)
            # pdb.set_trace()
            if len(speaker_id) != 0 and len(selected_points) == 0:
                invalid += len(speaker_id)
            
            # pdb.set_trace()
            
            min_distance_A_to_B, mean_distance_A_to_B, std_distance_A_to_B, \
            min_distance_B_to_A, mean_distance_B_to_A, std_distance_B_to_A = compute_distances(selected_points, speaker_id)

            # pdb.set_trace()
            if np.isnan(min_distance_A_to_B) or np.isnan(mean_distance_A_to_B) or np.isnan(std_distance_A_to_B) or\
            np.isnan(min_distance_B_to_A) or np.isnan(mean_distance_B_to_A) or np.isnan(std_distance_B_to_A):
                continue
            
            self.distance_A_2_B.append(mean_distance_A_to_B)
            self.distance_B_2_A.append(mean_distance_B_to_A)

            mean_distance_A_to_B_all += mean_distance_A_to_B
            std_distance_A_to_B_all += std_distance_A_to_B
            mean_distance_B_to_A_all += mean_distance_B_to_A
            std_distance_B_to_A_all += std_distance_B_to_A
            cnt += 1
        # pdb.set_trace()
        if cnt != 0:

            mean_distance_A_to_B_all /= cnt
            std_distance_A_to_B_all /= cnt
            mean_distance_B_to_A_all /= cnt
            std_distance_B_to_A_all /= cnt
        
        else:
            mean_distance_A_to_B_all, std_distance_A_to_B_all, mean_distance_B_to_A_all, std_distance_B_to_A_all \
            = None, None, None, None
                # = np.float64(-1), np.float64(-1), np.float64(-1), np.float64(-1)
                

        return mean_distance_A_to_B_all, std_distance_A_to_B_all, mean_distance_B_to_A_all, std_distance_B_to_A_all, invalid

        # return torch.tensor(torch.from_numpy(mean_distance_A_to_B_all), dtype=torch.float64), \
        #     torch.tensor(torch.from_numpy(std_distance_A_to_B_all), dtype=torch.float64), \
        #     torch.tensor(torch.from_numpy(mean_distance_B_to_A_all), dtype=torch.float64), \
        #     torch.tensor(torch.from_numpy(std_distance_B_to_A_all), dtype=torch.float64)


    # def on_validation_epoch_end(self):
    #     # Make the Progress Bar leave there
    #     self.print('')

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
            self.loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0]))
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
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
