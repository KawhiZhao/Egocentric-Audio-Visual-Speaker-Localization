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

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers import NeptuneLogger
import torch
from pytorch_lightning.loggers import WandbLogger
import pdb
from model import MInterface, MInterfaceWearer, MInterfaceAVST
from data import DInterface, DInterfaceAVST
from utils import load_model_path_by_args


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='test_loss',
        mode='min',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='test_loss',
        filename='best-{epoch:02d}-{test_loss:.3f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        every_n_epochs=1,
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(args.seed)
    load_path = args.load_path
   
    data_module = DInterfaceAVST(**vars(args))

    # model = MInterfaceWearer(**vars(args))
    model = MInterfaceAVST(**vars(args))
   



    args.callbacks = load_callbacks()
  
  
    if args.load_path is not None:
        trainer = Trainer.from_argparse_args(args, resume_from_checkpoint=load_path)
    else:
        trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, data_module)

    # test code 
    # checkpoint = torch.load(args.load_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # trainer.validate(model, data_module)
    if not args.ddp:
        trainer.test(model, data_module)
    else:
        trainer = Trainer(devices=1, num_nodes=1)
        trainer.test(model, data_module)




if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
 

    parser.add_argument('--load_path', default=None, type=str)
 


    # Training Info
    parser.add_argument('--dataset', default='avst_chunk', type=str)
    parser.add_argument('--data_dir', default='ref/data', type=str)
 
    parser.add_argument('--model_name', default='a_v_transformer', type=str)
 
    parser.add_argument('--loss', default='emd', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    

    parser.add_argument('--weight_ce', default=False, type=bool)
    parser.add_argument('--reshape_feature', default=True, type=bool)
    parser.add_argument('--freeze_vit', default=False, type=bool)

    parser.add_argument('--gccphat', default=True, type=bool)

    parser.add_argument('--ddp', default=False, type=bool)
    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)


    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(
        max_epochs=30, 
        accelerator='gpu', 
        devices=1, 
        precision='64', 
        # val_check_interval = 10,
        # fast_dev_run=50,
        logger=False,
        limit_val_batches=1.0,
        check_val_every_n_epoch=1,
        )

    args = parser.parse_args()

    if args.ddp:
        args.strategy = 'ddp'
        args.gpus = 2
        # args.accelerator = 'gpu'
        args.num_nodes = 1
   
    main(args)
