import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from model.common import *
from model.transformer import *
from model.position_encoding import *
import pdb
from transformers import AutoImageProcessor, ViTModel,  ViTConfig
from transformer.Models import PositionalEncoding

import re

vit_version = 'WinKawaks/vit-tiny-patch16-224'

class AudioTransformerWearer(nn.Module):

    def __init__(self, d_model=192, freeze_vit=False, gccphat=False,
                 ):
        super().__init__()

    
        self.position_enc = PositionalEncoding(d_model, n_position=185)
        self.gccphat = gccphat
        if gccphat:
            self.audio_split = nn.Sequential(
                nn.Conv2d(15, d_model, kernel_size=(8,6), stride=(8,6)),
                nn.BatchNorm2d(d_model),
            )
        else:
            self.audio_split = nn.Sequential(
                nn.Conv2d(12, 3, kernel_size=1, stride=1),
                nn.BatchNorm2d(3),
            )
            
     
        self.config = ViTConfig(
            image_size=224,
            patches_size=16,
            num_channels=3,
            hidden_size=192,
            num_hidden_layers=6, 
            num_attention_heads=3,
            intermediate_size=768,
        )

      
        self.audio_model = ViTModel(self.config)
        if freeze_vit:
            # Freeze the gradients of the image model
            for param in self.audio_split.parameters():
                param.requires_grad = False

            # Freeze the gradients of the audio model
            for param in self.audio_model.parameters():
                param.requires_grad = False
        
        self.audio_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
     

        self.classifier = nn.Linear(d_model, 2)
       
        self.dropout = nn.Dropout(p=0.1)
        self._reset_parameters()

     

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def load_pretrained_weights(self, path):
        model_ckpt = torch.load(path)
        weights = model_ckpt['state_dict']
        # remove `module.` prefix but not in the middle of the string
        weights = {re.sub(r'^model\.', '', k): v for k, v in weights.items()}
        model_dict = self.state_dict()
        overlapping_weights = {k: v for k, v in weights.items() if k in model_dict}
        model_dict.update(overlapping_weights)
        # pdb.set_trace()
        # check how many weights are loaded
        print("Loaded {}/{} parameters from checkpoint".format(len(overlapping_weights), len(model_dict)))
        # show the weights that failed to load
        missed_weights = [k for k, v in model_dict.items() if k not in weights]
        print("Missed weights: {}".format(missed_weights))
        
        self.load_state_dict(model_dict)
        print("Loaded pretrained weights from {}".format(path))


    
    def forward(self, stft, image):

       
        if self.gccphat:
            split_audio = self.audio_split(stft)
            split_audio = split_audio.reshape(split_audio.shape[0], split_audio.shape[1], split_audio.shape[2]*split_audio.shape[3])
            split_audio = split_audio.permute(0, 2, 1)
            audio_cls_tokens = self.audio_cls_token.expand(stft.shape[0], -1, -1).to(stft.device)
            split_audio = torch.cat((audio_cls_tokens, split_audio), dim=1)
            split_audio = self.dropout(self.position_enc(split_audio))
            features = self.audio_model.encoder(split_audio) ## (b, 197, 768)
            features = self.audio_model.layernorm(features.last_hidden_state)
            
            last_hidden_states_audio = features
            # features = self.audio_model.pooler(features)
        else:
            split_audio = self.audio_split(stft)


            features = self.audio_model(split_audio).last_hidden_state ## (b, 197, 768)
            
            
            last_hidden_states_audio = self.audio_model(split_audio).last_hidden_state ## (b, 197, 768)
       
      
        act = self.classifier(last_hidden_states_audio[:, 0, :])

        return act