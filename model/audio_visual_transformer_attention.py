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

# vit_version = 'google/vit-base-patch16-224-in21k'
vit_version = 'WinKawaks/vit-tiny-patch16-224'

class AudioVisualTransformerAttention(nn.Module):

    def __init__(self, d_model=192, nhead=4, freeze_vit=False, gccphat=False):
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

        self.image_processor = AutoImageProcessor.from_pretrained(vit_version)
   
        self.image_model = ViTModel(self.config)

   
        self.audio_model = ViTModel(self.config)
      
        if freeze_vit:
            # Freeze the gradients of the image model
            for param in self.image_model.parameters():
                param.requires_grad = False

            # Freeze the gradients of the audio model
            for param in self.audio_model.parameters():
                param.requires_grad = False

        
        
        self.audio_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # self.visual_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.attention = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_model, d_v=d_model)



        self.fc1 = nn.Linear(d_model*2, 4050)
        self.fc2 = nn.Linear(d_model*2, 4050)
        self.Upsample = nn.Upsample(scale_factor=2)
        self.dropout = nn.Dropout(p=0.1)
        self._reset_parameters()
       
        # self.load_pretrained_weights()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def load_pretrained_weights(self, path=''):
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
        # print weights that load
        # print("Loaded weights: {}".format(overlapping_weights.keys()))
        missed_weights = [k for k, v in model_dict.items() if k not in weights]
        print("Missed weights: {}".format(missed_weights))
        print("Loaded weights: {}".format(overlapping_weights.keys()))
        self.load_state_dict(model_dict)
        print("Loaded pretrained weights from {}".format(path))
    
    def forward(self, stft, image):

        image = self.image_processor(image, return_tensors="pt").to(stft.device)
        last_hidden_states_image = self.image_model(**image).last_hidden_state ## (b, 197, 768)


     
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
            # pdb.set_trace()
            
            last_hidden_states_audio = self.audio_model(split_audio).last_hidden_state ## (b, 197, 768)
        # pdb.set_trace()
        
        
        seq = torch.cat((last_hidden_states_audio[:, 0, :].unsqueeze(1), last_hidden_states_image[:, 0, :].unsqueeze(1)), dim=1)
        # pdb.set_trace()
        
        output= self.attention(seq, seq, seq, None)
        # # pdb.set_trace()
        output = torch.cat((output[:, 0, :], output[:, 1, :]), -1)
        
 
     
        o1 = self.fc1(output)
        o1 = o1.reshape(-1, 45, 90)
        o2 = self.fc2(output)
        o2 = o2.reshape(-1, 45, 90)
        o = torch.stack([o1, o2], dim=1)
        o = self.Upsample(o)
        return o