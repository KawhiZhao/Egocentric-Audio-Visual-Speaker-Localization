import pdb
import torch
from torch import nn


import numpy as np

from transformers import AutoImageProcessor, ViTModel,  ViTConfig
from transformer.Models import PositionalEncoding

vit_version = 'WinKawaks/vit-tiny-patch16-224'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from model.common import *


class AVTransformer(nn.Module):
    def __init__(self, d_model=128, n_head=4, in_audio=224, d_img_patches=16, dropout=0.1):
        super().__init__()
        
        self.gcc_emb = nn.Linear(in_audio, d_model)
        self.img_split = nn.Conv2d(3, d_model, kernel_size=(d_img_patches, d_img_patches), stride=(d_img_patches, d_img_patches))
        self.audio_position_enc = PositionalEncoding(d_model, n_position=97)
        self.visual_position_enc = PositionalEncoding(d_model, n_position=197)
      
        self.dropout = nn.Dropout(p=dropout)
        self.audio_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.config = ViTConfig(
            image_size=224,
            patches_size=16,
            num_channels=3,
            hidden_size=128,
            num_hidden_layers=2, 
            num_attention_heads=4,
            intermediate_size=256,
        )
       
        self.image_model = ViTModel(self.config)

       
        self.audio_model = ViTModel(self.config)
        self.audio_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.visual_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attention = MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_model, d_v=d_model)

        self.output_fc = nn.Linear(d_model * 2, 360)
        self.output_fc_audio = nn.Linear(d_model, 360)
   
        self.softmax = nn.Softmax(dim=-1)
    
   

        self._reset_parameters()
        # pdb.set_trace()
        self.d_model = d_model


    def AudioVisualAttention(self, audio_embedding, visual_embedding):
        seq = torch.cat((audio_embedding[:,0,:].unsqueeze(1),  visual_embedding[:,0,:].unsqueeze(1)), dim=1)
        
        output= self.attention(seq, seq, seq, None)
        # pdb.set_trace()
        feature = output[:, 0, :]
        output = torch.cat((output[:, 0, :], output[:, 1, :]), -1)
        output = torch.squeeze(output)
        doa = self.softmax(self.output_fc(output))
     
        return doa, feature
    
    def forward(self, audio, img, bbox):
        batch_size = audio.shape[0]
        av_index = []
        a_index = []
        fov = bbox[-1]
        for i in range(batch_size):
            if fov[i].item():
                av_index.append(i)
            else:
                a_index.append(i)
        gccphat = audio.permute(0, 2, 1)
        audio_emb = self.gcc_emb(gccphat)
        # pdb.set_trace()
        audio_cls_tokens = self.audio_cls_token.expand(audio.shape[0], -1, -1).to(audio.device)
        audio_co_embeds = torch.cat((audio_cls_tokens, audio_emb), dim=1)
        audio_co_embeds = self.audio_layer_norm(self.dropout(self.audio_position_enc(audio_co_embeds)))

        img_patches = self.img_split(img)
        batch_size, c, h, w = img_patches.shape
        img_emb = img_patches.reshape(batch_size, c, (h * w))
        img_emb = img_emb.permute(0, 2, 1)
        visual_cls_tokens = self.visual_cls_token.expand(batch_size, -1, -1).to(device)
        visual_co_embeds = torch.cat((visual_cls_tokens, img_emb), dim=1)
        visual_co_embeds = self.audio_layer_norm(self.dropout(self.visual_position_enc(visual_co_embeds)))

        if len(a_index) > 0:
            audio_tmp = self.audio_model.encoder(audio_co_embeds[av_index])
            audio_tmp = self.audio_model.layernorm(audio_tmp.last_hidden_state)
            visual_tmp = self.image_model.encoder(visual_co_embeds[av_index])
            visual_tmp = self.image_model.layernorm(visual_tmp.last_hidden_state)
         
        
            doa_0, feature_0 = self.AudioVisualAttention(audio_tmp, visual_tmp)
            feature_1 = self.audio_model.encoder(audio_co_embeds[a_index])
            feature_1 = self.audio_model.layernorm(feature_1.last_hidden_state)
            doa_1 = self.softmax(self.output_fc_audio(feature_1)[:,0,:])

            doa = torch.ones((batch_size, 360)).to(device)
            # pdb.set_trace()
            doa[a_index] = doa_1
            doa[av_index] = doa_0
           
            
        else:
            audio_tmp = self.audio_model.encoder(audio_co_embeds[av_index])
            audio_tmp = self.audio_model.layernorm(audio_tmp.last_hidden_state)
            visual_tmp = self.image_model.encoder(visual_co_embeds[av_index])
            visual_tmp = self.image_model.layernorm(visual_tmp.last_hidden_state)
            
            doa, feature = self.AudioVisualAttention(audio_tmp, visual_tmp)

     
        return doa

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)