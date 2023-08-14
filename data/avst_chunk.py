import torch
from torch import float32, nn
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import math
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
import pdb

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

class AvstChunk(nn.Module):
    def __init__(self, path_file, prefix, model, image_width, image_height, noise, bbox, rescale_index):
        super(AvstChunk, self).__init__()
        self.path_file = path_file
        tmp = np.load(self.path_file, allow_pickle=True)
        self.chunk_list = tmp.tolist()
        self.prefix = prefix
        self.model = model
        self.noise = noise
        self.bbox = bbox
        self.rescale_index = rescale_index
        self.width = image_width
        self.height = image_height

    def __getitem__(self, index):
        # import time
        # start = time.time()
        chunk_path = self.chunk_list[index]
        chunk_path = self.prefix + chunk_path.strip()
        chunk_path_split = chunk_path.split('simulated_AV_tracking/')
        # pdb.set_trace()
        chunk_path = os.path.join(chunk_path_split[0], 'simulated_AV_tracking/', chunk_path_split[1])
        img_path = os.path.join(chunk_path, 'image')
        img_file = os.listdir(img_path)[0]

        bbox_file = os.path.join(chunk_path, 'bbox.txt')
      
        img = Image.open(os.path.join(img_path, img_file)).convert('RGB')
      
        img = data_transform(img)
   
        gcc_path = os.path.join(chunk_path, 'gcc')
        gcc_files = os.listdir(gcc_path)
        for g in gcc_files:
            if 'npz' in g:
                gcc_file = g
                break
        gcc = np.load(os.path.join(gcc_path, gcc_file), allow_pickle=True)
        
        gccphat = gcc[:,:,None]
        
        if self.model == 'transformer':
            gccphat = cv2.resize(gccphat, (96, 224))
        
        elif self.model == 'resnet':
            gccphat = cv2.resize(gccphat, (224, 224))
        
        # img = []
        # gccphat = []
        files = os.listdir(chunk_path)
        for f in files:
            if f.endswith('txt'):
                if f.endswith('1.txt'):
                    ref = f
                else:
                    if not f.startswith('bbox'):
                        obj = f
        # print("4",time.time()-start); start = time.time()
        info_obj = np.zeros((1, 2))
        info_ref = np.zeros((1, 3))

        with open(os.path.join(chunk_path, obj), 'r') as f:
            content = f.readlines()[0]
            tmp = content.split(" ")
            obj_x, obj_z = tmp[1], tmp[3]
            info_obj[0][0] = obj_x
            info_obj[0][1] = obj_z
        
        with open(os.path.join(chunk_path, ref), 'r') as f:

            content = f.readlines()[0]
           
            tmp = content.split(" ")
            ref_x, ref_z, rotate = tmp[1], tmp[3], tmp[5]
            info_ref[0][0] = ref_x
            info_ref[0][1] = ref_z
            info_ref[0][2] = rotate
        
        converted_obj = np.zeros((1, 2))
        converted_ref = np.zeros((1, 2))
        converted_obj[:, 0] = info_obj[:, 0] * np.cos((info_ref[:, 2]) * math.pi / 180) - info_obj[:, 1] * np.sin((info_ref[:, 2]) * math.pi / 180)
        converted_obj[:, 1] = info_obj[:, 0] * np.sin((info_ref[:, 2]) * math.pi / 180) + info_obj[:, 1] * np.cos((info_ref[:, 2]) * math.pi / 180)

        converted_ref[:, 0] = info_ref[:, 0] * np.cos((info_ref[:, 2]) * math.pi / 180) - info_ref[:, 1] * np.sin((info_ref[:, 2]) * math.pi / 180)
        converted_ref[:, 1] = info_ref[:, 0] * np.sin((info_ref[:, 2]) * math.pi / 180) + info_ref[:, 1] * np.cos((info_ref[:, 2]) * math.pi / 180)

        gt = converted_obj - converted_ref
        
        doa = np.arctan2(gt[:, 1], gt[:, 0]) * 180 / np.pi


        for i in range(len(doa)):
            if doa[i] < 0:
                doa[i] = doa[i] + 360
        
        r = (gt[:, 0] ** 2 + gt[:, 1] ** 2) ** (0.5)
        # import pdb; pdb.set_trace()
        if self.bbox == True:
            with open(bbox_file, 'r') as f:
                content = f.readlines()[0]
                t, x1, y1, x2, y2 = content.split()
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                fov = 0
                if (x1 > 0 and y1 > 0 and x2 < self.width and y2 < self.height):
                    fov = 1
                    if self.rescale_index > 0:
                        x1, y1, x2, y2 = self.rescale(float(x1), float(y1), float(x2), float(y2), self.rescale_index)
                box = (x1, y1, x2, y2, fov)

        if self.noise == True:
            gcc_files = os.listdir(gcc_path)
            for gcc_file in gcc_files:
                if len(gcc_file.split('_')) == 1:
                    gcc_clean_tmp = gcc_file
                elif gcc_file.split('_')[1][:-4] == '0':
                    gcc_0_tmp = gcc_file
                elif gcc_file.split('_')[1][:-4] == '10':
                    gcc_10_tmp = gcc_file
                elif gcc_file.split('_')[1][:-4] == '20':
                    gcc_20_tmp = gcc_file
                elif gcc_file.split('_')[1][:-4] == '-10':
                    gcc_m10_tmp = gcc_file

            gcc_clean = np.load(os.path.join(gcc_path, gcc_clean_tmp), allow_pickle=True)
            gcc_0 = np.load(os.path.join(gcc_path, gcc_0_tmp), allow_pickle=True)
            gcc_10 = np.load(os.path.join(gcc_path, gcc_10_tmp), allow_pickle=True)
            gcc_20 = np.load(os.path.join(gcc_path, gcc_20_tmp), allow_pickle=True)
            gcc_m10 = np.load(os.path.join(gcc_path, gcc_m10_tmp), allow_pickle=True)

            gcc_clean = gcc_clean[:,:,None]
            gcc_0 = gcc_0[:,:,None]
            gcc_10 = gcc_10[:,:,None]
            gcc_20 = gcc_20[:,:,None]
            gcc_m10 = gcc_m10[:,:,None]

            if self.model == 'transformer':
                gcc_clean = cv2.resize(gcc_clean, (96, 224))
                gcc_0 = cv2.resize(gcc_0, (96, 224))
                gcc_10 = cv2.resize(gcc_10, (96, 224))
                gcc_20 = cv2.resize(gcc_20, (96, 224))
                gcc_m10 = cv2.resize(gcc_m10, (96, 224))
            # for training resnet
            elif self.model == 'resnet':
                gcc_clean = cv2.resize(gcc_clean, (224, 224))
                gcc_0 = cv2.resize(gcc_0, (224, 224))
                gcc_10 = cv2.resize(gcc_10, (224, 224))
                gcc_20 = cv2.resize(gcc_20, (224, 224))
                gcc_m10 = cv2.resize(gcc_m10, (224, 224))
            
            if self.bbox == True:
                
              
                return gcc_clean, gcc_0, gcc_10, gcc_20, gcc_m10, img, doa, r / 5, box
            else:
              
                return gcc_clean, gcc_0, gcc_10, gcc_20, gcc_m10, img, doa, r / 5

        if self.bbox == True:
          
            return img, gccphat, doa, r / 5, box, chunk_path
     
        return img, gccphat, doa, r / 5, chunk_path

    def __len__(self):

        return len(self.chunk_list)
    
    def rescale(self, x1, y1, x2, y2, scale = 0.85):
        # coordinates after transformation
        assert 0 < scale < 1.0
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y2 + y1)
        width = x2 - x1
        height = y2 - y1
        w = width * scale
        h = height * scale
        new_x1 = center_x - 0.5 * w
        new_x2 = center_x + 0.5 * w
        new_y1 = center_y - 0.5 * h
        new_y2 = center_y + 0.5 * h

        return new_x1, new_y1, new_x2, new_y2