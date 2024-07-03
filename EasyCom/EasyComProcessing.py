import torch
from torch import nn
import os
import librosa
import json
import imageio
import skimage
import numpy as np
import pdb
import cv2

class EasyCom(nn.Module):
    def __init__(self, dataset_path) -> None:
        super().__init__()
        self.main_path = '/mnt/fast/nobackup/scratch4weeks/jz01019/EasyComDataset/Main/'
        with open(dataset_path, 'r') as f:
            video_list = f.readlines()
        self.video_list = video_list

    def get_multi_channel_audio(self, index):
        video_path = self.video_list[index].strip()
        audio_path = video_path.replace('Video_Compressed', 'Glasses_Microphone_Array_Audio').replace('mp4', 'wav')
        multi_channel_audio, sr = librosa.load(audio_path, sr=48000, mono=False)
        
        return multi_channel_audio
    
    def get_poses(self, index):
        video_path = self.video_list[index].strip()
        pose_path = video_path.replace('Video_Compressed', 'Tracked_Poses').replace('mp4', 'json')
        with open(pose_path, 'r') as f:
            tracked_pose_contents = f.read()
        tracked_pose = json.loads(tracked_pose_contents) ## tracked pose over 1200 frame
        
        # [index]
        # frame_number = tracked_pose['Frame_Number']
        # nb_participants = len(tracked_pose['Participants'])
        # info_participants = tracked_pose['Participants']
        return tracked_pose
    
    def get_video(self, index):
        video_path = self.video_list[index].strip()
        vid = imageio.get_reader(video_path, 'ffmpeg')
        image_list = []
        for num, im in enumerate(vid):
            # pdb.set_trace()
            image = skimage.img_as_float(im).astype(np.float64)
            image_list.append(image)
        pdb.set_trace()
        return image_list
    
    def get_video_v2(self, index):
        cap = cv2.VideoCapture(self.video_list[index].strip())
        image_list = []
        while(cap.isOpened()):
            # ret返回布尔值
            ret, frame = cap.read()
            # pdb.set_trace()
            # 展示读取到的视频矩阵
            image_list.append(frame)
            # 键盘等待
            # k = cv2.waitKey(20)
            # # q键退出
            # if k & 0xFF == ord('q'):
            #     break
            del(ret)
            del(frame)
        
        cap.release()
        return image_list
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        multi_channel_audio = self.get_multi_channel_audio(index)
        tracked_pose = self.get_poses(index)
        image_list = self.get_video_v2(index)
        pdb.set_trace()
        return multi_channel_audio, tracked_pose, image_list
    

if __name__ == '__main__':
    with open('/mnt/fast/nobackup/scratch4weeks/jz01019/EasyComProcessing/video.txt', 'r') as f:
        contents = f.readlines()
    
    for content in contents:
        if '/Session_1/' in content or '/Session_2/' in content or '/Session_3/' in content:
            with open('/mnt/fast/nobackup/scratch4weeks/jz01019/EasyComProcessing/test.txt', 'a') as w:
                w.write(content)
        else:
            with open('/mnt/fast/nobackup/scratch4weeks/jz01019/EasyComProcessing/training.txt', 'a') as w:
                w.write(content)