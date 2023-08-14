import librosa
import os
from data.EasyComProcessing import EasyCom
from torch.utils.data import DataLoader
import cv2
import pdb
import librosa
import json
from tqdm import tqdm
import numpy as np

def split(dataset_path, output_path):
    with open(dataset_path, 'r') as f:
            video_list = f.readlines()

    for i in tqdm(range(14, len(video_list))):
        ## get video
        cap = cv2.VideoCapture(video_list[i].strip())
        image_list = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            # pdb.set_trace()
            if not ret:
                 break
            image_list.append(frame)

        cap.release()
        

        ## get audio

        video_path = video_list[i].strip()
        audio_path = video_path.replace('Video_Compressed', 'Glasses_Microphone_Array_Audio').replace('mp4', 'wav')
        multi_channel_audio, sr = librosa.load(audio_path, sr=48000, mono=False)


        ## get pose

        pose_path = video_path.replace('Video_Compressed', 'Tracked_Poses').replace('mp4', 'json')
        with open(pose_path, 'r') as f:
            tracked_pose_contents = f.read()
        tracked_pose = json.loads(tracked_pose_contents) ## tracked pose over 1200 frame

        fps = 20
        length = multi_channel_audio.shape[1] / 48000
        nb_frame = int(fps * length)

        audio_chunk_length = int(multi_channel_audio.shape[1] / nb_frame) ## 2400

        # multi_channel_audio = np.split(multi_channel_audio, nb_frame, axis=1)

        # pdb.set_trace()

        nb_audio_chunks_per_frame = 25

        for j in range(nb_frame):
            if j < 12 or j + 12 >= nb_frame:
                continue
            session_name = video_list[i].strip().split('/Session_')[1].replace('.mp4', '').replace('/', '-')
            session_name = session_name + '-frame-' + str(j)
            if not os.path.exists(os.path.join(output_path, session_name)):
                os.mkdir(os.path.join(output_path, session_name))
            
            audio_chunk = multi_channel_audio[:, (j - 12) * audio_chunk_length : (j + 12 + 1) * audio_chunk_length]

            image = image_list[j]

            pose = tracked_pose[j]

            # pdb.set_trace()
        
            np.save(os.path.join(output_path, session_name, 'features.npy'), {'audio': audio_chunk, 
                                                                              'image': image, 'pose': pose})



