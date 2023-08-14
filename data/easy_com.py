import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
import pdb
import math
import json
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms


data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def collect(batch):
    batch_size = len(batch)
    # pdb.set_trace()
    image = [item[1].unsqueeze(0) for item in batch]
    imgs = torch.cat(image, dim=0)
    stft = [item[2].unsqueeze(0) for item in batch]
    stfts = torch.cat(stft, dim=0)
    return batch, imgs, stfts

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def xz_to_horizontal_DOA(pitch_y, ref_x, ref_z, x, z):
    convert_ref_x = ref_x * np.cos(pitch_y) - ref_z * np.sin(pitch_y)
    convert_ref_z = ref_x * np.sin(pitch_y) + ref_z * np.cos(pitch_y)

    convert_x = x * np.cos(pitch_y) - z * np.sin(pitch_y)
    convert_z = x * np.sin(pitch_y) + z * np.cos(pitch_y)

    gt = (convert_x - convert_ref_x, convert_z - convert_ref_z)

    doa = np.arctan2(gt[1], gt[0]) * 180 / np.pi

    if doa < 0:
        doa = doa + 360

    return doa

def xyz_to_vertical_DOA(ref_x, ref_y, ref_z, x, y, z):
    
    delta_y = y - ref_y
    delta_horizontal = math.sqrt((x - ref_x) ** 2 + (z - ref_z) ** 2)

    doa = np.arctan(delta_y / delta_horizontal) + np.pi / 2

    doa = doa * 180 / np.pi

    return doa

def point_to_gt(vertical_DOA, horizontal_DOA, r, gt):
    x = vertical_DOA / 2
    y = horizontal_DOA / 2
   
    for i in range(90):
        for j in range(180):
            if math.sqrt((x - i) ** 2 + (y - j) ** 2) <= r:
                gt[i][j] = 1
    return gt


class EasyCom(nn.Module):
    def __init__(self, train, reshape_feature=False, gccphat=False, homography=False) -> None:
        super().__init__()
        if train == True:
            self.dataset_list = 'training_set.txt'
        elif train == False:
            self.dataset_list = 'test_set.txt'
        
        self.reshape_feature = reshape_feature
        self.gccphat = gccphat
        self.homography = homography
        with open(self.dataset_list, 'r') as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        speaker_id, active_wearer = self.get_active_speaker_id(index)
        feature = np.load(self.dataset[index].strip(), allow_pickle=True)
        
        # pdb.set_trace()

        if len(speaker_id) == 0 or (len(speaker_id) == 1 and speaker_id[0] == 1):
            gt = torch.zeros((90, 180))
            speaker_id_matrix = torch.zeros((90, 180))
        else:
            pose = feature.item()['pose']

            ## convert pose dict to GT

            info_participants = pose['Participants']
            nb_participants = len(info_participants)

            speakers = []
            for i in range(nb_participants):
                if int(info_participants[i]['Participant_ID']) == 2:
                    ref = info_participants[i]

                for j in range(len(speaker_id)):    
                    if int(info_participants[i]['Participant_ID']) == speaker_id[j] and speaker_id[j] != 1:
                        speakers.append(info_participants[i])

            
            ref_euler_r, ref_euler_p, ref_euler_y = euler_from_quaternion(ref['Quaternion_X'], ref['Quaternion_Y'], 
                                                    ref['Quaternion_Z'], ref['Quaternion_W'])

            tmp = torch.zeros((90, 180))
            tmp1 = torch.zeros((90, 180))
            for speaker in speakers:
                horizontal_DOA = xz_to_horizontal_DOA(ref_euler_p, ref['Position_X'], ref['Position_Z'], 
                                                    speaker['Position_X'], speaker['Position_Z'])
        
                assert 0<=horizontal_DOA<360

                vertical_DOA = xyz_to_vertical_DOA(ref['Position_X'], ref['Position_Y'], ref['Position_Z'], 
                                                speaker['Position_X'], speaker['Position_Y'], speaker['Position_Z'])
                assert 0 <=vertical_DOA<180

                tmp = point_to_gt(vertical_DOA, horizontal_DOA, 5, tmp)
                tmp1[int(vertical_DOA/2), int(horizontal_DOA/2)] = 1

            gt = tmp
            speaker_id_matrix = tmp1


        image_current = feature.item()['image']
        
        
        
        stft = np.load(os.path.join(os.path.dirname(self.dataset[index].strip()), 'stft.npy'))
        if not self.reshape_feature:
            stft_converted = np.zeros((stft.shape[0] * 2, stft.shape[1], stft.shape[2]))
            # stft_converted = np.zeros((stft.shape[0] * 2, 224, 224))
            for i in range(stft_converted.shape[0]):
                if i % 2 == 0:
                    # stft_converted[i, :, :] = cv2.resize(stft[int(i / 2), :, :].real, (224, 224))
                    stft_converted[i, :, :] = stft[int(i / 2), :, :].real
                elif i % 2 == 1:
                    # stft_converted[i, :, :] = cv2.resize(stft[int(i / 2), :, :].imag, (224, 224))
                    stft_converted[i, :, :] = stft[int(i / 2), :, :].imag
        
        elif self.reshape_feature:
            # stft_converted = np.zeros((stft.shape[0] * 2, stft.shape[1], stft.shape[2]))
            stft_converted = np.zeros((stft.shape[0] * 2, 224, 224))
            for i in range(stft_converted.shape[0]):
                if i % 2 == 0:
                    stft_converted[i, :, :] = cv2.resize(stft[int(i / 2), :, :].real, (224, 224))
                    # stft_converted[i, :, :] = stft[int(i / 2), :, :].real
                elif i % 2 == 1:
                    stft_converted[i, :, :] = cv2.resize(stft[int(i / 2), :, :].imag, (224, 224))
                    # stft_converted[i, :, :] = stft[int(i / 2), :, :].imag
    
        image_width = image_current.shape[1]
        image_current = image_current[:,(image_width - 1920):,:]
        
        image = cv2.cvtColor(image_current,cv2.COLOR_BGR2RGB)
       


        if self.gccphat:
            gccphat = np.load(os.path.join(os.path.dirname(self.dataset[index].strip()), 'gcc', 'gccphat.npy')) ## (15, 188, 48)
            # gccphat = np.transpose(gccphat, (1, 2, 0))
            # gccphat = torch.from_numpy(gccphat)
            return image, gccphat, gt, active_wearer, speaker_id_matrix
        # image = data_transform(image)

        # pdb.set_trace()
        
        # stft = torch.from_numpy(stft)
        # image = image.permute(2, 0, 1)
        return image, stft_converted, gt, active_wearer, speaker_id_matrix
    
    def get_active_speaker_id(self, index):

        feature_path = self.dataset[index].strip()
        folder_name = feature_path.split('/')[-2]
        nb_session = folder_name.split('-')[0]
        video_name = folder_name.split('-')[1] + '-' + folder_name.split('-')[2] + '-' + folder_name.split('-')[3] + '.json'
        nb_frame = int(folder_name.split('-')[-1])

        transcription_path = os.path.join('Speech_Transcriptions/',
                                          'Session_' + nb_session, video_name)
        try:
            with open(transcription_path, 'r', encoding="ISO-8859-1") as f:
                transcription = f.read()
        except Exception:
            pdb.set_trace()
        
        transcription_content = json.loads(transcription)
        # pdb.set_trace()
        speaker_id = []
        active_wearer = 0
        for i in range(len(transcription_content)):
            if nb_frame >= int(transcription_content[i]['Start_Frame']) and int(nb_frame <= transcription_content[i]['End_Frame']):
                if int(transcription_content[i]['Participant_ID']) != 1 and int(transcription_content[i]['Participant_ID']) != 2:
                    speaker_id.append(int(transcription_content[i]['Participant_ID']))
                if int(transcription_content[i]['Participant_ID']) == 2:
                    active_wearer = 1
        
        return speaker_id, active_wearer
    
    def get_nearby_frames(self, index, interval=20):
        vanilla_path = self.dataset[index].strip()
        folder_name = os.path.dirname(vanilla_path)
        frame_name = os.path.basename(folder_name)
        frame_nb = int(frame_name.split('-')[-1])
        frame_past = frame_nb - interval
        frame_future = frame_nb + interval
        frame_past_name = frame_name.split('-')[0] + '-' + frame_name.split('-')[1] + '-' + \
            frame_name.split('-')[2] + '-' + frame_name.split('-')[3] + '-' + 'frame-' + str(frame_past)
        frame_future_name = frame_name.split('-')[0] + '-' + frame_name.split('-')[1] + '-' + \
            frame_name.split('-')[2] + '-' + frame_name.split('-')[3] + '-' + 'frame-' + str(frame_future)
        # remove the top level folder from folder_name
        frame_past_path = os.path.join(os.path.dirname(folder_name), frame_past_name, 'features.npy')
        frame_future_path = os.path.join(os.path.dirname(folder_name), frame_future_name, 'features.npy')
        if os.path.exists(frame_past_path) and os.path.exists(frame_future_path):

            return frame_past_path, frame_future_path
        
        else:
            return None, None

def homography_transformation(image_A, image_B):
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for image A and B
    keypoints_A, descriptors_A = sift.detectAndCompute(image_A, None)
    keypoints_B, descriptors_B = sift.detectAndCompute(image_B, None)

    # Create a Brute Force Matcher
    bf = cv2.BFMatcher()

    # Find the best matches using KNN (k-nearest neighbors)
    k = 2
    matches = bf.knnMatch(descriptors_A, descriptors_B, k=k)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Get the matching points' coordinates in image A and image B
    points_A = np.float32([keypoints_A[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_B = np.float32([keypoints_B[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography matrix
    homography_matrix, _ = cv2.findHomography(points_A, points_B, cv2.RANSAC, 5.0)

    # Get the dimensions of image A
    height, width = image_A.shape[:2]

    # Apply the homography transformation to image B to align it with image A's perspective
    aligned_image_B = cv2.warpPerspective(image_B, homography_matrix, (width, height))
    

    return aligned_image_B
