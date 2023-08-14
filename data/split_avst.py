import argparse
import os
import shutil

import librosa
import soundfile

import cv2
import numpy as np
from tqdm import tqdm


def split(_dataset_path, _output_path, _groups, _nb_chunks):
    dataset_path = _dataset_path
    print('Processing ' + dataset_path)
    output_path = _output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    chunk_length = 0.04
    interval = 0.04
    # groups = 18
    groups = _groups

    seqs = os.listdir(dataset_path)
    seqs.sort()
    # seqs = seqs[11:]
    nb_chunks = _nb_chunks
    last_chunks = _nb_chunks

    # nb_chunks = 600
    for seq in seqs:
        if seq == '.DS_Store':
            continue
        # if seq == 'seq11':
        #     break
        print('Processing ' + seq)
        seq_path = os.path.join(dataset_path, seq)
        for f in os.listdir(seq_path):
            if f.startswith('Bbox'):
                bbox = f
                break
        bbox_file = os.path.join(seq_path, bbox)

        
        recording_folder = os.path.join(seq_path, 'Recordings')

        recording_files = os.listdir(recording_folder)
        for recording_file in recording_files:
            if recording_file.endswith(".wav"):
                audio_file = os.path.join(seq_path, 'Recordings', recording_file)
            elif recording_file.endswith(".mp4"):
                movie_file = os.path.join(seq_path, 'Recordings', recording_file)
        depth_folder = os.path.join(seq_path, 'depthImages')
        gt1 = os.path.join(seq_path, '3dGT_1.txt')
        if os.path.isfile(os.path.join(seq_path, '3dGT_2.txt')):
            gt2 = os.path.join(seq_path, '3dGT_2.txt')
        elif os.path.isfile(os.path.join(seq_path, '3dGT_3.txt')):
            gt2 = os.path.join(seq_path, '3dGT_3.txt')
        elif os.path.isfile(os.path.join(seq_path, '3dGT_4.txt')):
            gt2 = os.path.join(seq_path, '3dGT_4.txt')
        
        # split the sound file
        # i = 0
        print('Process the sound file...')
        nb_chunks = last_chunks

        audio, sr = librosa.load(audio_file, sr=48000, mono=False)
        step = 12
        for i in tqdm(range(groups)):
            
            nb_chunks += 1
            
            chunk_path = os.path.join(output_path, 'chunk_' + str(nb_chunks))
            if os.path.exists(chunk_path):
                shutil.rmtree(chunk_path)

            os.mkdir(chunk_path)
            
            if (i - step) < 0 or (i + step) >= groups:
                continue
            starting = (i - step) * chunk_length
            ending = (i + step) * chunk_length
            subaudio = audio[:, int(starting * sr): int(ending * sr)]
            subaudio = subaudio.transpose()
            soundfile.write(os.path.join(chunk_path, 'subaudio_' + str(starting) + '_' + str(ending) + '.wav'), subaudio, samplerate=sr)


        print('Process the first gt...')
        i = 0
        nb_chunks = last_chunks
        with open(gt1, 'r') as f1:
            contents = f1.readlines()
            contents = contents[3:]
            for content in tqdm(contents):
                
                k = i * chunk_length / 2
                

                if round(float(content.split(" ")[0]), 2) == round(k, 2) and (i % 2 == 0):
                    nb_chunks += 1
                
                    chunk_path = os.path.join(output_path, 'chunk_' + str(nb_chunks))
                    
                    with open(os.path.join(chunk_path, '3dGT_1.txt'), 'a') as fw1:
                        fw1.write(content)
                i += 1
                if (i == (groups * 2)):
                    break
                        
        print('Process the second gt...')
        i = 0
        nb_chunks = last_chunks    
        with open(gt2, 'r') as f2:
            contents = f2.readlines()
            contents = contents[3:]
            for content in tqdm(contents):
                
                k = i * chunk_length / 2 

                if round(float(content.split(" ")[0]), 2) == round(k, 2) and (i % 2 == 0):
                    nb_chunks += 1
                
                    chunk_path = os.path.join(output_path, 'chunk_' + str(nb_chunks))
                    with open(os.path.join(chunk_path, '3dGT_2.txt'), 'a') as fw2:
                        fw2.write(content)
                i += 1
                if (i == (groups * 2)):
                    break

            # with open(os.path.join(chunk_path, '3dGT_2.txt'), 'r') as fw:
            #     contents = fw.readlines()
            #     assert len(contents) == int(chunk_length / interval)
            # with open(os.path.join(chunk_path, '3dGT_1.txt'), 'r') as fw:
            #     contents = fw.readlines()
            #     assert len(contents) == int(chunk_length / interval)
            
        

        
        # split the video
        
        videoCapture = cv2.VideoCapture(movie_file)
        
        i = 0
        nb_chunks = last_chunks
        count = 0
        fps_video = videoCapture.get(cv2.CAP_PROP_FPS)
        print('Process the video....')
        while(videoCapture):   
            success, frame = videoCapture.read()
            if success == True:

                
                
                starting = count * chunk_length
                ending = (count + 1) * chunk_length
                # if count > (ending * fps_video):
                #     break
                
                
                # if (count >= (starting * fps_video / 2) and count < (ending * fps_video / 2)) and (count % 2 == 0):
                if (count % 2 == 0):
                    nb_chunks += 1
                    chunk_path = os.path.join(output_path, 'chunk_' + str(nb_chunks))
                    chunk_image_path = os.path.join(chunk_path, 'image')
                    if not os.path.exists(chunk_image_path):
                        os.mkdir(chunk_image_path)
                    cv2.imwrite(os.path.join(chunk_image_path, str(count) + '_' + str(count / fps_video) + '.jpg'), frame)
                count += 1
                if count == (groups * 2):
                    break
            else:
                videoCapture.release()
        
        print('Process the bbox gt...')
        i = 0
        nb_chunks = last_chunks    
        with open(bbox_file, 'r') as f2:
            contents = f2.readlines()
            contents = contents[3:]
            for content in tqdm(contents):
                
                k = i * chunk_length / 2 

                if round(float(content.split(" ")[0]), 2) == round(k, 2) and (i % 2 == 0):
                    nb_chunks += 1
                
                    chunk_path = os.path.join(output_path, 'chunk_' + str(nb_chunks))
                    # chunk_path = os.path.join(output_path, 'chunk_' + str(nb_chunks))
                    # if os.path.exists(chunk_path):
                    #     shutil.rmtree(chunk_path)

                    # os.mkdir(chunk_path)
                    with open(os.path.join(chunk_path, 'bbox.txt'), 'a') as fw2:
                        fw2.write(content)
                i += 1
                if (i == (groups * 2)):
                    break
        
        # split the depth image
        i = 0
        nb_chunks = last_chunks
        print('Process the depth map...')
        depthImages = os.listdir(depth_folder)
        depthImages.sort(key = lambda x : int(x.split('_')[0]))
        depthImages = depthImages[2:]
        # for j in range(round(starting * int(1 / interval)), round(ending * int(1 / interval))):
        #     k = j * interval
        for depthImage in tqdm(depthImages):
            
            k = i * chunk_length / 2
            
            if round(float(depthImage.split("_")[1][:-4]), 2) == round(k, 2) and (i % 2 == 0):
                nb_chunks += 1
                chunk_path = os.path.join(output_path, 'chunk_' + str(nb_chunks))
                chunk_depth_path = os.path.join(chunk_path, 'depthImages')
                if not os.path.exists(chunk_depth_path):
                    os.mkdir(chunk_depth_path)
                shutil.copy(os.path.join(depth_folder, depthImage), chunk_depth_path)
            i += 1
            if i == (groups * 2):
                break
        
        last_chunks = nb_chunks

