import shutil
import numpy as np
import librosa
import os
from tqdm import tqdm
import argparse
import pdb

nfft = 1024
hopsize = 320 # 640 for 20 ms
mel_bins = 96
window = 'hann'
fmin = 50

class LogMelGccExtractor():
    def __init__(self, fs, nfft, hopsize, mel_bins, window, fmin):

        self.nfft = nfft
        self.hopsize = hopsize
        self.window = window
        self.melW = librosa.filters.mel(sr=fs,
                                        n_fft=nfft,
                                        n_mels=mel_bins,
                                        fmin=fmin)

    def logmel(self, sig):

        S = np.abs(librosa.stft(y=sig,
                                n_fft=self.nfft,
                                hop_length=self.hopsize,
                                center=True,
                                window=self.window,
                                pad_mode='reflect'))**2        
        S_mel = np.dot(self.melW, S).T
        S_logmel = librosa.power_to_db(S_mel, ref=1.0, amin=1e-10, top_db=None)
        S_logmel = np.expand_dims(S_logmel, axis=0)

        return S_logmel

    def gcc_phat(self, sig, refsig):

        ncorr = 2*self.nfft - 1
        nfft = int(2**np.ceil(np.log2(np.abs(ncorr))))
        Px = librosa.stft(y=sig,
                        n_fft=nfft,
                        hop_length=self.hopsize,
                        center=True,
                        window=self.window, 
                        pad_mode='reflect')
        Px_ref = librosa.stft(y=refsig,
                            n_fft=nfft,
                            hop_length=self.hopsize,
                            center=True,
                            window=self.window,
                            pad_mode='reflect')
    
        R = Px*np.conj(Px_ref)

        n_frames = R.shape[1]
        gcc_phat = []
        for i in range(n_frames):
            spec = R[:, i].flatten()
            cc = np.fft.irfft(np.exp(1.j*np.angle(spec)))
            cc = np.concatenate((cc[-mel_bins//2:], cc[:mel_bins//2]))
            gcc_phat.append(cc)
        gcc_phat = np.array(gcc_phat)
        # gcc_phat = gcc_phat[None,:,:]

        return gcc_phat

    def transform(self, audio):

        channel_num = audio.shape[0]
        feature_logmel = []
        feature_gcc_phat = []
        for n in range(channel_num):
            feature_logmel.append(self.logmel(audio[n]))
            for m in range(n+1,channel_num):
                feature_gcc_phat.append(
                    self.gcc_phat(sig=audio[m], refsig=audio[n]))
        
        feature_logmel = np.concatenate(feature_logmel, axis=0)
        feature_gcc_phat = np.concatenate(feature_gcc_phat, axis=0)
        feature = np.concatenate([feature_logmel, feature_gcc_phat])

        return feature

def test(chunks_path):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--chunks_path", help="dataset path",type=str)
    
    # args = parser.parse_args()

    # # chunks_path = "/vol/research/AVTracking/simulated_AV_tracking/chunks/"
    # chunks_path = args.chunks_path

    chunks = os.listdir(chunks_path)
    for chunk in tqdm(chunks):
        audio_files = os.listdir(os.path.join(chunks_path, chunk))
        for audio_file in audio_files:
            if audio_file.endswith('wav'):
                audio = audio_file
                break
        gcc_path = os.path.join(chunks_path, chunk, 'gcc')
        if os.path.exists(gcc_path):    
            shutil.rmtree(gcc_path)
        
        os.mkdir(gcc_path)

        interval = 12
        fs = 48000
        fps = 25
        extractor = LogMelGccExtractor(fs, nfft, hopsize, mel_bins, window, fmin)
        unit_length = fs / fps
        audio, sr = librosa.load(os.path.join(chunks_path, chunk, audio), sr=fs, mono=False)
        nb_unit = len(audio[1]) / unit_length

        audio = np.split(audio, nb_unit, axis=1)
        result = []
        for i in range(int(nb_unit)):
            if (i < interval / 2) or (i + interval / 2 >= nb_unit):
                continue
            audio_tmp0 = np.array([])
            audio_tmp1 = np.array([])
            for j in range(-int(interval / 2), int(interval / 2) + 1):
                audio_tmp0 = np.concatenate([audio_tmp0, audio[i + j][0]], axis=0)
                audio_tmp1 = np.concatenate([audio_tmp1, audio[i + j][1]], axis=0)

            gccphat = extractor.gcc_phat(audio_tmp0, audio_tmp1)
            time_step = (1 / fps) * i
            result.append(gccphat)
        result = np.array(result)
        # pdb.set_trace()
        np.save(os.path.join(gcc_path, str(i) + '_' + str(time_step)), result)
