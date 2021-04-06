#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:03:12 2020

@author: vnaray29
"""
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display
from models.DDUnet.ddunet import DDUNet
import argparse
import matplotlib.pyplot as plt
import os


def normalize(ref):
    ref = ref/np.max(np.abs(ref))
    return ref

def compute_SNR(ref, est):
    ref = ref.reshape(-1)
    est = est.reshape(-1)
    snr =  np.sum(ref**2) / np.sum((ref-est)**2)  #
    snr = 10*np.log10(snr)
    return snr

def main(args):
    # Device configuration
    # Command to utilize either GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_snr = []
    reference_snr = []
    std = 0.1
    for index, i in enumerate(sorted(os.listdir(args.data_dir+'/'+args.dataset_name+'/clean'))):

        print(i, args.data_dir+'/'+args.dataset_name+'/clean/'+i)
        ref_audio = librosa.load(args.data_dir+'/'+args.dataset_name+'/clean/'+i, 16000)[0]

        if args.noise_type == 'gaussian':
            noise = np.load(args.data_dir+'/'+args.dataset_name+'/'+'gaussian_noise/'+i[:-4]+'.npy')
            noisy_audio = ref_audio + std*noise
        elif args.noise_type == 'env':
            noisy_audio = librosa.load(args.data_dir+'/'+args.dataset_name+'/'+'env_noise/'+i, 16000)[0]

        #noisy_audio = ref_audio + np.random.normal(0.0, 0.1, ref_audio.shape) # Gaussian Noise

        noisy_spec_numpy = librosa.stft(noisy_audio, n_fft=1023, hop_length=64)
        noisy_spec_real = np.expand_dims(noisy_spec_numpy.real, axis=0)
        noisy_spec_imag = np.expand_dims(noisy_spec_numpy.imag, axis=0)
        noisy_spec = np.concatenate((noisy_spec_real, noisy_spec_imag),axis=0)
        noisy_spec = np.expand_dims(noisy_spec, axis=0)
        print('Shape of Noisy Spectrogram', noisy_spec.shape)


        z = torch.FloatTensor(np.random.normal(0.0, 1.0, noisy_spec.shape)).to(device)
        ref_SNR = compute_SNR(ref_audio, noisy_audio)
        noisy_spec = torch.FloatTensor(noisy_spec).to(device)


        if args.model_type == 'unet':
            model = UNet().to(device)
            print('Model chosen {}'.format(args.model_type))
        elif args.model_type == 'dilated_unet':
            model = DilatedUNet(args).to(device)
            print('Model chosen {}'.format(args.model_type))
        elif args.model_type == 'ddunet':
            model = DDUNet(args).to(device)
            print('Model chosen {}'.format(args.model_type))
        elif args.model_type == 'harmonic_unet':
            model = HarmonicUNet().to(device)
            print('Model chosen {}'.format(args.model_type))
        elif args.model_type == 'harmonic_ddunet':
            model = HarmonicDDUNet(args).to(device)
            print('Model chosen {}'.format(args.model_type))


        results_dir = os.path.join(args.results_dir, args.dataset_name, args.model_type, args.noise_type, args.dilation_type)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) # ADAM Optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
        mse = nn.MSELoss()

        current_snr = -100

        ref_enhanced, n_enhanced = [], []
        model.train()
        for iter_ in range(args.num_epochs):
            scheduler.step()

            enhanced_spec = model(z)
            loss = mse(enhanced_spec, noisy_spec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            enhanced_spec_numpy = enhanced_spec.cpu().detach().numpy()[:,0,:,:] + 1j * enhanced_spec.cpu().detach().numpy()[:,1,:,:]
            enhanced_audio = librosa.istft(enhanced_spec_numpy[0],win_length=1022,hop_length=64, length=len(ref_audio))

            recon_snr = compute_SNR(ref_audio, np.clip(enhanced_audio,-1.0,1.0))
            n_snr = compute_SNR(noisy_audio, np.clip(enhanced_audio,-1.0,1.0))

            ref_enhanced.append(recon_snr)
            n_enhanced.append(n_snr)

            print(('Epoch: {} Rec. Loss: {}, Ref. SNR {} Nois. SNR {} Recon SNR {} Best SNR {}').format(iter_, loss.item(), ref_SNR, n_snr, recon_snr, current_snr))

            if recon_snr > current_snr:
                current_snr = recon_snr
                enhanced_version = enhanced_audio

        plt.figure(figsize=(12,8))
        plt.subplot(131)
        S_db = librosa.amplitude_to_db(np.abs(librosa.stft(ref_audio)), ref=np.max)
        librosa.display.specshow(S_db)
        plt.title('Clean Audio Spectrogram')
        plt.colorbar()

        plt.subplot(132)
        S_db = librosa.amplitude_to_db(np.abs(noisy_spec_numpy), ref=np.max)
        librosa.display.specshow(S_db)
        plt.title('Noisy Audio Spectrogram')
        plt.colorbar()

        plt.subplot(133)
        S_db = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_version)), ref=np.max)
        librosa.display.specshow(S_db)
        plt.title('Enhanced Audio Spectrogram')
        plt.colorbar()
        plt.savefig(results_dir+'/spec_'+i+'.png')
        plt.close()

        plt.figure()
        plt.plot(noisy_audio)
        plt.plot(enhanced_version)
        plt.title('Noisy Audio and Enhanced Audio {}'.format(i))
        plt.legend(['Noisy', 'Enhanced'])
        plt.savefig(results_dir+'/noisy_enhanced_audio_'+i+'.png')
        plt.close()

        plt.figure()
        plt.plot(ref_enhanced, color='red')
        plt.plot(n_enhanced, color='green')
        plt.title('Ref/Enhanced SNR  vs Noisy/Enhanced SNR  {}'.format(i))
        plt.legend(['Ref/Enhanced SNR', 'Noisy/Enhanced SNR'])
        plt.savefig(results_dir+'/snr_'+i+'.png')
        plt.close()

        librosa.output.write_wav(results_dir+'/enhanced_'+i+'.wav', enhanced_version, 16000, norm=False)
        librosa.output.write_wav(results_dir+'/noisy.wav_'+i+'.wav', noisy_audio, 16000, norm=False)

        best_snr.append(compute_SNR(ref_audio, enhanced_version))
        reference_snr.append(ref_SNR)


    np.savetxt(results_dir+'/best_snr.txt', best_snr)
    np.savetxt(results_dir+'/ref_snr.txt', reference_snr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='ddunet', help='Model type')
    parser.add_argument('--noise_type', type=str, default='gaussian', help='Noise type')
    parser.add_argument('--dataset_name', type=str, default='LJSpeech', help='Dataset')
    parser.add_argument('--num_input_channels', type=int, default=2, help='No of Input channels')
    parser.add_argument('--num_output_channels', type=int, default=2, help='No of output channels')
    parser.add_argument('--dilation_type', type=str, default = 'constant', help = 'Type of dilation')
    parser.add_argument('--data_dir', type=str, default = '../data', help = 'Directory of audio data')
    parser.add_argument('--results_dir', type=str, default = './results', help = 'Results Directory')
    parser.add_argument('--learning_rate', type=float, default = 0.001, help = 'learning rate')
    parser.add_argument('--num_epochs', type=int, default = 2000, help = 'number of epochs')

    main(parser.parse_args())
