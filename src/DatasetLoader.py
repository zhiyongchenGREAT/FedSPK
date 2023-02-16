#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def loadWAV(filename, max_frames, evalmode=True, num_eval=10, resample=None):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)
    # print(sample_rate)

    # Resample data if not 16k
    if (sample_rate != 16000):
        number_of_samples = round(len(audio) * float(16000) / sample_rate)
        audio = signal.resample(audio, number_of_samples)

    if resample == 'fast':
        audio = signal.resample_poly(audio, 9, 10)
    elif resample == 'slow':
        audio = signal.resample_poly(audio, 11, 10)
    else:
        pass

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0).astype(numpy.float)

    return feat
    
class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4)
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        fs, rir     = wavfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]


class voxceleb_loader(Dataset):
    def __init__(self, dataset_file_name, augment, musan_path, rir_path, max_frames, train_path, sox_aug=False):

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)

        self.dataset_file_name = dataset_file_name
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment

        self.sox_aug    = sox_aug
        
        ### Read Training Files...
        with open(dataset_file_name) as dataset_file:
            lines = dataset_file.readlines();

        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        len_dictkeys_ori = len(dictkeys)
        if self.sox_aug:
            len_dictkeys = len(dictkeys)
            for ii, key in enumerate(list(dictkeys.keys())):
                dictkeys[key+'_slow'] = ii + len_dictkeys
                dictkeys[key+'_fast'] = ii + len_dictkeys*2

            assert len(dictkeys) == 3*len_dictkeys_ori

        self.label_dict = {}
        self.data_list  = []
        self.data_label = []
        
        lidx = 0
        for _, line in enumerate(lines):
            data = line.strip().split();

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(train_path,data[1])

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []

            self.label_dict[speaker_label].append(lidx)
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

            lidx += 1  

            if self.sox_aug:
                speaker_label_aslow = dictkeys[data[0]+'_slow']
                if not (speaker_label_aslow in self.label_dict):
                    self.label_dict[speaker_label_aslow] = []

                self.label_dict[speaker_label_aslow].append(lidx)
                
                self.data_label.append(speaker_label_aslow)
                self.data_list.append(filename+'.slow')
                lidx += 1

                speaker_label_afast = dictkeys[data[0]+'_fast']
                if not (speaker_label_afast in self.label_dict):
                    self.label_dict[speaker_label_afast] = []

                self.label_dict[speaker_label_afast].append(lidx)
                
                self.data_label.append(speaker_label_afast)
                self.data_list.append(filename+'.fast')
                lidx += 1
                      

    def __getitem__(self, indices):

        feat = []

        for index in indices:
            if not self.sox_aug:
                audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
            else:
                if self.data_list[index][-4:] == 'fast':
                    wav_path = self.data_list[index][:-5]
                    audio = loadWAV(wav_path, self.max_frames, evalmode=False, resample='fast')
                elif self.data_list[index][-4:] == 'slow':
                    wav_path = self.data_list[index][:-5]
                    audio = loadWAV(wav_path, self.max_frames, evalmode=False, resample='slow')
                else:
                    audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False, resample=None)
            
            if self.augment:
                augtype = random.randint(0,4)
                if augtype == 1:
                    audio     = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio   = self.augment_wav.additive_noise('music',audio)
                elif augtype == 3:
                    audio   = self.augment_wav.additive_noise('speech',audio)
                elif augtype == 4:
                    audio   = self.augment_wav.additive_noise('noise',audio)
                    
            feat.append(audio);

        feat = numpy.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

class voxceleb_loader_speaker(Dataset):
    def __init__(self, fl_class_token, dataset_file_name, augment, musan_path, rir_path, max_frames, train_path, sox_aug=False):

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)

        self.dataset_file_name = dataset_file_name
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment    = augment

        self.sox_aug    = sox_aug
        
        ### Read Training Files...
        with open(dataset_file_name) as dataset_file:
            lines = dataset_file.readlines();

        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        # len_dictkeys_ori = len(dictkeys)
        if self.sox_aug:
            raise NotImplementedError

        self.label_dict = {}
        self.data_list  = []
        self.data_label = []
        
        lidx = 0
        for _, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]] + fl_class_token
            filename = os.path.join(train_path, data[1])

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []

            self.label_dict[speaker_label].append(lidx)
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

            lidx += 1

            if self.sox_aug:
                raise NotImplementedError
                      

    def __getitem__(self, indices):

        feat = []

        for index in indices:
            if not self.sox_aug:
                audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
            else:
                raise NotImplementedError

            if self.augment:
                augtype = random.randint(0,4)
                if augtype == 1:
                    audio   = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio   = self.augment_wav.additive_noise('music',audio)
                elif augtype == 3:
                    audio   = self.augment_wav.additive_noise('speech',audio)
                elif augtype == 4:
                    audio   = self.augment_wav.additive_noise('noise',audio)

            feat.append(audio)

        feat = numpy.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames;
        self.num_eval   = num_eval
        self.test_path  = test_path
        self.test_list  = test_list

    def __getitem__(self, index):
        audio = loadWAV(os.path.join(self.test_path,self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


class voxceleb_sampler(torch.utils.data.Sampler):
    ## class_strict_balance super-sampling data to max_seg_per_spk for every spks
    ## const_batch return a iterator with constant batch numbers everytime(maximum batch available for the dataset) 
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, \
                 class_strict_balance=False, const_batch=True):

        self.label_dict         = data_source.label_dict
        self.nPerSpeaker        = nPerSpeaker
        self.max_seg_per_spk    = max_seg_per_spk
        self.batch_size         = batch_size
        self.class_strict_balance = class_strict_balance
        self.const_batch = const_batch
        
    def __iter__(self):
        
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        if not self.class_strict_balance:
            for findex, key in enumerate(dictkeys):
                data    = self.label_dict[key]
                numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)

                rp      = lol(numpy.random.permutation(len(data))[:numSeg],self.nPerSpeaker)
                flattened_label.extend([findex] * (len(rp)))
                for indices in rp:
                    flattened_list.append([data[i] for i in indices])
        else:
            for findex, key in enumerate(dictkeys):
                ## numSeg = max_seg_per_spk - max_seg_per_spk % nPerSpeaker
                patched_data = []
                data    = self.label_dict[key]
                numSeg  = round_down(self.max_seg_per_spk, self.nPerSpeaker)
                
                if len(data) < numSeg:
                    residual = numSeg-len(data)                    
                    patch_indexs = numpy.random.randint(0, len(data), residual)
                    patched_data = data + [data[i] for i in patch_indexs]
                    assert len(patched_data) == numSeg
                else:
                    patched_data = data
                
                rp      = lol(numpy.random.permutation(len(patched_data))[:numSeg], self.nPerSpeaker)
                flattened_label.extend([findex] * (len(rp)))
                for indices in rp:
                    flattened_list.append([patched_data[i] for i in indices])
            

        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        mixlabel        = []
        self.mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                self.mixmap.append(ii)
                
        ## const batch
        if self.const_batch:
            data_length = len(flattened_label)
            # print(data_length)
            expect_return_length = data_length - data_length % self.batch_size
            # print(expect_return_length)
            # print(len(self.mixmap))
            if len(self.mixmap) >= expect_return_length:
                self.mixmap = self.mixmap[:expect_return_length]
            else:
                if len(self.mixmap) > self.batch_size:                
                    while(len(self.mixmap) < expect_return_length):
                        ii = numpy.random.randint(0, data_length)                       
                        startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
                        if flattened_label[ii] not in mixlabel[startbatch:]:
                            mixlabel.append(flattened_label[ii])
                            self.mixmap.append(ii)
                else:
                    while(len(self.mixmap) < expect_return_length):
                        ii = numpy.random.randint(0, data_length)                       
                        startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
                        mixlabel.append(flattened_label[ii])
                        self.mixmap.append(ii)                    


                assert len(self.mixmap) == expect_return_length
        
        return iter([flattened_list[i] for i in self.mixmap])
    
    def __len__(self):
        return len(self.mixmap)


def get_data_loader(dataset_file_name, batch_size, augment, musan_path, rir_path, max_frames, max_seg_per_spk, nDataLoaderThread, nPerSpeaker, train_path, sox_aug, **kwargs):
    
    train_dataset = voxceleb_loader(dataset_file_name, augment, musan_path, rir_path, max_frames, train_path, sox_aug)

    train_sampler = voxceleb_sampler(train_dataset, nPerSpeaker, max_seg_per_spk, batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    
    return train_loader

def get_data_loader_speaker(fl_class_token, dataset_file_name, batch_size, augment, musan_path, rir_path, max_frames, max_seg_per_spk, nDataLoaderThread, nPerSpeaker, train_path, sox_aug, **kwargs):
    
    train_dataset = voxceleb_loader_speaker(
        fl_class_token, dataset_file_name, 
        augment, musan_path, 
        rir_path, max_frames, 
        train_path, sox_aug
    )

    train_sampler = voxceleb_sampler(train_dataset, nPerSpeaker, max_seg_per_spk, batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    
    return train_loader