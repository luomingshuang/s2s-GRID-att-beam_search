# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance

    
class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, vid_path, anno_path, vid_pad, txt_pad, phase):
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase
        
        self.videos = glob.glob(os.path.join(vid_path, "*", "*", "*", "*"))
        self.videos = list(filter(lambda dir: len(os.listdir(dir)) == 75, self.videos))
        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)            
            self.data.append((vid, items[-4], items[-1]))        
        
                
    def __getitem__(self, idx):
        (vid, spk, name) = self.data[idx]
        vid = self._load_vid(vid)
        anno = self._load_anno(os.path.join(self.anno_path, spk, 'align', name + '.align'))
        #print('anno: ', anno)
        anno_len = anno.shape[0]
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        
        if(self.phase == 'train'):
            vid = HorizontalFlip(vid)
            vid = FrameRemoval(vid)
        vid = ColorNormalize(vid)
        
        return {'encoder_tensor': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), 
                'decoder_tensor': torch.LongTensor(anno)}
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, p): 
        #files = sorted(os.listdir(p))
        files = sorted(os.listdir(p), key=lambda x:int(x.split('.')[0]))        
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        # array = [cv2.resize(im, (50, 100)).reshape(50, 100, 3) for im in array]
        array = [cv2.resize(im, (100, 50)) for im in array]
        array = np.stack(array, axis=0)
        return array
    
    def _load_anno(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        return MyDataset.txt2arr(' '.join(txt).upper(), 1)
    
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    @staticmethod
    def txt2arr(txt, SOS=False):
        # SOS: 1, EOS: 2, P: 0, OTH: 3+x
        arr = []
        if(SOS):            
            tensor = [1]
        else:
            tensor = []
        for c in list(txt):
            tensor.append(3 + MyDataset.letters.index(c))
        tensor.append(2)
        return np.array(tensor)
        
    @staticmethod
    def arr2txt(arr):       
        # (B, T)
        result = []
        n = arr.size(0)
        T = arr.size(1)
        for i in range(n):
            text = []
            for t in range(T):
                c = arr[i,t]
                if(c >= 3):
                    text.append(MyDataset.letters[c - 3])
            text = ''.join(text)
            result.append(text)
        return result
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                txt.append(MyDataset.letters[n - start])
            pre = n
        return ''.join(txt)

    @staticmethod 
    def array_list_to_character(array_list):
        result = []
        txt = []
        for array in array_list:
            if array >=3:
                txt.append(MyDataset.letters[array-3])
        txt = ''.join(txt)
        result.append(txt)
        return result

    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()
    
    @staticmethod
    def wer_one(predict, truth):
        #print(predict)
        predict = predict[0].split(' ')
        truth = truth[0].split(' ')
        wer = 1.0*editdistance.eval(predict, truth)/len(truth)
        return wer

    @staticmethod
    def cer_one(predict, truth):
        cer = 1.0*editdistance.eval(predict[0], truth[0])/len(truth[0])
        return cer

    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer).mean()        
