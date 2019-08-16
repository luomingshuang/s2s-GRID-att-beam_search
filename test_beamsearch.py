#coding:utf-8
import torch
from torch.utils.data import Dataset, DataLoader
import tensorboardX
from data import MyDataset
from model import lipreading
import torch.nn as nn
from torch import optim
import os
import time
import tensorflow as tf
import numpy as np

if(__name__ == '__main__'):
    torch.manual_seed(55)
    torch.cuda.manual_seed_all(55)
    opt = __import__('options')
    
def data_from_opt(vid_path, phase):
    dataset = MyDataset(vid_path, 
        opt.anno_path,
        opt.vid_pad,
        opt.txt_pad,
        phase=phase)
    print('vid_path:{},num_data:{}'.format(vid_path,len(dataset.data)))
    loader = DataLoader(dataset, 
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,
        shuffle=True)   
    return (dataset, loader)


if(__name__ == '__main__'):
    model = lipreading(mode=opt.mode, nClasses=30).cuda()
    
    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)        
            
    (tst_dataset, tst_loader) = data_from_opt(opt.val_vid_path, 'test')

    predict_txt_total = []
    truth_txt_total = []
    
    wer_total = []
    cer_total = []

    for batch in tst_loader:
        (encoder_tensor, decoder_tensor) \
         = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
        outputs = model(encoder_tensor)
        
        truth_txt = MyDataset.arr2txt(decoder_tensor)

        wer_list= []
        cer_list= []
        pairs = []
        for array in outputs:
            #print(MyDataset.array_list_to_character(array[0]))
            predict_txt = MyDataset.array_list_to_character(array[0])
            wer = MyDataset.wer_one(predict_txt, truth_txt)
            cer = MyDataset.cer_one(predict_txt, truth_txt)
            pairs.append((wer, cer))
        pairs = sorted(pairs, key=lambda x : x[0], reverse=True)
        #print(pairs[0])
        print(pairs)
        wer_total.append(pairs[0][0])
        cer_total.append(pairs[0][1])
    
    wer = np.mean(wer_total)
    cer = np.mean(cer_total)
    print('wer: ', wer, 'cer: ', cer)
        #predict_array = 
        #predict_txt = MyDataset.arr2txt(outputs.argmax(-1))
    #     truth_txt = MyDataset.arr2txt(decoder_tensor)
    #     print(truth_txt)
    #     predict_txt_total.extend(predict_txt)
    #     truth_txt_total.extend(truth_txt)
        
                
    # print(''.join(101*'-'))                
    # print('{:<50}|{:>50}'.format('predict', 'truth'))
    # print(''.join(101*'-')) 
                
    # for (predict, truth) in list(zip(predict_txt_total, truth_txt_total))[:10]:
    #     print('{:<50}|{:>50}'.format(predict, truth))                
    # print(''.join(101 *'-'))
    # wer = MyDataset.wer(predict_txt_total, truth_txt_total)
    # cer = MyDataset.cer(predict_txt_total, truth_txt_total)                
    # print('cer:{}, wer:{}'.format(cer, wer))          
    # print(''.join(101*'-'))
            