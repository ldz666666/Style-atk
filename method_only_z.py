#list , update the noise iteratively 
import os
import argparse
import torch
import torch.nn as nn

import cv2
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from advertorch.utils import NormalizeByChannelMeanStd
import matplotlib.pyplot as plt
import numpy as np

from torchvision import utils
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow

from StyleGAN.model import StyledGenerator,new_StyledGenerator
from StyleGAN.generate import get_mean_style,sample, style_mixing

from pgd_baseline.utils import get_result

import math

class Attacker:

    def __init__(self,device='cuda',img_size=512,generator_weight_path='/hd1/lidongze/style_atk/StyleGAN/checkpoint/stylegan-512px-new.model',eps=0.004,delta=0.2,nb_iter=10,n_sample=1,loss_fn=None,noise=None,date=''):
        #first get the generator
        self.device='cuda'
        self.img_size=img_size
        self.eps=eps
        self.delta=delta
        self.nb_iter = nb_iter
        self.n_sample = n_sample
        self.date = date
        self.img_dir = '/hd5/lidongze/style_atk_imgs/'+date+'/'
        #step parameter in Style GAN , set to 8 for simplicity
        if loss_fn is None:
            self.loss_fn=nn.BCEWithLogitsLoss(reduction='none')
        if noise is not None:
            self.noise=noise
        else:
            self.noise=None
        self.generator=new_StyledGenerator(img_size)
        self.generator_weight_path=generator_weight_path
        self.generator.load_state_dict(torch.load(self.generator_weight_path)['generator'])
        self.generator.to(self.device)
        self.generator.eval()
        print('generator loaded')
        
        
    def Attack(self,target_model,name='',minimize=False):
        #declare style vector style and noise vector noise
        #first we define stp
        
        
        for i in range(8):
            now_path=os.path.join(self.img_dir,'style_{}'.format(i))
            if not os.path.exists(now_path):
                os.makedirs(now_path)
        
        stp = int(math.log(self.img_size, 2)) - 2
        mean_style = get_mean_style(self.generator, device=self.device)
                                
        style=torch.randn(self.n_sample, 512,requires_grad=True).to(self.device)
                          
        styles=[]               
                                
        #why was the same person
        #print(style)
        
        #after that we define noise if it is not given
        noise = []
        for i in range(stp + 1):
            size = 4 * 2 ** i
            noise.append(torch.randn(self.n_sample, 1, size, size, device=self.device,requires_grad=True))
            styles.append(style.clone().detach())
            #stack noise together maybe noise size different
            #print(noise.shape)
        #print('len styles',len(styles))
        #print('len noise',len(noise))
        #define ground truth label
        #print(style.requires_grad)
        '''
        for n in noise:
            print(n.requires_grad)
        '''
        #print(noise)
        
        true_label=torch.ones(self.n_sample,1).to(self.device)
        
        #g=self.generator(style,step=stp,alpha=1,mean_style=mean_style,style_weight=0.7,noise=noise)
        
        #declaration may change
  
        #print('len delta_style',len(delta_style))
        '''
        for d in delta_noise:
            print(d.requires_grad)
        '''
        
        #choose the level you need to update the gradient
        #update_style_index=[]
        update_noise_index=[0,1,2,3,4,5,6,7,8,9]
        
        for style_level in range(8):
            
            delta_style = [torch.zeros_like(s,requires_grad=True).cuda() for s in styles]
            delta_noise = [torch.zeros_like(n,requires_grad=True).cuda() for n in noise]
        
            for ii in range(self.nb_iter):
            
                temp_style=[delta_style[index]+styles[index] for index in range(len(styles))]
                temp_noise=[delta_noise[index]+noise[index] for index in range(len(noise))]         
                 
                #print('len_temp_style',len(temp_style))
                g_temp=self.generator(temp_style,step=stp,alpha=1,mean_style=mean_style,style_weight=0.7,noise=temp_noise)
                
                #remember to add resize g and input into the model
        
                prediction = target_model(g_temp)
                
                now_path=os.path.join(self.img_dir,'style_{}/'.format(style_level))
                #now_path=os.path.join(self.img_dir,'noise{}/step{}/'.format(noise_level,ii))

                if ii==0 or ii==9:
                    utils.save_image(g_temp,os.path.join(now_path,name+'style_{}_step_{}.png'.format(style_level,ii)),normalize=True, range=(-1, 1))
                #utils.save_image(g_temp, os.path.join(now_path,name+'noise{}_step{}.png'.format(noise_level,ii)) , normalize=True, range=(-1, 1))
                #print('step {} prediction'.format(ii),prediction)
                prediction=prediction.view(self.n_sample,-1)
                
                loss = self.loss_fn(prediction, true_label)
                
                if minimize:
                    loss = -loss
    
                loss.backward()
    
                for i,d in enumerate(delta_style):
                    if i==style_level:
                        delta_style_sign=d.grad.data.sign()
                        d.data=d.data+self.eps*delta_style_sign
                
                
                '''for i,d in enumerate(delta_noise):
                    #if i==noise_level:
                    delta_noise_sign=d.grad.data.sign()
                    d.data=d.data+self.delta*delta_noise_sign'''

        
if __name__ == '__main__':
    atker=Attacker()
    
    