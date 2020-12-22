import matplotlib.pyplot as plt
# matplotlib inline
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


from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow

from loader_resize import StyleGANDataset_resize
from utils import save_img , tensor_to_np , get_result

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH

import sys
sys.path.append('/hd1/lidongze/style_atk')
from detectors.face_detector.face_detector import FaceDetector
from detectors.fake_predictor.model.xception import Xception

#path
date='xception_pgd_l2_11_13_1'
batch_size=8
pics_root = '/hd1/lidongze/style_atk/output_imgs/10_15_ensemble_logits_0/img_atk_step0'
model_path='/hd1/lidongze/style_atk/detectors/weights/5GAN1024png15000_xception.ckpt'
face_detector_model_path='/hd1/lidongze/style_atk/detectors/weights/mobilenet0.25_Final.pth'
save_image_path='/hd1/lidongze/style_atk/pgd_baseline/output_img/imgs_{}'.format(date)

if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)


dataset=StyleGANDataset_resize(pics_root)
loader=torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size, 
                                         shuffle=False)
print('Datalaoder set')
                                                                                  
#load xception model

resize = nn.Upsample(size=(299, 299), mode='bilinear').cuda()
normalize = NormalizeByChannelMeanStd(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
model = Xception(num_classes=1)
model.load_state_dict(torch.load(model_path))                                             
model = nn.Sequential(resize,normalize, model)
model.cuda()
model.eval()
print('model loaded')

#face_detector = FaceDetector(face_detector_model_path, device = device)
#print('detector loaded')

from Attacker import MyPGDAttack

#adversary attacker
adversary = MyPGDAttack(
    model, loss_fn=nn.BCEWithLogitsLoss(), eps=0.3,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0, clip_max=1,
    targeted=False,order=np.inf)

cln_true,per_true=0,0
for i ,(img,label,image_name) in enumerate(loader):
    img=img.cuda()
    #print(img.size())
    #print(image_name)
    #print(label)
    label=label.view(label.shape[0],-1)
    #print(label.shape)
    label=label.cuda()
    img_perturbed, noise=adversary.perturb(img,label)
    #print(noise.shape)
    #print(model(img))
    #print(model(img_perturbed))
    output_cln=get_result(model(img))
    output_per=get_result(model(img_perturbed))
    cln_true+=torch.sum(output_cln==label)
    per_true+=torch.sum(output_per==label)
    print('true count',cln_true,'per count',per_true)
    
    #save images
    
    for j,name in enumerate(image_name):
        cln_img=cv2.imread(os.path.join(pics_root,name))
        adv_noise=noise[j].data.cpu().numpy()
        adv_noise=np.transpose(adv_noise, axes=[1, 2, 0]) * 255.0
        adv_noise = adv_noise[:, :, ::-1]
        adv_img=cln_img+adv_noise
        cv2.imwrite(os.path.join(save_image_path,image_name[j]),adv_img)
    
    #save_img(image_name,cln_batch=img.data.cpu(),perturb_batch=img_perturbed.data.cpu(),output_dir=save_image_path)
    
