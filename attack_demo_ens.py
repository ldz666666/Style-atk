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

#from loader_resize import StyleGANDataset
#from utils import save_img , tensor_to_np , get_result

#torch.manual_seed(0)
use_cuda = torch.cuda.is_available()

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH

import sys
sys.path.append('/hd1/lidongze/style_atk')
from detectors.face_detector.face_detector import FaceDetector
from detectors.fake_predictor.model.xception import Xception

from method_ensemble import Attacker

parser = argparse.ArgumentParser()
parser.add_argument('--order', type=int, default=0, help='Just an order')
args=parser.parse_args()

xception_path='/hd1/lidongze/style_atk/detectors/weights/5GAN1024png15000_xception.ckpt'
efficientnet_path="/hd1/fanhongxing/fake_detect/out/atack_efficientb3/0_efficient.ckpt"
model_type='efficient'

if __name__=='__main__': 
    #ensemble attack load xception and efficientnet model

    resize=nn.Upsample(size=(299, 299), mode='bilinear')
    normalize = NormalizeByChannelMeanStd(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    xception_model = Xception(num_classes=1)
    xception_model.load_state_dict(torch.load(xception_path))   
    xception_model = nn.Sequential(resize,normalize, xception_model)
    xception_model.cuda()
    xception_model.eval()
    print('xception model loaded')
        
    resize=nn.Upsample(size=(300, 300), mode='bilinear')
    normalize = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eff_model = EfficientNet.from_name('efficientnet-b3')
    eff_model._fc = nn.Linear(1536, 1)
    eff_model.load_state_dict(torch.load(efficientnet_path))   
    eff_model = nn.Sequential(resize,normalize, eff_model)
    eff_model.cuda()
    eff_model.eval()
    print('efficientnet model loaded')
    
    target_model_list=[xception_model,eff_model]

    atker=Attacker(date='10_28_ensemble_noise_{}'.format(args.order),ensemble='logits')
    print('attacker loaded')

    #print(atker)
    for i in range(5000):
        atker.Attack(name='id'+str(i)+'_',target_model_list=target_model_list)
        print('image {} generated'.format(i))