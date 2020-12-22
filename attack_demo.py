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

from method_w_only_noise import Attacker

parser = argparse.ArgumentParser()
parser.add_argument('--order', type=int, default=1, help='Just an order')
args=parser.parse_args()

xception_path='/hd1/lidongze/style_atk/detectors/weights/5GAN1024png15000_xception.ckpt'
efficientnet_path="/hd1/fanhongxing/fake_detect/out/atack_efficientb3/0_efficient.ckpt"
model_type='xception'

if __name__=='__main__': 
    #load xception model
    model_path,size,normalize,model,='',0,0,0
    
    if model_type=='xception':
        size=299
        model_path=xception_path
        normalize = NormalizeByChannelMeanStd(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        model = Xception(num_classes=1)
        
    elif model_type=='efficient':
        size=300
        model_path=efficientnet_path
        normalize = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = EfficientNet.from_name('efficientnet-b3')
        model._fc = nn.Linear(1536, 1)
    
    
    resize=nn.Upsample(size=(size, size), mode='bilinear')
    model.load_state_dict(torch.load(model_path))   
    model = nn.Sequential(resize,normalize, model)
    model.cuda()
    model.eval()
    print('model loaded')

    atker=Attacker(date='11_5_w_only_noise_xception_30_{}'.format(args.order))
    print('attacker loaded')

    #print(atker)
    for i in range(30):
        atker.Attack(name='id'+str(i)+'_',target_model=model)
        print('image {} generated'.format(i))