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

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH

import sys
sys.path.append('/hd1/lidongze/style_atk')
from detectors.face_detector.face_detector import FaceDetector
from detectors.fake_predictor.model.xception import Xception

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow

from loader import StyleGANDataset
from utils import save_img , tensor_to_np , get_result

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='xception', help='Type of the detector model')
 
parser.add_argument('--img_path', type=str, default='/hd5/lidongze/style_atk_imgs/11_3_only_noise_xception_100_1/noise_5_adv_step9', help='image path for manipulation detection')
parser.add_argument('--batch_size', type=int, default=2, help='input batchsize')
parser.add_argument('--date', type=str, default='10_12', help='date')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
date=args.date
batch_size=args.batch_size
pics_root = args.img_path
model_type=args.model_type

xception_path='/hd1/lidongze/style_atk/detectors/weights/5GAN1024png15000_xception.ckpt'
#xception_path='/hd1/lidongze/style_atk/0_xception.ckpt'
#xception_path='/hd1/fanhongxing/fake_detect/out/atack_xception_style_1101/2_efficient.ckpt'
efficientnet_path="/hd1/fanhongxing/fake_detect/out/atack_efficientb3/0_efficient.ckpt"
#efficientnet_path='/hd1/fanhongxing/fake_detect/out/atack_efficient_style_1101/2_efficient.ckpt'
#efficientnet_path="/hd1/fanhongxing/fake_detect/out/atack_efficientb3/3_efficient.ckpt"
#efficientnet_path="/data3/fanhongxing/GeekPwn2020/GeekPwn_CAAD_demo/weights/2_efficient_EndEpoch.ckpt"

normalize,model_path,model=0,'',0

if model_type=='efficient':
    normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = EfficientNet.from_name('efficientnet-b3')
    model._fc = nn.Linear(1536, 1)
    size=300
    model_path=efficientnet_path
       
elif model_type=='xception':
    normalize = NormalizeByChannelMeanStd(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    model = Xception(num_classes=1)
    size=299
    model_path=xception_path

model.load_state_dict(
    torch.load(model_path))
model = nn.Sequential(normalize,model)
model.cuda()
model.eval()
print('model loaded')

dataset=StyleGANDataset(pics_root,size)
print('total images',len(dataset))
loader=torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size, 
                                         shuffle=False)
print('Dataloader set')   
                                                                   
predict_true,predict_false,cnt=0,0,0
for i ,(img,label,image_name) in enumerate(loader):
    if cnt>500:
        break
    print(image_name)
    img=img.cuda()
    #print(img.size())
    #print(image_name)
    #print(label)
    label=label.view(img.shape[0],-1)
    #print(label)
    label=label.cuda()
    #print(model(img))
    prob=model(img)
    result=get_result(prob)
    predict_true+=torch.sum(result==label)
    predict_false+=torch.sum(result!=label)
    print(prob)
    print(predict_true)
    print(predict_false)
    cnt+=batch_size

length=0
if len(dataset)==0:
    length=1
else:
    length=len(dataset)
print('total {} imgs true {},false {}'.format(len(dataset),predict_true,predict_false))
print('accuracy {:.3f}'.format(float(predict_true)/cnt))

    
    

