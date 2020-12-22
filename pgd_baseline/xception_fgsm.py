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

from advertorch.attacks import FGSM
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow


torch.manual_seed(0)
use_cuda = torch.cuda.is_available()

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH
from utils import get_result
# filename = "mnist_lenet5_clntrained.pt"
# filename = "mnist_lenet5_advtrained.pt"

import sys
sys.path.append('/hd1/lidongze/style_atk')
from detectors.face_detector.face_detector import FaceDetector
from detectors.fake_predictor.model.xception import Xception

def tensor_to_np(img):
    tmp_img=img.detach().cpu().numpy()
    tmp_img=np.transpose(tmp_img,axes=[1,2,0])
    tmp_img=tmp_img[:,:,::-1]
    print(tmp_img)
    return (tmp_img*255.0).astype(np.uint8)


def save_img(cln_batch,perturb_batch,output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(cln_batch.shape[0]):
        perturbed_img=tensor_to_np(perturb_batch[i])
        clean_img=tensor_to_np(cln_batch[i])
        delta_img=tensor_to_np(perturb_batch[i]-cln_batch[i])
        cv2.imwrite(os.path.join(output_dir,'image_cln'+str(i)+'.png'),clean_img)
        cv2.imwrite(os.path.join(output_dir,'image_perturbed'+str(i)+'.png'),perturbed_img)
        cv2.imwrite(os.path.join(output_dir,'perturbed'+str(i)+'.png'),delta_img)
        
                                                                                 
#load xception model
model_path='/hd1/lidongze/style_atk/detectors/weights/5GAN1024png15000_xception.ckpt'
resize = nn.Upsample(size=(299, 299), mode='bilinear').cuda()
normalize = NormalizeByChannelMeanStd(
    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
model = Xception(num_classes=1)
model.load_state_dict(torch.load(model_path))                                             
model = nn.Sequential(resize,normalize, model)
model.cuda()
model.eval()
print('model loaded')

#path
date='xception_fgsm_11_13_1'
ori_image_path='/hd1/lidongze/style_atk/output_imgs/10_15_ensemble_logits_0/img_atk_step0'
save_image_path='/hd1/lidongze/style_atk/pgd_baseline/output_img/imgs_{}'.format(date)
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

batch_size=4
from loader_resize import StyleGANDataset_resize
dataset=StyleGANDataset_resize(ori_image_path)
loader=torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size, 
                                         shuffle=False)
print('Datalaoder set')

'''
from Attacker import MyPGDAttack
#adversary attacker
adversary = MyPGDAttack(
    model, loss_fn=nn.BCEWithLogitsLoss(), eps=0.3,
    nb_iter=60, eps_iter=0.01, rand_init=True, clip_min=0, clip_max=1,
    targeted=False,order=2)
'''
#adversary attacker
from Attacker import My_FGSM
adversary = My_FGSM(model,loss_fn=nn.BCEWithLogitsLoss(),eps=0.3,clip_min=0,clip_max=1,
targeted=False)


cln_true,per_true=0,0
for i ,(img,label,image_name) in enumerate(loader):
    img=img.cuda()
    #img=resize(img)
    #print(img.size())
    #print(image_name)
    #print(label)
    label=label.view(label.shape[0],-1)
    #print(label.shape)
    label=label.cuda()
    img_perturbed ,noise =adversary.perturb(img,label)
    

    for j,name in enumerate(image_name):
        cln_img=cv2.imread(os.path.join(ori_image_path,name))
        adv_noise=noise[j].data.cpu().numpy()
        adv_noise=np.transpose(adv_noise, axes=[1, 2, 0]) * 255.0
        adv_noise = adv_noise[:, :, ::-1]
        adv_img=cln_img+adv_noise
        cv2.imwrite(os.path.join(save_image_path,image_name[j]),adv_img)
        print('finish saving',name)
        
        test_image=cv2.imread(os.path.join(save_image_path,image_name[j]))
        test_image = Image.fromarray(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        # image = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(),
        #                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])(image)
        test_image = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(),
                             ])(test_image).cuda()
        test_image=test_image.unsqueeze(0)
        print('after perturbed,prediction is')
        tmp1=model[1](test_image)
        prediction=model[2](tmp1)
        print(prediction)
        print(get_result(prediction))
        print()      

















