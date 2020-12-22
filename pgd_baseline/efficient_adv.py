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


torch.manual_seed(0)
use_cuda = torch.cuda.is_available()

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH

# filename = "mnist_lenet5_clntrained.pt"
# filename = "mnist_lenet5_advtrained.pt"

normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = EfficientNet.from_name('efficientnet-b3')
model._fc = nn.Linear(1536, 1)

model.load_state_dict(
    torch.load("/data3/fanhongxing/GeekPwn2020/GeekPwn_CAAD_demo/weights/2_efficient_EndEpoch.ckpt"))
model = nn.Sequential(normalize, model)
model.cuda()
model.eval()
print('model loaded')

def get_pics(pics_root):
    images = []
    labels = []
    pics_path = os.listdir(pics_root)
    pics_path.sort()
    for pic_path in pics_path:
        if pic_path.split('_')[0] == 'fake':
            labels.append(1)
        else:
            labels.append(0)
        pic_path = os.path.join(pics_root,pic_path)
        pic = cv2.imread(pic_path, cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
        # image = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(),
        #                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])(image)
        image = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor(),
                             ])(image)
        images.append(image)
    cln_data = torch.stack(images, dim=0)
    labels = torch.Tensor(labels).unsqueeze(1)
    return cln_data, labels


# batch_size = 5
# loader = get_mnist_test_loader(batch_size=batch_size)
# for cln_data, true_label in loader:
#     break
# print(cln_data[0])

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






pics_root = '/data3/fanhongxing/DeepFake_Detection/data/adversarial/'
batch_size = len(os.listdir(pics_root))
cln_data, true_label = get_pics(pics_root)
cln_data, true_label = cln_data.cuda(), true_label.cuda()


from advertorch.attacks import LinfPGDAttack

adversary = LinfPGDAttack(
    model, loss_fn=nn.BCEWithLogitsLoss(), eps=0.15,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0, clip_max=1,
    targeted=False)

adv_untargeted = adversary.perturb(cln_data, true_label)
print(adv_untargeted)
print(cln_data)


save_img(cln_data,adv_untargeted,'./output_imgs/')

# target = torch.ones_like(true_label) * 3
# adversary.targeted = True
# adv_targeted = adversary.perturb(cln_data, target)

pred_cln = torch.sigmoid(model(cln_data)).cpu().detach().numpy()
pred_untargeted_adv = torch.sigmoid(model(adv_untargeted)).cpu().detach().numpy()
# pred_targeted_adv = torch.sigmoid(model(adv_targeted))
print('pred cln\n',pred_cln)
print('pred untargeted\n',pred_untargeted_adv)
print('true label\n',true_label)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(2, batch_size, ii + 1)
    _imshow(cln_data[ii])
    print('pred')
    print(pred_cln[ii])
    plt.title("clean \n label: {} \n pred: {} ".format(true_label[ii].cpu().detach().numpy(), pred_cln[ii]))
    plt.subplot(2, batch_size, ii + 1 + batch_size)
    _imshow(adv_untargeted[ii])
    plt.title("untargeted \n adv \n pred: {}".format(
        pred_untargeted_adv[ii]))
    print(ii)
    # plt.subplot(3, batch_size, ii + 1 + batch_size * 2)
    # _imshow(adv_targeted[ii])
    # plt.title("targeted to 3 \n adv \n pred: {}".format(
    #     pred_targeted_adv[ii]))

print('finished loop')
#plt.tight_layout()
#plt.show()

'''
# Construct defenses based on preprocessing

from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter

bits_squeezing = BitSqueezing(bit_depth=5)
median_filter = MedianSmoothing2D(kernel_size=3)
jpeg_filter = JPEGFilter(10)

defense = nn.Sequential(
    jpeg_filter,
    bits_squeezing,
    median_filter,
)


adv = adv_untargeted
adv_defended = defense(adv)
cln_defended = defense(cln_data)

pred_cln = predict_from_logits(model(cln_data))
pred_cln_defended = predict_from_logits(model(cln_defended))
pred_adv = predict_from_logits(model(adv))
pred_adv_defended = predict_from_logits(model(adv_defended))


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for ii in range(batch_size):
    plt.subplot(4, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(4, batch_size, ii + 1 + batch_size)
    _imshow(cln_data[ii])
    plt.title("defended clean \n pred: {}".format(pred_cln_defended[ii]))
    plt.subplot(4, batch_size, ii + 1 + batch_size * 2)
    _imshow(adv[ii])
    plt.title("adv \n pred: {}".format(
        pred_adv[ii]))
    plt.subplot(4, batch_size, ii + 1 + batch_size * 3)
    _imshow(adv_defended[ii])
    plt.title("defended adv \n pred: {}".format(
        pred_adv_defended[ii]))

plt.tight_layout()
plt.show()

from advertorch.bpda import BPDAWrapper
defense_withbpda = BPDAWrapper(defense, forwardsub=lambda x: x)
defended_model = nn.Sequential(defense_withbpda, model)
bpda_adversary = LinfPGDAttack(
    defended_model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
    nb_iter=1000, eps_iter=0.005, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)


bpda_adv = bpda_adversary.perturb(cln_data, true_label)
bpda_adv_defended = defense(bpda_adv)

pred_cln = predict_from_logits(model(cln_data))
pred_bpda_adv = predict_from_logits(model(bpda_adv))
pred_bpda_adv_defended = predict_from_logits(model(bpda_adv_defended))


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(pred_cln[ii]))
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(bpda_adv[ii])
    plt.title("bpda adv \n pred: {}".format(
        pred_bpda_adv[ii]))
    plt.subplot(3, batch_size, ii + 1 + batch_size * 2)
    _imshow(bpda_adv_defended[ii])
    plt.title("defended \n bpda adv \n pred: {}".format(
        pred_bpda_adv_defended[ii]))

plt.tight_layout()
plt.show()
'''