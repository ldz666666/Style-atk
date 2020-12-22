import os
import torch
from torch import nn
from torchvision import transforms
import collections
import numpy as np
import time
from PIL import Image
import cv2

from .model import EfficientNet
from .model import Xception

class BaseError(Exception):
    pass

class NoFrameError(BaseError):
    pass

class NetworkArchitectureError(BaseError):
    pass

class ImageChannelError(BaseError):
    pass


class Predictor(object):
    def __init__(self, model_path, arch = 'efficientnet-b3', classes_num = 1, thresh = 0.5, device = 'cuda'):
        if not os.path.isfile(model_path):
            raise BaseError('{:s} not found!'.format(model_path))
        
        # Set parameters
        self._arch = arch
        self._classes_num = classes_num
        self._thresh = thresh
        self._device = device

        # Load model
        print('Loading fake predictor model from {}'.format(model_path))
        if self._arch == 'efficientnet-b3':
            self._model = EfficientNet.from_arch(self._arch)
            # self._model._fc = nn.Linear(1792, self._classes_num)
            self._model._fc = nn.Linear(1536, self._classes_num)
            # model_para = collections.OrderedDict()
            # tmp_modelpara = torch.load(model_path)
            # for key in tmp_modelpara['state_dict'].keys():
            #     model_para[key[7 : ]] = tmp_modelpara['state_dict'][key]
            self._model.load_state_dict(torch.load(model_path))
        elif self._arch == 'xception_pre':
            self._model = Xception(num_classes=2)
            self._model.last_linear = self._model.fc
            del self._model.fc
            self._model.load_state_dict(torch.load(model_path))
        elif self._arch == 'xception':
            self._model = Xception(num_classes=1)
            self._model.load_state_dict(torch.load(model_path))
        elif self._arch == 'efficientnet-b4':
            self._model = EfficientNet.from_arch(self._arch)
            self._model._fc = nn.Linear(1792, self._classes_num)
            self._model.load_state_dict(torch.load(model_path))

        self._model = self._model.to(self._device)
        self._model.eval()
        print('Finished loading model!')

        # Determine transforms
        if self._arch == 'efficientnet-b3':
            self._tfms = transforms.Compose([transforms.Resize(300), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        elif self._arch == 'efficientnet-b4':
            self._tfms = transforms.Compose([transforms.Resize(380), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        else:
            self._tfms = transforms.Compose([transforms.Resize(299), transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
        
    
    def predict(self, img):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = self._tfms(image).unsqueeze(0)
        image = image.to(self._device)
 
        # Normal detection
        with torch.no_grad():
            outputs = self._model(image)
        # probs = torch.softmax(outputs, dim = 1)
        if self._arch == 'efficientnet-b3' or self._arch == 'xception' or self._arch == 'efficientnet-b4':
            probs = torch.sigmoid(outputs)
            predict = np.around(probs.cpu(), decimals=6)
            return predict.item()
        elif self._arch == 'xception_pre':
            probs = torch.softmax(outputs, dim=1)
            return probs[0, 1].item()
