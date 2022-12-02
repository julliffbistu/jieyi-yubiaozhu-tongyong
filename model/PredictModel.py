
import torch
import os
import numpy as np
from typing import List, Iterable, Dict, Union
import cv2
from nets.dunet import Dunet
from skimage import io as skio

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
basedir = os.path.abspath(os.path.dirname(__file__))

BATCHSIZE_PER_CARD  = 2

load_size = [512,512]
weights_file = basedir + '/weight/DuNet.th'
score_threshold = 0.3

rgb_value = {(255,255,255):0,(0,0,0):1}
actVal_val = {0:0,1:1}


class PredictModel:
    def __init__(self,n_class=2):
        # self.cuda = gpu_num != 0
        self.model = Dunet()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(weights_file, map_location=lambda storage, loc: storage))
        self.model = self.model.eval()

    def deal_data_img(self, image):
        '''
        获取得到（H,W,C）ndarray数组，进行处理，可以直接输入模型进行预测
        Args:
            img:

        Returns:

        '''
        img_means = (445.503477, 337.879574, 237.315930, 249.222868)
        img_stds = (129.563928, 114.740396, 121.430977, 131.762722)
        if image.shape[-1] == 4:
            img = (image - np.array(img_means).astype(np.float32)) / np.array(img_stds).astype(np.float32)
        else:
            img = image.astype('float32') / 255.
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        return img

    def deal_pred_result(self, pred_image):
        '''
        获取得到模型生成的结果，进行操作，将其修改为（H,W）数组
        Args:
            pre_img:

        Returns:

        '''
        pred = pred_image.cpu().detach().numpy()
        pred = np.squeeze(pred)
        pred = pred.astype(np.uint8)
        return pred

    #names和meta为可选参数，预留参数。函数返回类型是ndarray
    def predict(self, X: np.ndarray, names: Iterable[str] = None, meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
        # data = self.deal_data_img(X)
        img = X.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        outputs = self.model(img)
        pred = self.deal_pred_result(outputs)

        return pred

if __name__ == '__main__':
    # image = cv2.imread("./test.tif")
    image = skio.imread("./test.tif")
    #model对象在http服务中会一直存在
    model = PredictModel()
    pred = model.predict(image)
    print(pred.shape)
    cv2.imwrite('1.jpg', pred)


