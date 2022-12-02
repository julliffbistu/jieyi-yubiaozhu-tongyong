import json
import os
#os.system("pip uninstall opencv-python-headless==4.5.4.58")
#os.system("pip install cython matplotlib opencv-python==4.5.1.48 ")
#os.system("pip install opencv-python-headless")
#os.system("pip install opencv-contrib-python-headless")
os.system("pip install ninja ")
os.system("sh requirment.sh")
import torch
import numpy as np

#from nets.dunet import Dunet
from mmseg.apis import init_segmentor

from Predict_utils import prediction_image

# 设置参数
def parse_platform_arguments():
    platform_arguments = {
        'estimate_id': 'dtgthrth',  # 任务ID
        'image_path': None,  # 训练图片路径
        'device_ids': None,  # gpu device id 数
        'network_type': None,  # 预测的类别
        'load_size': None,  # 图片加载大小
        'value_name_map': None,  # 训练值对应英文名称
        'value_title_map': None,  # 真实value对应类别中文名称
        'name_title_map': None,  # 类别英文名称 对应类别中文名称
        'value_color': None,  # 真实value对应 rgb()颜色
        'background_value': None,  # 背景对应的训练值
        'class_name_list': None,  # 除背景外的类别列表
        'load_model': load_model,  # 模型加载方法
        'deal_data_img': deal_data_img,  # 处理进入到模型里面的数据方法
        'deal_pred_result': deal_pred_result,  # 处理模型预测后的结果方法
        'nms': 0.3,  # 目标识别
    }
    basedir = os.getcwd()
    config_path = basedir + "/traincfg.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    network_type = int(params['network_type'])
    load_size = params["load_size"].split(',')
    load_size[0] = round(float(load_size[0]) / 32) * 32
    load_size[1] = round(float(load_size[1]) / 32) * 32
    if load_size[1] > load_size[0]:
        load_size[1] = load_size[0]
    else:
        load_size[0] = load_size[1]
    platform_arguments['device_ids'] = []
    gpu_num = int(params["gpu_num"])
    if gpu_num != 0:
        for i in range(0, gpu_num):
            platform_arguments['device_ids'].append(i)
    platform_arguments['network_type'] = network_type
    platform_arguments['load_size'] = load_size
    labels = params['label']
    labels = eval(labels)
    value_name_map = {}
    value_title_map = {}
    name_title_map = {}
    value_color = {}
    class_list = []
    platform_arguments['background_value'] = None

    for i, val in enumerate(labels):
        for key in val.keys():
            if key == 'class_title':
                title_value = val[key]
            elif key == 'class_name':
                class_value = val[key]
                if class_value == 'background':
                    continue
                else:
                    class_list.append(class_value)
            elif key == "class_value":
                value = int(val[key])
                trainValue = int(i)
            elif key == "class_color":
                color_ = val[key]
                rgb = tuple(list(map(int, color_.split(','))))

        if rgb and value >= 0:
            value_name_map[trainValue] = class_value
            value_title_map[trainValue] = title_value
            name_title_map[class_value] = title_value
            if class_value == 'background':
                platform_arguments['background_value'] = trainValue
            value_color[trainValue] = 'rgb' + str(rgb)

    platform_arguments['value_name_map'] = value_name_map
    # 真实value对应类别中文名称
    platform_arguments['value_title_map'] = value_title_map
    # 类别英文名称 对应类别中文名称
    platform_arguments['name_title_map'] = name_title_map
    # 真实value对应 rgb()颜色
    platform_arguments['value_color'] = value_color
    # 背景的训练值

    platform_arguments['class_name_list'] = class_list
    conf = input()
    params_estimate = json.loads(conf)
    platform_arguments['estimate_id'] = params_estimate['estimate_id']
    platform_arguments['image_path'] = params_estimate["image_path"]
    return platform_arguments
def get_def_user():
    platform_arguments = {
        'load_model': load_model,  # 模型加载方法
        'deal_data_img': deal_data_img,  # 处理进入到模型里面的数据方法
        'deal_pred_result': deal_pred_result  # 处理模型预测后的结果方法
    }
    return platform_arguments

# 加载权重文件
def load_model():
    '''
        加载模型
    Returns:

    '''

    basedir = os.path.abspath(os.path.dirname(__file__))
    weight_root = basedir + '/weight/'
    checkpoint_file = weight_root + 'latest.pth'
    config_file = r"configs/upernet/VAN/upernet_van_large_512x512_160k_ade20k.py"
    #checkpoint_file = r"tools/rsb128/latest.pth"

    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    return model


def deal_data_img(image):
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

def deal_pred_result(pred_image):
    '''
    获取得到模型生成的结果，进行操作，将其修改为（H,W）数组
    Args:
        pre_img:

    Returns:

    '''
    # pred = torch.sigmoid(pred_image).cpu().detach().numpy()
    # pred = np.squeeze(pred)
    # pred = pred.astype(np.uint8)
    return pred_image


if __name__ == '__main__':
    platform_arguments = parse_platform_arguments()
    prediction_image(**platform_arguments)