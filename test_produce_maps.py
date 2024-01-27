import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import os, argparse
import cv2
# from Code.lib.model import SPNet
from Code.models.builder import EncoderDecoder as segmodel
from Code.utils.data import test_dataset
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id',   type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str, default='./Data/TestDataset/',help='test dataset path')
parser.add_argument('--checkpoint',type=str, default='Checkpoint/trained/DFormer_SOD_epoch_best.pth',help='checkpoint path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
 

#load the model
model = segmodel(is_test=True)
model.cuda()

model.load_state_dict(torch.load(opt.checkpoint),strict=True)
model.eval()
#test
test_datasets = ['NJU2K','NLPR', 'DES', 'SSD','SIP', 'STERE'] 

# test_datasets = ['STERE'] 


for dataset in test_datasets:
    save_path = './test_maps/DFormer-SOD/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    image_root  = dataset_path + dataset + '/RGB/'
    gt_root     = dataset_path + dataset + '/GT/'
    depth_root  = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    print('test dataset: ',dataset)
    for i in tqdm(range(test_loader.size), dynamic_ncols=True):
        image, gt,depth, name, image_for_post = test_loader.load_data()
        
        gt      = np.asarray(gt, np.float32)
        gt     /= (gt.max() + 1e-8)
        image   = image.cuda()
        depth   = depth.cuda()
        pre_res = model(image,depth)
        res     = pre_res  
        res     = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res     = res.sigmoid().data.cpu().numpy().squeeze()
        res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        # print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name,res*255)
    print('Test Done!')
