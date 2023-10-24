import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel



from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
import torch
from torchvision.models import resnet18
import time
# from Code.lib.model import SPNet
# from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from Code.models.builder import EncoderDecoder as segmodel
import torch
# from torchvision.models import AlexNet
# from torchviz import make_dot

if __name__ == '__main__':
    
    BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel()
    device = torch.device('cuda:0')
    model.eval()
    model.to(device)
    dump_input = torch.ones(1,3,352,352).to(device)


    # for i in tqdm(range(2000)):
    #     if i==50:
    #         start = time.time()
    #     outputs = model(dump_input,dump_input)
    #     torch.cuda.synchronize()
    #     if i==1999:
    #         end = time.time()
    #         print('Time:{}ms'.format((end-start)*1000/1950))

    from thop import profile
    # input = torch.randn(1,3,480,640)
    # print(model)

    input_shape = (1,3, 480, 640)
    flops,params = profile(model,inputs=(dump_input,dump_input[:,1,:,:].unsqueeze(1)))
    print('the flops is {}G,the params is {}M'.format(round(flops/(10**9),2), round(params/(10**6),2))) # 4111514624.0 25557032.0 res50

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    






    