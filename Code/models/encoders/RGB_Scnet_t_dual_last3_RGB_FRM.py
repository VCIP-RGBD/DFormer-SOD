# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from mmcv.utils import to_2tuple
from ..net_utils import FeatureFusionModule as FFM

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights

#FRM
class FRM_l(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FRM_l, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        # out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1 





# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
import math

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # self.norm = nn.BatchNorm2d(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x


class MemoryModulev3_230427(nn.Module):
    def __init__(self, dim, num_head=8, window=7, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.q = nn.Linear(dim, dim)
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        

        self.proj = nn.Linear(dim, dim)
        if window != 0:
            self.kv = nn.Linear(dim, dim)
            self.m = nn.Parameter(torch.zeros(1, window, window, dim // 2), requires_grad=True)
            self.proj = nn.Linear(dim // 2 * 3, dim)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")

    def forward(self, x):
        B, H, W, C = x.size()
        x = self.norm(x)

        q = self.q(x)        
        x = self.l(x).permute(0, 3, 1, 2)
        x = self.act(x)
            
        a = self.conv(x)
        a = a.permute(0, 2, 3, 1)
        a = self.a(a)

        if self.window != 0:
            b = x.permute(0, 2, 3, 1)
            kv = self.kv(b)
            kv = kv.reshape(B, H*W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

            m = self.m.reshape(1, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3).expand(B, -1, -1, -1)
            attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1) 
            attn = attn.softmax(dim=-1)
            # print(attn.shape,'============================')
            # to do: visualize
            attn = (attn @ v).reshape(B, self.num_head, self.window, self.window, C // self.num_head // 2).permute(0, 1, 4, 2, 3).reshape(B, C // 2, self.window, self.window)
            attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)#bilinaer
            # print(attn.shape)
        
        x = q * a

        if self.window != 0:
            x = torch.cat([x, attn], dim=3)
        x = self.proj(x)

        return x

class Block(nn.Module):
    def __init__(self, index, dim, num_head, norm_cfg=dict(type='SyncBN', requires_grad=True), mlp_ratio=4.,block_index=0, last_block_index=50, window=7, dropout_layer=None):
        super().__init__()
        
        self.index = index
        layer_scale_init_value = 1e-6  
        if block_index>last_block_index:
            window=0 
        self.attn = MemoryModulev3_230427(dim, num_head, window=window, norm_cfg=norm_cfg)
        self.mlp = MLP(dim, mlp_ratio, norm_cfg=norm_cfg)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
                 
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.dropout_layer(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.attn(x))
        x = x + self.dropout_layer(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x))
        return x


class SCNet(BaseModule):
    def __init__(self, in_channels=3, depths=(2, 2, 8, 2), dims=(32, 64, 128, 256), out_indices=(0,1, 2, 3), windows=[7, 7, 7, 7], norm_cfg=dict(type='SyncBN', requires_grad=True),
                 mlp_ratios=[8, 8, 4, 4], num_heads=(2, 4, 10, 16),last_block=[50,50,50,50], drop_path_rate=0.1, init_cfg=None):
        super().__init__()
        self.depths = depths
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.e_downsample_layers = nn.ModuleList()
        self.depth = depths
        stem = nn.Sequential( 
                nn.Conv2d(in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1),
                build_norm_layer(norm_cfg, dims[0] // 2)[1],
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                build_norm_layer(norm_cfg, dims[0])[1],
                # nn.Conv2d(dims[0] // 2, dims[0]//2, kernel_size=3, stride=2, padding=1),
                # build_norm_layer(norm_cfg, dims[0]//2)[1],
                # nn.GELU(),
                # nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
        )

        e_stem = nn.Sequential( 
                nn.Conv2d(in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1),
                build_norm_layer(norm_cfg, dims[0] // 2)[1],
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.downsample_layers.append(stem)
        self.e_downsample_layers.append(e_stem)
        for i in range(len(dims)-1):
            stride = 2
            downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, dims[i])[1],
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=stride, padding=1),
            )
            e_downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, dims[i])[1],
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers.append(downsample_layer)
            self.e_downsample_layers.append(e_downsample_layer)


        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.e_stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(index=cur+j, 
                        dim=dims[i], 
                        window=windows[i],
                        dropout_layer=dict(type='DropPath', drop_prob=dp_rates[cur + j]), 
                        num_head=num_heads[i], 
                        norm_cfg=norm_cfg,
                        block_index=depths[i]-j,
                        last_block_index=last_block[i],
                        mlp_ratio=mlp_ratios[i]) for j in range(depths[i])]
            )
            e_stage = nn.Sequential(
                *[Block(index=cur+j, 
                        dim=dims[i], 
                        window=windows[i],
                        dropout_layer=dict(type='DropPath', drop_prob=dp_rates[cur + j]), 
                        num_head=num_heads[i], 
                        norm_cfg=norm_cfg,
                        block_index=depths[i]-j,
                        last_block_index=last_block[i],
                        mlp_ratio=mlp_ratios[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            self.e_stages.append(e_stage)
            cur += depths[i]

        # Add a norm layer for each output
        for i in out_indices:
            layer = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

            e_layer = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            e_layer_name = f'e_norm{i}'
            self.add_module(e_layer_name, e_layer)

        self.FRMs = nn.ModuleList([
                    FRM_l(dim=dims[0], reduction=1),
                    FRM_l(dim=dims[1], reduction=1),
                    FRM_l(dim=dims[2], reduction=1),
                    FRM_l(dim=dims[3], reduction=1)])

        self.FFMs = nn.ModuleList([
                    FFM(dim=dims[0], reduction=1, num_heads=num_heads[0], norm_layer=nn.SyncBatchNorm),
                    FFM(dim=dims[1], reduction=1, num_heads=num_heads[1], norm_layer=nn.SyncBatchNorm),
                    FFM(dim=dims[2], reduction=1, num_heads=num_heads[2], norm_layer=nn.SyncBatchNorm),
                    FFM(dim=dims[3], reduction=1, num_heads=num_heads[3], norm_layer=nn.SyncBatchNorm)])


        # self.apply(self.init_weights)

    def init_weights(self,pretrained):
       
        _state_dict=torch.load(pretrained)['state_dict']

        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith('backbone.'):
                state_dict[k[9:]] = v
            else:
                state_dict[k] = v

        # strip prefix of state_dict
        print('--------------')
        print(state_dict.keys())
        dual_state_dict={}
        # if list(state_dict.keys())[0].startswith('module.'):
        for k, v in state_dict.items():
            dual_state_dict[k[:].replace('module.','')] =  v
            dual_state_dict['e_'+k[:].replace('module.','')] =  v
        print('--------------')
        print(dual_state_dict.keys())
        # load state_dict
        load_state_dict(self, dual_state_dict, strict=False)

    def forward(self, x,x_e):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = x.permute(0, 2, 3, 1) 
            

            x_e = self.e_downsample_layers[i](x_e)
            x_e = x_e.permute(0, 2, 3, 1)

            for j in range(self.depth[i]):
                x = self.stages[i][j](x)  
                x_e = self.e_stages[i][j](x_e)   
            x = x.permute(0, 3, 1, 2)
            x_e = x_e.permute(0, 3, 1, 2)
            if i==0:
                outs.append(x)
                continue

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                e_norm_layer = getattr(self, f'e_norm{i}')
                x = norm_layer(x)
                x_e=e_norm_layer(x_e)
                x = self.FRMs[i](x, x_e)
                
                x_fused = self.FFMs[i](x, x_e)
                outs.append(x_fused)

        return outs

def scnet_n(pretrained=False, **kwargs):   # 81.5
    model = SCNet(dims=[32, 64, 128, 256], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)
   
    if pretrained:
        model = load_model_weights(model, 'scnet', kwargs)
    return model