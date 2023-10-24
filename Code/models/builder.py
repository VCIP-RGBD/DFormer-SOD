import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

# from engine.# import get_#

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


# # = get_#()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 256, 512]
        self.norm_layer = norm_layer
        # backbone = 'chan4_rgbdn'
        backbone = 'rgbd_base_kuan'
        decoder = 'ham'
        decoder_embed_dim = 256
       
        if backbone == 'chan4_rgbdn':
            print('4chan_rgbdn')
            self.channels=[32, 64, 128, 256]
            from .encoders.chan4_rgbdn import rgbd_n as backbone
            self.backbone = backbone()
        elif backbone == 'rgbd_base_kuan':
            self.channels = [96,192,288,576]
            from .encoders.rgbd_v6 import rgbd_base_kuan as backbone
            self.backbone = backbone()
        elif backbone == 'chan4_rgbd_s':
            print('chan4_rgbd_s')
            self.channels=[64, 128, 256, 512]
            from .encoders.chan4_rgbdn import rgbd_s as backbone
            self.backbone = backbone()
        elif backbone == 'rgbd_base_v0_1':
            print('BASE:4chan_rgbdn')
            self.channels=[64, 128, 256, 512]
            from .encoders.chan4_rgbdn import rgbd_base as backbone
            if cfg.drop_path_rate is not None:
                self.backbone = backbone(drop_path_rate=cfg.drop_path_rate)
        elif backbone == 'rgbd_base_v1_nocutmix_onmixup_0.1drop':
            print('rgbd_base_v1_nocutmix_onmixup_0.1drop')
            self.channels=[64, 180, 360, 512]
            from .encoders.chan4_rgbdn_v1 import rgbd_base as backbone
            if cfg.drop_path_rate is not None:
                self.backbone = backbone(drop_path_rate=cfg.drop_path_rate)

        elif backbone == 'RGB_Scnet_t_dual_last3_4chan':
            self.channels = [32,64,128,256]
            print('RGB_Scnet_t_dual_last3_4chan')
            from .encoders.RGB_Scnet_t_dual_last3_4chan import scnet_n as backbone
            self.backbone = backbone()
        elif backbone == 'RGB_Scnet_t_dual_last3_noFRM':
            self.channels=[32, 64, 128, 256]
            print('RGB_Scnet_t_dual_last3_noFRM')
            from .encoders.RGB_Scnet_t_dual_last3_noFRM import scnet_n as backbone 
            self.backbone = backbone()
        elif backbone == 'RGB_Scnet_t_dual_last3_RGB_FRM':
            self.channels=[32, 64, 128, 256]
            print('++++++++++++++++++++')
            print('encoder:  RGB_Scnet_t_dual_last3_RGB_FRM')
            from .encoders.RGB_Scnet_t_dual_last3_RGB_FRM import scnet_n as backbone
            self.backbone = backbone()
        else:
            #.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if decoder == 'MLPDecoder':
            #.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=decoder_embed_dim)
        
        elif decoder == 'ham':
            #.info('Using MLP Decoder')
            from .decoders.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels[1:], num_classes=1, in_index=[1,2,3],norm_cfg=dict(type='BN', requires_grad=True), channels=decoder_embed_dim)
        
        elif decoder == 'UPernet':
            #.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif decoder == 'deeplabv3+':
            #.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        elif decoder == 'nl':
            #.info('Using Decoder: nl+')
            from .decoders.nl_head import NLHead as Head
            self.decode_head = Head(in_channels=self.channels[1:], in_index=[1,2,3],num_classes=cfg.num_classes, norm_cfg=dict(type='BN', requires_grad=True),channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            #.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained='/mnt/sda/repos/2023_RGBX/RGBD-Seg-3090/pretrained/rgbd/base/rgbd_base_kuan_82_9.pth.tar')#tiny/rgbdn_cutmix.pth.tar')
    
    def init_weights(self, cfg=None, pretrained=None):
        if pretrained:
            #.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        #.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, 1e-3, 0.1,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, 1e-3, 0.1,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x=None, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        # print(out.shape,label.shape)
        # print(torch.max(label),torch.min(label))
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss
        return out