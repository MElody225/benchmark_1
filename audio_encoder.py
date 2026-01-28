#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet音频特征提取器
处理Mel频谱图，提取音频特征
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ResNetAudioEncoder(nn.Module):
    """ResNet音频编码器（处理Mel频谱图）"""
    
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_dim: int = 512,
        input_channels: int = 1
    ):
        """
        参数:
            model_name: ResNet模型名称 ('resnet18', 'resnet34', 'resnet50')
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone
            feature_dim: 输出特征维度
            input_channels: 输入通道数（Mel频谱图通常是1通道）
        """
        super().__init__()
        
        # 加载ResNet模型
        if model_name == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif model_name == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif model_name == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"不支持的ResNet模型: {model_name}")
        
        # 修改第一层以接受单通道输入（Mel频谱图）
        if input_channels != 3:
            # 将第一层从3通道改为1通道
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # 如果使用预训练权重，初始化新层
            if pretrained:
                # 将3通道权重的平均值作为1通道权重
                with torch.no_grad():
                    resnet.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # 移除最后的分类层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 冻结backbone（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("ResNet backbone已冻结")
        
        # 特征投影层
        if backbone_dim != feature_dim:
            self.projection = nn.Linear(backbone_dim, feature_dim)
        else:
            self.projection = nn.Identity()
        
        self.feature_dim = feature_dim
        logger.info(f"ResNet音频编码器初始化完成: {model_name}, 输出维度={feature_dim}")
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        提取音频特征
        
        参数:
            audio: (B, C, H, W) Mel频谱图，C通常是1
        
        返回:
            features: (B, feature_dim) 音频特征
        """
        # ResNet提取特征
        x = self.backbone(audio)  # (B, backbone_dim, 1, 1)
        
        # 展平
        x = x.view(x.size(0), -1)  # (B, backbone_dim)
        
        # 投影到目标维度
        features = self.projection(x)  # (B, feature_dim)
        
        return features
