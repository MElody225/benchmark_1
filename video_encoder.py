#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoMAE视频特征提取器
使用预训练的VideoMAE模型提取视频特征
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VideoMAEEncoder(nn.Module):
    """VideoMAE视频编码器"""
    
    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_dim: int = 768
    ):
        """
        参数:
            model_name: VideoMAE模型名称
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone
            feature_dim: 输出特征维度
        """
        super().__init__()
        
        try:
            from transformers import VideoMAEModel, VideoMAEConfig
            
            if pretrained:
                logger.info(f"加载预训练VideoMAE模型: {model_name}")
                self.model = VideoMAEModel.from_pretrained(model_name)
            else:
                logger.info("使用随机初始化的VideoMAE模型")
                config = VideoMAEConfig()
                self.model = VideoMAEModel(config)
            
            self.feature_dim = feature_dim
            
            # 冻结backbone（可选）
            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info("VideoMAE backbone已冻结")
            
            # 获取隐藏层维度
            self.hidden_dim = self.model.config.hidden_size
            
            # 如果特征维度不匹配，添加投影层
            if self.hidden_dim != feature_dim:
                self.projection = nn.Linear(self.hidden_dim, feature_dim)
            else:
                self.projection = nn.Identity()
                
        except ImportError:
            logger.warning("transformers库未安装，使用简化的VideoMAE实现")
            # 如果transformers不可用，使用简化的实现
            self.model = None
            self.hidden_dim = 768
            self.feature_dim = feature_dim
            self.projection = nn.Linear(self.hidden_dim, feature_dim)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        提取视频特征
        
        参数:
            video: (B, C, T, H, W) 视频张量，已归一化
        
        返回:
            features: (B, feature_dim) 视频特征
        """
        if self.model is None:
            # 简化实现：使用3D CNN
            return self._simple_video_encoder(video)
        
        # VideoMAE期望输入格式: (B, T, C, H, W)
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        
        # 调整到VideoMAE期望的输入尺寸
        # VideoMAE通常期望224x224
        if H != 224 or W != 224:
            video = nn.functional.interpolate(
                video.reshape(B * T, C, H, W),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).reshape(B, T, C, 224, 224)
        
        # 转换为像素值 [0, 1] -> [0, 255]
        video = (video * 255).long()
        
        # 提取特征
        outputs = self.model(pixel_values=video)
        
        # 使用[CLS] token或平均池化
        # VideoMAE输出: last_hidden_state (B, T, hidden_dim)
        if hasattr(outputs, 'last_hidden_state'):
            # 使用时间维度的平均池化
            features = outputs.last_hidden_state.mean(dim=1)  # (B, hidden_dim)
        else:
            # 使用pooler_output
            features = outputs.pooler_output  # (B, hidden_dim)
        
        # 投影到目标维度
        features = self.projection(features)  # (B, feature_dim)
        
        return features
    
    def _simple_video_encoder(self, video: torch.Tensor) -> torch.Tensor:
        """
        简化的视频编码器（当transformers不可用时）
        使用3D ResNet作为backbone
        """
        B, C, T, H, W = video.shape
        
        # 简单的3D CNN特征提取
        # 这里使用一个简化的3D ResNet-like结构
        x = video
        
        # 3D卷积层
        conv1 = nn.Conv3d(C, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        x = nn.functional.relu(conv1(x))  # (B, 64, T, H/2, W/2)
        
        # 时间维度池化
        x = x.mean(dim=2)  # (B, 64, H/2, W/2)
        
        # 2D ResNet-like特征提取
        # 这里简化处理，实际可以使用预训练的ResNet
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # (B, 64, 1, 1)
        x = x.view(B, -1)  # (B, 64)
        
        # 投影到目标维度
        features = self.projection(x)  # (B, feature_dim)
        
        return features
