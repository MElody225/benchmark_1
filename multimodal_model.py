#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态模型：VideoMAE + ResNet + MBT融合 + 双分类头
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

from .video_encoder import VideoMAEEncoder
from .audio_encoder import ResNetAudioEncoder
from .mbt_fusion import MBTFusion

logger = logging.getLogger(__name__)


class MultiModalClassifier(nn.Module):
    """多模态分类模型"""
    
    def __init__(
        self,
        num_species_classes: int,
        num_activity_classes: int,
        video_feature_dim: int = 768,
        audio_feature_dim: int = 512,
        fusion_hidden_dim: int = 512,
        fusion_num_layers: int = 2,
        fusion_num_heads: int = 8,
        dropout: float = 0.1,
        video_encoder_config: Optional[Dict] = None,
        audio_encoder_config: Optional[Dict] = None
    ):
        """
        参数:
            num_species_classes: 物种类别数
            num_activity_classes: 行为类别数
            video_feature_dim: 视频特征维度
            audio_feature_dim: 音频特征维度
            fusion_hidden_dim: 融合模块隐藏维度
            fusion_num_layers: 融合Transformer层数
            fusion_num_heads: 注意力头数
            dropout: Dropout率
            video_encoder_config: 视频编码器配置
            audio_encoder_config: 音频编码器配置
        """
        super().__init__()
        
        # 视频编码器
        video_config = video_encoder_config or {}
        self.video_encoder = VideoMAEEncoder(
            model_name=video_config.get('model_name', 'MCG-NJU/videomae-base-finetuned-kinetics'),
            pretrained=video_config.get('pretrained', True),
            freeze_backbone=video_config.get('freeze_backbone', False),
            feature_dim=video_feature_dim
        )
        
        # 音频编码器
        audio_config = audio_encoder_config or {}
        self.audio_encoder = ResNetAudioEncoder(
            model_name=audio_config.get('model_name', 'resnet18'),
            pretrained=audio_config.get('pretrained', True),
            freeze_backbone=audio_config.get('freeze_backbone', False),
            feature_dim=audio_feature_dim,
            input_channels=audio_config.get('input_channels', 1)
        )
        
        # MBT融合模块
        self.fusion = MBTFusion(
            video_dim=video_feature_dim,
            audio_dim=audio_feature_dim,
            hidden_dim=fusion_hidden_dim,
            num_layers=fusion_num_layers,
            num_heads=fusion_num_heads,
            dropout=dropout
        )
        
        # 分类头
        self.species_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_species_classes)
        )
        
        self.activity_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_activity_classes)
        )
        
        logger.info(f"多模态模型初始化完成:")
        logger.info(f"  - 物种类别数: {num_species_classes}")
        logger.info(f"  - 行为类别数: {num_activity_classes}")
        logger.info(f"  - 视频特征维度: {video_feature_dim}")
        logger.info(f"  - 音频特征维度: {audio_feature_dim}")
        logger.info(f"  - 融合隐藏维度: {fusion_hidden_dim}")
    
    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            video: (B, C, T, H, W) 视频帧
            audio: (B, C, H, W) 音频Mel频谱图
        
        返回:
            {
                'species_logits': (B, num_species_classes)
                'activity_logits': (B, num_activity_classes)
                'video_feat': (B, video_feature_dim)
                'audio_feat': (B, audio_feature_dim)
                'fused_feat': (B, fusion_hidden_dim)
            }
        """
        # 提取特征
        video_feat = self.video_encoder(video)  # (B, video_feature_dim)
        audio_feat = self.audio_encoder(audio)  # (B, audio_feature_dim)
        
        # 特征融合
        fused_feat = self.fusion(video_feat, audio_feat)  # (B, fusion_hidden_dim)
        
        # 分类
        species_logits = self.species_classifier(fused_feat)  # (B, num_species_classes)
        activity_logits = self.activity_classifier(fused_feat)  # (B, num_activity_classes)
        
        return {
            'species_logits': species_logits,
            'activity_logits': activity_logits,
            'video_feat': video_feat,
            'audio_feat': audio_feat,
            'fused_feat': fused_feat
        }
