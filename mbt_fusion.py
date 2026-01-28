#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBT (Multimodal Bottleneck Transformer) 多模态融合模块
基于Transformer的跨模态注意力机制
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        参数:
            query: (B, N_q, d_model)
            key: (B, N_k, d_model)
            value: (B, N_v, d_model)
        
        返回:
            output: (B, N_q, d_model)
        """
        B = query.size(0)
        
        # 线性变换并分头
        Q = self.w_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, N_q, d_k)
        K = self.w_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, N_k, d_k)
        V = self.w_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, N_v, d_k)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, N_q, N_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        output = torch.matmul(attn_weights, V)  # (B, H, N_q, d_k)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # (B, N_q, d_model)
        output = self.w_o(output)  # (B, N_q, d_model)
        
        return output


class MBTFusion(nn.Module):
    """MBT多模态融合模块"""
    
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        参数:
            video_dim: 视频特征维度
            audio_dim: 音频特征维度
            hidden_dim: 融合后的隐藏维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        
        # 投影层：将不同模态的特征投影到相同维度
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # 模态嵌入（区分视频和音频）
        self.video_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.audio_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.hidden_dim = hidden_dim
        logger.info(f"MBT融合模块初始化: video_dim={video_dim}, audio_dim={audio_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, video_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        """
        融合视频和音频特征
        
        参数:
            video_feat: (B, video_dim) 视频特征
            audio_feat: (B, audio_dim) 音频特征
        
        返回:
            fused_feat: (B, hidden_dim) 融合后的特征
        """
        B = video_feat.size(0)
        
        # 投影到相同维度
        video_proj = self.video_proj(video_feat).unsqueeze(1)  # (B, 1, hidden_dim)
        audio_proj = self.audio_proj(audio_feat).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # 添加模态token
        video_token = self.video_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        audio_token = self.audio_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        
        # 拼接：video_token + video_feat + audio_token + audio_feat
        # 或者：video_feat + audio_feat
        sequence = torch.cat([video_proj, audio_proj], dim=1)  # (B, 2, hidden_dim)
        
        # Transformer编码
        fused = self.transformer(sequence)  # (B, 2, hidden_dim)
        
        # 使用平均池化或第一个token
        fused_feat = fused.mean(dim=1)  # (B, hidden_dim)
        
        # LayerNorm
        fused_feat = self.norm(fused_feat)
        
        return fused_feat
