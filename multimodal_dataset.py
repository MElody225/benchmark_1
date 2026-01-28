#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态数据集加载器
支持加载视频帧和音频Mel频谱图
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """多模态数据集：视频帧 + 音频Mel频谱图"""
    
    def __init__(
        self,
        metadata_path: str,
        video_dir: str,
        audio_dir: str,
        species_classes: List[str],
        activity_classes: List[str],
        transform=None,
        audio_transform=None
    ):
        """
        参数:
            metadata_path: CSV元数据文件路径
            video_dir: 视频帧文件目录（.pt文件）
            audio_dir: 音频Mel频谱图目录（.npy文件）
            species_classes: 物种类别列表
            activity_classes: 行为类别列表
            transform: 视频数据增强
            audio_transform: 音频数据增强
        """
        self.video_dir = Path(video_dir)
        self.audio_dir = Path(audio_dir)
        self.transform = transform
        self.audio_transform = audio_transform
        
        # 加载元数据
        self.metadata = pd.read_csv(metadata_path)
        
        # 类别映射
        self.species_classes = species_classes
        self.activity_classes = activity_classes
        self.species_to_idx = {cls: idx for idx, cls in enumerate(species_classes)}
        self.activity_to_idx = {cls: idx for idx, cls in enumerate(activity_classes)}
        
        # 构建文件路径映射
        self._build_file_mapping()
        
        logger.info(f"数据集加载完成: {len(self.metadata)} 个样本")
        logger.info(f"物种类别数: {len(species_classes)}, 行为类别数: {len(activity_classes)}")
    
    def _build_file_mapping(self):
        """构建视频和音频文件路径映射"""
        # 从video_path构建文件名
        # 例如: S1_C3_E144_V0060_ID1_T1/S1_C3_E144_V0060_ID1_T1_c0.mp4
        # -> S1_C3_E144_V0060_ID1_T1_S1_C3_E144_V0060_ID1_T1_c0.pt
        
        self.video_files = []
        self.audio_files = []
        self.species_labels = []
        self.activity_labels = []
        
        valid_indices = []
        
        for idx, row in self.metadata.iterrows():
            video_path_str = row['video_path']
            
            # 构建文件名（将路径分隔符替换为下划线）
            safe_name = video_path_str.replace('/', '_').replace('\\', '_')
            video_filename = safe_name.replace('.mp4', '.pt')
            audio_filename = safe_name.replace('.mp4', '.npy')
            
            video_file = self.video_dir / video_filename
            audio_file = self.audio_dir / audio_filename
            
            # 检查文件是否存在
            if video_file.exists() and audio_file.exists():
                self.video_files.append(video_file)
                self.audio_files.append(audio_file)
                
                # 标签编码
                species = row['species']
                activity = row['activity']
                
                self.species_labels.append(self.species_to_idx.get(species, 0))
                self.activity_labels.append(self.activity_to_idx.get(activity, 0))
                
                valid_indices.append(idx)
            else:
                if not video_file.exists():
                    logger.warning(f"视频文件不存在: {video_file}")
                if not audio_file.exists():
                    logger.warning(f"音频文件不存在: {audio_file}")
        
        # 过滤元数据
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        
        logger.info(f"有效样本数: {len(self.video_files)}")
    
    def __len__(self) -> int:
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个样本
        
        返回:
            {
                'video': (C, T, H, W) 视频帧张量
                'audio': (C, H, W) 音频Mel频谱图
                'species_label': 物种标签
                'activity_label': 行为标签
            }
        """
        # 加载视频帧
        video_tensor = torch.load(self.video_files[idx])
        # video_tensor shape: (C, T, H, W)
        
        # 加载音频Mel频谱图
        audio_spec = np.load(self.audio_files[idx])
        # audio_spec shape: (n_mels, n_frames) -> 需要转换为 (C, H, W)
        
        # 转换为torch tensor并添加通道维度
        if len(audio_spec.shape) == 2:
            # (H, W) -> (1, H, W)
            audio_tensor = torch.from_numpy(audio_spec).float().unsqueeze(0)
        else:
            audio_tensor = torch.from_numpy(audio_spec).float()
        
        # 数据增强
        if self.transform is not None:
            video_tensor = self.transform(video_tensor)
        
        if self.audio_transform is not None:
            audio_tensor = self.audio_transform(audio_tensor)
        
        return {
            'video': video_tensor,
            'audio': audio_tensor,
            'species_label': torch.tensor(self.species_labels[idx], dtype=torch.long),
            'activity_label': torch.tensor(self.activity_labels[idx], dtype=torch.long)
        }
