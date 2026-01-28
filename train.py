#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态模型训练脚本
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import MultiModalClassifier
from src.datasets import MultiModalDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_class_mapping(metadata_path: str):
    """从元数据获取类别映射"""
    import pandas as pd
    
    train_df = pd.read_csv(metadata_path)
    species_classes = sorted(train_df['species'].unique())
    activity_classes = sorted(train_df['activity'].unique())
    
    return species_classes, activity_classes


def train_epoch(model, dataloader, criterion_species, criterion_activity, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    species_correct = 0
    activity_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        species_label = batch['species_label'].to(device)
        activity_label = batch['activity_label'].to(device)
        
        # 前向传播
        outputs = model(video, audio)
        species_logits = outputs['species_logits']
        activity_logits = outputs['activity_logits']
        
        # 计算损失
        loss_species = criterion_species(species_logits, species_label)
        loss_activity = criterion_activity(activity_logits, activity_label)
        loss = loss_species + loss_activity
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        species_pred = species_logits.argmax(dim=1)
        activity_pred = activity_logits.argmax(dim=1)
        species_correct += (species_pred == species_label).sum().item()
        activity_correct += (activity_pred == activity_label).sum().item()
        total_samples += species_label.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'sp_acc': f'{species_correct/total_samples*100:.2f}%',
            'act_acc': f'{activity_correct/total_samples*100:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    species_acc = species_correct / total_samples
    activity_acc = activity_correct / total_samples
    
    return avg_loss, species_acc, activity_acc


def validate(model, dataloader, criterion_species, criterion_activity, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    species_correct = 0
    activity_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            species_label = batch['species_label'].to(device)
            activity_label = batch['activity_label'].to(device)
            
            # 前向传播
            outputs = model(video, audio)
            species_logits = outputs['species_logits']
            activity_logits = outputs['activity_logits']
            
            # 计算损失
            loss_species = criterion_species(species_logits, species_label)
            loss_activity = criterion_activity(activity_logits, activity_label)
            loss = loss_species + loss_activity
            
            # 统计
            total_loss += loss.item()
            species_pred = species_logits.argmax(dim=1)
            activity_pred = activity_logits.argmax(dim=1)
            species_correct += (species_pred == species_label).sum().item()
            activity_correct += (activity_pred == activity_label).sum().item()
            total_samples += species_label.size(0)
    
    avg_loss = total_loss / len(dataloader)
    species_acc = species_correct / total_samples
    activity_acc = activity_correct / total_samples
    
    return avg_loss, species_acc, activity_acc


def main():
    parser = argparse.ArgumentParser(description="多模态模型训练")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设备
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 获取类别映射
    train_metadata = Path(config['paths']['raw_data']) / 'metadata' / 'train.csv'
    val_metadata = Path(config['paths']['raw_data']) / 'metadata' / 'val.csv'
    
    species_classes, activity_classes = get_class_mapping(train_metadata)
    num_species = len(species_classes)
    num_activity = len(activity_classes)
    
    logger.info(f"物种类别: {species_classes} ({num_species}类)")
    logger.info(f"行为类别: {activity_classes} ({num_activity}类)")
    
    # 数据集
    train_dataset = MultiModalDataset(
        metadata_path=str(train_metadata),
        video_dir=config['paths']['video_frames'],
        audio_dir=config['paths']['mel_spectrograms'],
        species_classes=species_classes,
        activity_classes=activity_classes
    )
    
    val_dataset = MultiModalDataset(
        metadata_path=str(val_metadata),
        video_dir=config['paths']['video_frames'],
        audio_dir=config['paths']['mel_spectrograms'],
        species_classes=species_classes,
        activity_classes=activity_classes
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training', {}).get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('dataloader', {}).get('num_workers', 4),
        pin_memory=config.get('dataloader', {}).get('pin_memory', False)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('training', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('dataloader', {}).get('num_workers', 4),
        pin_memory=config.get('dataloader', {}).get('pin_memory', False)
    )
    
    # 模型
    model = MultiModalClassifier(
        num_species_classes=num_species,
        num_activity_classes=num_activity,
        video_feature_dim=config.get('model', {}).get('video_feature_dim', 768),
        audio_feature_dim=config.get('model', {}).get('audio_feature_dim', 512),
        fusion_hidden_dim=config.get('model', {}).get('fusion_hidden_dim', 512),
        fusion_num_layers=config.get('model', {}).get('fusion_num_layers', 2),
        fusion_num_heads=config.get('model', {}).get('fusion_num_heads', 8),
        dropout=config.get('model', {}).get('dropout', 0.1),
        video_encoder_config=config.get('model', {}).get('video_encoder', {}),
        audio_encoder_config=config.get('model', {}).get('audio_encoder', {})
    ).to(device)
    
    # 损失函数
    criterion_species = nn.CrossEntropyLoss()
    criterion_activity = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training', {}).get('learning_rate', 1e-4),
        weight_decay=config.get('training', {}).get('weight_decay', 1e-5)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('training', {}).get('num_epochs', 50)
    )
    
    # 恢复训练
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.info(f"从epoch {start_epoch}恢复训练")
    
    # 训练循环
    num_epochs = config.get('training', {}).get('num_epochs', 50)
    save_dir = Path(config.get('paths', {}).get('results', './results')) / 'checkpoints'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("开始训练...")
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, train_sp_acc, train_act_acc = train_epoch(
            model, train_loader, criterion_species, criterion_activity, optimizer, device
        )
        
        # 验证
        val_loss, val_sp_acc, val_act_acc = validate(
            model, val_loader, criterion_species, criterion_activity, device
        )
        
        # 学习率调度
        scheduler.step()
        
        # 日志
        logger.info(f"Train - Loss: {train_loss:.4f}, Species Acc: {train_sp_acc*100:.2f}%, Activity Acc: {train_act_acc*100:.2f}%")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Species Acc: {val_sp_acc*100:.2f}%, Activity Acc: {val_act_acc*100:.2f}%")
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': max(best_val_acc, (val_sp_acc + val_act_acc) / 2),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_sp_acc': train_sp_acc,
            'train_act_acc': train_act_acc,
            'val_sp_acc': val_sp_acc,
            'val_act_acc': val_act_acc
        }
        
        # 保存最新检查点
        torch.save(checkpoint, save_dir / 'latest.pth')
        
        # 保存最佳模型
        current_val_acc = (val_sp_acc + val_act_acc) / 2
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(checkpoint, save_dir / 'best.pth')
            logger.info(f"保存最佳模型 (val_acc: {best_val_acc*100:.2f}%)")
    
    logger.info("训练完成！")


if __name__ == '__main__':
    main()
