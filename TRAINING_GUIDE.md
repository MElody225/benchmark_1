# 多模态模型训练指南

## 📋 概述

本项目实现了基于VideoMAE + ResNet + MBT融合的多模态野生动物行为识别模型。

### 模型架构
- **视频编码器**: VideoMAE (预训练) → 提取视频特征
- **音频编码器**: ResNet18 (预训练) → 处理Mel频谱图
- **融合模块**: MBT (Multimodal Bottleneck Transformer) → 跨模态特征融合
- **分类头**: 双分类头（物种分类 + 行为分类）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**重要依赖**:
- `transformers>=4.20.0` - VideoMAE模型
- `torch>=1.10.0` - PyTorch
- `torchvision>=0.11.0` - ResNet模型

### 2. 数据准备

确保已完成数据预处理：
- ✅ 视频帧文件：`data/processed/video_frames/*.pt`
- ✅ 音频Mel频谱图：`data/processed/mel_spectrograms/*.npy`
- ✅ 元数据文件：`data/raw/metadata/train.csv`, `val.csv`, `test.csv`

### 3. 配置模型

编辑 `config.yaml` 中的模型配置：

```yaml
model:
  video_feature_dim: 768      # 视频特征维度
  audio_feature_dim: 512      # 音频特征维度
  fusion_hidden_dim: 512      # 融合模块隐藏维度
  fusion_num_layers: 2        # Transformer层数
  fusion_num_heads: 8          # 注意力头数
  dropout: 0.1
  video_encoder:
    model_name: "MCG-NJU/videomae-base-finetuned-kinetics"
    pretrained: true
    freeze_backbone: false
  audio_encoder:
    model_name: "resnet18"
    pretrained: true
    freeze_backbone: false
    input_channels: 1

training:
  num_epochs: 50
  batch_size: 8               # 根据GPU内存调整
  learning_rate: 1e-4
  weight_decay: 1e-5
```

### 4. 开始训练

```bash
python src/training/train.py --config config.yaml
```

**使用GPU**:
```bash
python src/training/train.py --config config.yaml --device cuda
```

**恢复训练**:
```bash
python src/training/train.py --config config.yaml --resume results/checkpoints/latest.pth
```

## 📊 训练输出

训练过程中会生成：
- **日志文件**: `training_YYYYMMDD_HHMMSS.log`
- **检查点**: `results/checkpoints/latest.pth` (最新)
- **最佳模型**: `results/checkpoints/best.pth` (验证集最佳)

## 🔧 模型组件说明

### 1. VideoMAEEncoder (`src/models/video_encoder.py`)
- 使用预训练的VideoMAE模型提取视频特征
- 输入: `(B, C, T, H, W)` 视频帧张量
- 输出: `(B, feature_dim)` 视频特征

### 2. ResNetAudioEncoder (`src/models/audio_encoder.py`)
- 使用ResNet处理Mel频谱图
- 输入: `(B, 1, H, W)` Mel频谱图
- 输出: `(B, feature_dim)` 音频特征

### 3. MBTFusion (`src/models/mbt_fusion.py`)
- 基于Transformer的多模态融合
- 使用跨模态注意力机制融合视频和音频特征
- 输出: `(B, hidden_dim)` 融合特征

### 4. MultiModalClassifier (`src/models/multimodal_model.py`)
- 完整的端到端模型
- 包含两个分类头：物种分类和行为分类

## 📈 性能优化建议

### 内存优化
1. **减小batch_size**: 如果GPU内存不足，减小`batch_size`（默认8）
2. **冻结backbone**: 设置`freeze_backbone: true`减少内存占用
3. **梯度累积**: 可以修改训练脚本实现梯度累积

### 速度优化
1. **多进程加载**: 增加`dataloader.num_workers`
2. **混合精度训练**: 可以添加`torch.cuda.amp`支持
3. **减少融合层数**: 减小`fusion_num_layers`

## 🐛 常见问题

### Q1: VideoMAE模型下载失败
**A**: 如果无法从HuggingFace下载，代码会自动使用简化的3D CNN实现。也可以手动下载模型权重。

### Q2: 内存不足 (OOM)
**A**: 
- 减小`batch_size`到4或2
- 设置`freeze_backbone: true`
- 使用CPU模式（虽然会很慢）

### Q3: 训练速度慢
**A**:
- 确保使用GPU (`--device cuda`)
- 增加`num_workers`
- 考虑使用更小的模型（如resnet18）

### Q4: 类别数量不匹配
**A**: 代码会自动从metadata中读取实际类别，无需手动配置。

## 📝 下一步

训练完成后，可以：
1. 在测试集上评估模型
2. 可视化注意力权重
3. 分析不同模态的贡献
4. 进行模型蒸馏或量化

## 🔗 参考

- VideoMAE: https://github.com/MCG-NJU/VideoMAE
- MBT: Multimodal Bottleneck Transformer
- ResNet: https://pytorch.org/vision/stable/models.html
